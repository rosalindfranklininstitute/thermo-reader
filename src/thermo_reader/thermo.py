# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NamedTuple, Literal
from enum import Enum
from dataclasses import dataclass
import math

import numpy as np
import random

from ms_nexus_tools.api import (
    image_plot as nxtic,
    spectrum_plot as nxts,
    image_and_spectrum_plot as nxisp,
    kendrick_mass_defect_plot as nxkdm,
)
from ms_nexus_tools.api.args import arg_field, ArgType, ConfigFileArgs
from ms_nexus_tools.api.formula_args import FormulaArgs
from ms_nexus_tools.api.mass_range_args import MassRangeArgs
from ms_nexus_tools.api.image_args import (
    LayerSliceArgs,
    WidthAndHeightSliceArgs,
    MassSliceArgs,
)

from ms_nexus_tools.lib.bounds import ContainedBounds, Chunk, Shape
from ms_nexus_tools.lib.filter import Filter, TotalImages, MassRangeTotalImage
from ms_nexus_tools.lib.nxs import (
    NexusFile,
    create_group,
    GenericAxis,
    Axis,
    create_chunked_subentry,
)


from pathlib import Path
import re

import matplotlib.pyplot as plt

from .load_thermo import (
    RawFileReaderAdapter,
    Device,
    DataUnits,
    GetSpectrum,
    IScanFilter,
    IRawDataExtended,
    ScanStatistics,
)

from icecream import ic


class PixelMetric(Enum):
    TIME = "time"
    DISTANCE = "distance"


class TimeBounds(Enum):
    SUBSET = "subset"
    SUPERSET = "superset"


@dataclass
class ProcessArgs(
    ConfigFileArgs,
    MassSliceArgs,
    WidthAndHeightSliceArgs,
    LayerSliceArgs,
    FormulaArgs,
    MassRangeArgs,
):
    in_path: Path = arg_field(
        "-d",
        "--directory",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The input directory.",
        default=None,
    )
    nxs_out_path: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The output file.",
        default=None,
    )

    ignore_prefix: str = arg_field(
        doc="A prefix to ignore on the filenames.", default=""
    )

    pixel_width: float = arg_field(
        doc="The amount of time or distance per pixel. The units are determined by --width-metric",
        default=0.025,
    )

    pixel_metric: PixelMetric = arg_field(
        doc="The type metric used to define the width of a pixel.",
        choices=[t for t in PixelMetric],
        default=PixelMetric.TIME,
    )

    time_bounds: TimeBounds = arg_field(
        doc="What timing to use for aligning the lines. Subset: data between the largest start time and the smallest end time will be used, and the rest discarded. Superset: all data will be included from the smallest start tim eto the largest end time. Lines which do not fully span this will have their data padded with zeros.",
        choices=[t for t in TimeBounds],
        default=TimeBounds.SUBSET,
    )

    micron_per_second: float = arg_field(
        doc="The line speed of the scan. Used to convert the line times into spacial dimensions. Each line is stored along the X axis.",
        default=1.0,
    )

    micron_per_line: float = arg_field(
        doc="The distance between each line. Used to convert the seperate files into the spacial dimensions. The lines are concatonated aling the Y axis.",
        default=1.0,
    )

    mass_decimal_places: int = arg_field(
        "--mass-decimals",
        doc="The number of decimal places to which to round off the mass vallues.",
        default=5,
    )
    mass_bin_width: float = arg_field(
        doc="The mz width of a mass bin.",
        default=0.5,
    )

    plot_spectra: bool = arg_field(
        "--no-plot-spec",
        arg_type=ArgType.EXPLICIT_ONLY,
        action="store_false",
        doc="If present will not plot the Total Spectra per layer.",
    )
    plot_tic: bool = arg_field(
        "--no-plot-tic",
        arg_type=ArgType.EXPLICIT_ONLY,
        action="store_false",
        doc="If present will not plot the Total Ion Count per layer.",
    )
    plot_kdm: bool = arg_field(
        "--no-plot-kfm",
        arg_type=ArgType.EXPLICIT_ONLY,
        action="store_false",
        doc="If present will not plot the Kendrick Mass Defect per layer.",
    )


class RawLine(NamedTuple):
    file: Path
    number: int


class Spectrum(NamedTuple):
    time: float
    mass: np.ndarray
    intensity: np.ndarray


class MSInstrumentData:
    def __init__(self, rawFile: IRawDataExtended, index):
        self.raw_file = rawFile
        self.raw_file.SelectInstrument(Device.MS, index)
        self._data = self.raw_file.GetInstrumentData()

        self.first_scan_number = self.raw_file.RunHeaderEx.FirstSpectrum
        self.last_scan_number = self.raw_file.RunHeaderEx.LastSpectrum

        self.first_scan_statistics = rawFile.GetScanStatsForScanNumber(
            self.first_scan_number
        )

    def is_centroid_scan(self):
        return self.first_scan_statistics.IsCentroidScan

    def scan_range(self) -> range:
        return range(self.first_scan_number, self.last_scan_number)

    def get_centroid_stream(
        self, scan_number: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centroidStream = self.raw_file.GetCentroidStream(scan_number, False)

        masses = np.array([m for m in centroidStream.Masses])
        intensities = np.array([m for m in centroidStream.Intensities])
        charges = np.array([m for m in centroidStream.Charges])

        return masses, intensities, charges

    def get_scan_number(self, scan_index) -> int:
        return int(self.first_scan_number + scan_index)

    def get_mass_range(self) -> tuple[float, float]:
        stats = self.raw_file.GetScanStatsForScanNumber(self.first_scan_number)
        return stats.LowMass, stats.HighMass

    def get_scan_times(self) -> np.ndarray:
        return np.array(
            [
                self.raw_file.GetScanStatsForScanNumber(ii).StartTime
                for ii in self.scan_range()
            ]
        )

    def get_spectra_on_mass(
        self, scan_number: int, mass_axis: np.ndarray
    ) -> np.ndarray:
        scan_statistics = ScanStatistics()
        segmentedScan = self.raw_file.GetSegmentedScanFromScanNumber(
            scan_number, scan_statistics
        )

        if segmentedScan.Positions is None:
            raise IndexError(f"{scan_number} not a valid scan number.")
        spec, _ = np.histogram(
            segmentedScan.Positions, bins=mass_axis, weights=segmentedScan.Intensities
        )
        return spec


@dataclass
class ProcessedLine:
    file: Path
    line_number: int

    times: np.ndarray
    masses: np.ndarray
    spectra: np.ndarray

    @staticmethod
    def from_spectra(line: RawLine, spectra: list[Spectrum]) -> "ProcessedLine":
        masses = np.array([])
        times = []
        for ii, spec in enumerate(spectra):
            masses = np.union1d(masses, spec.mass)
            times.append(spec.time)
        times = np.array(times)

        line_spectra = np.zeros((len(spectra), len(masses)))

        for ii, spec in enumerate(spectra):
            values = np.isin(masses, spec.mass, assume_unique=True)
            line_spectra[ii, values] = spec.intensity[:]

        return ProcessedLine(
            file=line.file,
            line_number=line.number,
            times=times,
            masses=masses,
            spectra=line_spectra,
        )


def process(args: ProcessArgs, config: dict[str, Any] = {}):

    digits = re.compile("\\d+")
    error = False
    lines: list[RawLine] = []
    for fle in args.in_path.iterdir():
        if not fle.is_file():
            continue
        if fle.suffix.lower() != ".raw":
            continue

        parts = [
            int(num)
            for num in digits.findall(fle.name.removeprefix(args.ignore_prefix))
        ]
        if len(parts) != 1:
            if len(parts) == 0:
                print(f"Could not parse filename {fle.name}: no line number included.")
            else:
                print(f"Could not parse filename {fle.name}: Multiple numbers in name.")
            error = True
        lines.append(RawLine(fle, parts[0]))

    if error:
        raise RuntimeError("Could not parse all filenames")

    lines.sort(key=lambda x: x.number)

    array = np.array([line.number for line in lines])
    diffs = np.diff(array)
    for ii, diff in enumerate(diffs[1:]):
        if diff != diffs[0]:
            print(
                f"The numbers {lines[ii].number} and {lines[ii + 1].number} are not consistently {diffs[0]} apart."
            )
    first_number = lines[0].number
    all_times = []
    min_mass: None | float = None
    max_mass: None | float = None
    min_time: None | float = None
    max_time: None | float = None
    for ii, line in enumerate(lines):
        lines[ii] = RawLine(line.file, line.number - first_number)

        rawFile = RawFileReaderAdapter.FileFactory(str(lines[ii].file))
        if not rawFile.IsOpen or rawFile.IsError:
            print("Unable to access the RAW file using the RawFileReader class!")
            exit()
        if rawFile.IsError:
            print("Error opening ({}) - {}".format(rawFile.FileError, lines[ii].file))
            exit()
        if rawFile.InAcquisition:
            print("RAW file still being acquired - {}".format(lines[ii].file))
            exit()

        instrument_data = MSInstrumentData(rawFile, 1)
        line_times = instrument_data.get_scan_times()
        tmp_min, tmp_max = instrument_data.get_mass_range()
        if min_mass is None or max_mass is None or min_time is None or max_time is None:
            min_mass = tmp_min
            max_mass = tmp_max
            min_time = np.min(line_times)
            max_time = np.max(line_times)
        else:
            match args.time_bounds:
                case TimeBounds.SUBSET:
                    min_mass = max(min_mass, tmp_min)
                    max_mass = min(max_mass, tmp_max)
                    min_time = max(min_time, np.min(line_times))
                    max_time = min(max_time, np.max(line_times))
                case TimeBounds.SUPERSET:
                    min_mass = min(min_mass, tmp_min)
                    max_mass = max(max_mass, tmp_max)
                    min_time = min(min_time, np.min(line_times))
                    max_time = max(max_time, np.max(line_times))
        all_times.append(line_times)
    if len(lines) == 0:
        return
    assert not (
        min_mass is None or max_mass is None or min_time is None or max_time is None
    )
    if args.use_mass:
        min_mass = args.start_mass
        max_mass = args.end_mass
    mass_count = math.ceil((max_mass - min_mass) / args.mass_bin_width)
    mass_edges = np.array(
        [ii * args.mass_bin_width + min_mass for ii in range(mass_count + 1)]
    )
    mass_axis = mass_edges[:-1]

    match args.pixel_metric:
        case PixelMetric.TIME:
            delta_t = args.pixel_width
            delta_m = args.pixel_width * args.micron_per_second
        case PixelMetric.DISTANCE:
            delta_t = args.pixel_width / args.micron_per_second
            delta_m = args.pixel_width

    time_count = math.ceil((max_time - min_time) / delta_t) + 1
    x_time_axis = np.array([ii * delta_t + min_time for ii in range(time_count)])
    x_distance_axis = np.array([ii * delta_m for ii in range(time_count)])
    y_distance_axis = np.array([ii * args.micron_per_line for ii in range(len(lines))])

    print(f"Reading {len(all_times)} lines.")
    print(
        f" Output will be {x_distance_axis[-1] - x_distance_axis[0]}micron wide and {y_distance_axis[-1] - y_distance_axis[0]} micron high."
    )
    print(
        f" Output will be between {min_time:.4f} - {max_time:.4f}s, every {delta_t:.4f}s giving {len(x_time_axis)} pixels."
    )
    print(
        f" Mass values will be from {min_mass} - {max_mass}m/z, every {args.mass_bin_width}m/z giving {mass_count} mass bins."
    )

    image = np.zeros((len(x_time_axis), len(lines), len(mass_axis)))
    totals = TotalImages(image.shape)
    formula_data, formula_images = args.get_formulae_filters(image.shape, mass_axis)
    mass_range_data, mass_images = args.get_mass_filters(image.shape, mass_axis)
    all_totals = [totals, *formula_images, *mass_images]

    for ll, line in enumerate(lines):
        rawFile = RawFileReaderAdapter.FileFactory(str(line.file))
        instrument_data = MSInstrumentData(rawFile, 1)

        line_scan_indices = np.array(
            [inx for inx, t in enumerate(all_times[ll]) if min_time <= t <= max_time]
        )
        line_times = all_times[ll][line_scan_indices]
        x_time_axis_indices = np.searchsorted(x_time_axis, line_times, side="left")

        for inx, scan_inx in enumerate(line_scan_indices):
            tt = x_time_axis_indices[inx]
            scan_number = instrument_data.get_scan_number(scan_inx)
            spec = instrument_data.get_spectra_on_mass(scan_number, mass_edges)
            image[tt, ll, :] += spec[:]
            for total in all_totals:
                total.add_spectra(tt, ll, spec[:])

    layer_slice = args.calculate_layer_slice(1)
    width_slice, height_slice = args.calculate_width_and_height_slice(
        image.shape[0], image.shape[1]
    )
    spectra_slice = args.calculate_mass_slice(mass_axis)
    mass_values = mass_axis[spectra_slice]

    cbounds = ContainedBounds.from_chunk(
        outer_shape=(1, *image.shape),
        inner_chunk=Chunk(
            [
                layer_slice,
                width_slice,
                height_slice,
                spectra_slice,
            ]
        ),
    )

    axes = GenericAxis(
        [
            [
                Axis.create(
                    name="layer",
                    values=np.arange(layer_slice.start, layer_slice.stop, 1),
                    indices=[0],
                )
            ],
            [
                Axis.create(
                    name="x_dist",
                    values=x_distance_axis,
                    indices=[1],
                    unit="um",
                ),
                Axis.create(
                    name="x_time",
                    values=x_time_axis,
                    indices=[1],
                    unit="s",
                ),
            ],
            [
                Axis.create(
                    name="y_dist", values=y_distance_axis, indices=[2], unit="um"
                )
            ],
            [Axis.create(name="mass", values=mass_values, indices=[3])],
        ]
    )

    nxs = NexusFile(args.nxs_out_path, mode="w")
    with nxs.as_context():
        min_items_per_chunk = 46000

        spectra_chunks, spectra = create_chunked_subentry(
            nxs,
            "spectra",
            min_items_per_chunk=min_items_per_chunk,
            memory_shape=cbounds.outer_shape,
            data_shape=cbounds.inner_shape,
            priorities=(3, 2, 2, 1),
            axes=axes,
        )

        total_spectra_chunks, total_spectra = create_chunked_subentry(
            nxs,
            "total_spectra",
            min_items_per_chunk=min_items_per_chunk,
            memory_shape=(cbounds.inner_shape[0], cbounds.inner_shape[3]),
            data_shape=(cbounds.inner_shape[0], cbounds.inner_shape[3]),
            priorities=(2, 1),
            axes=GenericAxis([axes[0], axes[3]]),
        )

        image_chunks, images = create_chunked_subentry(
            nxs,
            "images",
            min_items_per_chunk=min_items_per_chunk,
            memory_shape=cbounds.outer_shape,
            data_shape=cbounds.inner_shape,
            priorities=(3, 1, 1, 2),
            axes=axes,
        )

        total_image_chunks, total_images = create_chunked_subentry(
            nxs,
            "total_ion_count",
            min_items_per_chunk=min_items_per_chunk,
            memory_shape=(
                cbounds.inner_shape[0],
                cbounds.inner_shape[1],
                cbounds.inner_shape[2],
            ),
            data_shape=(
                cbounds.inner_shape[0],
                cbounds.inner_shape[1],
                cbounds.inner_shape[2],
            ),
            priorities=(2, 1, 1),
            axes=GenericAxis([axes[0], axes[1], axes[2]]),
        )

        print("Writing data:")
        spectra.data.signal[0, :] = image[width_slice, height_slice, spectra_slice]
        images.data.signal[0, :] = image[width_slice, height_slice, spectra_slice]

        total_spectra.data.signal[0] = totals.total_spectrum[spectra_slice]
        total_images.data.signal[0] = totals.total_image[width_slice, height_slice]

    print("Plotting:")
    tic_config = nxtic.PlotKwArgs.read_config(config, "total_ion_count")
    ts_config = nxts.PlotKwArgs.read_config(config, "total_spectra")
    kdm_config = nxkdm.PlotKwArgs.read_config(config, "kendrick_mass_defect")
    isp_config = nxisp.PlotKwArgs.read_config(config, "calibration_plot")
    path_parts = args.nxs_out_path.parts
    title = f"{args.nxs_out_path.stem} "
    if args.plot_tic:
        filename = f"{args.nxs_out_path.stem}.tic.png"
        nxtic.process(
            nxtic.ProcessArgs(
                title,
                totals.total_image,
                Path(*path_parts[:-1], filename),
                plot_args=tic_config,
            )
        )

    if args.plot_spectra:
        filename = f"{args.nxs_out_path.stem}.ts.png"
        nxts.process(
            nxts.ProcessArgs(
                title,
                mass_axis,
                totals.total_spectrum,
                Path(*path_parts[:-1], filename),
                plot_args=ts_config,
            )
        )

    if args.plot_kdm:
        filename = f"{args.nxs_out_path.stem}.kdm.png"
        nxkdm.process(
            nxkdm.ProcessArgs(
                title,
                mass_axis,
                totals.total_spectrum,
                Path(*path_parts[:-1], filename),
                normalisation=nxkdm.Normalisation.QUADRATIC,
                plot_args=kdm_config,
            )
        )

    args.plot_formulae_ranges(
        mass_axis, formula_data, formula_images, args.nxs_out_path, isp_config
    )

    args.plot_mass_ranges(
        mass_axis, mass_range_data, mass_images, args.nxs_out_path, isp_config
    )

    print("Done plotting")
