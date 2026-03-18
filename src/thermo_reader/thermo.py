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
    ConfigFileArgs, MassSliceArgs, WidthAndHeightSliceArgs, LayerSliceArgs, FormulaArgs
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

    def get_scan_times(self) -> np.ndarray:
        return np.array(
            [
                self.raw_file.GetScanStatsForScanNumber(ii).StartTime
                for ii in self.scan_range()
            ]
        )

    def get_spectra(self, scan_number: int, decimal_places: int) -> Spectrum:
        scan_statistics = ScanStatistics()
        segmentedScan = self.raw_file.GetSegmentedScanFromScanNumber(
            scan_number, scan_statistics
        )

        mass = np.round([p for p in segmentedScan.Positions], decimals=decimal_places)
        unique_mass = np.unique_inverse(mass)

        spec = np.zeros(unique_mass.values.shape)
        for ii, inv in enumerate(unique_mass.inverse_indices):
            spec[inv] += segmentedScan.Intensities[ii]

        return Spectrum(
            time=scan_statistics.StartTime, mass=unique_mass.values, intensity=spec
        )

    def get_all_spectra(self, decimal_places: int) -> list[Spectrum]:
        return [self.get_spectra(ii, decimal_places) for ii in self.scan_range()]


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
        all_times.append(instrument_data.get_scan_times())

    line_starts = np.array([np.min(t) for t in all_times])
    line_ends = np.array([np.max(t) for t in all_times])
    match args.time_bounds:
        case TimeBounds.SUBSET:
            start_time = np.max(line_starts)
            end_time = np.min(line_ends)
        case TimeBounds.SUPERSET:
            start_time = np.min(line_starts)
            end_time = np.max(line_ends)
    match args.pixel_metric:
        case PixelMetric.TIME:
            delta_t = args.pixel_width
            delta_m = args.pixel_width * args.micron_per_second
        case PixelMetric.DISTANCE:
            delta_t = args.pixel_width / args.micron_per_second
            delta_m = args.pixel_width

    time_count = math.ceil((end_time - start_time) / delta_t) + 1
    x_time_axis = np.array([ii * delta_t + start_time for ii in range(time_count)])
    x_distance_axis = np.array([ii * delta_m for ii in range(time_count)])
    y_distance_axis = np.array([ii * args.micron_per_line for ii in range(len(lines))])

    print(f"Reading {len(all_times)} lines.")
    print(
        f" Output will be between {start_time}s and {end_time}s in {len(x_time_axis)} pixels."
    )
    print(
        f" Output will be {x_distance_axis[-1] - x_distance_axis[0]}micron wide and {y_distance_axis[-1] - y_distance_axis[0]} micron high."
    )
    print(
        f" Mass values will be binned into groups width {math.pow(10, -args.mass_decimal_places)}m/z."
    )

    spectra: list[list[Spectrum]] = []
    mass_axis = np.array([])

    for ii, line in enumerate(lines):
        print(line)
        rawFile = RawFileReaderAdapter.FileFactory(str(lines[ii].file))
        instrument_data = MSInstrumentData(rawFile, 1)
        spectra.append(instrument_data.get_all_spectra(args.mass_decimal_places))
        for spec in spectra[-1]:
            mass_axis = np.union1d(mass_axis, spec.mass)

    formula_data = args.calculate_formulae_ranges(mass_axis)
    ic(len(formula_data))
    image = np.zeros((len(x_time_axis), len(lines), len(mass_axis)))

    totals = TotalImages(image.shape)
    formula_images = [
        MassRangeTotalImage(image.shape, m.start_mass_index, m.stop_mass_index)
        for m in formula_data
        if m.mass_index_width > 0
    ]
    ic(len(formula_images))
    all_totals = [totals, *formula_images]

    for ll, spectra_in_line in enumerate(spectra):
        indices = np.array(
            [inx for inx, t in enumerate(all_times[ll]) if start_time <= t <= end_time]
        )
        times = all_times[ll][indices]
        time_indices = np.searchsorted(x_time_axis, times, side="left")

        for inx, jj in enumerate(indices):
            spec = spectra_in_line[jj]
            int_indices = np.isin(mass_axis, spec.mass, assume_unique=True)
            tt = time_indices[inx]
            image[tt, ll, int_indices] += spec.intensity[:]
            for total in all_totals:
                total.add_spectra(tt, ll, image[tt, ll, :])

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

    for fd, fi in zip(formula_data, formula_images):
        filename = f"{args.nxs_out_path.stem}.{fd.formula}.png"
        title = f"{args.nxs_out_path.stem}: {fd.formula}"
        ic(filename)

        isp_config.plot_axes_commands_and_kw_args.update(
            dict(axvline=dict(x=fd.mass, linewidth=0.5, linestyle=":"))
        )

        spec_slice = slice(fd.start_mass_index, fd.stop_mass_index)
        nxisp.process(
            nxisp.ProcessArgs(
                title,
                mass_axis[spec_slice],
                totals.total_spectrum[spec_slice],
                fi.total_image,
                Path(*path_parts[:-1], filename),
                plot_args=isp_config,
            )
        )
    print("Done plotting")
