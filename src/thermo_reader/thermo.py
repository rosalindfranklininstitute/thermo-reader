# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NamedTuple, Literal
from enum import Enum
from dataclasses import dataclass
import math
import os

import numpy as np

import h5py
from colorama import just_fix_windows_console, Fore, Style


from ms_nexus_tools.api import (
    image_plot as nxtic,
    spectrum_plot as nxts,
    image_and_spectrum_plot as nxisp,
    kendrick_mass_defect_plot as nxkdm,
    imzml as nxml,
)
from datargs import arg_field, ArgType, ConfigFileArgs, InteractiveArgs

from ms_nexus_tools.api.formula_args import FormulaArgs
from ms_nexus_tools.api.mass_range_args import MassRangeArgs
from ms_nexus_tools.api.image_args import (
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
    create_standard_file,
)


from pathlib import Path
import re

import matplotlib.pyplot as plt

from .load_thermo import (
    RawFileReaderAdapter,
    Device,
    IRawDataExtended,
    ScanStatistics,
    ListTrailerExtraFields,
    to_py_datetime,
)

from icecream import ic

just_fix_windows_console()


class PixelMetric(Enum):
    TIME = "time"
    DISTANCE = "distance"


class TimeBounds(Enum):
    SUBSET = "subset"
    SUPERSET = "superset"


@dataclass
class ProcessArgs(
    InteractiveArgs,
    ConfigFileArgs,
    MassSliceArgs,
    WidthAndHeightSliceArgs,
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

    filename_prefix: str = arg_field(
        doc="A prefix to ignore on the filenames.", default=None
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

    mass_bin_width: float = arg_field(
        doc="The mz width of a mass bin. If not specified will use the value stored inthee raw files.",
        default=None,
    )

    write_unidec: bool = arg_field(
        "--no-write-unidec",
        arg_type=ArgType.EXPLICIT_ONLY,
        action="store_false",
        doc="If present will not write out the total spectra and line spectra to a format UniDec can read.",
    )

    write_imzml: bool = arg_field(
        "--no-write-imzml",
        arg_type=ArgType.EXPLICIT_ONLY,
        action="store_false",
        doc="If present will not write out the imzMl file.",
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

    def stored_mass_resolution(self) -> float:
        return self.raw_file.RunHeaderEx.MassResolution

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
                self.raw_file.GetScanStatsForScanNumber(ii).StartTime * 60
                for ii in self.scan_range()
            ]
        )

    def get_spectra_on_mass(
        self, scan_number: int, mass_axis: np.ndarray
    ) -> tuple[np.ndarray, int]:
        scan_statistics = ScanStatistics()
        segmentedScan = self.raw_file.GetSegmentedScanFromScanNumber(
            scan_number, scan_statistics
        )

        if segmentedScan.Positions is None:
            raise IndexError(f"{scan_number} not a valid scan number.")
        spec, _ = np.histogram(
            segmentedScan.Positions, bins=mass_axis, weights=segmentedScan.Intensities
        )
        return spec, len(segmentedScan.Intensities)


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
    if args.filename_prefix is None:
        paths = []
        for fle in args.in_path.iterdir():
            if not fle.is_file():
                continue
            if fle.suffix.lower() != ".raw":
                continue
            paths.append(fle.name)
        filename_prefix = os.path.commonprefix(paths)
    else:
        filename_prefix = args.filename_prefix

    for fle in args.in_path.iterdir():
        if not fle.is_file():
            continue
        if fle.suffix.lower() != ".raw":
            continue

        parts = [
            int(num) for num in digits.findall(fle.name.removeprefix(filename_prefix))
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

    print(f"Inspecting {len(lines)} files.")

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
    mass_resolution: None | float = None
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
        if (
            min_mass is None
            or max_mass is None
            or min_time is None
            or max_time is None
            or mass_resolution is None
        ):
            min_mass = tmp_min
            max_mass = tmp_max
            min_time = np.min(line_times)
            max_time = np.max(line_times)
            mass_resolution = instrument_data.stored_mass_resolution()
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
            mass_resolution = min(
                mass_resolution, instrument_data.stored_mass_resolution()
            )
        all_times.append(line_times)

    time_percentiles = np.percentile(
        [np.percentile(np.diff(line_times), [0, 50, 100]) for line_times in all_times],
        [0, 50, 100],
        axis=0,
    )
    time_percentiles = np.array([time_percentiles[ii, ii] for ii in range(3)])
    scan_percentiles = np.percentile([len(lt) for lt in all_times], [0, 50, 100])

    if len(lines) == 0:
        return
    assert not (
        min_mass is None or max_mass is None or min_time is None or max_time is None
    )
    if args.mass_bin_width is not None:
        mass_resolution = args.mass_bin_width
    assert mass_resolution is not None
    ic(max_mass, min_mass)
    mass_count = math.ceil((max_mass - min_mass) / mass_resolution)
    mass_edges = np.array(
        [ii * mass_resolution + min_mass for ii in range(mass_count + 1)]
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

    print(f"Reading {len(all_times)} lines:")
    print(Style.DIM, end="")
    print(Fore.CYAN, end="")
    for line in lines:
        print(f"   {line.file.name}")
    print(Style.RESET_ALL, end="")

    print(
        f" Output will be {x_distance_axis[-1] - x_distance_axis[0]}micron wide and {y_distance_axis[-1] - y_distance_axis[0]} micron high."
    )
    print(
        f" Output will be between {min_time:.2f} - {max_time:.2f}s, every {delta_t:.4f}s giving {len(x_time_axis)} pixels."
    )
    print(
        f"   ({min_time / 60:.4f} - {max_time / 60:.4f}min, Approximate scan times: min {time_percentiles[0]:.2f}s, median {time_percentiles[1]:.2f}s, max {time_percentiles[2]:.2f}s)"
    )
    print(
        f"   (Actual scan count: min {int(scan_percentiles[0])}, median {int(scan_percentiles[1])}, max {int(scan_percentiles[2])})"
    )
    if len(x_time_axis) > scan_percentiles[2]:
        print(
            Style.BRIGHT
            + Fore.YELLOW
            + "WARNING: "
            + Style.NORMAL
            + "The pixel width may be set incorrectly:"
        )
        print(
            f"    The number of pixels requested ({len(x_time_axis)}) is greater than the higest resolution line (with {scan_percentiles[2]} scans)"
        )
        print(
            "    This will result in each scan being in only one pixel with surrounding enpty pixels.",
            Style.RESET_ALL,
        )

    print(
        f" Mass values will be from {min_mass} - {max_mass}m/z, every {mass_resolution}m/z giving {mass_count} mass bins."
    )

    image = np.zeros((len(x_time_axis), len(lines), len(mass_axis)))
    if args.write_unidec:
        line_spectra = np.zeros((len(lines), len(mass_axis)))
    totals = TotalImages(image.shape)
    all_totals = [totals]

    all_lengths = []
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
            spec, scan_length = instrument_data.get_spectra_on_mass(
                scan_number, mass_edges
            )
            all_lengths.append(scan_length)
            image[tt, ll, :] += spec[:]
            if args.write_unidec:
                line_spectra[ll] += spec[:]
            for total in all_totals:
                total.add_spectra(tt, ll, spec[:])
    bin_percentiles = np.percentile(all_lengths, [0, 50, 100])
    print(
        f"   (Actual bin counts: min {bin_percentiles[0]}, median {bin_percentiles[1]}, max {bin_percentiles[2]})"
    )
    if mass_count > bin_percentiles[2]:
        print(
            Style.BRIGHT
            + Fore.YELLOW
            + "WARNING: "
            + Style.NORMAL
            + "The mass resolution may be set incorrectly:"
        )
        print(
            f"    The number of bins requested ({mass_count}) was greater than the higest resolution line (with {bin_percentiles[2]} mass bins)"
        )
        print(
            "    This will result in each mass being in only one pixel with surrounding enpty pixels.",
            Style.RESET_ALL,
        )

    layer_slice = slice(0, 1)
    width_slice, height_slice = args.calculate_width_and_height_slice(
        image.shape[0], image.shape[1]
    )
    spectra_slice = args.calculate_mass_slice(mass_axis)
    mass_values = mass_axis[spectra_slice]

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

    data_shape = (1, *image.shape)
    out_chunk = Chunk(
        [
            layer_slice,
            width_slice,
            height_slice,
            spectra_slice,
        ]
    )

    min_items_per_chunk = 256 * 1024  # 1Kb
    (
        cbounds,
        (spectra_chunks, total_spectra_chunks, image_chunks, total_image_chunks),
    ) = create_standard_file(
        data_shape,
        out_chunk,
        args.nxs_out_path,
        axes=axes,
        min_items_per_chunk=min_items_per_chunk,
    )

    nxs = NexusFile(args.nxs_out_path, mode="a")
    with nxs.as_context():
        print("Writing data:")
        nxs.root.spectra.data.signal[0, :] = image[
            width_slice, height_slice, spectra_slice
        ]
        nxs.root.images.data.signal[0, :] = image[
            width_slice, height_slice, spectra_slice
        ]

        nxs.root.total_spectra.data.signal[0, 0] = totals.tic_spectrum[spectra_slice]
        nxs.root.total_images.data.signal[0, 0] = totals.tic_image[
            width_slice, height_slice
        ]

        nxs.root.total_spectra.data.signal[1, 0] = totals.max_spectrum[spectra_slice]
        nxs.root.total_images.data.signal[1, 0] = totals.max_image[
            width_slice, height_slice
        ]

    path_parts = args.nxs_out_path.parts
    if args.write_unidec:
        total_spectra_data = np.array(
            [mass_values, totals.tic_spectrum[spectra_slice]]
        ).T
        filename = f"{args.nxs_out_path.stem}.ts.txt"
        np.savetxt(args.nxs_out_path.parent / filename, total_spectra_data)
        filename = f"{args.nxs_out_path.stem}.unidec.hdf5"
        with h5py.File(args.nxs_out_path.parent / filename, "w") as fle:
            dataset = fle.create_group("ms_dataset")
            dataset.attrs["num"] = len(lines)
            dataset.attrs["num"] = len(lines)
            dataset.attrs["v1name"] = "Variable 1"
            dataset.attrs["v2name"] = "Variable 2"
            for ll in range(len(lines)):
                line_dataset = fle.create_group(f"ms_dataset/{ll}")
                line_dataset.attrs["name"] = lines[ii].file.name
                raw_line = np.sum(image[:, ll, spectra_slice], axis=0)
                line_stats = np.percentile(raw_line, [0, 100])
                normal_line = (raw_line - line_stats[0]) / (
                    line_stats[1] - line_stats[0]
                )
                line_dataset.create_dataset(
                    name="raw_data",
                    data=np.array([mass_values, raw_line[:]]).T,
                    chunks=(len(mass_values), 2),
                )
                line_dataset.create_dataset(
                    name="processed_data",
                    data=np.array([mass_values, normal_line[:]]).T,
                    chunks=(len(mass_values), 2),
                )

    if args.write_imzml:
        print("Writing imzML:")
        nxml.process(
            nxml.ProcessArgs(
                in_path=args.nxs_out_path, out_path=args.nxs_out_path, one_indexed=True
            )
        )
