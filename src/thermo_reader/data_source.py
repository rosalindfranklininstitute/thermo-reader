from ms_nexus_tools.lib.dtypes import Float1D32, Int1D32, Int3D32
import math
from dataclasses import dataclass
from enum import Enum
from logging import warning, info
import re
import os
from pathlib import Path
import datetime as dt

from typing import Any, Callable, NamedTuple
import numpy as np
import numpy.typing as npt

from icecream import ic

from ms_nexus_tools.lib.bounds import Shape, Chunk

from ms_nexus_tools.lib.data_source import (
    AbstractDataSource,
    DataShape,
    Axis,
    AxisDensity,
)
from ms_nexus_tools.lib.sparse_sampling import SparseSampling
from ms_nexus_tools.lib.multi_coo import (
    MultiCOO,
)

from .load_thermo import (
    RawFileReaderAdapter,
)

from .msi_instrument import MSInstrumentData


class PixelMetric(Enum):
    TIME = "time"
    DISTANCE = "distance"


class TimeBounds(Enum):
    SUBSET = "subset"
    SUPERSET = "superset"


class RawLine(NamedTuple):
    file: Path
    number: int
    ignore: bool


@dataclass
class MinMax:
    min: float
    max: float


class DataLines(NamedTuple):
    lines: list[RawLine]
    scan_times: list[list[float]]
    scan_dates: list[dt.datetime]
    mass: MinMax
    times: MinMax
    mass_resolution: float

    @staticmethod
    def from_lines(lines: list[RawLine], time_bounds: TimeBounds) -> "DataLines":
        first_number = lines[0].number
        all_creation_times = []
        all_times = []
        mm_mass: None | MinMax = None
        mm_time: None | MinMax = None
        mass_resolution: None | float = None

        for ii, line in enumerate(lines):
            lines[ii] = RawLine(line.file, line.number - first_number, False)

            rawFile = RawFileReaderAdapter.FileFactory(str(lines[ii].file))
            if not rawFile.IsOpen or rawFile.IsError:
                raise RuntimeError(
                    "Unable to access the RAW file using the RawFileReader class!"
                )
            if rawFile.IsError:
                raise IOError(
                    "Error opening ({}) - {}".format(rawFile.FileError, lines[ii].file)
                )
            if rawFile.InAcquisition:
                raise IOError(
                    "RAW file still being acquired - {}".format(lines[ii].file)
                )

            instrument_data = MSInstrumentData(rawFile, 1)
            line_times = instrument_data.get_scan_times()
            if len(line_times) == 0:
                warning(f"{lines[ii].file.name}: -- No data found, ignoring")
                lines[ii] = RawLine(lines[ii].file, lines[ii].number, True)
                continue

            tmp_min_mz, tmp_max_mz = instrument_data.get_mass_range()
            mass_resolution = instrument_data.stored_mass_resolution()
            tmp_min_s = np.min(line_times)
            tmp_max_s = np.max(line_times)
            info(
                f"{lines[ii].file.name}: ({tmp_min_s / 60: >.4f} - {tmp_max_s / 60: >.4f}min)"
            )

            if mm_mass is None or mm_time is None or mass_resolution is None:
                mm_mass = MinMax(tmp_min_mz, tmp_max_mz)
                mm_time = MinMax(np.min(line_times), np.max(line_times))
                mass_resolution = instrument_data.stored_mass_resolution()
            else:
                match time_bounds:
                    case TimeBounds.SUBSET:
                        mm_mass.min = max(mm_mass.min, tmp_min_mz)
                        mm_mass.max = min(mm_mass.max, tmp_max_mz)
                        mm_time.min = max(mm_time.min, tmp_min_s)
                        mm_time.max = min(mm_time.max, tmp_max_s)
                    case TimeBounds.SUPERSET:
                        mm_mass.min = min(mm_mass.min, tmp_min_mz)
                        mm_mass.max = max(mm_mass.max, tmp_max_mz)
                        mm_time.min = min(mm_time.min, tmp_min_s)
                        mm_time.max = max(mm_time.max, tmp_max_s)
                mass_resolution = min(
                    mass_resolution, instrument_data.stored_mass_resolution()
                )
            all_times.append(line_times)
            all_creation_times.append(instrument_data.creatoin_time)

        lines = [ll for ll in lines if not ll.ignore]

        if mm_mass is None or mm_time is None or mass_resolution is None:
            raise ValueError(f"No data found in {time_bounds.value} of lines.")

        return DataLines(
            lines=lines,
            scan_times=all_times,
            scan_dates=all_creation_times,
            mass=mm_mass,
            times=mm_time,
            mass_resolution=mass_resolution,
        )

    def get_line_data(
        self, line_index: int, instrument_index
    ) -> tuple[MSInstrumentData, list[float]]:

        rawFile = RawFileReaderAdapter.FileFactory(str(self.lines[line_index].file))
        return MSInstrumentData(rawFile, instrument_index), self.scan_times[line_index]


def inspect(thing):
    print(f"{thing}")
    for ss in dir(thing):
        if len(ss) > 0 and ss[0].isupper() and "_" not in ss:
            try:
                value = getattr(thing, ss)
            except Exception as e:
                print(f"{ss}: {e}")
            else:
                print(f"{ss}: {value}")


class ThermoDataSource(AbstractDataSource):
    def __init__(
        self,
        in_path: Path,
        filename_prefix: str | None,
        time_bounds: TimeBounds,
        pixel_metric: PixelMetric,
        pixel_width: float,
        micron_per_second: float,
        micron_per_line: float,
        sampling: SparseSampling = SparseSampling(),
    ):

        filename_prefix = (
            filename_prefix
            if filename_prefix is not None
            else self._find_prefix(in_path)
        )

        self.data_lines = DataLines.from_lines(
            self._read_lines_raw(in_path, filename_prefix), time_bounds
        )

        mass_count = math.ceil(
            (self.data_lines.mass.max - self.data_lines.mass.min)
            / self.data_lines.mass_resolution
        )
        self.mz_edges = sampling.get_edges(
            self.data_lines.mass.min, self.data_lines.mass.max, mass_count
        )
        match pixel_metric:
            case PixelMetric.TIME:
                delta_t = pixel_width
                delta_m = pixel_width * micron_per_second
            case PixelMetric.DISTANCE:
                delta_t = pixel_width / micron_per_second
                delta_m = pixel_width

        time_count = (
            math.ceil((self.data_lines.times.max - self.data_lines.times.min) / delta_t)
            + 1
        )
        self.time_values = np.array(
            [ii * delta_t + self.data_lines.times.min for ii in range(time_count)]
        )
        self.x_values = np.array([ii * delta_m for ii in range(time_count)])
        self.y_values = np.array(
            [ii * micron_per_line for ii in range(len(self.data_lines.lines))]
        )

        self.total_shape = (
            len(self.x_values),
            len(self.y_values),
            len(self.mz_edges) - 1,
        )

        start = min(self.data_lines.scan_dates)
        end = max(self.data_lines.scan_dates) + dt.timedelta(
            seconds=self.data_lines.times.max
        )

        self.experiment_data = dict(
            pixel_width=delta_m,
            pixel_time=delta_t,
            pixel_height=micron_per_line,
            creation_date=str(start),
            end_date=str(end),
            duration=str(end - start),
            files=[str(line.file.name) for line in self.data_lines.lines],
            file_numbers=[line.number for line in self.data_lines.lines],
        )

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _find_prefix(self, in_path: Path) -> str:
        paths = []
        for fle in in_path.iterdir():
            if not fle.is_file():
                continue
            if fle.suffix.lower() != ".raw":
                continue
            paths.append(fle.name)
        return os.path.commonprefix(paths)

    def _read_lines_raw(self, in_path: Path, filename_prefix: str) -> list[RawLine]:
        digits = re.compile("\\d+")
        error = False
        lines: list[RawLine] = []
        for fle in in_path.iterdir():
            if not fle.is_file():
                continue
            if fle.suffix.lower() != ".raw":
                continue

            parts = [
                int(num)
                for num in digits.findall(fle.name.removeprefix(filename_prefix))
            ]
            if len(parts) != 1:
                if len(parts) == 0:
                    print(
                        f"Could not parse filename {fle.name}: no line number included."
                    )
                else:
                    print(
                        f"Could not parse filename {fle.name}: Multiple numbers in name."
                    )
                error = True
            lines.append(RawLine(fle, parts[0], False))

        if error:
            raise RuntimeError("Could not parse all filenames")

        lines.sort(key=lambda x: x.number)
        array = np.array([line.number for line in lines])
        diffs = np.diff(array)
        for ii, diff in enumerate(diffs[1:]):
            if diff != diffs[0]:
                warning(
                    f"The numbers {lines[ii].number} and {lines[ii + 1].number} are not consistently {diffs[0]} apart."
                )
        return lines

    def instrament_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the instrament metadata.
        """
        # TODO: DMD: what can we do here?
        return {}

    def experiment_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the experiment metadata.
        """
        return self.experiment_data

    def shape(self) -> DataShape:
        """
        Return the shape of the data.
        """
        # TODO: DMD: see if we can find the density.
        return DataShape(self.total_shape, 1.0)

    def signal_type(self) -> npt.DTypeLike:
        """
        Returns the type for data.
        """
        return np.int32

    def output_chunks(self) -> dict[str, Shape]:
        """
        Returns the names and chunking priorities of the desired output array.
        """
        return dict(images=(1, 1, 2), spectra=(2, 2, 1))

    def chunk_read_count(self, memory_chunk: Shape) -> int:
        """
        Returns the number of read operations needed to fill the provided memory chunk.
        """
        return np.prod(memory_chunk[0:2])

    def axis_definitions(self) -> list[Axis]:
        """
        Returns the axis that should be used when storing the data.
        """
        return [
            Axis(
                name="x",
                primary_axis=0,
                density=AxisDensity.CONTINUOUS,
                units="m",
                dtype=np.float32,
            ),
            Axis(
                name="time",
                primary_axis=0,
                density=AxisDensity.CONTINUOUS,
                units="s",
                dtype=np.float32,
            ),
            Axis(
                name="y",
                primary_axis=1,
                density=AxisDensity.CONTINUOUS,
                units="m",
                dtype=np.float32,
            ),
            Axis(
                name="mz",
                primary_axis=2,
                density=AxisDensity.BINNED,
                units="mz",
                dtype=np.float32,
            ),
        ]

    def continuous_axis_values(self, axis: Axis) -> np.ndarray:
        """
        Returns the values for the specified continuous axis.
        """
        match axis.name:
            case "time":
                return self.time_values
            case "x":
                return self.x_values
            case "y":
                return self.y_values
            case _:
                raise ValueError(f"Unknown continuous axis requested: {axis.name}")

    def binned_axis_edges(self, axis: Axis) -> np.ndarray:
        """
        Returns the bin edges used to histogram the given sparse axis.
        This is used for generting the output accumulations accros this axis, if required.
        """
        if axis.name != "mz":
            raise ValueError(f"Unknown sparse axis requested: {axis.name}")
        return self.mz_edges

    def output_accumulations(self) -> dict[str, tuple[str, ...]]:
        """
        Returns the names and lists of axis that should be
        accumulated (summed and max).
        For examlpe simple image data (x,y, spectra):
        might produce:
        'total_images':     ('mz') # Accumulate over the spectra
        'total_spectra':    ('x','y') # Accumulate over the images
        """
        return dict(total_image=("mz",), total_spectra=("x", "y"))

    def fill_chunk(
        self,
        memory_chunk: Chunk,
        fill_axis: list[Axis],
        update: Callable[[int], None],
    ) -> np.ndarray | MultiCOO:
        """
        Read data from the source in the region specified by
        memory_chunk and return that data. Also return the data
        any sparse axis.

        Parameters:
        memory_chunk:   The bounds of the data to read.
        fill_axis:      The list of sparce axis to fill.
        update:         A callback to update progress.
                        The total of the progress counter is
                        sum([chunk_read_count(mc) for mc in all_memory_chunks])
        Returns:
        The data from the source, and the data for all the sparse axes, ordered in the same order as in the fill_axis.
        If dense :
        -> return_data.shape == self.shape()
        If sparse there is an extra dimension for storing signal and each sparse axis:
        -> return_data.shape[0:-1] == self.shape() and return_data.shape[-1] = len(fill_axis)+1

        """

        min_time = self.time_values[memory_chunk[0].start]
        max_time = self.time_values[memory_chunk[0].stop - 1]

        coords: list[Int3D32] = []
        data: list[Int1D32] = []
        mz_data: list[Float1D32] = []

        for yy in memory_chunk.range(1):
            line_data, line_times = self.data_lines.get_line_data(yy, 1)
            line_chunk_indices = np.array(
                [inx for inx, t in enumerate(line_times) if min_time <= t <= max_time]
            )
            time_axis_indices = np.searchsorted(
                self.time_values, line_times[line_chunk_indices], side="left"
            )
            for inx, scan_inx in enumerate(line_chunk_indices):
                tt = time_axis_indices[inx]
                scan_number = line_data.get_scan_number(scan_inx)
                mass, spec = line_data.get_segmented_scan(scan_number)

                count = len(spec)
                scan_coords = np.tile(
                    np.array([tt, yy, 0]).reshape(3, 1),
                    (1, count),
                )
                coords.append(scan_coords)
                data.append(spec)
                mz_data.append(mass)
                update(1)

        axis = np.concatenate(mz_data)
        labels = np.searchsorted(self.mz_edges[1:], axis)
        labels[labels == self.total_shape[-1]] = self.total_shape[-1] - 1
        final_coords = np.concatenate(coords, axis=1)
        final_coords[2, :] = labels
        return MultiCOO(
            coords=final_coords,
            signal=np.concatenate(data),
            axis=[axis],
        )
