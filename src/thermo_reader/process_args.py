from dataclasses import dataclass
from datargs import arg_field, ArgType
from datargs.extra_types import DirPathType, FilePathType
from ms_nexus_tools.api import data_convert

from .data_source import PixelMetric, TimeBounds


@dataclass
class ProcessArgs(
    data_convert.ProcessArgs,
):
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
