# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import sys
import os
import argparse

from . import thermo
from .data_source import ThermoDataSource
from .process_args import ProcessArgs

import numpy as np

from pathlib import Path

import datargs as nxargs

from ms_nexus_tools.api import data_convert
from ms_nexus_tools.lib.sparse_sampling import SparseSampling


def main():

    partial_args = thermo.ProcessArgs.parse_config("thermo")
    process_args = thermo.ProcessArgs.parse_interactive(
        "thermo", args=partial_args.remaining_args, exclude=["config"]
    )
    thermo.process(process_args, partial_args.config)


def source():
    partial_args = ProcessArgs.parse_config("thermo")
    process_args = ProcessArgs.parse_interactive(
        "thermo", args=partial_args.remaining_args, exclude=["config"]
    )
    sampling = SparseSampling(
        downsample_count=10,
        area_positions=np.array([50, 75, 100]),
        area_volumes=np.array([75, 20, 5]),
    )
    process_args.data_source = ThermoDataSource(
        in_path=process_args.in_path.parent,
        filename_prefix=process_args.filename_prefix,
        time_bounds=process_args.time_bounds,
        pixel_metric=process_args.pixel_metric,
        pixel_width=process_args.pixel_width,
        micron_per_second=process_args.micron_per_second,
        micron_per_line=process_args.micron_per_line,
        sampling=sampling,
    )
    data_convert.process(process_args, partial_args.config)
