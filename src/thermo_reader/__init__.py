# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import sys
import os
import argparse

from . import thermo

from pathlib import Path

import datargs as nxargs


def main():

    partial_args = thermo.ProcessArgs.parse_config("thermo")
    process_args = thermo.ProcessArgs.parse_interactive(
        "thermo", args=partial_args.remaining_args, exclude=["config"]
    )
    thermo.process(process_args, partial_args.config)
