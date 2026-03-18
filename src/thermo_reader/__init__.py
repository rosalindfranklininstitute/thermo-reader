# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import sys
import os
import argparse

from . import thermo

from pathlib import Path

from ms_nexus_tools.api import args as nxargs


def main():
    parser = argparse.ArgumentParser(prog="thermo")

    nxargs.add_arguments(parser, thermo.ProcessArgs)

    args, config_dict = thermo.ProcessArgs.parse_args(parser)

    process_args = thermo.ProcessArgs(**vars(args))

    thermo.process(process_args, config_dict)
