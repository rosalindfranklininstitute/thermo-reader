# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import sys
import os
import argparse

from . import thermo
from . import collect_figs

from pathlib import Path

from ms_nexus_tools.api import args as nxargs


def main():
    parser = argparse.ArgumentParser(prog="thermo")

    nxargs.add_arguments(parser, thermo.ProcessArgs)

    args, config_dict = thermo.ProcessArgs.parse_args(parser)

    process_args = thermo.ProcessArgs(**vars(args))

    thermo.process(process_args, config_dict)


def images():
    parser = argparse.ArgumentParser(prog="figs")

    nxargs.add_arguments(parser, collect_figs.ProcessArgs)

    args, config_dict = collect_figs.ProcessArgs.parse_args(parser)

    process_args = collect_figs.ProcessArgs(**vars(args))

    collect_figs.process(process_args, config_dict)
