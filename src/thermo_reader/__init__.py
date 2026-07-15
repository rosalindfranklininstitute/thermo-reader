# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

from typing import reveal_type, cast

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

import datargs as nxargs

from ms_nexus_tools.api import data_convert, imzml
from ms_nexus_tools.lib.sparse_sampling import SparseSampling
from ms_nexus_tools.lib import unidec, image, utils

from . import thermo
from .data_source import ThermoDataSource
from .process_args import ProcessArgs


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
        downsample_count=process_args.down_sampling,
        area_positions=np.array([100]),
        area_volumes=np.array([100]),
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
    with utils.FileGuard(
        process_args.out_path, delete_on_failure=True, check_exist_on_success=True
    ):
        data_convert.process(process_args, partial_args.config)

    if process_args.write_unidec:
        file_names = process_args.data_source.experiment_data["files"]
        if not isinstance(file_names, list):
            raise ValueError("Expected 'files' to be a list.")
        if not all([isinstance(s, str) for s in file_names]):
            raise ValueError("Expected 'files' to be a list[str].")
        file_names = cast(list[str], file_names)

        with h5py.File(process_args.out_path, "r") as fle:
            total_int = fle["entry/total_spectra/data/signal"][0, :]
            total_mz = fle["entry/total_spectra/data/mz"][:]

            total_spectra = unidec.Spectra("total", total_mz, total_int)

            individual_spectra = []
            row_mz = fle["entry/spectra/data/mz"][:]
            for row, name in enumerate(file_names):
                row_int = np.sum(fle["entry/spectra/data/signal"][:, row, :], axis=0)
                individual_spectra.append(unidec.Spectra(name, row_mz, row_int))

            with utils.FileGuard(
                process_args.out_path.with_suffix(".total_spectrum.txt"),
                process_args.out_path.with_suffix(".unidec.hdf5"),
                delete_on_failure=True,
                check_exist_on_success=True,
            ):
                unidec.write_unidec(
                    process_args.out_path, total_spectra, individual_spectra
                )

    if process_args.write_imzml:
        imzml_args = imzml.ProcessArgs(
            in_path=process_args.out_path,
            out_path=process_args.out_path.with_suffix(".imzml"),
            entry_name="images",
            signal="signal",
            mass="mz",
            x_axis=0,
            y_axis=1,
            z_axis=-1,
            mz_axis=2,
            one_indexed=True,
        )

        with utils.FileGuard(
            imzml_args.out_path,
            imzml_args.out_path.with_suffix(".ibd"),
            delete_on_failure=True,
            check_exist_on_success=True,
        ):
            imzml.process(imzml_args, {})

    if process_args.write_tic:
        with h5py.File(process_args.out_path, "r") as fle:
            fig, ax = plt.subplots()
            image.plot_image(
                ax,
                fle["entry/total_image/data/signal"][0, :],
                fle["entry/total_image/data/x"][:],
                fle["entry/total_image/data/y"][:],
            )
            fig.savefig(process_args.out_path.with_suffix(".total_ion_count.png"))
