# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0
import logging

from typing import cast

import h5py
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from xartus.lib.multi_linspace import MultiLinspace
from xartus.lib import unidec, image, utils
from xartus.api import data_convert

from .data_source import ThermoDataSource
from .process_args import ProcessArgs
from .imzml_writer import write_imzml

from icecream import ic


logging.basicConfig(format="%(module)s: %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def source() -> None:
    logger.setLevel(logging.INFO)
    logging.getLogger(data_convert.__name__).setLevel(logging.INFO)
    partial_args = ProcessArgs.parse_config("thermo")
    process_args = ProcessArgs.parse_interactive(
        "thermo", args=partial_args.remaining_args, exclude=["config"]
    )
    sampling = MultiLinspace(
        downsample_count=process_args.down_sampling,
        area_positions=np.array([100]),
        area_volumes=np.array([100]),
    )
    process_args.data_source = ThermoDataSource(
        in_path=process_args.in_path,
        filename_prefix=process_args.filename_prefix,
        time_bounds=process_args.time_bounds,
        pixel_metric=process_args.pixel_metric,
        pixel_width=process_args.pixel_width,
        micron_per_second=process_args.micron_per_second,
        micron_per_line=process_args.micron_per_line,
        sampling=sampling,
    )

    with utils.FileGuard(
        process_args.out_path,
        on_failure=utils.FileAction.DELETE,
        on_success=utils.FileAction.CHECK_EXISTS
        if not process_args.dry_run
        else utils.FileAction.DO_NOTHING,
    ):
        data_convert.process(process_args, partial_args.config)

    if process_args.dry_run:
        return

    if process_args.write_unidec:
        file_names = process_args.data_source.experiment_data["files"]
        if not isinstance(file_names, list):
            raise ValueError("Expected 'files' to be a list.")
        if not all(isinstance(s, str) for s in file_names):
            raise ValueError("Expected 'files' to be a list[str].")
        file_names = cast(list[str], file_names)

        with utils.DeferredAction() as defer:
            data = xr.open_datatree(process_args.out_path, engine="nexus")
            defer(data.close)

            total_int = data["entry/accumulations/total_spectra/"].sel(
                accumulator=b"sum"
            )
            total_mz = data["entry/accumulations/total_spectra/"].coords["mz"]

            total_spectra = unidec.Spectra("total", total_mz.values, total_int.values)

            individual_spectra = []
            row_mz = data["entry/data/spectra/"].coords["mz"]
            for row, name in enumerate(file_names):
                row_int = data["entry/data/spectra"].isel(y=row).sum(dim="x")
                individual_spectra.append(
                    unidec.Spectra(name, row_mz.values, row_int.values)
                )

            with utils.FileGuard(
                process_args.out_path.with_suffix(".total_spectrum.txt"),
                process_args.out_path.with_suffix(".unidec.hdf5"),
                on_failure=utils.FileAction.DELETE,
                on_success=utils.FileAction.CHECK_EXISTS,
            ):
                unidec.write_unidec(
                    process_args.out_path, total_spectra, individual_spectra
                )

    if process_args.write_tic:
        with utils.DeferredAction() as defer:
            data = xr.open_datatree(process_args.out_path, engine="nexus")
            defer(data.close)
            da = data["/entry/accumulations/total_image"]
            assert isinstance(da, xr.DataArray)
            fig, ax = plt.subplots()
            image.imshow_sparse(
                ax,
                da.sel(accumulator=b"sum").as_numpy(),
                x="x",
                y="y",
            )
            fig.savefig(process_args.out_path.with_suffix(".total_ion_count.png"))

    if process_args.write_imzml:
        write_imzml(
            process_args.out_path,
            process_args.out_path.with_suffix(".imzML"),
            one_indexed=True,
        )
