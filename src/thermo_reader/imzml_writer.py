# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import itertools

from tqdm import tqdm
import xarray as xr
import numpy as np

from xartus.lib import utils

from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.compression import NoCompression


def write_imzml(nx_path: Path, out_path: Path, one_indexed=True) -> None:
    with utils.DeferredAction() as defer:
        data = xr.open_datatree(nx_path, engine="nexus")
        defer(data.close)
        da = data["/entry/data/spectra"]

        shape = da.shape

        required_ndim = 4

        if "y" not in da.coords:
            required_ndim -= 1
        if "z" not in da.coords:
            required_ndim -= 1

        ndim = len(shape)
        if required_ndim != ndim:
            raise ValueError(
                f"Expected the data to have {required_ndim} dimensions, but found {ndim}."
            )

        with ImzMLWriter(
            out_path,
            mz_dtype=np.float32,
            intensity_dtype=np.float32,
            mode="processed",
            mz_compression=NoCompression(),
            intensity_compression=NoCompression(),
            spec_type="profile",
        ) as writer:
            xyz = [
                (x, y, z)
                for x, y, z in itertools.product(
                    range(da.coords["x"].size),
                    range(da.coords["y"].size) if "y" in da.coords else [0],
                    range(da.coords["z"].size) if "z" in da.coords else [0],
                )
            ]
            for coords in tqdm(xyz, desc="Writing imzML"):
                select = {"x": coords[0]}
                if "y" in da.coords:
                    select["y"] = coords[1]
                if "z" in da.coords:
                    select["z"] = coords[2]

                intensity = da.isel(**select)
                assert intensity.shape == da.coords["mz"].shape
                np_int = intensity.as_numpy().data
                inx = np_int.nonzero()[0]
                if len(inx) == 0:
                    continue
                intensity = np_int[inx]
                mz = da.coords["mz"].as_numpy().data[inx]
                if one_indexed:
                    coords = tuple(int(cc + 1) for cc in coords)
                else:
                    coords = tuple(int(cc) for cc in coords)
                writer.addSpectrum(
                    mz,
                    intensity,
                    coords,
                )
