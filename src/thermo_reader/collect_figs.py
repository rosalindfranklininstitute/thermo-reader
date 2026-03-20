# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from dataclasses import dataclass

from pathlib import Path

import matplotlib.pyplot as plt
import scipy
import numpy as np

from ms_nexus_tools.api.args import arg_field, ArgType, ConfigFileArgs


@dataclass
class ProcessArgs(ConfigFileArgs):
    in_path: Path = arg_field(
        "-d",
        "--directory",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The input directory.",
        default=None,
    )
    out_path: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The output directory.",
        default=None,
    )

    colormap: str = arg_field(
        "--color-map", doc="The color map to use for the plotting", default="plasma"
    )


def recurse_children(
    d, depth=0, max_depth=-1, types=[]
) -> list[tuple[list[str], np.ndarray]]:
    if max_depth >= 0 and depth >= max_depth:
        return []
    results = []
    if isinstance(d, dict):
        if "type" in d:
            tps = [*types, d["type"]]
            if "properties" in d and "CData" in d["properties"]:
                results.append((tps, d["properties"]["CData"]))
        else:
            tps = types[:]

        for k, v in d.items():
            results.extend(recurse_children(v, depth=depth + 1, types=tps))

    elif isinstance(d, list):
        for ii, v in enumerate(d):
            results.extend(recurse_children(v, depth=depth + 1, types=types))

    if len(results) > 0:
        return results
    else:
        return []


def process(args: ProcessArgs, config: dict[str, Any] = {}):
    if not args.in_path.is_dir():
        raise ValueError(
            f"Input ({args.in_path}) should be a directory with .fig files."
        )
    if not args.out_path.is_dir():
        raise ValueError(
            f"Output ({args.out_path}) should be a directory with .fig files."
        )

    images = []
    for file in args.in_path.iterdir():
        if file.suffix == ".fig":
            mat = scipy.io.loadmat(
                file,
                simplify_cells=True,
            )
            data_children = recurse_children(mat, depth=0, max_depth=3)
            found_data = False
            for c in data_children:
                if "axes" in c[0] and "image" in c[0]:
                    if len(c[1].shape) == 2:
                        images.append((c[1], file))
                        found_data = True
            print(file.name)
            if not found_data:
                print(f"-> Did not find any 2d data in {file.name}")

    shape = None
    for ii, (image, file) in enumerate(images):
        if shape is None:
            shape = image.shape
        else:
            assert shape == image.shape
        np.savetxt(args.out_path / f"{file.stem}.csv", image, delimiter=",")

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_title(file.name)
        im = ax.imshow(image, cmap=args.colormap)
        fig.colorbar(im, ax=ax, location="right")
        fig.savefig(args.out_path / f"{file.stem}.png")

    assert shape is not None
    total_image = np.sum([img[0] for img in images], axis=0)
    np.savetxt(args.out_path / "Total Image.csv", total_image, delimiter=",")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("Total Image")
    im = ax.imshow(total_image, cmap=args.colormap)
    fig.colorbar(im, ax=ax, location="right")
    fig.savefig(args.out_path / "Total Image.png")
