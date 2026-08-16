#!/usr/bin/env python3
"""Drop splats with non-finite values from a Gaussian Splatting .ply.

Training occasionally leaves a handful of gaussians with NaN or inf positions.
They are harmless to the numbers but some viewers refuse to load a file that
contains them, so strip them before export.

    python tools/filter_nan_splats.py input.ply                  # writes input_clean.ply
    python tools/filter_nan_splats.py input.ply -o out.ply
    python tools/filter_nan_splats.py input.ply --in-place
"""

import argparse
import os
import shutil

import numpy as np
from plyfile import PlyData, PlyElement


def filter_ply(in_path, out_path, in_place=False):
    ply = PlyData.read(in_path)

    cleaned = []
    total_dropped = 0
    for element in ply.elements:
        data = element.data
        # A row is kept only if every one of its properties is finite. Checking
        # all properties rather than just x/y/z also catches NaN scales,
        # rotations and SH coefficients, which break viewers just as easily.
        keep = np.ones(len(data), dtype=bool)
        for name in data.dtype.names:
            column = data[name]
            if np.issubdtype(column.dtype, np.floating):
                keep &= np.isfinite(column)

        dropped = int((~keep).sum())
        total_dropped += dropped
        print(
            f"{element.name}: {len(data)} -> {int(keep.sum())} ({dropped} dropped)"
        )
        cleaned.append(PlyElement.describe(data[keep], element.name))

    if total_dropped == 0:
        print("nothing to drop, file is already clean")

    # Preserve the original binary/text format and byte order.
    out = PlyData(cleaned, text=ply.text, byte_order=ply.byte_order)

    if in_place:
        backup = in_path + ".bak"
        shutil.copy2(in_path, backup)
        print(f"backup written to {backup}")
        out.write(in_path)
        print(f"wrote {in_path}")
    else:
        out.write(out_path)
        print(f"wrote {out_path}")

    return total_dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to point_cloud.ply")
    parser.add_argument("-o", "--output", help="output path")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite the input, keeping a .bak copy alongside it",
    )
    args = parser.parse_args()

    if args.in_place and args.output:
        parser.error("--in-place and --output are mutually exclusive")

    output = args.output
    if not output:
        root, ext = os.path.splitext(args.input)
        output = root + "_clean" + ext

    filter_ply(args.input, output, in_place=args.in_place)


if __name__ == "__main__":
    main()
