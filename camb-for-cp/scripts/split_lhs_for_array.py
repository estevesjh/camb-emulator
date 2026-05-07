#!/usr/bin/env python
"""
Split a CosmoSIS list-sampler file into N slices for a SLURM job array.

Each slice file keeps the one-line header from the source list, then a
contiguous block of rows. The last slice absorbs any remainder.

Usage:
  python split_lhs_for_array.py SOURCE OUTDIR N_SLICES
"""
import os
import sys


def main():
    src, outdir, n_slices = sys.argv[1], sys.argv[2], int(sys.argv[3])
    os.makedirs(outdir, exist_ok=True)

    with open(src) as f:
        header = f.readline()
        rows = f.readlines()

    n = len(rows)
    chunk = n // n_slices
    print(f"Source: {src}")
    print(f"Rows: {n}, slices: {n_slices}, base chunk: {chunk}")

    for i in range(n_slices):
        lo = i * chunk
        hi = (i + 1) * chunk if i < n_slices - 1 else n
        out = os.path.join(outdir, f"slice_{i:03d}.list")
        with open(out, "w") as f:
            f.write(header)
            f.writelines(rows[lo:hi])
        print(f"  slice_{i:03d}: rows [{lo}, {hi}) -> {out}")


if __name__ == "__main__":
    main()
