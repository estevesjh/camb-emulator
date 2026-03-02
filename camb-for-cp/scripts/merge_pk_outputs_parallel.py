#!/usr/bin/env python
"""
Merge rank-specific output files from MPI cosmosis run using streaming I/O.

Usage:
    python merge_pk_outputs_parallel.py [--clean]

This script combines linear_rank*.dat and boost_rank*.dat files
into single linear.dat and boost.dat files by streaming concatenation.
No data is loaded into memory — rank files are copied byte-by-byte
into the output, so memory usage is negligible (~64KB buffer).
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys


def count_lines(filepath):
    """Count lines in a file using wc -l (fast, no memory)."""
    result = subprocess.run(
        ["wc", "-l", filepath], capture_output=True, text=True
    )
    return int(result.stdout.split()[0])


def merge_files(pattern, output_file):
    """Merge all files matching pattern into output_file via streaming cat."""
    files = sorted(
        glob.glob(pattern),
        key=lambda x: int(x.split('rank')[1].split('.')[0])
    )

    if not files:
        print("No files found matching: {}".format(pattern))
        return 0

    print("Merging {} files into {}".format(len(files), output_file))

    # Remove existing output file
    if os.path.exists(output_file):
        os.remove(output_file)

    # Stream-concatenate: copy each rank file into the output sequentially.
    # This uses a 64KB buffer and never holds more than one buffer in memory.
    total_bytes = 0
    with open(output_file, 'wb') as fout:
        for i, f in enumerate(files):
            sz = os.path.getsize(f)
            total_bytes += sz
            with open(f, 'rb') as fin:
                shutil.copyfileobj(fin, fout)
            if (i + 1) % 32 == 0 or (i + 1) == len(files):
                print("  {}/{} files merged ({:.1f} GB written)".format(
                    i + 1, len(files), total_bytes / 1e9))

    # Count rows in merged file
    total_rows = count_lines(output_file)
    print("  Total: {} rows, {:.1f} GB\n".format(total_rows, total_bytes / 1e9))
    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Merge MPI rank output files (streaming, low memory)"
    )
    parser.add_argument("--clean", action="store_true",
                        help="Delete rank files after merging")
    args = parser.parse_args()

    # Merge linear files
    n_linear = merge_files("linear_rank*.dat", "linear.dat")

    # Merge boost files
    n_boost = merge_files("boost_rank*.dat", "boost.dat")

    if n_linear == 0 and n_boost == 0:
        print("No rank files found. Nothing to merge.")
        return

    # Verify counts match
    if n_linear != n_boost:
        print("WARNING: linear ({}) and boost ({}) row counts don't match!".format(
            n_linear, n_boost))

    # Clean up rank files if requested
    if args.clean:
        removed = 0
        for f in glob.glob("linear_rank*.dat") + glob.glob("boost_rank*.dat"):
            os.remove(f)
            removed += 1
        print("Cleaned up {} rank files".format(removed))

    print("\nDone! Output files: linear.dat, boost.dat")


if __name__ == "__main__":
    main()
