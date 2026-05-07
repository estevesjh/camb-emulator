#!/usr/bin/env python
"""
Merge rank-specific output files from MPI cosmosis run.

Usage:
    python merge_pk_outputs.py [--nranks N]

This script combines linear_rank*.dat and boost_rank*.dat files
into single linear.dat and boost.dat files.
"""

import argparse
import glob
import numpy as np
import os


def merge_files(pattern, output_file):
    """Merge all files matching pattern into output_file."""
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split('rank')[1].split('.')[0]))

    if not files:
        print("No files found matching: {}".format(pattern))
        return 0

    print("Merging {} files into {}".format(len(files), output_file))

    # Remove existing output file
    if os.path.exists(output_file):
        os.remove(output_file)

    total_rows = 0
    with open(output_file, 'ab') as fout:
        for f in files:
            data = np.loadtxt(f)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            np.savetxt(fout, data, fmt='%.8e')
            total_rows += len(data)
            print("  {} -> {} rows".format(f, len(data)))

    print("Total: {} rows\n".format(total_rows))
    return total_rows


def main():
    parser = argparse.ArgumentParser(description="Merge MPI rank output files")
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
        print("Cleaning up rank files...")
        for f in glob.glob("linear_rank*.dat") + glob.glob("boost_rank*.dat"):
            os.remove(f)
            print("  Removed {}".format(f))

    print("\nDone! Output files: linear.dat, boost.dat")


if __name__ == "__main__":
    main()
