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


def _rank_key(path):
    """Sort helper: extract the integer after 'rank' or 'slice' in a filename."""
    base = os.path.basename(path)
    for tag in ("rank", "slice"):
        if tag in base:
            rest = base.split(tag, 1)[1]
            digits = ""
            for ch in rest:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                return int(digits)
    return base


def merge_files(pattern, output_file):
    """Merge all files matching pattern into output_file via streaming cat."""
    files = sorted(glob.glob(pattern), key=_rank_key)

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
    parser.add_argument("--names", nargs="+",
                        default=["linear", "boost"],
                        help="Base names to merge; expects {name}_rank*.dat "
                             "-> {name}.dat (default: linear boost)")
    parser.add_argument("--glob-template", default="{name}_rank*.dat",
                        help="Glob template with {name} placeholder for the "
                             "input pattern (default: {name}_rank*.dat). For "
                             "array-task slices use 'slice*_{name}.dat'.")
    parser.add_argument("--output-template", default="{name}.dat",
                        help="Output filename template with {name} placeholder "
                             "(default: {name}.dat).")
    args = parser.parse_args()

    counts = {}
    for name in args.names:
        pattern = args.glob_template.format(name=name)
        out = args.output_template.format(name=name)
        counts[name] = merge_files(pattern, out)

    if all(v == 0 for v in counts.values()):
        print("No rank files found. Nothing to merge.")
        return

    distinct = set(v for v in counts.values() if v > 0)
    if len(distinct) > 1:
        print("WARNING: row counts differ across outputs: {}".format(counts))

    if args.clean:
        removed = 0
        for name in args.names:
            for f in glob.glob(args.glob_template.format(name=name)):
                os.remove(f)
                removed += 1
        print(f"Cleaned up {removed} input files")

    outs = ", ".join(args.output_template.format(name=name)
                     for name, n in counts.items() if n > 0)
    print(f"\nDone! Output files: {outs}")


if __name__ == "__main__":
    main()
