#!/usr/bin/env python
"""
Convert v2 LHS samples to CosmoSIS .list files.

Produces two files:
- LHS_params_v2_100k.list: first 100k rows (initial CAMB run)
- LHS_params_v2_500k.list: full 500k rows (for the eventual fill-in)

The first 100k rows of a 500k LHS are not themselves a strict LHS, but they
are still quasi-uniform enough for initial training. See notes in the CLAUDE
refactor plan.
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="LHS_params_v2.npz")
args = parser.parse_args()

# npz key -> CosmoSIS parameter name. Order matches save_pk_training_v2.py.
PARAM_MAP = [
    ("h0",        "cosmological_parameters--h0"),
    ("omega_m",   "cosmological_parameters--omega_m"),
    ("omega_b",   "cosmological_parameters--omega_b"),
    ("n_s",       "cosmological_parameters--n_s"),
    ("log1e10As", "cosmological_parameters--log1e10as"),
    ("mnu",       "cosmological_parameters--mnu"),
]

data = np.load(args.input)
n_total = len(data[PARAM_MAP[0][0]])
print(f"Loaded {n_total} samples from {args.input}")

columns = [data[npz_key] for npz_key, _ in PARAM_MAP]
data_array = np.column_stack(columns)
header = "# " + " ".join(name for _, name in PARAM_MAP)


def write_list(path, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(" ".join(f"{val:.10e}" for val in row) + "\n")
    print(f"Wrote {len(rows):,} rows -> {path}")


write_list("LHS_params_v2_100k.list", data_array[:100_000])
write_list("LHS_params_v2_500k.list", data_array)
