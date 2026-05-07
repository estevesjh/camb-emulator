#!/usr/bin/env python
"""
Convert 1M LHS .npz (wide + dense + ultra) into CosmoSIS .list files.

Produces:
  LHS_params_1M_wide.list    (300k rows)
  LHS_params_1M_dense.list   (500k rows)
  LHS_params_1M_ultra.list   (200k rows)

Row format (header + data, space-separated):
  # cosmological_parameters--<param>  ...
  <h0> <omega_m> <omega_b> <n_s> <log1e10as> <mnu>

Matches the save_pk_training_v2.py PARAM_NAMES order.
"""
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--wide",        default="LHS_params_1M_wide.npz")
parser.add_argument("--dense",       default="LHS_params_1M_dense.npz")
parser.add_argument("--ultra",       default="LHS_params_1M_ultra.npz")
parser.add_argument("--wide-list",   default="LHS_params_1M_wide.list")
parser.add_argument("--dense-list",  default="LHS_params_1M_dense.list")
parser.add_argument("--ultra-list",  default="LHS_params_1M_ultra.list")
args = parser.parse_args()

PARAM_MAP = [
    ("h0",        "cosmological_parameters--h0"),
    ("omega_m",   "cosmological_parameters--omega_m"),
    ("omega_b",   "cosmological_parameters--omega_b"),
    ("n_s",       "cosmological_parameters--n_s"),
    ("log1e10As", "cosmological_parameters--log1e10as"),
    ("mnu",       "cosmological_parameters--mnu"),
]
header = "# " + " ".join(cs for _, cs in PARAM_MAP)


def write_list(npz_path, list_path):
    data = np.load(npz_path)
    cols = [data[key] for key, _ in PARAM_MAP]
    arr = np.column_stack(cols)
    with open(list_path, "w") as f:
        f.write(header + "\n")
        for row in arr:
            f.write(" ".join(f"{v:.10e}" for v in row) + "\n")
    print(f"Wrote {len(arr):,} rows -> {list_path}")


write_list(args.wide,  args.wide_list)
write_list(args.dense, args.dense_list)
write_list(args.ultra, args.ultra_list)
