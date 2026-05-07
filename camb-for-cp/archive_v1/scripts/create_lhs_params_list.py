#!/usr/bin/env python
"""
Convert LHS parameters from .npz to CosmoSIS list format.

The list sampler in CosmoSIS reads a file with:
- Header line: # param1 param2 param3 ...
- Data lines: value1 value2 value3 ...

Parameter names must match CosmoSIS convention:
  section--parameter (e.g., cosmological_parameters--omega_m)
"""

import numpy as np

# -----------------------------
# Configuration
# -----------------------------
npz_file = "LHS_params.npz"
list_file = "LHS_params.list"

# Parameter mapping: npz_key -> cosmosis_name
# Order matters and must match save_pk.py PARAM_NAMES!
PARAM_MAP = [
    ("h0",        "cosmological_parameters--h0"),
    ("omega_m",   "cosmological_parameters--omega_m"),
    ("omega_b",   "cosmological_parameters--omega_b"),
    ("n_s",       "cosmological_parameters--n_s"),
    ("log1e10As", "cosmological_parameters--log1e10as"),
    ("mnu",       "cosmological_parameters--mnu"),
]

# -----------------------------
# Load and convert
# -----------------------------
data = np.load(npz_file)
n_samples = len(data[PARAM_MAP[0][0]])

print(f"Loading {n_samples} samples from {npz_file}")

# Build header
header = "# " + " ".join([cosmosis_name for _, cosmosis_name in PARAM_MAP])

# Build data array
columns = [data[npz_key] for npz_key, _ in PARAM_MAP]
data_array = np.column_stack(columns)

# -----------------------------
# Save list file
# -----------------------------
with open(list_file, "w") as f:
    f.write(header + "\n")
    for row in data_array:
        f.write(" ".join(f"{val:.10e}" for val in row) + "\n")

print(f"Saved CosmoSIS list file to {list_file}")
print(f"Header: {header}")
print(f"Samples: {n_samples}")
