import numpy as np
import pyDOE

# -----------------------------
# User configuration
# -----------------------------
n_samples = 10  # number of LHS samples

# Define your parameters and ranges here
# Format: "parameter_name": (min, max)
param_ranges = {
    "omega_m": (0.1, 0.9),
    # "h0": (0.6, 0.8),
    # "n_s": (0.92, 1.0),
    # "A_s": (1.8e-9, 2.4e-9),
}

# Output file
npz_file = "test_LHS_CAMB.npz"

# -----------------------------
# Generate Latin Hypercube
# -----------------------------
param_names = list(param_ranges.keys())
n_params = len(param_names)

lhs = pyDOE.lhs(n_params, samples=n_samples)

# Scale LHS to parameter ranges
params = {}
for i, name in enumerate(param_names):
    pmin, pmax = param_ranges[name]
    params[name] = pmin + (pmax - pmin) * lhs[:, i]

# Save parameters
np.savez(npz_file, **params)
print(f"Saved LHS parameters to {npz_file}")

# -----------------------------
# Print summary
# -----------------------------
print("Generated parameter samples:")
for name in param_names:
    print(f"{name}: {params[name]}")

