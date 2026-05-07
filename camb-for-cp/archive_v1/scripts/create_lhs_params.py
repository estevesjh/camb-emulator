#!/usr/bin/env python
"""
Generate Latin Hypercube Samples for CAMB emulator training.

Parameters (CosmoSIS convention - physical density fractions, not h^2 units):
- h0: reduced Hubble constant H0/100
- omega_m: total matter density fraction Omega_m
- omega_b: baryon density fraction Omega_b
- n_s: scalar spectral index
- log1e10As: log(10^10 A_s) primordial amplitude
- mnu: sum of neutrino masses in eV

Note: Redshift z is NOT sampled here - CAMB outputs P(k) at multiple z values
in a single run (configured via zmin, zmax, nz in camb_pipeline.ini).

Usage:
    python create_lhs_params.py [--use-pydoe]

By default, uses scipy.stats.qmc.LatinHypercube (fast).
Use --use-pydoe flag to use pyDOE with maximin criterion (slow but traditional).
"""

import argparse
import numpy as np

# -----------------------------
# Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Generate LHS samples for CAMB emulator")
parser.add_argument("--use-pydoe", action="store_true",
                    help="Use pyDOE with maximin criterion (slow) instead of scipy (fast)")
args = parser.parse_args()

# -----------------------------
# Configuration
# -----------------------------
n_samples = 200000  # Number of LHS samples (cosmologies)

# Parameter ranges (min, max)
# These use CosmoSIS convention: Omega (density fractions), not omega = Omega*h^2
param_ranges = {
    "h0":        (0.4, 1.0),      # H0/100, wide range
    "omega_m":   (0.02, 1.0),     # Omega_m, matter density fraction
    "omega_b":   (0.05, 0.06),    # Omega_b, baryon density fraction
    "n_s":       (0.87, 1.07),    # scalar spectral index
    "log1e10As": (-3.0, 7.0),     # log(10^10 A_s), primordial amplitude
    "mnu":       (0.0, 0.2),      # Sum(m_nu) in eV, neutrino mass sum
}

# Output files
npz_file = "LHS_params.npz"
summary_file = "LHS_params_summary.txt"

# -----------------------------
# Generate Latin Hypercube
# -----------------------------
param_names = list(param_ranges.keys())
n_params = len(param_names)

print("Generating {} LHS samples for {} parameters...".format(n_samples, n_params))

# Generate LHS in unit hypercube [0,1]^n_params
if args.use_pydoe:
    import pyDOE
    print("Using pyDOE with maximin criterion (this may take a long time)...")
    lhs = pyDOE.lhs(n_params, samples=n_samples, criterion='maximin')
else:
    from scipy.stats.qmc import LatinHypercube
    print("Using scipy.stats.qmc.LatinHypercube (fast)...")
    sampler = LatinHypercube(d=n_params, seed=42)
    lhs = sampler.random(n=n_samples)

# Scale to physical parameter ranges
params = {}
for i, name in enumerate(param_names):
    pmin, pmax = param_ranges[name]
    params[name] = pmin + (pmax - pmin) * lhs[:, i]

# -----------------------------
# Save parameters
# -----------------------------
np.savez(npz_file, **params)
print(f"Saved LHS parameters to {npz_file}")

# -----------------------------
# Print summary
# -----------------------------
print("\n" + "="*60)
print("Parameter Summary")
print("="*60)

summary_lines = []
for name in param_names:
    pmin, pmax = param_ranges[name]
    actual_min = params[name].min()
    actual_max = params[name].max()
    actual_mean = params[name].mean()
    line = f"{name:12s}: range=[{pmin:8.4f}, {pmax:8.4f}]  sampled=[{actual_min:8.4f}, {actual_max:8.4f}]  mean={actual_mean:8.4f}"
    print(line)
    summary_lines.append(line)

print("="*60)
print(f"Total samples: {n_samples}")
print(f"Parameters: {n_params}")

# Save summary
with open(summary_file, 'w') as f:
    f.write(f"LHS Parameter Summary\n")
    f.write(f"Samples: {n_samples}\n")
    f.write(f"Parameters: {n_params}\n\n")
    for line in summary_lines:
        f.write(line + "\n")

print(f"Summary saved to {summary_file}")
