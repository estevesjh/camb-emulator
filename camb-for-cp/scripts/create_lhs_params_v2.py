#!/usr/bin/env python
"""
Generate Latin Hypercube Samples for the z=0 linear emulator (v2).

Changes vs v1:
- 500k samples (vs 200k)
- No redshift dimension; z=0 only. Downstream inference recovers P_lin(k, z)
  via the growth factor D(z) from CAMB.
- Reparametrized: sample omega_cdm directly, compute omega_m = omega_cdm + omega_b.
  Avoids wasted samples in the unphysical omega_b > omega_m zone.
- log1e10As tightened to Planck +/- 100 sigma.
- omega_m effective range pushed to [0.08, 0.7] via choice of omega_cdm range.

Emulator-visible parameters (what gets written to .dat and .list) are the same
as v1: (h0, omega_m, omega_b, n_s, log1e10As, mnu).
"""

import argparse
import numpy as np

from scipy.stats.qmc import LatinHypercube

parser = argparse.ArgumentParser(description="Generate LHS samples (v2)")
parser.add_argument("--nsamples", type=int, default=500_000,
                    help="Number of LHS samples (default: 500000)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default="LHS_params_v2.npz")
args = parser.parse_args()

# Sampled parameters (reparametrized: omega_cdm instead of omega_m)
# Ranges per locked spec.
sampled_ranges = {
    "h0":         (0.4,  1.0),
    "omega_cdm":  (0.02, 0.65),   # => omega_m in [0.07, 0.71] given omega_b
    "omega_b":    (0.05, 0.06),
    "n_s":        (0.87, 1.07),
    "log1e10As":  (1.64, 4.44),   # Planck 2018: 3.044 +/- 0.014, +/-100 sigma
    "mnu":        (0.0,  0.2),
}

sampled_names = list(sampled_ranges.keys())
n_params = len(sampled_names)
n_samples = args.nsamples

print(f"Generating {n_samples} LHS samples over {n_params} parameters...")

sampler = LatinHypercube(d=n_params, seed=args.seed)
lhs = sampler.random(n=n_samples)

sampled = {}
for i, name in enumerate(sampled_names):
    pmin, pmax = sampled_ranges[name]
    sampled[name] = pmin + (pmax - pmin) * lhs[:, i]

# Derive omega_m = omega_cdm + omega_b. This is what the emulator sees.
omega_m = sampled["omega_cdm"] + sampled["omega_b"]

# Emulator-visible parameters
emulator_params = {
    "h0":        sampled["h0"],
    "omega_m":   omega_m,
    "omega_b":   sampled["omega_b"],
    "n_s":       sampled["n_s"],
    "log1e10As": sampled["log1e10As"],
    "mnu":       sampled["mnu"],
}

np.savez(args.output, **emulator_params)
print(f"Saved emulator-space parameters to {args.output}")

print("\nParameter summary:")
print("=" * 72)
for name, arr in emulator_params.items():
    print(f"  {name:12s}: [{arr.min():.5f}, {arr.max():.5f}]  "
          f"mean={arr.mean():.5f}")
print(f"\nDerived omega_cdm: [{sampled['omega_cdm'].min():.5f}, "
      f"{sampled['omega_cdm'].max():.5f}]")
print(f"Samples: {n_samples}")
