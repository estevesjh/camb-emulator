#!/usr/bin/env python
"""
Planck-dense LHS for the v2 emulators.

Same six parameters as the v2 LHS, but tightened around Planck 2018
(TT,TE,EE+lowE+lensing) by pm 10 sigma on (Omega_m, n_s, log1e10As).
h0, Omega_b, and m_nu keep the v2 broad ranges so the Planck box is
nested inside v2 space, letting us concatenate datasets for retraining.

Reparametrization: sample Omega_cdm so Omega_m = Omega_cdm + Omega_b is
bounded near the Planck target. The min/max of Omega_m is approximate
because Omega_b varies in [0.05, 0.06]; that slop is tolerated.

Output (dict npz; same schema as LHS_params_v2.npz):
  h0, omega_m, omega_b, n_s, log1e10As, mnu
"""
import argparse

import numpy as np
from scipy.stats.qmc import LatinHypercube

parser = argparse.ArgumentParser()
parser.add_argument("--nsamples", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=4242)
parser.add_argument("--output", type=str, default="LHS_params_v2_planck.npz")
args = parser.parse_args()

# Planck 2018 TT,TE,EE+lowE+lensing central values and 1-sigma.
# Width = pm 10 sigma for (Omega_m, n_s, log1e10As).
# Others keep the v2 broad prior.
PLANCK = {
    "omega_m":   (0.3153, 0.0073),
    "n_s":       (0.9649, 0.0042),
    "log1e10As": (3.044,  0.014),
}
N_SIGMA = 10

# Target Omega_m range (pm 10 sigma around Planck)
om_lo = PLANCK["omega_m"][0] - N_SIGMA * PLANCK["omega_m"][1]
om_hi = PLANCK["omega_m"][0] + N_SIGMA * PLANCK["omega_m"][1]

# Omega_b widened relative to v2's overly-tight [0.05, 0.06].
# Planck 2018: Omega_b h^2 = 0.02208 pm 0.00052. At h~=0.67 this gives
# Omega_b ~= 0.0376-0.0608 for +/-10 sigma on Omega_b h^2, which we use
# as a broad box (no h-coupling at sampling time).
ob_lo, ob_hi = 0.0376, 0.0608

# Corresponding Omega_cdm bounds (so Omega_m ~= target range).
oc_lo = max(0.005, om_lo - ob_hi)     # worst-case Omega_b adds to the top
oc_hi = om_hi - ob_lo                 # worst-case Omega_b adds to the bottom

sampled_ranges = {
    "h0":        (0.40, 1.00),
    "omega_cdm": (oc_lo, oc_hi),
    "omega_b":   (ob_lo, ob_hi),
    "n_s":       (PLANCK["n_s"][0]       - N_SIGMA * PLANCK["n_s"][1],
                  PLANCK["n_s"][0]       + N_SIGMA * PLANCK["n_s"][1]),
    "log1e10As": (PLANCK["log1e10As"][0] - N_SIGMA * PLANCK["log1e10As"][1],
                  PLANCK["log1e10As"][0] + N_SIGMA * PLANCK["log1e10As"][1]),
    "mnu":       (0.00, 0.20),
}

names = list(sampled_ranges.keys())

print(f"Generating {args.nsamples} Planck-dense LHS samples "
      f"({N_SIGMA} sigma):")
for n, (lo, hi) in sampled_ranges.items():
    print(f"  {n:12s} [{lo:.5f}, {hi:.5f}]")

sampler = LatinHypercube(d=len(names), seed=args.seed)
u = sampler.random(n=args.nsamples)
raw = {n: sampled_ranges[n][0]
          + (sampled_ranges[n][1] - sampled_ranges[n][0]) * u[:, i]
       for i, n in enumerate(names)}

omega_m = raw["omega_cdm"] + raw["omega_b"]

out = {
    "h0":        raw["h0"],
    "omega_m":   omega_m,
    "omega_b":   raw["omega_b"],
    "n_s":       raw["n_s"],
    "log1e10As": raw["log1e10As"],
    "mnu":       raw["mnu"],
}

np.savez(args.output, **out)
print(f"\nSaved {args.output}")
print(f"Derived Omega_m: [{omega_m.min():.5f}, {omega_m.max():.5f}]  "
      f"mean={omega_m.mean():.5f}")
for n, arr in out.items():
    print(f"  {n:12s} min={arr.min():.5f}  max={arr.max():.5f}")
