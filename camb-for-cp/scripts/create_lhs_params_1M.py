#!/usr/bin/env python
"""
1M-sample LHS for high-precision v3 emulator training.

Parameters are sampled directly in log1e10As (no sigma_8 book-keeping) --
this keeps the emulator API identical to v2c and avoids the 2x CAMB cost
of sigma_8_input=T. The corresponding approximate sigma_8 range per box
is printed at generation time for transparency.

Three boxes (nested, SPT-3G + DES Y3 + cluster joint constraints):
- 300k wide:    +/-20 sigma  -> sigma_8 ~ [0.60, 1.10] (Planck-fid)
- 500k dense:   +/-10 sigma  -> sigma_8 ~ [0.70, 0.95]
- 200k ultra:   +/- 5 sigma  -> sigma_8 ~ [0.76, 0.88]

Sampling is in *physical densities* (omega_b h^2, omega_cdm h^2) to avoid
the omega_b > omega_m unphysical corners that v2c suffered from. Derived
(omega_m, omega_b) go to the .npz; the emulator input schema matches v2c:

    h0, omega_m, omega_b, n_s, log1e10As, mnu

Usage:
    python create_lhs_params_1M.py \
        --nsamples-wide 300000 --nsamples-dense 500000 \
        --nsamples-ultra 200000 \
        --output-wide  LHS_params_1M_wide.npz \
        --output-dense LHS_params_1M_dense.npz \
        --output-ultra LHS_params_1M_ultra.npz
"""
import argparse

import numpy as np
from scipy.stats.qmc import LatinHypercube

parser = argparse.ArgumentParser()
parser.add_argument("--nsamples-wide",  type=int, default=300_000)
parser.add_argument("--nsamples-dense", type=int, default=500_000)
parser.add_argument("--nsamples-ultra", type=int, default=200_000)
parser.add_argument("--output-wide",  type=str, default="LHS_params_1M_wide.npz")
parser.add_argument("--output-dense", type=str, default="LHS_params_1M_dense.npz")
parser.add_argument("--output-ultra", type=str, default="LHS_params_1M_ultra.npz")
parser.add_argument("--seed-wide",  type=int, default=20260507)
parser.add_argument("--seed-dense", type=int, default=20260508)
parser.add_argument("--seed-ultra", type=int, default=20260509)
args = parser.parse_args()

# SPT-3G + DES Y3 + cluster joint constraints (rough mean / 1-sigma).
CENTRAL = {
    "h0":        (0.673,   0.010 ),
    "omch2":     (0.1200,  0.0025),  # Omega_cdm * h^2
    "ombh2":     (0.02230, 0.00050), # Omega_b   * h^2
    "n_s":       (0.963,   0.007 ),
    "log1e10As": (3.050,   0.030 ),  # SPT+DES+Cl weak constraint
}
# mnu upper bound is ~0.12 eV (Planck+clusters 95%); sample [0, 0.2] to
# cover with margin.
MNU_RANGE = (0.00, 0.20)

# sigma_8 reference for the translation log1e10As -> approximate sigma_8.
# At Planck-fid (h=0.673, Om=0.315, Ob=0.0493, ns=0.965, mnu=0.06) CosmoSIS
# CAMB reports sigma_8 = 0.81149 for log1e10As = 3.044.
SIGMA8_REF     = 0.81149
LOG1E10AS_REF  = 3.044


def box(n_sigma):
    """Box at +/- n_sigma on all Gaussian params; mnu stays [0, 0.2]."""
    b = {p: (c - n_sigma * s, c + n_sigma * s) for p, (c, s) in CENTRAL.items()}
    b["mnu"] = MNU_RANGE
    return b


def sample_box(ranges, nsamples, seed):
    """Draw LHS in (h0, omch2, ombh2, n_s, log1e10As, mnu) and
    derive (omega_m, omega_b)."""
    names = list(ranges.keys())
    sampler = LatinHypercube(d=len(names), seed=seed)
    u = sampler.random(n=nsamples)

    raw = {n: ranges[n][0] + (ranges[n][1] - ranges[n][0]) * u[:, i]
           for i, n in enumerate(names)}

    h2 = raw["h0"] ** 2
    omega_m = (raw["omch2"] + raw["ombh2"]) / h2
    omega_b = raw["ombh2"] / h2

    return {
        "h0":        raw["h0"],
        "omega_m":   omega_m,
        "omega_b":   omega_b,
        "n_s":       raw["n_s"],
        "log1e10As": raw["log1e10As"],
        "mnu":       raw["mnu"],
    }


def approx_sigma8_from_logAs(log1e10As):
    """Scaling sigma_8 proportional to sqrt(A_s) at fixed other params.

    True only at Planck-fid cosmology; gives an order-of-magnitude sigma_8
    range per box. Actual sigma_8 at LHS corners can differ by a factor 2-3.
    """
    return SIGMA8_REF * np.exp(0.5 * (log1e10As - LOG1E10AS_REF))


def report(name, d):
    print(f"\n{name}:")
    for n, arr in d.items():
        print(f"  {n:12s} min={arr.min():.5f}  max={arr.max():.5f}  "
              f"mean={arr.mean():.5f}")
    # Health check: omega_b < omega_m always?
    nbad = int((d["omega_b"] >= d["omega_m"]).sum())
    if nbad:
        print(f"  WARNING: {nbad} samples have omega_b >= omega_m")


def run(name, n_sigma, nsamples, seed, output):
    ranges = box(n_sigma)
    print(f"\n=== {name} (+/-{n_sigma} sigma, n={nsamples:,}) ===")
    print("Sampled ranges:")
    for n, (lo, hi) in ranges.items():
        print(f"  {n:12s} [{lo:.5f}, {hi:.5f}]")
    # Approximate sigma_8 range from log1e10As (Planck-fid scaling only)
    la_lo, la_hi = ranges["log1e10As"]
    s8_lo = approx_sigma8_from_logAs(la_lo)
    s8_hi = approx_sigma8_from_logAs(la_hi)
    print(f"  -> approx sigma_8 at Planck-fid: [{s8_lo:.3f}, {s8_hi:.3f}]")
    d = sample_box(ranges, nsamples, seed)
    report("derived", d)
    np.savez(output, **d)
    print(f"Saved {output}")


print("=== 1M SPT+DES+Cluster LHS (log1e10As-space sampling) ===")
print("Physical-density coords (omega_b h^2, omega_cdm h^2);")
print("(omega_m, omega_b) are derived for emulator input.")
run("Wide",  20, args.nsamples_wide,  args.seed_wide,  args.output_wide)
run("Dense", 10, args.nsamples_dense, args.seed_dense, args.output_dense)
run("Ultra",  5, args.nsamples_ultra, args.seed_ultra, args.output_ultra)
