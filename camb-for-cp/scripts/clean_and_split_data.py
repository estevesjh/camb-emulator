#!/usr/bin/env python
"""
Convert pre-split .dat files to .npy format for CosmoPower training.

Expects train/test .dat files already created via head/tail split:
  linear_train.dat, linear_test.dat, boost_train.dat, boost_test.dat

Uses polars for fast parallel CSV parsing (~5-10x faster than pandas).
Saves as .npy (uncompressed) for fast memory-mapped loading.

Data format (per row):
  [h0, omega_m, omega_b, n_s, log1e10As, mnu, z, P(k1), P(k2), ..., P(kN)]
"""

import gc
import numpy as np
import os
import time

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    import pandas as pd
    HAS_POLARS = False
    print("WARNING: polars not installed, falling back to pandas (slower)")

# -----------------------------
# Configuration
# -----------------------------
K_FILE = "k_modes.txt"
OUTPUT_DIR = "./training_data"

PARAM_NAMES = ['h0', 'omega_m', 'omega_b', 'n_s', 'log1e10As', 'mnu', 'z']
N_PARAMS = len(PARAM_NAMES)

# Files to convert: (data_type, split, input_file)
FILES = [
    ("linear", "train", "linear_train.dat"),
    ("linear", "test",  "linear_test.dat"),
    ("boost",  "train", "boost_train.dat"),
    ("boost",  "test",  "boost_test.dat"),
]


def read_dat_polars(filename):
    """Read whitespace-separated dat file using polars."""
    df = pl.read_csv(
        filename,
        separator=' ',
        has_header=False,
        dtypes=[pl.Float64],
        ignore_errors=True,
    )
    # polars can sometimes add empty columns from trailing whitespace
    df = df.select([c for c in df.columns if not df[c].is_null().all()])
    return df.to_numpy()


def read_dat_pandas(filename):
    """Fallback: read using pandas."""
    df = pd.read_csv(filename, sep=r'\s+', header=None, dtype=np.float64, engine='c')
    return df.values


def convert_dat(filename, data_type, split, n_k, k_modes):
    """Read a .dat file, apply cleaning, save as .npy and .npz."""
    t0 = time.time()

    print(f"  Reading {filename}...")
    if HAS_POLARS:
        data = read_dat_polars(filename)
    else:
        data = read_dat_pandas(filename)

    t_read = time.time() - t0
    total_rows = len(data)
    print(f"  Read {total_rows:,} rows in {t_read:.1f}s")

    params = data[:, :N_PARAMS]
    spectra = data[:, N_PARAMS:]
    del data
    gc.collect()

    # Validity mask: remove NaN/inf and unphysical values
    valid = np.isfinite(spectra).all(axis=1) & np.isfinite(params).all(axis=1)

    if data_type == "linear":
        # P(k) must be positive; also drop extreme outliers where
        # CAMB produced near-zero values from marginal cosmologies
        valid &= (spectra > 0).all(axis=1)
        valid &= (spectra > 1e-12).all(axis=1)  # log10(P) > -12
    elif data_type == "boost":
        valid &= (spectra > 0.1).all(axis=1) & (spectra < 100).all(axis=1)

    n_removed = (~valid).sum()
    params = params[valid].astype(np.float32)
    spectra = np.log10(spectra[valid]).astype(np.float32)

    assert spectra.shape[1] == n_k, \
        f"Spectrum cols {spectra.shape[1]} != k-modes {n_k}"

    n_valid = len(params)
    print(f"  Valid: {n_valid:,}, removed: {n_removed:,} ({100*n_removed/total_rows:.2f}%)")

    # Save as .npy (fast, supports memory mapping)
    np.save(os.path.join(OUTPUT_DIR, f"camb_{data_type}_params_{split}.npy"), params)
    np.save(os.path.join(OUTPUT_DIR, f"camb_{data_type}_logpower_{split}.npy"), spectra)
    if split == "train":
        np.save(os.path.join(OUTPUT_DIR, f"camb_{data_type}_modes.npy"), k_modes)

    # Also save as .npz for backwards compatibility
    params_dict = {name: params[:, i] for i, name in enumerate(PARAM_NAMES)}
    np.savez(
        os.path.join(OUTPUT_DIR, f"camb_{data_type}_params_{split}.npz"),
        **params_dict
    )
    np.savez(
        os.path.join(OUTPUT_DIR, f"camb_{data_type}_logpower_{split}.npz"),
        features=spectra, modes=k_modes
    )

    t_total = time.time() - t0
    print(f"  Saved .npy + .npz for camb_{data_type}_{split} ({t_total:.1f}s total)")

    del params, spectra, params_dict
    gc.collect()

    return n_valid


def main():
    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    k_modes = np.loadtxt(K_FILE)
    n_k = len(k_modes)
    print(f"Loaded {n_k} k-modes from {K_FILE}")
    print(f"Backend: {'polars' if HAS_POLARS else 'pandas'}\n")

    for data_type, split, filename in FILES:
        if not os.path.exists(filename):
            print(f"SKIP {filename} (not found)")
            continue
        params_out = os.path.join(OUTPUT_DIR, f"camb_{data_type}_params_{split}.npy")
        features_out = os.path.join(OUTPUT_DIR, f"camb_{data_type}_logpower_{split}.npy")
        if os.path.exists(params_out) and os.path.exists(features_out):
            print(f"SKIP {data_type} {split} (output already exists)")
            continue
        print(f"Converting {data_type} {split}:")
        convert_dat(filename, data_type, split, n_k, k_modes)
        print()

    # Print parameter statistics
    print("=" * 60)
    print("Parameter Statistics (linear train)")
    print("=" * 60)
    pf = os.path.join(OUTPUT_DIR, "camb_linear_params_train.npy")
    if os.path.exists(pf):
        params = np.load(pf)
        for i, name in enumerate(PARAM_NAMES):
            arr = params[:, i]
            print(f"  {name:12s}: [{arr.min():.4f}, {arr.max():.4f}], mean={arr.mean():.4f}")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
