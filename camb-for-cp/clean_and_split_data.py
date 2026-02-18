#!/usr/bin/env python
"""
Clean and split CAMB training data for emulator.

Input: linear.dat, boost.dat from CosmoSIS CAMB runs
Output: Training and test .npz files for CosmoPower

Data format (per row):
  [h0, omega_m, omega_b, n_s, log1e10As, mnu, z, P(k1), P(k2), ..., P(kN)]
"""

import numpy as np
import os

# -----------------------------
# Configuration
# -----------------------------
# Input files
LINEAR_FILE = "linear.dat"
BOOST_FILE = "boost.dat"
K_FILE = "k_modes.txt"

# Output directory
OUTPUT_DIR = "./training_data"

# Train/test split ratio
TEST_FRACTION = 0.2

# Parameter names (must match save_pk_training.py order)
PARAM_NAMES = ['h0', 'omega_m', 'omega_b', 'n_s', 'log1e10As', 'mnu', 'z']
N_PARAMS = len(PARAM_NAMES)

# -----------------------------
# Main processing
# -----------------------------
def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load k-modes
    k_modes = np.loadtxt(K_FILE)
    n_k = len(k_modes)
    print(f"Loaded {n_k} k-modes from {K_FILE}")

    # Process both linear and boost data
    for data_type, filename in [("linear", LINEAR_FILE), ("boost", BOOST_FILE)]:
        print(f"\nProcessing {data_type} data from {filename}...")

        # Load data
        data = np.loadtxt(filename)
        print(f"  Loaded shape: {data.shape}")

        # Split into parameters and spectra
        params = data[:, :N_PARAMS]
        spectra = data[:, N_PARAMS:]

        # Verify dimensions
        assert spectra.shape[1] == n_k, f"Spectrum length {spectra.shape[1]} != k-modes {n_k}"

        # Clean: remove rows with NaN or inf
        valid_mask = np.isfinite(spectra).all(axis=1) & np.isfinite(params).all(axis=1)

        # For linear P(k), also remove non-positive values before log
        if data_type == "linear":
            positive_mask = (spectra > 0).all(axis=1)
            valid_mask = valid_mask & positive_mask

        # For boost, remove extreme values
        if data_type == "boost":
            reasonable_mask = (spectra > 0.1).all(axis=1) & (spectra < 100).all(axis=1)
            valid_mask = valid_mask & reasonable_mask

        n_removed = len(valid_mask) - valid_mask.sum()
        print(f"  Removed {n_removed} invalid rows ({100*n_removed/len(valid_mask):.2f}%)")

        params_clean = params[valid_mask]
        spectra_clean = spectra[valid_mask]

        # Take log of spectra for training (NN predicts log P(k))
        log_spectra = np.log10(spectra_clean)

        # Random shuffle
        np.random.seed(42)
        indices = np.random.permutation(len(params_clean))
        params_clean = params_clean[indices]
        log_spectra = log_spectra[indices]

        # Train/test split
        n_test = int(len(params_clean) * TEST_FRACTION)
        n_train = len(params_clean) - n_test

        params_train = params_clean[:n_train]
        params_test = params_clean[n_train:]
        spectra_train = log_spectra[:n_train]
        spectra_test = log_spectra[n_train:]

        print(f"  Train samples: {n_train}")
        print(f"  Test samples: {n_test}")

        # Save as .npz files
        # Parameters file (dict with each parameter as key)
        params_train_dict = {name: params_train[:, i] for i, name in enumerate(PARAM_NAMES)}
        params_test_dict = {name: params_test[:, i] for i, name in enumerate(PARAM_NAMES)}

        np.savez(
            os.path.join(OUTPUT_DIR, f"camb_{data_type}_params_train.npz"),
            **params_train_dict
        )
        np.savez(
            os.path.join(OUTPUT_DIR, f"camb_{data_type}_params_test.npz"),
            **params_test_dict
        )

        # Features file (spectra + k-modes)
        np.savez(
            os.path.join(OUTPUT_DIR, f"camb_{data_type}_logpower_train.npz"),
            features=spectra_train,
            modes=k_modes
        )
        np.savez(
            os.path.join(OUTPUT_DIR, f"camb_{data_type}_logpower_test.npz"),
            features=spectra_test,
            modes=k_modes
        )

        print(f"  Saved to {OUTPUT_DIR}/camb_{data_type}_*.npz")

    # Print parameter statistics
    print("\n" + "="*60)
    print("Parameter Statistics (training set, linear)")
    print("="*60)

    params_train_dict = np.load(os.path.join(OUTPUT_DIR, "camb_linear_params_train.npz"))
    for name in PARAM_NAMES:
        arr = params_train_dict[name]
        print(f"{name:12s}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")


if __name__ == "__main__":
    main()
