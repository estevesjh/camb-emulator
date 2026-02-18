#!/usr/bin/env python
"""
Test trained CosmoPower emulator against held-out CAMB spectra.

Computes accuracy metrics and generates comparison plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cosmopower import cosmopower_NN

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "./training_data"
SPECTRA_TYPE = "linear"  # or "boost"
MODEL_NAME = f"camb_{SPECTRA_TYPE}_emulator"

MODEL_PARAMETERS = ['h0', 'omega_m', 'omega_b', 'n_s', 'log1e10As', 'mnu', 'z']

# Output directory for plots
PLOT_DIR = "./plots"

# -----------------------------
# Load model and test data
# -----------------------------
print(f"Loading model: {MODEL_NAME}")
cp_nn = cosmopower_NN(restore=True, restore_filename=MODEL_NAME)

print(f"Loading test data...")
params_test = np.load(os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_params_test.npz"))
features_test = np.load(os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_logpower_test.npz"))

k_modes = features_test['modes']
true_spectra = features_test['features']  # log10(P(k))
n_test = true_spectra.shape[0]

print(f"Test samples: {n_test}")
print(f"k-modes: {len(k_modes)}")

# -----------------------------
# Generate predictions
# -----------------------------
print("Generating predictions...")

# Build input dict for CosmoPower
input_params = {name: params_test[name] for name in MODEL_PARAMETERS}
pred_spectra = cp_nn.predictions_np(input_params)

# -----------------------------
# Compute accuracy metrics
# -----------------------------
print("\nAccuracy Metrics:")
print("="*60)

# Relative error in linear space: |P_pred - P_true| / P_true
P_true = 10**true_spectra
P_pred = 10**pred_spectra

rel_error = np.abs(P_pred - P_true) / P_true

# Statistics
mean_rel_error = np.mean(rel_error)
median_rel_error = np.median(rel_error)
max_rel_error = np.max(rel_error)
pct_below_1 = 100 * np.mean(rel_error < 0.01)
pct_below_5 = 100 * np.mean(rel_error < 0.05)

print(f"Mean relative error:   {mean_rel_error:.4f} ({100*mean_rel_error:.2f}%)")
print(f"Median relative error: {median_rel_error:.4f} ({100*median_rel_error:.2f}%)")
print(f"Max relative error:    {max_rel_error:.4f} ({100*max_rel_error:.2f}%)")
print(f"Samples with <1% error:  {pct_below_1:.1f}%")
print(f"Samples with <5% error:  {pct_below_5:.1f}%")

# Per-k statistics
mean_rel_error_per_k = np.mean(rel_error, axis=0)
max_rel_error_per_k = np.max(rel_error, axis=0)

print(f"\nPer k-mode (mean over samples):")
print(f"  Best k:  {k_modes[np.argmin(mean_rel_error_per_k)]:.4f} h/Mpc ({100*np.min(mean_rel_error_per_k):.3f}%)")
print(f"  Worst k: {k_modes[np.argmax(mean_rel_error_per_k)]:.4f} h/Mpc ({100*np.max(mean_rel_error_per_k):.3f}%)")

# -----------------------------
# Generate plots
# -----------------------------
os.makedirs(PLOT_DIR, exist_ok=True)

# Plot 1: Relative error vs k
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(k_modes, mean_rel_error_per_k, 'b-', label='Mean', linewidth=2)
ax.fill_between(k_modes,
                np.percentile(rel_error, 16, axis=0),
                np.percentile(rel_error, 84, axis=0),
                alpha=0.3, label='16-84 percentile')
ax.axhline(0.01, color='r', linestyle='--', label='1% target')
ax.set_xlabel('k [h/Mpc]')
ax.set_ylabel('Relative Error |ΔP/P|')
ax.set_title(f'Emulator Accuracy: {SPECTRA_TYPE.capitalize()} Power Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, f'{SPECTRA_TYPE}_error_vs_k.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: {PLOT_DIR}/{SPECTRA_TYPE}_error_vs_k.png")

# Plot 2: Example spectra comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
np.random.seed(123)
sample_indices = np.random.choice(n_test, 6, replace=False)

for idx, ax in zip(sample_indices, axes.flat):
    ax.loglog(k_modes, P_true[idx], 'b-', label='CAMB', linewidth=2)
    ax.loglog(k_modes, P_pred[idx], 'r--', label='Emulator', linewidth=2)

    # Show parameters
    z = params_test['z'][idx]
    om = params_test['omega_m'][idx]
    h0 = params_test['h0'][idx]
    ax.set_title(f'z={z:.2f}, Ωm={om:.3f}, h={h0:.2f}')
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k) [(Mpc/h)³]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f'{SPECTRA_TYPE}_comparison.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {PLOT_DIR}/{SPECTRA_TYPE}_comparison.png")

# Plot 3: Error distribution histogram
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(rel_error.flatten(), bins=100, density=True, alpha=0.7, edgecolor='black')
ax.axvline(mean_rel_error, color='r', linestyle='--', label=f'Mean: {100*mean_rel_error:.2f}%')
ax.axvline(median_rel_error, color='g', linestyle='--', label=f'Median: {100*median_rel_error:.2f}%')
ax.set_xlabel('Relative Error')
ax.set_ylabel('Density')
ax.set_title(f'Error Distribution: {SPECTRA_TYPE.capitalize()} P(k)')
ax.set_xlim(0, 0.1)
ax.legend()
plt.savefig(os.path.join(PLOT_DIR, f'{SPECTRA_TYPE}_error_hist.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {PLOT_DIR}/{SPECTRA_TYPE}_error_hist.png")

print("\nDone!")
