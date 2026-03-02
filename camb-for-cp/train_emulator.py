#!/usr/bin/env python
"""
Train CosmoPower neural network emulator on CAMB power spectra.

Input: Cleaned training data from clean_and_split_data.py
Output: Trained CosmoPower model

Parameters: h0, omega_m, omega_b, n_s, log1e10As, mnu, z

Usage:
    python train_emulator.py [--nsamples N] [--spectra linear|boost]

    --nsamples N    Use only N samples for training (default: all)
    --spectra TYPE  Train on 'linear' or 'boost' spectra (default: linear)
"""

import argparse
import os
import time
import warnings

# Suppress noisy messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from cosmopower import cosmopower_NN

# -----------------------------
# Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Train CosmoPower emulator")
parser.add_argument("--nsamples", type=int, default=0,
                    help="Number of training samples (0 = use all)")
parser.add_argument("--spectra", type=str, default="linear",
                    choices=["linear", "boost"],
                    help="Spectra type to train on (default: linear)")
args = parser.parse_args()

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "./training_data"
SPECTRA_TYPE = args.spectra
MODEL_NAME = f"camb_{SPECTRA_TYPE}_emulator"

MODEL_PARAMETERS = ['h0', 'omega_m', 'omega_b', 'n_s', 'log1e10As', 'mnu', 'z']

# Network architecture
HIDDEN_LAYERS = [512, 512, 512, 512]

# Training hyperparameters - scaled for large datasets (16M samples)
# Batch sizes are large to keep steps/epoch manageable on GPU
# Note: lr=1e-2 caused loss explosions; start at 1e-3 for stability.
# Batch sizes capped at 1M to stay within A100-80GB memory.
LEARNING_RATES = [1e-3, 3e-4, 1e-4]
BATCH_SIZES_LARGE = [200000, 500000, 1000000]
BATCH_SIZES_MEDIUM = [5000, 10000, 50000]
BATCH_SIZES_SMALL = [1000, 5000, 10000]
MAX_EPOCHS = [100, 200, 300]
PATIENCE = [30, 30, 30]

# Thresholds to switch between batch schedules
LARGE_DATASET_THRESHOLD = 5_000_000
MEDIUM_DATASET_THRESHOLD = 500_000

# -----------------------------
# Setup
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
n_gpus = len(gpus)
print(f"GPUs available: {n_gpus}")
if gpus:
    for gpu in gpus:
        print(f"  {gpu.name}")
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")

# Use single GPU — CosmoPower's optimizer is not compatible with MirroredStrategy
# Restrict to GPU:0 for reliable training
if n_gpus >= 1:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"Using single GPU: {gpus[0].name}")
GPU_SCALE = 1

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Load data (try .npy first, fallback to .npz)
# -----------------------------
print(f"\nLoading {SPECTRA_TYPE} training data...")
t0 = time.time()

params_npy = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_params_train.npy")
features_npy = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_logpower_train.npy")
modes_npy = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_modes.npy")

params_npz = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_params_train.npz")
features_npz = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_logpower_train.npz")

if os.path.exists(params_npy) and os.path.exists(features_npy):
    print("Loading .npy files (fast, memory-mapped)...")
    params_arr = np.load(params_npy, mmap_mode='r')
    training_features_arr = np.load(features_npy, mmap_mode='r')
    modes = np.load(modes_npy)
    training_parameters = {name: params_arr[:, i] for i, name in enumerate(MODEL_PARAMETERS)}
elif os.path.exists(params_npz) and os.path.exists(features_npz):
    print("Loading .npz files (slower)...")
    params_data = np.load(params_npz)
    features_data = np.load(features_npz)
    training_parameters = {name: params_data[name] for name in MODEL_PARAMETERS}
    training_features_arr = features_data['features']
    modes = features_data['modes']
else:
    raise FileNotFoundError(
        f"No training data found in {DATA_DIR}. Run clean_and_split_data.py first."
    )

t_load = time.time() - t0
n_total = len(training_features_arr)
print(f"Loaded {n_total:,} samples, {len(modes)} k-modes in {t_load:.1f}s")

# Subsample if requested
if args.nsamples > 0 and args.nsamples < n_total:
    print(f"Subsampling to {args.nsamples:,} samples...")
    idx = np.random.choice(n_total, args.nsamples, replace=False)
    idx.sort()
    training_features_arr = np.array(training_features_arr[idx])
    training_parameters = {name: np.array(training_parameters[name])[idx]
                           for name in MODEL_PARAMETERS}
    n_total = args.nsamples

# Verify parameter ranges
for name in MODEL_PARAMETERS:
    arr = training_parameters[name]
    print(f"  {name}: [{arr.min():.4f}, {arr.max():.4f}]")

# -----------------------------
# Select batch schedule based on dataset size
# -----------------------------
if n_total >= LARGE_DATASET_THRESHOLD:
    batch_sizes = [bs * GPU_SCALE for bs in BATCH_SIZES_LARGE]
    print(f"\nUsing LARGE batch schedule (dataset >= {LARGE_DATASET_THRESHOLD:,})")
elif n_total >= MEDIUM_DATASET_THRESHOLD:
    batch_sizes = [bs * GPU_SCALE for bs in BATCH_SIZES_MEDIUM]
    print(f"\nUsing MEDIUM batch schedule (dataset >= {MEDIUM_DATASET_THRESHOLD:,})")
else:
    batch_sizes = [bs * GPU_SCALE for bs in BATCH_SIZES_SMALL]
    print(f"\nUsing SMALL batch schedule (dataset < {MEDIUM_DATASET_THRESHOLD:,})")

if GPU_SCALE > 1:
    print(f"  Batch sizes scaled {GPU_SCALE}x for {n_gpus} GPUs")

n_train = int(n_total * 0.8)  # 80% train after validation split
steps_per_epoch = [n_train // bs for bs in batch_sizes]
print(f"  Steps per epoch by phase: {steps_per_epoch}")

# -----------------------------
# Create and train model
# -----------------------------
print(f"\nInitializing CosmoPower NN...")
print(f"  Parameters: {MODEL_PARAMETERS}")
print(f"  Hidden layers: {HIDDEN_LAYERS}")
print(f"  Training samples: {n_total:,}")

# Check for existing checkpoint to resume from
checkpoint_file = MODEL_NAME

def build_model():
    if os.path.exists(checkpoint_file + '.pkl'):
        print(f"  Found existing model checkpoint: {checkpoint_file}.pkl")
        print(f"  Loading checkpoint for resume...")
        return cosmopower_NN(
            parameters=MODEL_PARAMETERS,
            modes=modes,
            n_hidden=HIDDEN_LAYERS,
            verbose=True,
            restore=True,
            restore_filename=checkpoint_file,
        )
    else:
        return cosmopower_NN(
            parameters=MODEL_PARAMETERS,
            modes=modes,
            n_hidden=HIDDEN_LAYERS,
            verbose=True,
        )

cp_nn = build_model()

print(f"\nStarting training...")
print(f"  Learning rates: {LEARNING_RATES}")
print(f"  Batch sizes: {batch_sizes}")
print(f"  Max epochs per phase: {MAX_EPOCHS}")

t_train = time.time()
cp_nn.train(
    training_parameters=training_parameters,
    training_features=training_features_arr,
    filename_saved_model=MODEL_NAME,

    validation_split=0.2,

    learning_rates=LEARNING_RATES,
    batch_sizes=batch_sizes,
    gradient_accumulation_steps=[1, 1, 1],
    patience_values=PATIENCE,
    max_epochs=MAX_EPOCHS,
)

print(f"\nTraining complete! ({time.time() - t_train:.1f}s)")
print(f"Model saved as: {MODEL_NAME}")

# -----------------------------
# Evaluate on held-out test set
# -----------------------------
print(f"\n{'='*60}")
print(f"Evaluating on test set...")

test_params_npy = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_params_test.npy")
test_features_npy = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_logpower_test.npy")

test_params_npz = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_params_test.npz")
test_features_npz = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_logpower_test.npz")

if os.path.exists(test_params_npy) and os.path.exists(test_features_npy):
    test_params_arr = np.load(test_params_npy)
    test_features_true = np.load(test_features_npy)
    test_parameters = {name: test_params_arr[:, i] for i, name in enumerate(MODEL_PARAMETERS)}
elif os.path.exists(test_params_npz) and os.path.exists(test_features_npz):
    test_params_data = np.load(test_params_npz)
    test_features_data = np.load(test_features_npz)
    test_parameters = {name: test_params_data[name] for name in MODEL_PARAMETERS}
    test_features_true = test_features_data['features']
else:
    print("No test data found, skipping evaluation.")
    test_features_true = None

if test_features_true is not None:
    n_test = len(test_features_true)
    print(f"  Test samples: {n_test:,}")

    # Predict in batches to avoid OOM
    EVAL_BATCH = 50000
    predictions = []
    for i in range(0, n_test, EVAL_BATCH):
        batch_params = {name: test_parameters[name][i:i+EVAL_BATCH]
                        for name in MODEL_PARAMETERS}
        predictions.append(cp_nn.predictions_np(batch_params))
    test_features_pred = np.concatenate(predictions, axis=0)

    # RMSE in log10(P(k)) space
    residuals = test_features_pred - test_features_true
    rmse_log = np.sqrt(np.mean(residuals**2))
    rmse_log_per_mode = np.sqrt(np.mean(residuals**2, axis=0))

    # Fractional error in linear P(k) space: |10^(pred-true) - 1|
    # This avoids overflow from computing 10^pred and 10^true separately
    frac_error = np.abs(np.power(10.0, np.clip(residuals, -30, 30)) - 1.0)
    median_frac = np.median(frac_error)
    pct95_frac = np.percentile(frac_error, 95)
    pct99_frac = np.percentile(frac_error, 99)

    # Fraction of predictions within accuracy thresholds
    within_01 = np.mean(frac_error < 0.01) * 100  # < 1%
    within_05 = np.mean(frac_error < 0.05) * 100  # < 5%
    within_10 = np.mean(frac_error < 0.10) * 100  # < 10%

    print(f"\n  Results on test set ({n_test:,} samples, {len(modes)} k-modes):")
    print(f"  -----------------------------------------------")
    print(f"  RMSE [log10 P(k)]:        {rmse_log:.6f}")
    print(f"  RMSE [log10 P(k)] range:  [{rmse_log_per_mode.min():.6f}, {rmse_log_per_mode.max():.6f}]")
    print(f"  Fractional error |dP/P|:")
    print(f"    median:                  {median_frac:.4%}")
    print(f"    95th percentile:         {pct95_frac:.4%}")
    print(f"    99th percentile:         {pct99_frac:.4%}")
    print(f"  Samples within accuracy:")
    print(f"    |dP/P| < 1%%:            {within_01:.1f}%")
    print(f"    |dP/P| < 5%%:            {within_05:.1f}%")
    print(f"    |dP/P| < 10%%:           {within_10:.1f}%")
    print(f"  -----------------------------------------------")
