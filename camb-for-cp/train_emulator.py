#!/usr/bin/env python
"""
Train CosmoPower neural network emulator on CAMB power spectra.

Input: Cleaned training data from clean_and_split_data.py
Output: Trained CosmoPower model

Parameters: h0, omega_m, omega_b, n_s, log1e10As, mnu, z
"""

import os
import warnings

# Suppress noisy messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from cosmopower import cosmopower_NN

# -----------------------------
# Configuration
# -----------------------------
# Data directory
DATA_DIR = "./training_data"

# Which spectra to train (linear or boost)
SPECTRA_TYPE = "linear"  # or "boost"

# Model output name
MODEL_NAME = f"camb_{SPECTRA_TYPE}_emulator"

# Parameters (must match clean_and_split_data.py order)
MODEL_PARAMETERS = ['h0', 'omega_m', 'omega_b', 'n_s', 'log1e10As', 'mnu', 'z']

# Network architecture
HIDDEN_LAYERS = [512, 512, 512, 512]

# Training hyperparameters
LEARNING_RATES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
BATCH_SIZES = [1000, 10000, 20000, 40000, 50000]
MAX_EPOCHS = [100, 200, 300, 500, 1000]
PATIENCE = [20, 20, 20, 20, 20]

# -----------------------------
# Setup
# -----------------------------
# Check GPU
gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'
print(f"Using device: {device}")
if gpus:
    print(f"GPU(s) available: {[gpu.name for gpu in gpus]}")

# Enable mixed precision for speedup on GPU
if gpus:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")

# Seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Load data
# -----------------------------
print(f"\nLoading {SPECTRA_TYPE} training data...")

params_file = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_params_train.npz")
features_file = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_logpower_train.npz")

training_parameters = np.load(params_file)
training_features = np.load(features_file)

print(f"Parameters: {training_parameters.files}")
print(f"Features shape: {training_features['features'].shape}")
print(f"Number of k-modes: {len(training_features['modes'])}")

# Verify parameter order
for i, name in enumerate(MODEL_PARAMETERS):
    if name not in training_parameters.files:
        raise ValueError(f"Parameter '{name}' not found in training data!")
    print(f"  {name}: [{training_parameters[name].min():.4f}, {training_parameters[name].max():.4f}]")

# -----------------------------
# Create and train model
# -----------------------------
print(f"\nInitializing CosmoPower NN...")
print(f"  Parameters: {MODEL_PARAMETERS}")
print(f"  Hidden layers: {HIDDEN_LAYERS}")

cp_nn = cosmopower_NN(
    parameters=MODEL_PARAMETERS,
    modes=training_features['modes'],
    n_hidden=HIDDEN_LAYERS,
    verbose=True
)

print(f"\nStarting training...")
print(f"  Learning rates: {LEARNING_RATES}")
print(f"  Batch sizes: {BATCH_SIZES}")
print(f"  Max epochs per phase: {MAX_EPOCHS}")

with tf.device(device):
    cp_nn.train(
        training_parameters=training_parameters,
        training_features=training_features['features'],
        filename_saved_model=MODEL_NAME,

        validation_split=0.2,

        learning_rates=LEARNING_RATES,
        batch_sizes=BATCH_SIZES,
        gradient_accumulation_steps=[1, 1, 1, 1, 1],
        patience_values=PATIENCE,
        max_epochs=MAX_EPOCHS,
    )

print(f"\nTraining complete!")
print(f"Model saved as: {MODEL_NAME}")
