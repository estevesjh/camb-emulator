# CAMB Emulator Training Pipeline

Generate cosmological power spectra with CAMB via CosmoSIS and train a neural network emulator using CosmoPower.

## Overview

This pipeline:
1. Generates Latin Hypercube Samples (LHS) of cosmological parameters
2. Runs CAMB via CosmoSIS to compute P(k,z) for each cosmology
3. Cleans and splits data into training/test sets
4. Trains a CosmoPower neural network emulator
5. Tests emulator accuracy

## Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| h0 | [0.4, 1.0] | Reduced Hubble constant H₀/100 |
| omega_m | [0.02, 1.0] | Matter density fraction Ω_m |
| omega_b | [0.05, 0.06] | Baryon density fraction Ω_b |
| n_s | [0.87, 1.07] | Scalar spectral index |
| log1e10As | [-3.0, 7.0] | Primordial amplitude log(10¹⁰A_s) |
| mnu | [0.0, 0.2] | Neutrino mass sum Σm_ν (eV) |
| z | [0.0, 2.0] | Redshift (100 bins from CAMB) |

**Convention**: Uses CosmoSIS physical density fractions (Ω), NOT h² units.

## Quick Start

```bash
# Full pipeline (requires CosmoSIS environment)
./run_training_pipeline.sh

# Skip CAMB if data already exists
./run_training_pipeline.sh --skip-camb
```

## Step-by-Step Usage

### 1. Generate LHS parameters

```bash
python create_lhs_params.py
```

Creates `LHS_params.npz` with 200,000 cosmology samples.

### 2. Convert to CosmoSIS list format

```bash
python create_lhs_params_list.py
```

Creates `LHS_params.list` for the CosmoSIS list sampler.

### 3. Run CAMB via CosmoSIS

```bash
# Single process
cosmosis camb_pipeline_training.ini

# With MPI (recommended for large runs)
mpirun -n 16 cosmosis camb_pipeline_training.ini
```

Outputs:
- `linear.dat` - Linear P(k) for each (cosmology, z)
- `boost.dat` - Non-linear boost P_nl/P_lin
- `k_modes.txt` - k-mode values in h/Mpc

Data format per row: `[h0, omega_m, omega_b, n_s, log1e10As, mnu, z, P(k₁), P(k₂), ..., P(k_N)]`

### 4. Clean and split data

```bash
python clean_and_split_data.py
```

Creates `training_data/` directory with:
- `camb_linear_params_train.npz`, `camb_linear_params_test.npz`
- `camb_linear_logpower_train.npz`, `camb_linear_logpower_test.npz`
- Same for boost spectra

### 5. Train emulator

```bash
python train_emulator.py
```

Trains CosmoPower NN with:
- 4 hidden layers × 512 neurons
- Progressive learning rates: 1e-2 → 1e-6
- Progressive batch sizes: 1k → 50k
- Mixed precision on GPU

Output: `camb_linear_emulator` model files

### 6. Test emulator

```bash
python test_emulator.py
```

Generates accuracy metrics and plots in `plots/`:
- `linear_error_vs_k.png` - Relative error vs k
- `linear_comparison.png` - Example spectra comparisons
- `linear_error_hist.png` - Error distribution

## File Reference

### Training Pipeline (new)

| File | Description |
|------|-------------|
| `create_lhs_params.py` | Generate 200k LHS samples (6 params) |
| `create_lhs_params_list.py` | Convert .npz to CosmoSIS .list format |
| `values_training.ini` | Parameter ranges for training |
| `camb_pipeline_training.ini` | CosmoSIS pipeline config (z=0-2) |
| `save_pk_training.py` | CosmoSIS module to save P(k) |
| `clean_and_split_data.py` | Data cleaning and 80/20 split |
| `train_emulator.py` | CosmoPower NN training |
| `test_emulator.py` | Accuracy testing and plots |
| `run_training_pipeline.sh` | Master script for full pipeline |

### Test Pipeline (original)

| File | Description |
|------|-------------|
| `make_test_lhs_params.py` | Generate small test LHS (10 samples) |
| `make_test_lhs_params_list.py` | Convert test params to .list |
| `camb_pipeline.ini` | Test pipeline config (z=0-1) |
| `save_pk.py` | Test save module |
| `y1_mock_values.ini` | Test parameter values |
| `check_camb_spectra_10.py` | Plot test spectra |

## Output Structure

```
camb-for-cp/
├── LHS_params.npz              # Parameter samples
├── LHS_params.list             # CosmoSIS input
├── linear.dat                  # CAMB linear P(k)
├── boost.dat                   # CAMB non-linear boost
├── k_modes.txt                 # k-mode values
├── training_data/
│   ├── camb_linear_params_train.npz
│   ├── camb_linear_params_test.npz
│   ├── camb_linear_logpower_train.npz
│   ├── camb_linear_logpower_test.npz
│   └── ... (same for boost)
├── camb_linear_emulator*       # Trained model
├── camb_boost_emulator*        # Trained model (boost)
└── plots/
    ├── linear_error_vs_k.png
    ├── linear_comparison.png
    └── linear_error_hist.png
```

## Expected Output Size

- 200k cosmologies × 100 z-values = **20M training rows**
- Each row: 7 params + 200 k-modes = 207 floats
- `linear.dat` size: ~30 GB (text format)

## Requirements

- Python 3.8+
- CosmoSIS with CAMB module
- pyDOE (LHS sampling)
- numpy, matplotlib
- tensorflow 2.x
- cosmopower

## Notes

- CAMB computation is the bottleneck; use MPI for large runs
- GPU recommended for NN training (mixed precision enabled)
- The emulator predicts log₁₀P(k), not P(k) directly
