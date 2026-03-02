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
python scripts/create_lhs_params.py
```

Creates `LHS_params.npz` with 200,000 cosmology samples.

### 2. Convert to CosmoSIS list format

```bash
python scripts/create_lhs_params_list.py
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
python scripts/clean_and_split_data.py
```

Creates `training_data/` directory with:
- `camb_linear_params_train.npz`, `camb_linear_params_test.npz`
- `camb_linear_logpower_train.npz`, `camb_linear_logpower_test.npz`
- Same for boost spectra

### 5. Train emulator

```bash
# Debug run (1M samples, 30 min on debug queue)
sbatch slurm/submit_train_debug.sh

# Full run (16M samples, ~7 hours on A100-80GB)
sbatch slurm/submit_train.sh
```

Trains CosmoPower NN with:
- 4 hidden layers x 512 neurons
- 3-phase learning rate schedule: 1e-3, 3e-4, 1e-4
- Progressive batch sizes (dataset-size dependent):
  - Large (>5M): 200k, 500k, 1M
  - Medium (500k-5M): 5k, 10k, 50k
  - Small (<500k): 1k, 5k, 10k
- Mixed precision (float16) on GPU
- Requires A100-80GB for large batch schedule (`-C gpu&hbm80g`)

Output: `camb_linear_emulator.pkl` model file

### 6. Evaluate emulator

The training script automatically evaluates on the held-out test set and prints
RMSE and fractional error statistics. For detailed analysis and plots, see:

```
evaluate_emulator.ipynb
```

This notebook reproduces Figure 2 from the
[CosmoPower paper](https://arxiv.org/abs/2106.03846) and compares accuracy.

## File Reference

### Python Scripts (`scripts/`)

| File | Description |
|------|-------------|
| `create_lhs_params.py` | Generate 200k LHS samples (6 params) |
| `create_lhs_params_list.py` | Convert .npz to CosmoSIS .list format |
| `save_pk_training.py` | CosmoSIS module to save P(k) |
| `clean_and_split_data.py` | Data cleaning and 80/20 split |
| `merge_pk_outputs_parallel.py` | Merge MPI rank outputs into single files |
| `train_emulator.py` | CosmoPower NN training |
| `test_emulator.py` | Accuracy testing and plots |
| `make_test_lhs_params.py` | Generate small test LHS (10 samples) |
| `save_pk.py` | Test save module |
| `check_camb_spectra_10.py` | Plot test spectra |

### SLURM Job Scripts (`slurm/`)

| File | Description |
|------|-------------|
| `submit_camb_training.sh` | Run CAMB via CosmoSIS with MPI |
| `submit_merge.sh` | Merge parallel CAMB outputs |
| `submit_clean_split.sh` | Clean and split data |
| `submit_train.sh` | Full training (16M samples, A100-80GB) |
| `submit_train_debug.sh` | Debug training (1M samples, 30 min) |

### Config and Pipeline

| File | Description |
|------|-------------|
| `camb_pipeline_training.ini` | CosmoSIS pipeline config (z=0-2) |
| `camb_pipeline.ini` | Test pipeline config (z=0-1) |
| `values_training.ini` | Parameter ranges for training |
| `y1_mock_values.ini` | Test parameter values |
| `run_training_pipeline.sh` | Master script for full pipeline |
| `evaluate_emulator.ipynb` | Evaluation notebook with plots |

## Directory Structure

```
camb-for-cp/
├── scripts/                        # Python pipeline scripts
│   ├── create_lhs_params.py
│   ├── clean_and_split_data.py
│   ├── train_emulator.py
│   └── ...
├── slurm/                          # NERSC SLURM job scripts
│   ├── submit_train.sh
│   ├── submit_camb_training.sh
│   └── ...
├── logs/                           # SLURM job output files
├── training_data/                  # Processed .npy arrays
│   ├── camb_linear_params_train.npy
│   ├── camb_linear_logpower_train.npy
│   └── ... (train/test for linear & boost)
├── plots/                          # Evaluation plots
│   ├── fig2_error_vs_k.png
│   ├── error_per_kmode.png
│   ├── error_vs_redshift.png
│   └── example_spectra.png
├── camb_linear_emulator.pkl        # Trained model
├── evaluate_emulator.ipynb         # Evaluation notebook
├── run_training_pipeline.sh        # Master pipeline script
├── *.ini                           # CosmoSIS config files
├── linear.dat                      # Raw CAMB output (~62 GB)
└── boost.dat                       # Raw CAMB output (~62 GB)
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

## Training Results

Trained on 16M samples (200k cosmologies x 100 redshifts), evaluated on 4M held-out test samples.

### Test set accuracy

| Metric | Value |
|--------|-------|
| RMSE [log10 P(k)] | 0.0398 |
| Median |dP/P| | 0.26% |
| 95th percentile |dP/P| | 2.45% |
| 99th percentile |dP/P| | 12.7% |
| Samples < 1% error | 86.3% |
| Samples < 5% error | 97.7% |
| Samples < 10% error | 98.8% |

### Training configuration

| Phase | Learning rate | Batch size | Max epochs | Val loss |
|-------|--------------|------------|------------|----------|
| 1 | 1e-3 | 200,000 | 100 | 0.0376 |
| 2 | 3e-4 | 500,000 | 200 | 0.0351 |
| 3 | 1e-4 | 1,000,000 | 300 | 0.0345 |

Total training time: ~7 hours on a single A100-SXM4-80GB.

### Comparison with CosmoPower

See `evaluate_emulator.ipynb` for a detailed comparison with
[Spurio Mancini et al. (2022)](https://arxiv.org/abs/2106.03846).
Note that this emulator covers a wider parameter space (h in [0.4, 1.0],
Omega_m in [0.02, 1.0]) compared to the CosmoPower paper, and includes
neutrino mass as an additional parameter.

## Notes

- CAMB computation is the bottleneck; use MPI for large runs
- GPU recommended for NN training (mixed precision enabled)
- The emulator predicts log10 P(k), not P(k) directly
- Full training requires A100-80GB GPUs (`-C gpu&hbm80g` on Perlmutter)
