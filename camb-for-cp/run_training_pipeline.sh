#!/bin/bash
#
# CAMB Emulator Training Pipeline
#
# Run this script to execute the full training pipeline:
# 1. Generate LHS parameter samples
# 2. Create CosmoSIS list file
# 3. Run CAMB via CosmoSIS (this is the slow part)
# 4. Clean and split data
# 5. Train neural network emulator
# 6. Test emulator accuracy
#
# Prerequisites:
# - CosmoSIS installed and configured
# - pyDOE, numpy, tensorflow, cosmopower installed
#
# Usage:
#   ./run_training_pipeline.sh [--skip-camb]
#
# Options:
#   --skip-camb  Skip CAMB generation (use existing linear.dat, boost.dat)

set -e  # Exit on error

# -----------------------------
# Configuration
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for skip flag
SKIP_CAMB=false
if [[ "$1" == "--skip-camb" ]]; then
    SKIP_CAMB=true
    echo "Skipping CAMB generation (using existing data)"
fi

# -----------------------------
# Step 1: Generate LHS samples
# -----------------------------
echo ""
echo "=========================================="
echo "Step 1: Generating LHS parameter samples"
echo "=========================================="
python create_lhs_params.py

# -----------------------------
# Step 2: Create CosmoSIS list
# -----------------------------
echo ""
echo "=========================================="
echo "Step 2: Creating CosmoSIS list file"
echo "=========================================="
python create_lhs_params_list.py

# -----------------------------
# Step 3: Run CAMB via CosmoSIS
# -----------------------------
if [[ "$SKIP_CAMB" == false ]]; then
    echo ""
    echo "=========================================="
    echo "Step 3: Running CAMB via CosmoSIS"
    echo "=========================================="
    echo "This will take a long time for large sample sizes!"
    echo "Consider running on HPC with MPI parallelization."
    echo ""

    # Check if CosmoSIS is available
    if ! command -v cosmosis &> /dev/null; then
        echo "ERROR: cosmosis not found in PATH"
        echo "Please activate your CosmoSIS environment first."
        exit 1
    fi

    # Create output directory
    mkdir -p output

    # Run CosmoSIS
    # For MPI: mpirun -n $NPROCS cosmosis camb_pipeline_training.ini
    cosmosis camb_pipeline_training.ini

    echo "CAMB generation complete!"
else
    echo ""
    echo "=========================================="
    echo "Step 3: SKIPPED (using existing CAMB data)"
    echo "=========================================="
fi

# -----------------------------
# Step 4: Clean and split data
# -----------------------------
echo ""
echo "=========================================="
echo "Step 4: Cleaning and splitting data"
echo "=========================================="
python clean_and_split_data.py

# -----------------------------
# Step 5: Train emulator
# -----------------------------
echo ""
echo "=========================================="
echo "Step 5: Training neural network emulator"
echo "=========================================="
python train_emulator.py

# -----------------------------
# Step 6: Test emulator
# -----------------------------
echo ""
echo "=========================================="
echo "Step 6: Testing emulator accuracy"
echo "=========================================="
python test_emulator.py

# -----------------------------
# Done
# -----------------------------
echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - LHS_params.npz: Parameter samples"
echo "  - training_data/: Train/test data"
echo "  - camb_linear_emulator*: Trained model"
echo "  - plots/: Accuracy plots"
