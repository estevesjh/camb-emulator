#!/bin/bash
#SBATCH --job-name=train_v2_dbg
#SBATCH --qos=debug
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:29:00
#SBATCH --account=des_g
#SBATCH --output=logs/train_v2_dbg_%j.out
#SBATCH --error=logs/train_v2_dbg_%j.err

# Pass SPECTRA via --export, e.g.
#   sbatch --export=SPECTRA=linear_v2 slurm/submit_train_v2_debug.sh

cd "${SLURM_SUBMIT_DIR:-$PSCRATCH/camb-emulator/camb-for-cp}"

# Training uses NERSC's tensorflow/2.15 module as the python interpreter,
# with cosmopower + polars installed per-user via pip install --user.
# The module sets PYTHONUSERBASE=$HOME/.local/perlmutter/tensorflow2.15.0
# so --user packages land there and auto-import.
#
# FIRST-TIME SETUP (one-off, outside this script):
#   module load tensorflow/2.15.0
#   pip install --user cosmopower polars
# After that every training job just works.
module load tensorflow/2.15.0

SPECTRA=${SPECTRA:-linear_v2}

echo "=== train_emulator_v2 (spectra=${SPECTRA}) [debug] ==="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python --version
python -c "import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))"
python -c "import cosmopower; print('cosmopower OK')" || {
    echo ""
    echo "ERROR: cosmopower missing in this TF module env."
    echo "Run once (interactively, not in SLURM):"
    echo "    module load tensorflow/2.15.0"
    echo "    pip install --user cosmopower polars"
    echo "then resubmit this job."
    exit 1
}

python scripts/train_emulator_v2.py --spectra ${SPECTRA}

date
echo "Finished"
