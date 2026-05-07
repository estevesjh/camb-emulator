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

module load tensorflow/2.15.0

SPECTRA=${SPECTRA:-linear_v2}

echo "=== train_emulator_v2 (spectra=${SPECTRA}) [debug] ==="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))"

python scripts/train_emulator_v2.py --spectra ${SPECTRA}

date
echo "Finished"
