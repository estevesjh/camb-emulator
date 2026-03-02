#!/bin/bash
#SBATCH --job-name=train_emulator
#SBATCH --qos=regular
#SBATCH -C gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=09:00:00
#SBATCH --account=des_g
#SBATCH --output=logs/train_emulator_%j.out
#SBATCH --error=logs/train_emulator_%j.err

cd $SLURM_SUBMIT_DIR/..

# Load NERSC TensorFlow module (includes CUDA 12.2, cuDNN, NCCL)
module load tensorflow/2.15.0

echo "Starting emulator training"
echo "Date: $(date)"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))"

# Full training run with all samples
python scripts/train_emulator.py --spectra linear

echo "Finished: $(date)"
