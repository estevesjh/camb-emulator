#!/bin/bash
#SBATCH --job-name=train_debug
#SBATCH --qos=debug
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --account=des_g
#SBATCH --output=logs/train_debug_%j.out
#SBATCH --error=logs/train_debug_%j.err

cd $SLURM_SUBMIT_DIR/..

# Load NERSC TensorFlow module (includes CUDA 12.2, cuDNN, NCCL)
module load tensorflow/2.15.0

echo "Starting emulator training (DEBUG)"
echo "Date: $(date)"
echo "GPUs: $SLURM_GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))"

# Debug run with 1M samples
python scripts/train_emulator.py --spectra linear --nsamples 1000000

echo "Finished: $(date)"
