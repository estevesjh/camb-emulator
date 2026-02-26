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
#SBATCH --output=train_debug_%j.out
#SBATCH --error=train_debug_%j.err

cd $SLURM_SUBMIT_DIR

# Activate cosmopower environment
conda activate /global/common/software/des/common/Conda_Envs/jesteves_cosmopower

echo "Starting emulator training (DEBUG)"
echo "Date: $(date)"
echo "GPUs: $SLURM_GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Debug run with 100k samples
python train_emulator.py --spectra linear --nsamples 100000

echo "Finished: $(date)"
