#!/bin/bash
#SBATCH --job-name=train_emulator
#SBATCH --qos=regular
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --time=09:00:00
#SBATCH --account=des_g
#SBATCH --output=train_emulator_%j.out
#SBATCH --error=train_emulator_%j.err

cd $SLURM_SUBMIT_DIR

# Activate cosmopower environment
conda activate /global/common/software/des/common/Conda_Envs/jesteves_cosmopower

echo "Starting emulator training"
echo "Date: $(date)"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Full training run with all samples
python train_emulator.py --spectra linear

echo "Finished: $(date)"
