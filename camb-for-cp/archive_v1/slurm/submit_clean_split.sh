#!/bin/bash
#SBATCH --job-name=clean_split
#SBATCH --qos=debug
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:15:00
#SBATCH --account=des
#SBATCH --output=logs/clean_split_%j.out
#SBATCH --error=logs/clean_split_%j.err

cd $SLURM_SUBMIT_DIR/..

conda activate /global/common/software/des/common/Conda_Envs/jesteves_cosmopower

echo "Starting clean_and_split_data.py"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python scripts/clean_and_split_data.py

echo "Finished: $(date)"
