#!/bin/bash
#SBATCH --job-name=merge_pk
#SBATCH --qos=debug
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --account=des
#SBATCH --output=merge_pk_%j.out
#SBATCH --error=merge_pk_%j.err

cd $SLURM_SUBMIT_DIR

echo "Starting streaming merge"
echo "Date: $(date)"

python merge_pk_outputs_parallel.py --clean

echo "Finished: $(date)"
