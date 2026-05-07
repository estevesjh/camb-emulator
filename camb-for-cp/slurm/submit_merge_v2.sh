#!/bin/bash
#SBATCH --job-name=merge_v2
#SBATCH --qos=debug
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --account=des
#SBATCH --output=logs/merge_v2_%j.out
#SBATCH --error=logs/merge_v2_%j.err

cd $SLURM_SUBMIT_DIR/..

echo "Starting v2 streaming merge"
echo "Date: $(date)"

python scripts/merge_pk_outputs_parallel.py --clean \
    --names linear_v2 linear_nonu_v2

echo "Finished: $(date)"
