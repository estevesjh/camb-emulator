#!/bin/bash
#SBATCH --job-name=clean_v2
#SBATCH --qos=debug
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:29:00
#SBATCH --account=des
#SBATCH --output=logs/clean_v2_%j.out
#SBATCH --error=logs/clean_v2_%j.err

cd "$(dirname "$(realpath "$0")")/.."

# Use jesteves_cosmopower env directly (no conda activate needed, just
# call its python binary).
PYBIN=/global/common/software/des/common/Conda_Envs/jesteves_cosmopower/bin/python
echo "Using python: ${PYBIN}"
${PYBIN} --version

echo "=== clean_and_split_data_v2 ==="
date
echo "CPUs: $SLURM_CPUS_PER_TASK"

${PYBIN} scripts/clean_and_split_data_v2.py

date
echo "Finished"
