#!/bin/bash
#SBATCH --job-name=camb_v2
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=07:00:00
#SBATCH --account=des
#SBATCH --output=logs/camb_v2_%j.out
#SBATCH --error=logs/camb_v2_%j.err

# v2 CAMB training: z=0, delta_tot + delta_nonu linear spectra.
# LHS: first 100k rows of LHS_params_v2_500k.list, set via
# camb_pipeline_training_v2.ini [list] filename.

export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
export OMP_NUM_THREADS=1
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc /global/common/software/des/common/Conda_Envs/y3cl_je

cd "$(dirname "$(realpath "$0")")/.."

echo "Starting v2 CAMB training"
echo "Date: $(date)"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Working dir: $(pwd)"

srun --mpi=cray_shasta -n $SLURM_NTASKS cosmosis configs/camb_pipeline_training_v2.ini --mpi

echo "Finished: $(date)"
