#!/bin/bash
#SBATCH --job-name=camb_v2_dbg
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --account=des
#SBATCH --output=logs/camb_v2_debug_%j.out
#SBATCH --error=logs/camb_v2_debug_%j.err

# Debug smoke test: 200 LHS rows, z=0, delta_tot + delta_nonu.
# Output files are prefixed with 'debug_' so they don't clobber the main run.

export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
export OMP_NUM_THREADS=1
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc /global/common/software/des/common/Conda_Envs/y3cl_je

cd /pscratch/sd/j/jesteves/github/camb-emulator/camb-for-cp

export SAVE_PK_PREFIX=debug_

echo "Starting v2 CAMB debug smoke test"
echo "Date: $(date)"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Working dir: $(pwd)"
echo "SAVE_PK_PREFIX: $SAVE_PK_PREFIX"

srun --mpi=cray_shasta -n $SLURM_NTASKS cosmosis configs/camb_pipeline_training_v2_debug.ini --mpi

echo "Finished: $(date)"
