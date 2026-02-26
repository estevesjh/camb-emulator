#!/bin/bash
#SBATCH --job-name=camb_training
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=2
#SBATCH --ntasks=256
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --account=des
#SBATCH --output=camb_training_%j.out
#SBATCH --error=camb_training_%j.err

# Load cosmosis environment
export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
export OMP_NUM_THREADS=4
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc /global/common/software/des/common/Conda_Envs/y3cl_je

# Change to working directory
cd $SLURM_SUBMIT_DIR

echo "Starting CAMB training run"
echo "Date: $(date)"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Working dir: $(pwd)"

# Run cosmosis with MPI
srun -n $SLURM_NTASKS cosmosis camb_pipeline_training.ini --mpi

echo "Finished: $(date)"
