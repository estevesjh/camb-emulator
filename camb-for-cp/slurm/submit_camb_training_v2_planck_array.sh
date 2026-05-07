#!/bin/bash
#SBATCH --job-name=camb_planck
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=des
#SBATCH --output=logs/camb_planck_%A_%a.out
#SBATCH --error=logs/camb_planck_%A_%a.err
#SBATCH --array=0-99%20

# Planck-dense CAMB array: 100 serial tasks, 1000 samples each.
# Uses slices_v2_planck/ and writes slice{NNN}planck_ prefixed outputs.

export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
export OMP_NUM_THREADS=1
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc /global/common/software/des/common/Conda_Envs/y3cl_je

cd "$(dirname "$(realpath "$0")")/.."

SLICE_ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
export SLICE_ID
export LHS_SLICE_FILE=./slices_v2_planck/slice_${SLICE_ID}.list
export SAVE_PK_PREFIX=slice${SLICE_ID}planck_

echo "========== camb_v2_planck array task =========="
echo "Array: ${SLURM_ARRAY_JOB_ID}, task: ${SLURM_ARRAY_TASK_ID}"
echo "Slice file: ${LHS_SLICE_FILE}"
echo "Output prefix: ${SAVE_PK_PREFIX}"
echo "Date: $(date)"
echo "==============================================="

cosmosis configs/camb_pipeline_training_v2_planck.ini

echo "Finished task ${SLURM_ARRAY_TASK_ID}: $(date)"
