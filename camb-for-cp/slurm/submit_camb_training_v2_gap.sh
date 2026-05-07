#!/bin/bash
#SBATCH --job-name=camb_v2_gap
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:29:00
#SBATCH --account=des
#SBATCH --output=logs/camb_v2_gap_%A_%a.out
#SBATCH --error=logs/camb_v2_gap_%A_%a.err
#SBATCH --array=0-3

# Gap-fill for slices 6 and 7 (failed in array 52193197).
# 4 half-slices of 500 samples each:
#   0: slice_006_half1 (rows 0-499)   -> prefix slice006a_
#   1: slice_006_half2 (rows 500-999) -> prefix slice006b_
#   2: slice_007_half1 (rows 0-499)   -> prefix slice007a_
#   3: slice_007_half2 (rows 500-999) -> prefix slice007b_

export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
export OMP_NUM_THREADS=1
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc /global/common/software/des/common/Conda_Envs/y3cl_je

cd "$(dirname "$(realpath "$0")")/.."

case ${SLURM_ARRAY_TASK_ID} in
    0) SLICE_FILE=./slices_v2/slice_006_half1.list; PREFIX=slice006a_ ;;
    1) SLICE_FILE=./slices_v2/slice_006_half2.list; PREFIX=slice006b_ ;;
    2) SLICE_FILE=./slices_v2/slice_007_half1.list; PREFIX=slice007a_ ;;
    3) SLICE_FILE=./slices_v2/slice_007_half2.list; PREFIX=slice007b_ ;;
esac

export LHS_SLICE_FILE=${SLICE_FILE}
export SAVE_PK_PREFIX=${PREFIX}
export SLICE_ID=gap${SLURM_ARRAY_TASK_ID}

echo "========== camb_v2 gap task =========="
echo "Array: ${SLURM_ARRAY_JOB_ID}, task: ${SLURM_ARRAY_TASK_ID}"
echo "Slice file: ${LHS_SLICE_FILE}"
echo "Output prefix: ${SAVE_PK_PREFIX}"
echo "Date: $(date)"
echo "======================================"

cosmosis configs/camb_pipeline_training_v2_array.ini

echo "Finished task ${SLURM_ARRAY_TASK_ID}: $(date)"
