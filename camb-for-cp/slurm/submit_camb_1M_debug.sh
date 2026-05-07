#!/bin/bash
#SBATCH --job-name=camb_1M_dbg
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:25:00
#SBATCH --account=des
#SBATCH --output=logs/camb_1M_dbg_%A_%a.out
#SBATCH --error=logs/camb_1M_dbg_%A_%a.err
#SBATCH --array=0-2

# Step 4a -- debug-queue smoke test for the 1M CAMB pipeline.
# Runs one small task per box (wide, dense, ultra) on real LHS data:
# the first 100 samples of each box's slice_000. Fits comfortably in
# debug qos's 30 min / MaxJobsPU=2 caps because tasks run 2-at-a-time.
#
# Use this BEFORE the full arrays (submit_camb_training_1M_array.sh) to
# confirm the env + ini + paths all work. Expected wall per task:
# ~4-5 min (100 samples x ~2 s CAMB).
#
# Submit (after Step 2 sliced the LHSes into slices_1M_{wide,dense,ultra}):
#   sbatch slurm/submit_camb_1M_debug.sh

export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
export OMP_NUM_THREADS=1

# Initialise conda in the fresh SLURM shell (.bashrc is NOT sourced in batch).
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"

source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc \
    /global/common/software/des/common/Conda_Envs/y3cl_je

cd "${SLURM_SUBMIT_DIR:-$PSCRATCH/camb-emulator/camb-for-cp}"

# Task id -> box name
BOX_LIST=(wide dense ultra)
BOX=${BOX_LIST[${SLURM_ARRAY_TASK_ID}]}

# Build a 100-sample debug slice from the real LHS (first slice, first 100 rows).
SRC=./slices_1M_${BOX}/slice_000.list
DST=./slices_1M_${BOX}/slice_debug.list
if [ ! -f "${SRC}" ]; then
    echo "ERROR: ${SRC} not found. Run Step 2 (split_lhs_for_array) first."
    exit 1
fi
head -101 "${SRC}" > "${DST}"   # 1 header + 100 data rows

export LHS_SLICE_FILE=${DST}
export SAVE_PK_PREFIX=slice_debug_${BOX}_
export SLICE_ID=debug

# Clean any prior attempt so writes start fresh
for f in "${SAVE_PK_PREFIX}linear_v2.dat" \
         "${SAVE_PK_PREFIX}linear_nonu_v2.dat" \
         "${SAVE_PK_PREFIX}k_modes_v2.txt"; do
    [ -f "$f" ] && rm -f "$f"
done

echo "========== camb_1M_debug (${BOX}) =========="
echo "task id:     ${SLURM_ARRAY_TASK_ID}"
echo "slice file:  ${LHS_SLICE_FILE} (100 samples)"
echo "out prefix:  ${SAVE_PK_PREFIX}"
echo "date:        $(date)"
echo "============================================="

cosmosis configs/camb_pipeline_training_1M.ini

echo ""
echo "Finished (${BOX}): $(date)"

# Sanity check on output
LIN="${SAVE_PK_PREFIX}linear_v2.dat"
NON="${SAVE_PK_PREFIX}linear_nonu_v2.dat"
if [ -f "${LIN}" ] && [ -f "${NON}" ]; then
    nl=$(wc -l < "${LIN}"); nn=$(wc -l < "${NON}")
    nc=$(head -1 "${LIN}" | awk '{print NF}')
    echo "Output rows: linear=${nl}, nonu=${nn}, cols=${nc}"
    if [ "${nl}" = "${nn}" ] && [ "${nc}" = "512" ]; then
        echo "OK: matching rows, 512 cols"
    else
        echo "WARN: inconsistent rows or cols"
    fi
else
    echo "ERROR: output .dat files missing"
fi
