#!/bin/bash
#SBATCH --job-name=camb_smoke
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --account=des
#SBATCH --output=logs/camb_smoke_%A_%a.out
#SBATCH --error=logs/camb_smoke_%A_%a.err
#SBATCH --array=0-0

# 10-sample smoke test for the 1M pipeline on the debug qos.
# Usage:
#   sbatch --export=BOX=wide  slurm/submit_camb_smoke.sh
#   sbatch --export=BOX=dense slurm/submit_camb_smoke.sh
#   sbatch --export=BOX=ultra slurm/submit_camb_smoke.sh
#
# Reads slices_smoke_${BOX}/slice_000.list (10 samples), writes
# smoke${BOX}_linear_v2.dat etc. Exit 0 + nonempty .dat = pipeline OK.

export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
export OMP_NUM_THREADS=1
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc \
    /global/common/software/des/common/Conda_Envs/y3cl_je

cd "${SLURM_SUBMIT_DIR:-$PSCRATCH/camb-emulator/camb-for-cp}"

BOX=${BOX:-wide}
case "${BOX}" in
    wide|dense|ultra) ;;
    *) echo "ERROR: BOX must be wide/dense/ultra (got '${BOX}')"; exit 1 ;;
esac

export LHS_SLICE_FILE=./slices_smoke_${BOX}/slice_000.list
export SAVE_PK_PREFIX=smoke${BOX}_
export SLICE_ID=smoke

for f in "${SAVE_PK_PREFIX}linear_v2.dat" \
         "${SAVE_PK_PREFIX}linear_nonu_v2.dat" \
         "${SAVE_PK_PREFIX}k_modes_v2.txt"; do
    [ -f "$f" ] && rm -f "$f"
done

echo "========== camb_smoke (${BOX}) =========="
echo "slice file:  ${LHS_SLICE_FILE}"
echo "out prefix:  ${SAVE_PK_PREFIX}"
echo "date:        $(date)"
echo "=========================================="

cosmosis configs/camb_pipeline_training_1M.ini

echo "Finished (${BOX}): $(date)"
