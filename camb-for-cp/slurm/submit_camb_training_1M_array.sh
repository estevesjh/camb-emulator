#!/bin/bash
#SBATCH --job-name=camb_1M
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:30:00
#SBATCH --account=des
#SBATCH --output=logs/camb_1M_%A_%a.out
#SBATCH --error=logs/camb_1M_%A_%a.err
#SBATCH --array=0-0%50

# 1M-sample CAMB generation on the shared qos.
# Runs *one* box at a time; choose by exporting BOX and overriding --array:
#
#   sbatch --array=0-299%50 --export=BOX=wide   slurm/submit_camb_training_1M_array.sh
#   sbatch --array=0-499%50 --export=BOX=dense  slurm/submit_camb_training_1M_array.sh
#   sbatch --array=0-199%50 --export=BOX=ultra  slurm/submit_camb_training_1M_array.sh
#
# (Defaults above are placeholders; the driver script run_1M_pipeline.sh
#  issues the correct --array per box.)
#
# Each task processes 1000 samples serially via cosmosis (no MPI --
# mpi4py in the shared env is not Cray-MPICH-linked). Output files are
# prefixed 'slice{NNN}{box}_' so boxes don't collide and concurrent tasks
# don't touch the same inode. Wall is 2.5 h (safe above the 2 h median we
# measured for Planck-dense cosmologies on the v2c run).

export OMP_NUM_THREADS=1

# Use the y3_cluster_cpp prescription (~/cosmosis_init.sh convention):
# source setup-cosmosis-nersc directly. It internally runs
# 'module load python/3.9' (which in turn 'module load conda' + activates
# nersc-python) and then 'conda activate y3cl_je'. No manual conda init
# needed -- doing it before 'module load python' was actively breaking
# things.
export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc \
    /global/common/software/des/common/Conda_Envs/y3cl_je

cd "${SLURM_SUBMIT_DIR:-$PSCRATCH/camb-emulator/camb-for-cp}"

# -- Box selection --------------------------------------------------------
BOX=${BOX:-wide}
case "${BOX}" in
    wide|dense|ultra) ;;
    *)
        echo "ERROR: BOX must be 'wide', 'dense', or 'ultra' (got '${BOX}')"
        exit 1
        ;;
esac
SLICE_DIR=./slices_1M_${BOX}
SLICE_ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
export SLICE_ID
export LHS_SLICE_FILE=${SLICE_DIR}/slice_${SLICE_ID}.list
export SAVE_PK_PREFIX=slice${SLICE_ID}${BOX}_

# -- Re-run safety --------------------------------------------------------
# If a previous (preempted) attempt left partial output, clean it so
# append-mode writes start fresh. This is idempotent per slice because
# the slice file is the only input.
for f in "${SAVE_PK_PREFIX}linear_v2.dat" \
         "${SAVE_PK_PREFIX}linear_nonu_v2.dat" \
         "${SAVE_PK_PREFIX}k_modes_v2.txt"; do
    [ -f "$f" ] && rm -f "$f"
done

echo "========== camb_1M array task =========="
echo "Box:          ${BOX}"
echo "Array:        ${SLURM_ARRAY_JOB_ID}, task: ${SLURM_ARRAY_TASK_ID}"
echo "Slice file:   ${LHS_SLICE_FILE}"
echo "Output prefix: ${SAVE_PK_PREFIX}"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "========================================="

# Uses the v2_array.ini config (reads ${LHS_SLICE_FILE} from env).
cosmosis configs/camb_pipeline_training_1M.ini

echo "Finished task ${SLURM_ARRAY_TASK_ID} (${BOX}): $(date)"
