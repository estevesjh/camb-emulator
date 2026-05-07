#!/bin/bash
# ---------------------------------------------------------------------------
# 1M-sample CAMB + emulator pipeline on Perlmutter. Commented step-by-step.
# Run these manually, one block at a time -- do NOT execute the whole script
# end to end. The SLURM steps return control before the cluster work finishes;
# you must wait for sacct to show COMPLETED before running the next block.
#
# Expected wall time end to end: ~40-50 h (CAMB bound; training is <30 min).
# Expected CPU cost: ~1100 CPU-h on 'preempt' qos (0.5x CPU charge after 2 h).
# ---------------------------------------------------------------------------

set -eu

REPO="$(dirname "$(realpath "$0")")"
cd "$REPO"

PYBIN_CAMB=/global/common/software/des/common/Conda_Envs/y3cl_je/bin/python
PYBIN_TRAIN=/global/common/software/des/common/Conda_Envs/jesteves_cosmopower/bin/python

# ===========================================================================
# Step 1 -- Generate the 1M LHS (wide + dense + ultra)
# ===========================================================================
# Three nested boxes based on SPT-3G + DES Y3 + cluster joint constraints:
#   wide:  300k at +/- 20 sigma  (broad coverage, tails)
#   dense: 500k at +/- 10 sigma  (dense core)
#   ultra: 200k at +/-  5 sigma  (cluster-relevant cluster core)
#
# Sampling is in physical densities (omega_b h^2, omega_cdm h^2); the derived
# (omega_m, omega_b) are written to the .npz so the downstream CAMB module
# doesn't need any changes. Amplitude is sampled in log1e10As (not sigma_8)
# to keep a single CAMB call per sample.

${PYBIN_CAMB} scripts/create_lhs_params_1M.py \
    --output-wide  data/LHS_params_1M_wide.npz \
    --output-dense data/LHS_params_1M_dense.npz \
    --output-ultra data/LHS_params_1M_ultra.npz

# Convert each .npz to a CosmoSIS .list.
${PYBIN_CAMB} scripts/make_lhs_lists_1M.py \
    --wide  data/LHS_params_1M_wide.npz \
    --dense data/LHS_params_1M_dense.npz \
    --ultra data/LHS_params_1M_ultra.npz \
    --wide-list  data/LHS_params_1M_wide.list \
    --dense-list data/LHS_params_1M_dense.list \
    --ultra-list data/LHS_params_1M_ultra.list

# ===========================================================================
# Step 2 -- Slice for SLURM array (1000 samples/task -> 300+500+200 = 1000 tasks)
# ===========================================================================
# Each SLURM array task consumes one slice serially -- no MPI. This sidesteps
# the mpi4py-vs-Cray-MPICH issue in the shared cosmosis env.

${PYBIN_CAMB} scripts/split_lhs_for_array.py \
    data/LHS_params_1M_wide.list  slices_1M_wide  300
${PYBIN_CAMB} scripts/split_lhs_for_array.py \
    data/LHS_params_1M_dense.list slices_1M_dense 500
${PYBIN_CAMB} scripts/split_lhs_for_array.py \
    data/LHS_params_1M_ultra.list slices_1M_ultra 200

# ===========================================================================
# Step 3 -- Submit the three CAMB arrays on the shared qos
# ===========================================================================
# shared qos on Perlmutter: 1 CPU per task, proven stable in the v2c run.
# %50 concurrency throttle keeps us within the per-user soft limit.
# Wall budget 2.5h/task matches what we measured for Planck-dense cosmologies.
#
# Submit all three arrays; they run in parallel because tasks are only
# 1 CPU each and shared has ample concurrency room.

JOB_WIDE=$(sbatch  --parsable --array=0-299%50 --export=BOX=wide  \
                   slurm/submit_camb_training_1M_array.sh)
echo "submitted WIDE  array: ${JOB_WIDE}  (300 tasks)"

JOB_DENSE=$(sbatch --parsable --array=0-499%50 --export=BOX=dense \
                   slurm/submit_camb_training_1M_array.sh)
echo "submitted DENSE array: ${JOB_DENSE}  (500 tasks)"

JOB_ULTRA=$(sbatch --parsable --array=0-199%50 --export=BOX=ultra \
                   slurm/submit_camb_training_1M_array.sh)
echo "submitted ULTRA array: ${JOB_ULTRA}  (200 tasks)"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER --format='%.14i %.20j %.10q %.2t %.10M %.10l %R'"
echo "  sacct  -j ${JOB_WIDE},${JOB_DENSE},${JOB_ULTRA} --format='JobID,State,ExitCode,Elapsed' --allocations"

# After all arrays COMPLETED, run the integrity check before merging.
echo ""
echo "WAIT for COMPLETION, then run:"
echo "  bash -c 'declare -A N=([wide]=300 [dense]=500 [ultra]=200)"
echo "  for box in wide dense ultra; do"
echo "    n=\${N[\$box]}"
echo "    for i in \$(seq -f \"%03g\" 0 \$((n-1))); do"
echo "      lin=slice\${i}\${box}_linear_v2.dat"
echo "      non=slice\${i}\${box}_linear_nonu_v2.dat"
echo "      [ -f \"\$lin\" ] || { echo MISSING_\$lin; continue; }"
echo "      nl=\$(wc -l < \"\$lin\"); nn=\$(wc -l < \"\$non\")"
echo "      [ \"\$nl\" = \"\$nn\" ] || echo MISMATCH slice\${i}\${box}: lin=\$nl nonu=\$nn"
echo "    done"
echo "  done'"

# ===========================================================================
# Steps 4-8 -- Merge, split, clean, train, report
# ===========================================================================
# Once all three CAMB arrays show COMPLETED and the integrity check is clean,
# continue the pipeline with run_1M_post_camb.sh (one step per invocation):
#
#   ./run_1M_post_camb.sh merge     # step 4: concat .dat files, ~5 min login
#   ./run_1M_post_camb.sh split     # step 5: shuffled 90/10 train/test
#   ./run_1M_post_camb.sh clean     # step 6: .dat -> .npy (SLURM debug)
#   ./run_1M_post_camb.sh train     # step 7: 2 GPU training jobs
#   ./run_1M_post_camb.sh report    # step 8: regenerate plots + PDF

echo ""
echo "When arrays finish, continue with:"
echo "  ./run_1M_post_camb.sh merge && ./run_1M_post_camb.sh split"
echo "  ./run_1M_post_camb.sh clean  (then wait for SLURM)"
echo "  ./run_1M_post_camb.sh train  (then wait for SLURM)"
echo "  ./run_1M_post_camb.sh report"
