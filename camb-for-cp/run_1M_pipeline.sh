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

REPO=/pscratch/sd/j/jesteves/github/camb-emulator/camb-for-cp
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
# Step 4 -- Merge per-slice outputs
# ===========================================================================
# After both arrays COMPLETED and the integrity check is clean, merge each
# box into one pair of (linear, linear_nonu) .dat files, then concatenate
# into the v3c training set.

# (Run each of the following on the login node; each is a ~5 min streaming
# cat, not SLURM-sized work.)

cat <<'MERGE' > /dev/null
${PYBIN_CAMB} scripts/merge_pk_outputs_parallel.py \
    --glob-template 'slice*wide_{name}.dat' \
    --output-template '{name}_1M_wide.dat' \
    --names linear_v2 linear_nonu_v2

${PYBIN_CAMB} scripts/merge_pk_outputs_parallel.py \
    --glob-template 'slice*dense_{name}.dat' \
    --output-template '{name}_1M_dense.dat' \
    --names linear_v2 linear_nonu_v2

${PYBIN_CAMB} scripts/merge_pk_outputs_parallel.py \
    --glob-template 'slice*ultra_{name}.dat' \
    --output-template '{name}_1M_ultra.dat' \
    --names linear_v2 linear_nonu_v2

cat linear_v2_1M_wide.dat      linear_v2_1M_dense.dat      linear_v2_1M_ultra.dat      > linear_v3c.dat
cat linear_nonu_v2_1M_wide.dat linear_nonu_v2_1M_dense.dat linear_nonu_v2_1M_ultra.dat > linear_nonu_v3c.dat
MERGE

# ===========================================================================
# Step 5 -- Shuffled 90/10 train/test split
# ===========================================================================
# Shared permutation across linear and linear_nonu so row i is the same
# cosmology in both files. Seed is hardcoded so re-running reproduces the
# same split.

cat <<'SPLIT' > /dev/null
${PYBIN_TRAIN} -c "
import numpy as np, polars as pl
# Read both with the SAME row order -- cat preserves slice order, slices
# are parallel across linear/nonu, so they are already aligned.
A = pl.read_csv('linear_v3c.dat', separator=' ', has_header=False,
                schema_overrides=[pl.Float64], ignore_errors=True)
A = A.select([c for c in A.columns if not A[c].is_null().all()]).to_numpy()
B = pl.read_csv('linear_nonu_v3c.dat', separator=' ', has_header=False,
                schema_overrides=[pl.Float64], ignore_errors=True)
B = B.select([c for c in B.columns if not B[c].is_null().all()]).to_numpy()
assert A.shape == B.shape, f'{A.shape} vs {B.shape}'
rng = np.random.default_rng(20260507)
perm = rng.permutation(len(A))
A, B = A[perm], B[perm]
n_train = int(len(A) * 0.9)
np.savetxt('linear_v3c_train.dat',      A[:n_train], fmt='%.8e')
np.savetxt('linear_v3c_test.dat',       A[n_train:], fmt='%.8e')
np.savetxt('linear_nonu_v3c_train.dat', B[:n_train], fmt='%.8e')
np.savetxt('linear_nonu_v3c_test.dat',  B[n_train:], fmt='%.8e')
print(f'train={n_train} test={len(A)-n_train}')
"
SPLIT

# ===========================================================================
# Step 6 -- .dat -> .npy  (submits clean job on the debug qos)
# ===========================================================================
# Reuse submit_clean_split_v2c.sh but point it at the v3c files. If you keep
# the same naming convention (linear_v3c / linear_nonu_v3c) you may need a
# new slurm script; otherwise, rename the merged files to linear_v2c*.dat
# so the existing script works unchanged.

cat <<'CLEAN' > /dev/null
sbatch slurm/submit_clean_split_v2c.sh  # (adapt SPECTRA env or file paths)
CLEAN

# ===========================================================================
# Step 7 -- Train both emulators on GPU
# ===========================================================================
# 1M samples move the pipeline into the MEDIUM batch schedule in
# train_emulator_v2.py: batches 5k / 10k / 50k, max 400/800/1200 epochs,
# patience 30. On A100-80GB: ~20-40 min per emulator.

cat <<'TRAIN' > /dev/null
sbatch --export=SPECTRA=linear_v3c      slurm/submit_train_v2_debug.sh
sbatch --export=SPECTRA=linear_nonu_v3c slurm/submit_train_v2_debug.sh
TRAIN

# ===========================================================================
# Step 8 -- Regenerate report and update plots
# ===========================================================================

cat <<'REPORT' > /dev/null
${PYBIN_TRAIN} scripts/make_report_plots.py   # adapt to load v3c emulators
cd report && pdflatex emulator_v2_report.tex && pdflatex emulator_v2_report.tex
REPORT

echo ""
echo "NOTE: Step 1-3 are live. Steps 4-8 are printed above as templates."
echo "      Execute them manually after each prior step is verified COMPLETED."
