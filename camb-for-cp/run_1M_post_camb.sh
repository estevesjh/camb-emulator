#!/bin/bash
# ---------------------------------------------------------------------------
# Post-CAMB steps for the 1M pipeline: merge -> shuffled split -> .npy cache
# -> GPU training -> regen report.
#
# Run this AFTER:
#   (a) All three arrays from run_1M_pipeline.sh are COMPLETED.
#   (b) The integrity check at the bottom of run_1M_pipeline.sh reported
#       zero MISSING and zero MISMATCH.
#
# Run each step as a separate invocation so you can inspect outputs.
# Usage:
#   ./run_1M_post_camb.sh merge     # step 4
#   ./run_1M_post_camb.sh split     # step 5 (depends on merge)
#   ./run_1M_post_camb.sh clean     # step 6 (submits SLURM job on debug qos)
#   ./run_1M_post_camb.sh train     # step 7 (submits 2 SLURM jobs on gpu_debug)
#   ./run_1M_post_camb.sh report    # step 8 (regens plots+PDF)
# ---------------------------------------------------------------------------
set -eu

REPO="$(dirname "$(realpath "$0")")"
cd "$REPO"

PYBIN_CAMB=/global/common/software/des/common/Conda_Envs/y3cl_je/bin/python
PYBIN_TRAIN=/global/common/software/des/common/Conda_Envs/jesteves_cosmopower/bin/python

STEP=${1:-}

case "$STEP" in

merge)
    # Streaming concat: ~5 min on a login node, no SLURM needed.
    # Merges the 300 / 500 / 200 per-slice .dat files per box into one
    # pair of merged .dat files per box, then concatenates into linear_v3c
    # and linear_nonu_v3c.
    echo "[1/3] Merging wide..."
    ${PYBIN_CAMB} scripts/merge_pk_outputs_parallel.py \
        --glob-template 'slice*wide_{name}.dat' \
        --output-template '{name}_1M_wide.dat' \
        --names linear_v2 linear_nonu_v2

    echo "[2/3] Merging dense..."
    ${PYBIN_CAMB} scripts/merge_pk_outputs_parallel.py \
        --glob-template 'slice*dense_{name}.dat' \
        --output-template '{name}_1M_dense.dat' \
        --names linear_v2 linear_nonu_v2

    echo "[3/3] Merging ultra..."
    ${PYBIN_CAMB} scripts/merge_pk_outputs_parallel.py \
        --glob-template 'slice*ultra_{name}.dat' \
        --output-template '{name}_1M_ultra.dat' \
        --names linear_v2 linear_nonu_v2

    echo "Concatenating boxes into v3c..."
    cat linear_v2_1M_wide.dat      linear_v2_1M_dense.dat      linear_v2_1M_ultra.dat      > linear_v3c.dat
    cat linear_nonu_v2_1M_wide.dat linear_nonu_v2_1M_dense.dat linear_nonu_v2_1M_ultra.dat > linear_nonu_v3c.dat

    # Sanity: row counts must match across linear and nonu.
    n_lin=$(wc -l < linear_v3c.dat)
    n_nonu=$(wc -l < linear_nonu_v3c.dat)
    echo ""
    echo "linear_v3c rows:     $n_lin"
    echo "linear_nonu_v3c rows: $n_nonu"
    if [ "$n_lin" != "$n_nonu" ]; then
        echo "ERROR: merged row counts differ"
        exit 1
    fi
    echo "Merge OK."
    ;;

split)
    # Shuffled 90/10 train/test split. Shared permutation across linear
    # and linear_nonu so row i is the same cosmology in both.
    echo "Shuffling + splitting 90/10..."
    ${PYBIN_TRAIN} - <<'PY'
import numpy as np, polars as pl
for f in ('linear_v3c', 'linear_nonu_v3c'):
    assert __import__('os').path.exists(f + '.dat'), f'missing {f}.dat (run merge step first)'

# Load both (same row order from merge/cat).
dfs = {}
for name in ('linear_v3c', 'linear_nonu_v3c'):
    df = pl.read_csv(f'{name}.dat', separator=' ', has_header=False,
                     schema_overrides=[pl.Float64], ignore_errors=True)
    dfs[name] = df.select([c for c in df.columns if not df[c].is_null().all()]).to_numpy()

A, B = dfs['linear_v3c'], dfs['linear_nonu_v3c']
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
PY
    echo "Split OK."
    ;;

clean)
    # .dat -> .npy on the debug qos (~30 s). Uses the v2c clean/split
    # submit script, which already reads linear_v3c* names if you rename
    # accordingly, or you can copy submit_clean_split_v2c.sh and change
    # SPECTRA_BASE to linear_v3c.
    echo "Submitting clean_split on debug qos..."
    sbatch slurm/submit_clean_split_v2c.sh
    ;;

train)
    # Two GPU-debug jobs on A100-80GB, ~20-40 min each (1M samples move the
    # pipeline into the MEDIUM batch schedule).
    echo "Submitting two GPU training jobs..."
    sbatch --export=SPECTRA=linear_v3c      slurm/submit_train_v2_debug.sh
    sbatch --export=SPECTRA=linear_nonu_v3c slurm/submit_train_v2_debug.sh
    ;;

report)
    ${PYBIN_TRAIN} scripts/make_report_plots.py
    cd report
    pdflatex emulator_v2_report.tex
    pdflatex emulator_v2_report.tex
    echo "Report at report/emulator_v2_report.pdf"
    ;;

*)
    echo "Usage: $0 <merge|split|clean|train|report>"
    exit 1
    ;;

esac
