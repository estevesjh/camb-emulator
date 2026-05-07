# Pipeline on NERSC Perlmutter

End-to-end reproduction of the `v2c` emulators from scratch. Assumes
you have a NERSC account, membership in a DES-like project with CPU
(`des`) and GPU (`des_g`) allocations, and access to the shared cosmosis
env at `/global/common/software/des/common/Conda_Envs/y3cl_je` and the
cosmopower env at
`/global/common/software/des/common/Conda_Envs/jesteves_cosmopower`.

All SLURM scripts below assume the working directory
`/pscratch/sd/<u>/<user>/.../camb-for-cp/` is your repo root, and should
be submitted with `sbatch slurm/<name>.sh` from that directory. Output
files land at the repo root; intermediate slice files under
`slices_v2[_planck]/` and merged `.dat` files at the root are all
gitignored.

## 0. Environment

CosmoSIS (for CAMB generation) is activated via
`source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc <env>` in the SLURM
scripts; CosmoPower/polars is activated by calling the env's python
binary directly (we do **not** `conda activate` in batch — it fails
because `.bashrc` is not sourced).

If you need to regenerate a Python script interactively:

```bash
source /global/common/software/des/jesteves/cosmosis/setup-cosmosis-nersc \
    /global/common/software/des/common/Conda_Envs/y3cl_je
# ... or for polars / cosmopower analysis:
PYBIN=/global/common/software/des/common/Conda_Envs/jesteves_cosmopower/bin/python
${PYBIN} --version   # sanity check
```

## 1. Generate the LHS of cosmologies

Two LHS are used:

- **v2** — broad, uniform prior (initial coverage)
- **v2_planck** — Planck ±10σ on $\Omega_m$, $n_s$, $\ln(10^{10}A_s)$;
  broader $\Omega_b$ consistent with $\Omega_b h^2 = 0.022 \pm 0.0005$

```bash
PYBIN=/global/common/software/des/common/Conda_Envs/y3cl_je/bin/python

# v2 (broad prior, 500k LHS; we train on the first 100k):
${PYBIN} scripts/create_lhs_params_v2.py
${PYBIN} scripts/create_lhs_params_list_v2.py

# v2_planck (100k, Planck-tightened):
${PYBIN} scripts/create_lhs_params_v2_planck.py
# (list file is written directly by the script above)
```

Outputs (already present under `data/`):

```
data/LHS_params_v2.npz
data/LHS_params_v2_100k.list
data/LHS_params_v2_500k.list
data/LHS_params_v2_planck.npz
data/LHS_params_v2_planck.list
```

## 2. Split the LHS into SLURM-array slices

CAMB is expensive enough that we run each sample serially in a SLURM
array job, 1000 samples per task, 100 tasks per LHS.

```bash
${PYBIN} scripts/split_lhs_for_array.py \
    data/LHS_params_v2_100k.list  slices_v2         100
${PYBIN} scripts/split_lhs_for_array.py \
    data/LHS_params_v2_planck.list slices_v2_planck 100
```

This writes 100 files `slice_000.list` … `slice_099.list` under
`slices_v2/` and `slices_v2_planck/`.

**Why not use MPI?** The shared cosmosis env's `mpi4py` is not linked
to Cray MPICH on Perlmutter; under `srun` every rank reports
`Get_size() == 1` and writes to the same file, silently corrupting the
output. Running 100 serial tasks sidesteps this without modifying the
shared env.

## 3. CAMB generation (SLURM array on `shared` qos)

```bash
# v2 broad-prior run (100 tasks, 20 concurrent, ~2 h wall):
sbatch slurm/submit_camb_training_v2_array.sh

# v2_planck dense run (100 tasks, 20 concurrent, ~2 h wall):
sbatch slurm/submit_camb_training_v2_planck_array.sh
```

Each task:

1. Sources the cosmosis env.
2. Sets `SAVE_PK_PREFIX=slice{NNN}[planck]_` and
   `LHS_SLICE_FILE=./slices_v2[_planck]/slice_NNN.list`.
3. Runs `cosmosis configs/camb_pipeline_training_v2_array.ini`
   (or `_planck.ini`), which uses the list sampler with no MPI.
4. Appends one row per cosmology to
   `slice{NNN}[planck]_linear_v2.dat` and
   `slice{NNN}[planck]_linear_nonu_v2.dat`.
5. Writes a per-slice `slice{NNN}[planck]_k_modes_v2.txt`; these should
   be byte-identical across slices (safe to cross-check with `md5sum`).

### Monitoring

```bash
squeue -u $USER --format="%.14i %.20j %.10q %.2t %.10M %R"
sacct  -j <array_id> --format="JobID,State,ExitCode,Elapsed"
```

Expected: 98–100 % yield. A handful of outlier cosmologies hit CAMB
convergence issues and fail; the surviving 99k/100k is fine.

### Integrity check (critical — do not skip)

Before merging, verify per-slice row counts match between the two
quantities. A mismatch means `save_pk_training_v2.py` wrote one file
but not the other (usually a Lustre `PermissionError` transient or a
mid-write cancel); fix it by truncating the longer file to the shorter
length.

```bash
for i in $(seq -f "%03g" 0 99); do
  lin=slice${i}[planck_]linear_v2.dat
  non=slice${i}[planck_]linear_nonu_v2.dat
  [ -f "$lin" ] || continue
  nl=$(wc -l < "$lin"); nn=$(wc -l < "$non")
  [ "$nl" = "$nn" ] || echo "MISMATCH slice${i}: lin=$nl nonu=$nn"
done
```

## 4. Merge slices

```bash
sbatch slurm/submit_merge_v2_slices.sh            # for v2
# (for v2_planck: adapt the script or run manually)
```

Or inline (small, ~3 min):

```bash
${PYBIN} scripts/merge_pk_outputs_parallel.py \
    --glob-template 'slice*_{name}.dat' \
    --output-template '{name}.dat' \
    --names linear_v2 linear_nonu_v2
```

Merged files: `linear_v2.dat`, `linear_nonu_v2.dat`, each ~180k rows
(after both arrays).

## 5. Combine v2 + v2_planck and shuffled 90/10 split

```bash
cat linear_v2.dat linear_v2_planck.dat      > linear_v2c.dat
cat linear_nonu_v2.dat linear_nonu_v2_planck.dat > linear_nonu_v2c.dat

# Shuffle both with the same permutation, then split 90/10:
PYBIN=/global/common/software/des/common/Conda_Envs/jesteves_cosmopower/bin/python
${PYBIN} -c "
import numpy as np, polars as pl
for name in ('linear_v2c', 'linear_nonu_v2c'):
    df = pl.read_csv(f'{name}.dat', separator=' ', has_header=False,
                     schema_overrides=[pl.Float64], ignore_errors=True)
    df = df.select([c for c in df.columns if not df[c].is_null().all()])
    arr = df.to_numpy()
    rng = np.random.default_rng(20260430)
    perm = rng.permutation(len(arr))  # re-seed per file? use shared if desired
    arr = arr[perm]
    n_train = int(len(arr) * 0.9)
    np.savetxt(f'{name}_train.dat', arr[:n_train], fmt='%.8e')
    np.savetxt(f'{name}_test.dat',  arr[n_train:], fmt='%.8e')
"
```

(The actual production run used a *shared* shuffle across linear and
nonu so row `i` of both files is the same cosmology — see the session
notes. The simplest way is to read both, permute once, save both.)

## 6. Convert .dat → .npy

```bash
sbatch slurm/submit_clean_split_v2c.sh
```

This runs `scripts/clean_and_split_data_v2.py` under the cosmopower env
(polars-backed CSV parse, then `np.save` to `training_data_v2c/`). ~30 s
for 200k rows. Output: 10 `.npy` files (params train/test, logpower
train/test, modes) for each of `linear_v2c` and `linear_nonu_v2c`.

## 7. Train the emulators (gpu_debug qos)

```bash
sbatch --export=SPECTRA=linear_v2c      slurm/submit_train_v2_debug.sh
sbatch --export=SPECTRA=linear_nonu_v2c slurm/submit_train_v2_debug.sh
```

Each job: 1× A100-80GB, 7 min wall. Three-phase schedule
(lr = $10^{-3}, 3\times 10^{-4}, 10^{-4}$; batch = $10^3, 5\times 10^3,
10^4$; epochs = 400/800/1200) with patience 30. Checkpoint auto-resume
from `camb_{spectra}_emulator.pkl` if present; delete the pkl first to
retrain from scratch.

Exports a pickle + numpy-wrapper weights:

- `camb_linear_v2c_emulator.pkl`  (cosmopower native)
- `camb_linear_v2c_emulator.npz`  (numpy weights for `cp_numpy.py`)

Move into `models/` and update downstream consumers as needed.

## 8. Evaluation & report

```bash
${PYBIN} scripts/make_report_plots.py      # regenerate plots/
cd report && pdflatex emulator_v2_report.tex && pdflatex emulator_v2_report.tex
```

Outputs `report/emulator_v2_report.pdf` (5 pp). Downstream
number-counts validation lives at `hmf_report/` and
`/global/common/software/des/jesteves/y3_cluster_cpp/validations/hmf_report/`.

---

## NERSC-specific pitfalls observed during v2c development

1. **`conda activate` fails in batch** — `.bashrc` is not sourced.
   Call `<env>/bin/python` directly, or `source
   setup-cosmosis-nersc`.
2. **`mpi4py` in the shared env is not Cray-MPICH-linked** —
   every rank sees `size=1`. Do not rely on `cosmosis --mpi` without
   first confirming rank counts in stdout. We use SLURM arrays.
3. **Debug QoS is tight**: `MaxWall=30 min`, `MaxJobsPU=2`,
   `MaxSubmitPU=5`. Not usable for 100-task arrays. Use `shared`.
4. **CAMB config with `zmax=0` crashes** in `save_distances`
   (geomspace(0, …)). Set `zmax_background = 3.0, nz_background = 100`
   even for a z=0 training run.
5. **Lustre write races** — concurrent appends by 64 MPI ranks
   corrupted files during v1 development. Partition by rank/slice
   prefix (`SAVE_PK_PREFIX`) rather than relying on file locking.
6. **Integrity check every time** — compare row counts per slice,
   across sibling files, before merging.

## Contact

J. Esteves — NERSC username `jesteves`. Historical v1 pipeline
preserved at `archive_v1/` for reference (do not use for new work).
