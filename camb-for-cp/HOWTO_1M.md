# HOWTO: generate 1M training samples on Perlmutter

Target audience: a Perlmutter-experienced collaborator reproducing or
extending the v2c emulators toward a **0.1 % accuracy** v3 emulator.

## Setup (one-time, any user on Perlmutter)

The repo is self-contained and path-portable -- clone it anywhere in
your own `$PSCRATCH` and run from there. All SLURM scripts resolve
paths relative to their own location, so no edits are required.

```bash
cd $PSCRATCH
git clone git@github.com:estevesjh/camb-emulator.git
cd camb-emulator/camb-for-cp
```

**Always `sbatch` from this directory** (`$PSCRATCH/camb-emulator/camb-for-cp`).
Every SLURM script does `cd "${SLURM_SUBMIT_DIR}"` on the compute node,
so outputs land under your own clone. The fallback if you forget to
`cd` first is `$PSCRATCH/camb-emulator/camb-for-cp/` — override with:

```bash
export PSCRATCH=/pscratch/sd/<first>/<your_user>   # if not already set
```

The cosmosis and cosmopower conda envs live under
`/global/common/software/des/common/Conda_Envs/` and are world-readable,
so any `des`-group member can use them directly without copying.
No write access to `/pscratch/sd/j/jesteves/` is required.

## Scale and budget

| Quantity | Value |
|---|---|
| LHS size | 1,000,000 (300k wide + 500k dense + 200k ultra) |
| CAMB cost | ~1000 CPU-hours @ ~60 s/sample |
| Wall time | ~30 h (all three arrays in parallel, `%50` each) |
| Charge | ~1000 CPU-h on `shared` qos (1× factor) |
| Storage | ~20 GB raw .dat + ~6 GB .npy |
| Training | 2 × A100-80GB jobs @ ~30 min each |

## Accuracy target

v2c (200k samples) achieves:
- median `|eps|` = 0.07–0.08%
- 95th percentile `|eps|` = 0.5%
- 99th percentile `|eps|` = 1.3%

1M samples with a 50/50 broad/dense split should bring:
- median `|eps|` ≈ 0.035% (~2× improvement via density scaling)
- **95th percentile `|eps|` ≈ 0.1%** (10× density in the dense core)
- 99th percentile `|eps|` ≈ 0.4%

The 95th-percentile target is the realistic "0.1 %" goal. Pushing
harder requires wider networks, not just more data.

## Parameter box

Sampling is in **physical densities** (`omega_b h^2`, `omega_cdm h^2`)
with SPT-3G + DES Y3 + cluster joint constraints as the reference.
The derived `(omega_m, omega_b)` go into the `.npz` so the downstream
CosmoSIS module is unchanged.

Central values and 1σ widths:

| Parameter | Central | 1σ |
|---|---|---|
| h₀ | 0.673 | 0.010 |
| Ω_cdm·h² | 0.1200 | 0.0025 |
| Ω_b·h² | 0.02230 | 0.00050 |
| n_s | 0.963 | 0.007 |
| ln(10¹⁰A_s) | 3.050 | 0.030 |
| Σm_ν [eV] | — | [0, 0.2] uniform |

Three nested LHS boxes:

- **Wide** (300k, ±20σ): broad coverage; captures non-Planck/non-SPT
  cosmologies for downstream robustness.
- **Dense** (500k, ±10σ): centred on the joint best-fit region where
  the cluster likelihood lives.
- **Ultra** (200k, ±5σ): cluster-relevant core, the focus region that
  drives the 0.1% accuracy target.

Derived Ω_m ranges:

| Box | Ω_m min | Ω_m max |
|---|---|---|
| Wide (±20σ) | ~0.11 | ~0.90 |
| Dense (±10σ) | ~0.19 | ~0.52 |
| Ultra (±5σ) | ~0.24 | ~0.40 |

## Pipeline

The driver script `run_1M_pipeline.sh` at repo root contains the
commands **and** comments for every step. Execute **one block at a
time** — do not run the script end-to-end, the SLURM steps return
control before the cluster work finishes.

```
camb-for-cp/
├── run_1M_pipeline.sh                     ← step-by-step driver, comments
├── scripts/
│   ├── create_lhs_params_1M.py            ← generate 1M LHS (wide+dense)
│   ├── make_lhs_lists_1M.py               ← .npz → cosmosis .list
│   └── split_lhs_for_array.py             ← slice into 500 chunks/box
├── slurm/
│   └── submit_camb_training_1M_array.sh   ← CAMB array, preempt qos
└── configs/
    └── camb_pipeline_training_v2_array.ini ← reused from v2c (no changes)
```

### Step-by-step

1. **Generate LHS** — `scripts/create_lhs_params_1M.py` — produces
   `data/LHS_params_1M_{wide,dense}.npz`.
2. **Convert to cosmosis .list** — `scripts/make_lhs_lists_1M.py`.
3. **Slice each box into 1000-sample chunks** — `split_lhs_for_array.py`.
   Writes `slices_1M_{wide,dense,ultra}/slice_NNN.list` (300, 500, 200
   slices respectively).
4. **Submit three CAMB arrays on shared qos**:
   ```bash
   sbatch --array=0-299%50 --export=BOX=wide   slurm/submit_camb_training_1M_array.sh
   sbatch --array=0-499%50 --export=BOX=dense  slurm/submit_camb_training_1M_array.sh
   sbatch --array=0-199%50 --export=BOX=ultra  slurm/submit_camb_training_1M_array.sh
   ```
   Each runs at `%50` concurrency. All three can be submitted at once --
   tasks use 1 CPU each and shared has ample concurrency room.
5. **Wait for COMPLETED** on all three arrays before merging.
6. **Integrity check**: per-slice row counts must match between
   `linear_v2.dat` and `linear_nonu_v2.dat`. The `run_1M_pipeline.sh`
   file prints the exact snippet.
7. **Post-CAMB steps** use the driver `run_1M_post_camb.sh` (one
   invocation per step):
   ```bash
   ./run_1M_post_camb.sh merge     # step 7: cat all slices per box, then boxes
   ./run_1M_post_camb.sh split     # step 8: shuffled 90/10 train/test
   ./run_1M_post_camb.sh clean     # step 9: .dat -> .npy (SLURM debug qos)
   ./run_1M_post_camb.sh train     # step 10: 2 GPU training jobs
   ./run_1M_post_camb.sh report    # step 11: regen plots + PDF
   ```
   Each step is idempotent and safe to re-run. The merge concatenates
   wide + dense + ultra into `linear_v3c.dat` and `linear_nonu_v3c.dat`;
   the split uses a shared RNG across the two files so row `i` is always
   the same cosmology in both.

## shared qos — what to know

- **Charge factor**: 1×.
- **Per-task footprint**: 1 CPU, up to 64 cores per node shared with
  other users. Our 1-CPU tasks fit comfortably.
- **Concurrency**: soft cap around `MaxJobsPU ≈ 2000`. Three arrays at
  `%50` each (150 concurrent) is polite and proven stable.
- **Wait**: usually <30 min to first task start on a typical week;
  check the NERSC queue heatmap for the 2–4 h time bucket before
  committing to the wall time budget.
- **Alternative**: `preempt` qos (0.5× charge after 2 h) is cheaper
  but can wait 24 h for first start on a busy week. Use only if the
  3-box cost becomes a concern; add `--requeue` and set
  `--qos=preempt` in the SLURM script.

## Monitoring

```bash
# Queue status
squeue -u $USER --format='%.14i %.20j %.10q %.2t %.10M %.10l %R'

# Per-task history
sacct -j <array_job_id> --format='JobID%-20,State,ExitCode,Elapsed,Start,End'

# Output growth -- do all boxes produce the expected ~1000 rows per slice?
for box in wide dense ultra; do
    total=$(cat slice*${box}_linear_v2.dat 2>/dev/null | wc -l)
    echo "${box}: ${total} rows"
done
```

## Failure modes (from the v2c session, flagged for prevention)

1. **conda activate in SLURM batch fails** (`.bashrc` not sourced).
   Our scripts call the env's `python` binary directly, or `source
   setup-cosmosis-nersc`. Do not add `conda activate`.

2. **`mpi4py` silently sees rank 0 on every node** — do not use
   `cosmosis --mpi`. The SLURM array sidesteps this.

3. **CAMB crashes with `zmax=0`** — `save_distances` calls
   `np.geomspace(zmax_background, zmax_logz, ...)`. The `v2_array.ini`
   config sets `zmax_background=3.0, nz_background=100` to work around
   this; do not change those values.

4. **Lustre write races** — concurrent appends corrupt files. Our
   per-slice-prefix naming (`SAVE_PK_PREFIX=slice{NNN}{box}_`) gives
   each task its own inodes.

5. **3-second "COMPLETED" means failure**. Always check output file
   growth after the first wave of tasks finishes. A properly-running
   1000-sample slice produces two ~7.7 MB `.dat` files.

6. **debug qos is tiny** (30 min wall, `MaxJobsPU=2`). Do not try to
   run the CAMB arrays there.

## References

- `README.md` — high-level overview of the v2c emulators
- `PIPELINE.md` — v2c reproduction walkthrough (shared qos)
- `report/emulator_v2_report.pdf` — v2c accuracy note
- NERSC queue-wait heatmap: <https://www.nersc.gov/users/live-status/queue-wait-times/>
