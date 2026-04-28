# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active working directory

`camb-for-cp/` is the current pipeline. The top-level `*.py` files (`cleaning_and_split_data.py`, `cp_NNtraining_opt.py`, `create_training_spectra_mpi.py`, etc.) and `README.md` are the legacy (pre-reorganization) versions — do not edit them when implementing new features. The authoritative README is `camb-for-cp/readme.md`.

## Environment (NERSC Perlmutter)

Two conda environments, loaded in different SLURM jobs:

- **CAMB generation** (`slurm/submit_camb_training.sh`): sources `${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc` with `/global/common/software/des/common/Conda_Envs/y3cl_je`. Account: `des` (CPU).
- **Data cleaning / training** (`slurm/submit_clean_split.sh`, `submit_train*.sh`): `conda activate /global/common/software/des/common/Conda_Envs/jesteves_cosmopower`, plus `module load tensorflow/2.15.0` for the GPU jobs. Account: `des_g`. Full training requires `-C gpu&hbm80g` (A100-80GB).

SLURM scripts `cd $SLURM_SUBMIT_DIR/..`, so submit from inside `camb-for-cp/slurm/` (not from the scripts' own directory).

## Pipeline architecture

End-to-end flow (see `camb-for-cp/run_training_pipeline.sh`):

1. `scripts/create_lhs_params.py` — generates 200k Latin-Hypercube cosmologies into `LHS_params.npz`.
2. `scripts/create_lhs_params_list.py` — converts `.npz` to CosmoSIS `.list` format for the list sampler.
3. `cosmosis camb_pipeline_training.ini` via MPI — runs CAMB per cosmology. The CosmoSIS module `scripts/save_pk_training.py` appends one row per `(cosmology, z)` to `linear.dat` and `boost.dat`. Under MPI each rank writes to `linear_rank{N}.dat` / `boost_rank{N}.dat`; `k_modes.txt` is written only by rank 0.
4. `scripts/merge_pk_outputs_parallel.py --clean` — streams rank files into single `linear.dat` / `boost.dat` (byte-level concat via `shutil.copyfileobj`, no memory load).
5. **Train/test split is done externally via `head`/`tail` on the merged `.dat`** to produce `linear_train.dat`, `linear_test.dat`, `boost_train.dat`, `boost_test.dat`. `scripts/clean_and_split_data.py` does NOT split — it only converts the pre-split `.dat` files to `.npy` (uses polars for parsing, falls back to pandas).
6. `scripts/train_emulator.py --spectra linear|boost` — trains a CosmoPower NN. Predicts `log10 P(k)`, not `P(k)`.

Raw `.dat` files are ~60 GB each; `.npy` is preferred downstream because it supports `mmap_mode='r'`.

## Conventions and contracts

- **Parameter convention**: CosmoSIS physical density fractions (Ω), **NOT** h²-units. So `omega_m = Ω_m`, `omega_b = Ω_b`. This is why `values_training.ini` uses `omega_b = 0.05 0.055 0.06` rather than the usual 0.0224.
- **Parameter order must match across**:
  - `PARAM_NAMES` in `scripts/save_pk_training.py` (the CosmoSIS writer)
  - `PARAM_NAMES` in `scripts/clean_and_split_data.py`
  - `MODEL_PARAMETERS` in `scripts/train_emulator.py`
  - The list order in `LHS_params.list` / `create_lhs_params_list.py`
  Row layout in `.dat`: `[h0, omega_m, omega_b, n_s, log1e10As, mnu, z, P(k1)...P(kN)]`. `z` is row 7 because CAMB outputs `nz` redshifts per cosmology.
- **CosmoSIS lowercases parameter names** when reading the datablock — `save_pk_training.py` must query `"log1e10as"` (lowercase) even though the config file declares `log1e10As`.
- `save_pk_training.py` is a CosmoSIS module (has `setup`/`execute`/`cleanup`), not a standalone script. Do not run it directly.

## Training specifics

- Single-GPU only — CosmoPower's custom optimizer is not compatible with `tf.distribute.MirroredStrategy`. `train_emulator.py` explicitly restricts visible devices to `gpus[0]`.
- Mixed precision (`float16`) is enabled globally when a GPU is present.
- Batch schedule is selected by dataset size (`train_emulator.py:144-152`): LARGE (≥5M → 200k/500k/1M), MEDIUM (≥500k → 5k/10k/50k), SMALL (1k/5k/10k). The LARGE schedule requires A100-80GB — do not use it on smaller GPUs.
- Training auto-resumes from `camb_linear_emulator.pkl` / `camb_boost_emulator.pkl` if present (`build_model()` calls `cosmopower_NN(restore=True, ...)`). Delete the `.pkl` to force a fresh run.
- Learning rate 1e-2 caused loss explosions historically; start phase 1 at 1e-3.

## Common commands

```bash
# From camb-for-cp/
sbatch slurm/submit_camb_training.sh     # generate CAMB spectra (MPI, 2 nodes, ~1.5h)
sbatch slurm/submit_merge.sh             # merge rank files
# (manually split merged .dat into _train / _test with head/tail)
sbatch slurm/submit_clean_split.sh       # .dat -> .npy
sbatch slurm/submit_train_debug.sh       # 1M-sample smoke test (30 min, debug queue)
sbatch slurm/submit_train.sh             # full training (9h, A100-80GB)

# Local/interactive equivalents
python scripts/train_emulator.py --spectra linear --nsamples 1000000
python scripts/clean_and_split_data.py
```

No test framework, no linter — this is a research pipeline.

## Planned refactor: three-quantity emulator (P_lin total, boost, P_lin no-ν)

Goal: emulate three quantities so the downstream CosmoSIS halo-model pipeline gets all the linear / non-linear spectra it needs:

1. **`linear`** — `delta_tot` linear P(k) (total matter, **includes neutrinos**). From CAMB `matter_power_lin`.
2. **`boost`** — `P_nl / P_lin` of `delta_tot`. From `matter_power_nl / matter_power_lin`.
3. **`linear_nonu`** — `delta_nonu` linear P(k) (CDM+baryon, **no neutrinos**). From CAMB `cdm_baryon_power_lin`. **NEW** — needed by `mf_tinker` with `matter_power_lin_version = 2`, which reads `cdm_baryon_power_lin` instead of `matter_power_lin` (see `cosmosis-standard-library/mass_function/mf_tinker/interface_tools.f90:49`). Without it, the Tinker mass function is called with the wrong input for massive-neutrino cosmologies.

All remaining halo-model quantities (σ(R, z), dn/dM, b(M), ξ_NL(r, z)) derive at inference from these three — see "Derived quantities" below.

Current code hard-codes `linear`/`boost` literals across `save_pk_training.py`, `clean_and_split_data.py`, `train_emulator.py`, and the SLURM scripts. A registry-based refactor removes that duplication and is what makes adding the third quantity (`linear_nonu`) cheap.

### Backup
Branch `backup/pk-emulator-only` at commit `367f56d` preserves the working P(k)-only pipeline.

### Derived quantities (NOT emulated)
The downstream CosmoSIS pipeline at `/global/common/software/des/jesteves/y1_mock_emcee.ini` reconstructs these from the emulated spectra:
- **σ(R, z)** — `cosmosis-standard-library/boltzmann/sigma_cpp/sigma_cpp.py`, a GSL top-hat integral over P(k). ~1700 integrations for a full grid.
- **dn/dM (Tinker)** — `cosmosis-standard-library/mass_function/mf_tinker/tinker_mf_module.so`; with `matter_power_lin_version = 2` it consumes **`cdm_baryon_power_lin`** (i.e. the new `linear_nonu` emulator), not `matter_power_lin`.
- **b(M)** — `y3_cluster_cpp/y3_buzzard/haloModel.py:134`, `cluster_toolkit.peak_height.nu_at_M(M, k, P_lin, Ω_m)` + Tinker-2010 eq. 6. Consumes `matter_power_lin` (total).
- **ξ_NL(r, z)** — `y3_cluster_cpp/y3_buzzard/haloModel.py:181`, `cluster_toolkit.xi.xi_mm_at_r(R, k, P_nl)`. Consumes `matter_power_nl` (total).

### Refactor — registry-based rewrite
- Introduce `scripts/spectra_registry.py` declaring each quantity: name, CAMB datablock source, grid file, cleaning predicate, dtype, optional per-quantity network overrides.
- Rewrite `save_pk_training.py` to iterate the registry and write one `.dat` per entry (shared `[params, grid, values...]` row layout). Add the `cdm_baryon_power_lin` datablock read for `linear_nonu`.
- Rewrite `clean_and_split_data.py` to loop the registry instead of the hard-coded `FILES` list.
- Rewrite `train_emulator.py` to take `--quantity NAME`; pull config from the registry.
- Update `camb_pipeline_training.ini` to add `delta_nonu` to `power_spectra` (currently `power_spectra = delta_tot`).
- Keep byte-compatible output for `linear` so the existing `camb_linear_emulator.pkl` and `evaluate_emulator.ipynb` still load.

### Follow-ups after the refactor lands
- Re-run CAMB generation once with `power_spectra = delta_tot delta_nonu` so all three `.dat` files are produced from the same 200k LHS.
- Train `boost` and `linear_nonu` to completion (`sbatch slurm/submit_train.sh --spectra <name>`).
- Sanity-check the "no new emulators needed beyond these three" claim: load all three `.pkl` files, feed predicted spectra into `cluster_toolkit.peak_height.nu_at_M`, `ct.xi.xi_mm_at_r`, and the Tinker mass function, and compare against true-CAMB outputs on held-out cosmologies. Target: <1% on b(M), <2% on ξ_NL at r < 100 Mpc/h, <2% on dn/dM over the cluster-mass range.
