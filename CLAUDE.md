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
- **D(z)** — linear growth factor. Because the v2 emulators train at z=0 only (see `camb_pipeline_training_v2.ini`: `zmin=0, zmax=0.05, nz=2`), any z>0 P(k) must be reconstructed as `P(k, z) = D(z)² · P(k, 0)`. D(z) comes from `cosmosis-standard-library/structure/growth_factor/interface.so` — a standalone ODE solver (`params:` `zmin, zmax, dz`). Measured timing on Perlmutter login: **~3 ms per sample** for 201 z-bins (z ∈ [0, 2], dz = 0.01); negligible compared to mf_tinker. Ini order must be `consistency → growth_factor → cp_camb → mf_tinker → …`. Note: `structure/extract_growth` is NOT usable here because it derives D(z) from an existing P(k, z) grid — circular when the emulator only produces P(k, z=0).
- **σ(R, z)** — `cosmosis-standard-library/boltzmann/sigma_cpp/sigma_cpp.py`, a GSL top-hat integral over P(k). ~1700 integrations for a full grid. **Bottleneck** in the emulator-fed pipeline (~200 ms/sample vs the P(k) emulator's ~10 ms). The "Planned: direct σ(M, z) emulator" section below proposes replacing this.
- **dn/dM (Tinker)** — `cosmosis-standard-library/mass_function/mf_tinker/tinker_mf_module.so`; with `matter_power_lin_version = 2` it consumes **`cdm_baryon_power_lin`** (i.e. the new `linear_nonu` emulator), not `matter_power_lin`. Currently mf_tinker runs its own σ integral internally; the σ emulator (below) aims to replace that too.
- **b(M)** — `y3_cluster_cpp/y3_buzzard/haloModel.py:134`, `cluster_toolkit.peak_height.nu_at_M(M, k, P_lin, Ω_m)` + Tinker-2010 eq. 6. Consumes `matter_power_lin` (total).
- **ξ_NL(r, z)** — `y3_cluster_cpp/y3_buzzard/haloModel.py:181`, `cluster_toolkit.xi.xi_mm_at_r(R, k, P_nl)`. Consumes `matter_power_nl` (total).

### Planned: direct σ(M, z) emulator (follow-on to the three P(k) spectra)

**Motivation.** End-to-end timing of the `y3_cluster_cpp` smoke pipeline with the cp_camb + λ-emulator replacements (see
`/global/common/software/des/jesteves/y3_cluster_cpp/CLAUDE.md`) showed that `mf_tinker` is now the bottleneck (~200 ms/sample,
~2/3 of total pipeline time), because it runs `sigma_cpp`-style top-hat integrals over the emulated P(k) grid for every mass and
redshift requested. Replacing that integration with a direct NN emulator of σ(M, z) skips the `sigma_cpp` call entirely and
should bring `mf_tinker` (or its replacement) closer to the ~10 ms range of the P(k) emulators.

**Specification.**
1. **Quantity**: `sigma` — σ(M, z) where M is halo mass in M☉/h and z is redshift. Output is log σ (or log₁₀ σ — match the sign convention of the P(k) emulators for consistency, default log₁₀).
2. **Training source**: run `cosmosis-standard-library/boltzmann/sigma_cpp/sigma_cpp.py` from a CAMB-fed datablock and capture the `sigma_r` / `sigma_m` grid per cosmology. Alternative cheaper path: derive σ(R) analytically from each CAMB P(k) in the existing training pipeline (same top-hat integral sigma_cpp does, but embedded in `save_pk_training.py`). The CAMB P(k) training data is already generated — no new MPI-CAMB job needed, only an extra sigma top-hat integral per (cosmology, z).
3. **Grid axes**:
   - M: 200 log-spaced points in [1e10, 1e16] M☉/h (cluster mass range with margin on each side). Match this to whatever `mf_tinker` internally samples so the NN output can be consumed by the downstream Tinker formula without re-interpolation.
   - z: same 100 redshifts as the P(k) training (z ∈ [0, 2]).
4. **Parameter order**: reuse the P(k) convention `[h0, omega_m, omega_b, n_s, log1e10As, mnu, z]` — M is NOT an input; the NN emits σ on the full M grid as a vector output (analogous to k being the output axis for the P(k) emulators). This keeps the architecture identical (4×512 hidden, CosmoPower activation) and the batch inference shape matches.
5. **Registry entry** (when the registry refactor lands): `sigma` with source `sigma_cpp/sigma_m`, grid file `M_modes.txt`, cleaning predicate = same-finite-and-positive check, log₁₀ transform.

**Downstream integration.**
- Add a new inference module alongside `cp_camb`: e.g. `cp_sigma` that writes `sigma_r`/`sigma_m` to the datablock in the exact format `mf_tinker` and `tinker_mass_function_2008.c` expect. Then wire `mf_tinker` to skip its own σ computation — or swap `mf_tinker` for a lightweight wrapper that consumes the pre-computed σ directly.
- Cross-check: run `mf_tinker` twice, once fed by real sigma_cpp, once fed by the σ emulator, on held-out cosmologies. Target: <1% on dn/dM across the cluster-mass range at z ∈ [0.1, 0.8] — the same budget as the ξ_NL / b(M) checks planned for the P(k) emulators.

**Trade-off**: this costs one more emulator to train, maintain, and validate, but removes the single largest runtime cost in the post-CAMB pipeline. Only attempt after the three P(k) emulators are accuracy-validated (<1% on b(M), ξ_NL, dn/dM) so the σ emulator isn't masking a P(k) training deficit.

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
