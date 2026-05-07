# camb-for-cp — Linear P(k) emulators for DES halo-model pipelines

CosmoPower neural-network emulators of the CAMB linear matter power
spectrum at z = 0, over a six-parameter $\Lambda$CDM cosmology. Two
independent networks:

- `camb_linear_v2c_emulator` — total-matter $P^{\rm mm}(k)$ (includes
  massive neutrinos), consumed as `matter_power_lin` in CosmoSIS.
- `camb_linear_nonu_v2c_emulator` — CDM + baryon $P^{\rm cb}(k)$,
  consumed as `cdm_baryon_power_lin` by `mf_tinker` with
  `matter_power_lin_version = 2`.

Redshift dependence of the linear power spectrum is separable:
$P_{\rm lin}(k, z) = D(z)^2 \, P_{\rm lin}(k, 0)$, so the emulator runs
at $z = 0$ only and downstream CosmoSIS uses the
`structure/growth_factor` module for $D(z)$.

## Accuracy (v2c, test set = 19,994 cosmologies × 506 k-modes)

|                          | $P^{\rm mm}(k)$ | $P^{\rm cb}(k)$ |
|--------------------------|-----------------|-----------------|
| median $|\epsilon|$      | 0.068 %         | 0.076 %         |
| 95th percentile          | 0.46 %          | 0.54 %          |
| 99th percentile          | 1.27 %          | 1.49 %          |
| fraction $< 1 \%$        | 98.5 %          | 98.1 %          |

$\epsilon(k) \equiv P_{\rm pred}(k)/P_{\rm true}(k) - 1$.

## Layout

```
camb-for-cp/
├── README.md           ← you are here
├── PIPELINE.md         ← Perlmutter step-by-step reproduction guide
├── configs/            ← CAMB & CosmoSIS configs (v2 + v2_planck)
├── scripts/            ← Python scripts (LHS, CAMB saver, cleaner, trainer)
├── slurm/              ← SLURM submission scripts
├── models/             ← trained emulators (.pkl + .npz numpy)
├── data/               ← LHS input files (gitignored; regenerable)
├── report/             ← accuracy note (5 pp, LaTeX + plots)
├── hmf_report/         ← downstream number-counts validation
├── cp_numpy.py                   ← numpy-only inference wrapper (used by cp_camb)
├── export_cosmopower_numpy.py    ← .pkl → .npz export for the wrapper
└── archive_v1/         ← historical v1 pipeline (do not use for new work)
```

Large intermediate files (slice .dat, merged .dat, training-data .npy
caches) live at `/pscratch/sd/j/jesteves/camb-emulator-archive/` —
regeneratable from the scripts in this repo.

## Quick start

Load a trained emulator in a downstream Python pipeline:

```python
from cosmopower import cosmopower_NN

emu = cosmopower_NN(restore=True,
                    restore_filename="/path/to/camb-for-cp/models/camb_linear_v2c_emulator")
params = {"h0": [0.67], "omega_m": [0.315], "omega_b": [0.0493],
          "n_s": [0.965], "log1e10As": [3.044], "mnu": [0.06]}
log10pk = emu.predictions_np(params)  # shape (1, 506) on the k-grid
pk_mm = 10 ** log10pk[0]
```

Or use the numpy-only wrapper `cp_numpy.CosmoPowerNumpy` in
`cp_numpy.py` — no TensorFlow dependency, suitable for CosmoSIS
inline use. See `export_cosmopower_numpy.py` for how the `.npz` files
under `models/` are produced.

## Retraining or extending

See `PIPELINE.md` for the end-to-end reproduction walkthrough on NERSC
Perlmutter, including environment setup, CAMB generation, data
cleaning, and GPU training.

## References

- `report/emulator_v2_report.pdf` — accuracy note
- `hmf_report/report.pdf` — number-counts validation (Planck cosmology
  scan) from the downstream `y3_cluster_cpp` pipeline
- CosmoPower: [Spurio Mancini et al. 2022, MNRAS 511, 1771](https://arxiv.org/abs/2106.03846)
