"""
CosmoSIS module to save CAMB linear spectra at z=0 for emulator training (v2).

Reads two datablocks per cosmology:
- matter_power_lin  (delta_tot,  includes neutrinos) -> linear_v2.dat
- cdm_baryon_power_lin (delta_nonu, no neutrinos)    -> linear_nonu_v2.dat

Row format (6 params, no z):
  [h0, omega_m, omega_b, n_s, log1e10As, mnu, P(k1), P(k2), ..., P(kN)]

z is dropped because the linear spectrum is separable:
  P_lin(k, z) = (D(z)/D(0))^2 * P_lin(k, 0)
and downstream CosmoSIS already exposes D(z) via the growth_parameters block.

Files produced:
- k_modes_v2.txt                       (rank 0 only)
- linear_v2.dat | linear_v2_rank{N}.dat
- linear_nonu_v2.dat | linear_nonu_v2_rank{N}.dat
"""

import os
import numpy as np
from cosmosis.datablock import names

try:
    from mpi4py import MPI
    # Force MPI initialization in case the launcher didn't. Without this,
    # Cray MPICH can leave each srun-launched rank thinking it is rank 0.
    if not MPI.Is_initialized():
        MPI.Init()
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1

print(f"[save_pk_training_v2] MPI rank={RANK} size={SIZE}", flush=True)

# Must match create_lhs_params_list_v2.py and clean_and_split_data_v2.py.
# CosmoSIS lowercases parameter names in the datablock.
PARAM_NAMES = [
    "h0",
    "omega_m",
    "omega_b",
    "n_s",
    "log1e10as",
    "mnu",
]

PREFIX = os.environ.get("SAVE_PK_PREFIX", "")
if SIZE > 1:
    LINEAR_FILE      = f"{PREFIX}linear_v2_rank{RANK}.dat"
    LINEAR_NONU_FILE = f"{PREFIX}linear_nonu_v2_rank{RANK}.dat"
else:
    LINEAR_FILE      = f"{PREFIX}linear_v2.dat"
    LINEAR_NONU_FILE = f"{PREFIX}linear_nonu_v2.dat"
K_FILE = f"{PREFIX}k_modes_v2.txt"

CDM_BARYON_LIN = "cdm_baryon_power_lin"


def write_rows(filename, rows):
    with open(filename, "ab") as f:
        np.savetxt(f, rows, fmt="%.8e")


def setup(options):
    for f in [LINEAR_FILE, LINEAR_NONU_FILE]:
        if os.path.exists(f):
            os.remove(f)
    if RANK == 0 and os.path.exists(K_FILE):
        os.remove(K_FILE)
    return {"k_saved": False, "count": 0}


def execute(block, config):
    try:
        # delta_tot linear
        pk_lin_tot = block[names.matter_power_lin, "P_k"]   # (nz, nk)
        z_arr      = block[names.matter_power_lin, "z"]
        k          = block[names.matter_power_lin, "k_h"]

        # delta_nonu linear (CDM + baryons only)
        pk_lin_nonu = block[CDM_BARYON_LIN, "P_k"]
        # k and z are shared with matter_power_lin because CAMB evaluates
        # both on the same output grid.

        # Expect z[0] == 0.0. nz may be > 1 (CAMB needs >=2 to build the
        # growth-factor array), but we only keep the z=0 slice — linear
        # P(k, z) = D(z)^2 * P(k, 0), so higher-z rows are redundant.
        if z_arr[0] != 0.0:
            raise RuntimeError(
                f"v2 save_pk expected z[0]=0 but got z[0]={z_arr[0]}. "
                f"Check camb_pipeline_training_v2.ini."
            )

        if not config["k_saved"]:
            if RANK == 0:
                np.savetxt(K_FILE, k, fmt="%.10e")
                print(f"Saved {len(k)} k-modes to {K_FILE}")
            config["k_saved"] = True

        cosmo = np.array([
            block[names.cosmological_parameters, p] for p in PARAM_NAMES
        ])

        # One row per quantity for the z=0 slice
        row_lin_tot  = np.hstack((cosmo, pk_lin_tot[0, :]))
        row_lin_nonu = np.hstack((cosmo, pk_lin_nonu[0, :]))

        write_rows(LINEAR_FILE,      row_lin_tot[np.newaxis, :])
        write_rows(LINEAR_NONU_FILE, row_lin_nonu[np.newaxis, :])

        config["count"] += 1
        if config["count"] % 1000 == 0:
            print(f"Rank {RANK}: processed {config['count']} cosmologies")

        return 0

    except Exception as e:
        print(f"Error in save_pk_training_v2.execute: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cleanup(config):
    print(f"\nRank {RANK} finished: {config['count']} cosmologies")
    print(f"Output: {LINEAR_FILE}, {LINEAR_NONU_FILE}")
    if RANK == 0:
        print(f"k-modes: {K_FILE}")
        if SIZE > 1:
            print("Merge with merge_pk_outputs_parallel.py --pattern linear_v2 "
                  "and --pattern linear_nonu_v2")
    return 0
