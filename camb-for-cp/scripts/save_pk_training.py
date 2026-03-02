"""
CosmoSIS module to save CAMB power spectra for emulator training.

Output format (per row):
  [h0, omega_m, omega_b, n_s, log1e10As, mnu, z, P(k1), P(k2), ..., P(kN)]

Files produced:
  - k_modes.txt: k values in h/Mpc
  - linear.dat (or linear_rank{N}.dat in MPI mode): linear P(k) for each (cosmology, z) combination
  - boost.dat (or boost_rank{N}.dat in MPI mode): non-linear boost P_nl/P_lin for each (cosmology, z) combination

In MPI mode, each rank writes to separate files. Use merge_pk_outputs.py to combine them.
"""

import numpy as np
from cosmosis.datablock import names
import os

# Check for MPI
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1

# ============================================================
# PARAMETER NAMES - ORDER MUST MATCH create_lhs_params_list.py
# ============================================================
PARAM_NAMES = [
    "h0",
    "omega_m",
    "omega_b",
    "n_s",
    "log1e10as",  # CosmoSIS lowercases parameter names
    "mnu",
]

# Output files - use rank-specific names in MPI mode
if SIZE > 1:
    LINEAR_FILE = "linear_rank{}.dat".format(RANK)
    BOOST_FILE = "boost_rank{}.dat".format(RANK)
else:
    LINEAR_FILE = "linear.dat"
    BOOST_FILE = "boost.dat"
K_FILE = "k_modes.txt"

# -----------------------------
# Utility functions
# -----------------------------
def write_rows(filename, rows):
    """Append rows to file in binary mode for speed."""
    with open(filename, "ab") as f:
        np.savetxt(f, rows, fmt="%.8e")


def setup(options):
    """
    Initialize module. Delete old output files if they exist.
    """
    # Each rank cleans up its own files
    for f in [LINEAR_FILE, BOOST_FILE]:
        if os.path.exists(f):
            os.remove(f)

    # Only rank 0 handles k_modes.txt
    if RANK == 0 and os.path.exists(K_FILE):
        os.remove(K_FILE)

    return {"k_saved": False, "count": 0}


def execute(block, config):
    """
    Extract P(k,z) from CAMB and save to files.

    Each cosmology sample produces nz rows (one per redshift).
    """
    try:
        # --- Fetch CAMB outputs ---
        pk_lin = block[names.matter_power_lin, "P_k"]   # shape (nz, nk)
        pk_nl = block[names.matter_power_nl, "P_k"]     # shape (nz, nk)
        z_arr = block[names.matter_power_lin, "z"]      # shape (nz,)
        k = block[names.matter_power_lin, "k_h"]        # shape (nk,)

        # --- Save k-modes once (rank 0 only) ---
        if not config["k_saved"]:
            if RANK == 0:
                np.savetxt(K_FILE, k, fmt="%.10e")
                print("Saved {} k-modes to {}".format(len(k), K_FILE))
            config["k_saved"] = True

        # --- Get cosmological parameters ---
        cosmo = np.array([
            block[names.cosmological_parameters, p]
            for p in PARAM_NAMES
        ])

        # --- Build output rows ---
        linear_rows = []
        boost_rows = []

        for iz, z in enumerate(z_arr):
            Plin = pk_lin[iz, :]
            Pnl = pk_nl[iz, :]

            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                boost = np.where(Plin > 0, Pnl / Plin, 1.0)

            # Row format: [params..., z, P(k)...]
            row_base = np.hstack((cosmo, z))
            linear_rows.append(np.hstack((row_base, Plin)))
            boost_rows.append(np.hstack((row_base, boost)))

        # --- Write to disk ---
        write_rows(LINEAR_FILE, linear_rows)
        write_rows(BOOST_FILE, boost_rows)

        # --- Progress counter ---
        config["count"] += 1
        if config["count"] % 1000 == 0:
            print(f"Processed {config['count']} cosmologies...")

        return 0  # success

    except Exception as e:
        print(f"Error in save_pk_training.execute: {e}")
        import traceback
        traceback.print_exc()
        return 1  # failure


def cleanup(config):
    """Print final summary."""
    print("\nRank {} finished processing {} cosmologies".format(RANK, config['count']))
    print("Output files: {}, {}".format(LINEAR_FILE, BOOST_FILE))
    if RANK == 0:
        print("k-modes saved to: {}".format(K_FILE))
        if SIZE > 1:
            print("\nTo merge rank files, run: python merge_pk_outputs.py")
    return 0
