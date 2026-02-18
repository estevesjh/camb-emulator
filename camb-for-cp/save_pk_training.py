"""
CosmoSIS module to save CAMB power spectra for emulator training.

Output format (per row):
  [h0, omega_m, omega_b, n_s, log1e10As, mnu, z, P(k1), P(k2), ..., P(kN)]

Files produced:
  - k_modes.txt: k values in h/Mpc
  - linear.dat: linear P(k) for each (cosmology, z) combination
  - boost.dat: non-linear boost P_nl/P_lin for each (cosmology, z) combination
"""

import numpy as np
from cosmosis.datablock import names
import os

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

# Output files
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
    for f in [LINEAR_FILE, BOOST_FILE]:
        if os.path.exists(f):
            os.remove(f)

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

        # --- Save k-modes once ---
        if not config["k_saved"]:
            np.savetxt(K_FILE, k, fmt="%.10e")
            print(f"Saved {len(k)} k-modes to {K_FILE}")
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
    print(f"\nFinished processing {config['count']} cosmologies")
    print(f"Output files: {LINEAR_FILE}, {BOOST_FILE}, {K_FILE}")
    return 0
