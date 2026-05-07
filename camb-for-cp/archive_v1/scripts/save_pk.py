import numpy as np
from cosmosis.datablock import names
import os

# ============================================================
# USER-DEFINED: parameters used as emulator inputs
# (order MATTERS and must never change)
# ============================================================
PARAM_NAMES = [
    "omega_m",
    #"h0",
    #"omega_b",
    #"n_s",
    #"sigma8_input",
    #"w",
    #"wa",
]

# -----------------------------
# Utility to append rows to file
# -----------------------------
def write_rows(filename, rows):
    with open(filename, "ab") as f:
        np.savetxt(f, rows, fmt="%.6e")

# -----------------------------
# Setup function
# -----------------------------
def setup(options):
    """
    Nothing to pre-load; we'll read k-modes and parameters from the block in execute()
    """
    return {"k_saved": False}

# -----------------------------
# Main execution
# -----------------------------
def execute(block, config):
    try:
        # --- Fetch CAMB outputs ---
        pk_lin = block[names.matter_power_lin, "P_k"]  # shape (nz, nk)
        pk_nl  = block[names.matter_power_nl, "P_k"]   # shape (nz, nk)
        z_arr  = block[names.matter_power_lin, "z"]    # shape (nz,)
        k      = block[names.matter_power_lin, "k_h"]  # shape (nk,)

        # --- Save k only once ---
        if not config["k_saved"]:
            np.savetxt("k_modes.txt", k, fmt="%.8e")
            config["k_saved"] = True

        # --- Cosmological parameters (fixed order!) ---
        cosmo = np.array([
            block[names.cosmological_parameters, p]
            for p in PARAM_NAMES
        ])        

        # --- Prepare rows ---
        linear_rows = []
        boost_rows  = []

        for iz, z in enumerate(z_arr):
            Plin  = pk_lin[iz, :]
            Pnl   = pk_nl[iz, :]
            boost = Pnl / Plin
            row_base = np.hstack((cosmo, z))  # cosmo params + redshift

            linear_rows.append(np.hstack((row_base, Plin)))
            boost_rows.append(np.hstack((row_base, boost)))

        # --- Write to disk ---
        write_rows("linear.dat", linear_rows)
        write_rows("boost.dat", boost_rows)

        return 0  # success

    except Exception as e:
        print("Error in save_pk.execute:", e)
        return 1  # failure

# -----------------------------
# Cleanup function
# -----------------------------
def cleanup(config):
    return 0

