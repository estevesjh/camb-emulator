import numpy as np

# Load your .npz file
data = np.load("test_LHS_CAMB.npz")

# Only the free parameter
param_name = "cosmological_parameters--omega_m"

# Extract samples
samples = data['omega_m']  # 1D array

# Save as Cosmosis list with proper header
with open("test_LHS_CAMB.list", "w") as f:
    f.write(f"# {param_name}\n")
    for val in samples:
        f.write(f"{val:.8f}\n")
print("Saved Cosmosis-compatible list file with proper header")        
