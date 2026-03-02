import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("linear.dat")
k = np.loadtxt("k_modes.txt")

nrows = data.shape[0]
nsample = 10

# Pick random rows
rng = np.random.default_rng(1234)
indices = rng.choice(nrows, size=nsample, replace=False)

plt.figure(figsize=(7, 5))

for i in indices:
    row = data[i]
    omega_m = row[0]
    z = row[1]
    Pk = row[2:]

    label = rf"$\Omega_m={omega_m:.3f},\ z={z:.2f}$"
    plt.loglog(k, Pk, alpha=0.8, label=label)

print("k-modes #:", len(k))
print("Pk len:", len(Pk))

plt.xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$P(k)$")
plt.title("Linear matter power spectra (10 random samples)")
plt.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("test_camb_spectra_10.pdf")
plt.close()

