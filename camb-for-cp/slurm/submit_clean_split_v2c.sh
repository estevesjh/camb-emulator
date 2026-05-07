#!/bin/bash
#SBATCH --job-name=clean_v2c
#SBATCH --qos=debug
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:29:00
#SBATCH --account=des
#SBATCH --output=logs/clean_v2c_%j.out
#SBATCH --error=logs/clean_v2c_%j.err

cd "$(dirname "$(realpath "$0")")/.."

PYBIN=/global/common/software/des/common/Conda_Envs/jesteves_cosmopower/bin/python
echo "Using python: ${PYBIN}"
${PYBIN} --version

echo "=== clean_and_split_data_v2c ==="
date

${PYBIN} <<'PY'
import gc, os, time
import numpy as np
import polars as pl

K_FILE = "k_modes_v2.txt"
OUTDIR = "./training_data_v2c"
os.makedirs(OUTDIR, exist_ok=True)

PARAM_NAMES = ["h0", "omega_m", "omega_b", "n_s", "log1e10As", "mnu"]
N_P = len(PARAM_NAMES)

FILES = [
    ("linear_v2c",      "train", "linear_v2c_train.dat"),
    ("linear_v2c",      "test",  "linear_v2c_test.dat"),
    ("linear_nonu_v2c", "train", "linear_nonu_v2c_train.dat"),
    ("linear_nonu_v2c", "test",  "linear_nonu_v2c_test.dat"),
]


def read_dat(fn):
    df = pl.read_csv(fn, separator=" ", has_header=False,
                     dtypes=[pl.Float64], ignore_errors=True)
    df = df.select([c for c in df.columns if not df[c].is_null().all()])
    return df.to_numpy()


def convert(fn, quant, split, n_k, k_modes):
    t0 = time.time()
    print(f"  Reading {fn}...")
    data = read_dat(fn)
    total = len(data)
    params  = data[:, :N_P]
    spectra = data[:, N_P:]
    del data; gc.collect()

    valid = (np.isfinite(spectra).all(axis=1)
             & np.isfinite(params).all(axis=1)
             & (spectra > 0).all(axis=1)
             & (spectra > 1e-12).all(axis=1))
    n_rm = (~valid).sum()
    params  = params[valid].astype(np.float32)
    spectra = np.log10(spectra[valid]).astype(np.float32)
    assert spectra.shape[1] == n_k, f"cols {spectra.shape[1]} != {n_k}"
    print(f"  Valid: {len(params):,}, removed: {n_rm:,} ({100*n_rm/total:.2f}%)")

    np.save(os.path.join(OUTDIR, f"camb_{quant}_params_{split}.npy"),   params)
    np.save(os.path.join(OUTDIR, f"camb_{quant}_logpower_{split}.npy"), spectra)
    if split == "train":
        np.save(os.path.join(OUTDIR, f"camb_{quant}_modes.npy"), k_modes)
    print(f"  Saved .npy ({time.time()-t0:.1f}s)")


k_modes = np.loadtxt(K_FILE)
n_k = len(k_modes)
print(f"Loaded {n_k} k-modes")

for q, s, fn in FILES:
    print(f"\n{q} {s}:")
    convert(fn, q, s, n_k, k_modes)
PY

date
echo "Finished"
