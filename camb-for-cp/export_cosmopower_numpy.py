"""Export a CosmoPower-trained camb emulator .pkl to a .npz usable in
environments without TensorFlow.

The .pkl stores `parameters` and `n_hidden` as TensorFlow ListWrapper
objects, so unpickling requires `tensorflow` on sys.path. This script
therefore runs in the training env (jesteves_cosmopower). It converts
the pickle to a plain numpy-serializable .npz that `cp_numpy.py` can
load anywhere (numpy-only).

Output .npz schema (version 1):
    schema_version     : int64, =1
    source_pkl_sha256  : bytes (hex str) of the original .pkl, for provenance
    parameters         : array of str (7 entries: h0, omega_m, omega_b, n_s,
                          log1e10As, mnu, z)
    n_parameters       : int64
    modes              : float64 (n_modes,)  — k-values in h/Mpc
    n_modes            : int64
    n_layers           : int64
    n_hidden           : int64 (n_layers-1,)
    architecture       : int64 (n_layers+1,)
    parameters_mean    : float32 (n_parameters,)  — training mean, used by NN
    parameters_std     : float32 (n_parameters,)
    features_mean      : float32 (n_modes,)
    features_std       : float32 (n_modes,)
    parameters_min     : float64 (n_parameters,)  — LHS training box min
    parameters_max     : float64 (n_parameters,)  — LHS training box max
    W0..W{n-1}         : float32 weight matrices
    b0..b{n-1}         : float32 bias vectors
    alpha0..alpha{n-2} : float32 activation alphas
    beta0..beta{n-2}   : float32 activation betas

Prior box (parameters_min/max) is sourced from:
  (1) --param-min / --param-max CLI flags (preferred, explicit), OR
  (2) LHS_params_summary.txt sitting next to the pkl (fallback).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

# tensorflow is imported only so that pickle.load can resolve
# ListWrapper. We don't run any tf ops.
try:
    import tensorflow  # noqa: F401
except ImportError:
    print(
        "WARNING: tensorflow not importable. If the pickle contains "
        "ListWrapper objects, unpickling will fail. Proceeding anyway.",
        file=sys.stderr,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _coerce_list(x) -> list:
    # TF ListWrapper → list; already-list pass-through.
    return list(x)


def _parse_kv_bounds(spec: str | None) -> dict[str, float]:
    if spec is None:
        return {}
    out = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, val = entry.split("=", 1)
        out[name.strip()] = float(val.strip())
    return out


def _read_lhs_summary(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Parse LHS_params_summary.txt lines like
    `h0  : range=[  0.4000,   1.0000]  ...`
    and return (mins, maxs) as name-keyed dicts.
    """
    lo, hi = {}, {}
    if not path.is_file():
        return lo, hi
    row_re = re.compile(
        r"^\s*(\w+)\s*:\s*range\s*=\s*\[\s*([-\d\.eE+]+)\s*,\s*([-\d\.eE+]+)\s*\]"
    )
    for line in path.read_text().splitlines():
        m = row_re.match(line)
        if m:
            name, lo_s, hi_s = m.groups()
            lo[name] = float(lo_s)
            hi[name] = float(hi_s)
    return lo, hi


_DEFAULT_Z_BOUNDS = (0.0, 2.0)


def export(pkl_path: Path, out_path: Path,
           param_min: dict[str, float],
           param_max: dict[str, float]) -> None:
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)

    # cosmopower_NN.restore unpacks (W_, b_, alphas_, betas_,
    # parameters_mean_, parameters_std_, features_mean_, features_std_,
    # n_parameters, parameters, n_modes, modes, n_hidden, n_layers,
    # architecture).
    if len(obj) != 15:
        raise RuntimeError(
            f"Expected 15 items in pickle, got {len(obj)}. "
            f"CosmoPower format may have drifted."
        )
    (W_, b_, alphas_, betas_,
     parameters_mean_, parameters_std_,
     features_mean_, features_std_,
     n_parameters, parameters,
     n_modes, modes,
     n_hidden, n_layers, architecture) = obj

    parameters = [str(s) for s in _coerce_list(parameters)]
    n_hidden = [int(x) for x in _coerce_list(n_hidden)]
    architecture = [int(x) for x in _coerce_list(architecture)]

    n_params_int = int(n_parameters)
    n_layers_int = int(n_layers)
    n_modes_int = int(n_modes)

    if len(parameters) != n_params_int:
        raise RuntimeError(
            f"parameters list length {len(parameters)} != n_parameters {n_params_int}"
        )
    if len(W_) != n_layers_int or len(b_) != n_layers_int:
        raise RuntimeError(
            f"W/b length {len(W_)}/{len(b_)} != n_layers {n_layers_int}"
        )
    if len(alphas_) != n_layers_int - 1 or len(betas_) != n_layers_int - 1:
        raise RuntimeError(
            f"alphas/betas length {len(alphas_)}/{len(betas_)} "
            f"!= n_layers - 1 {n_layers_int - 1}"
        )

    # Prior box: every declared parameter must be supplied.
    missing = [p for p in parameters if p not in param_min or p not in param_max]
    if missing:
        raise SystemExit(
            "Missing prior-box entries for: "
            + ", ".join(missing)
            + ".\nSupply via --param-min / --param-max or via "
            "LHS_params_summary.txt sitting next to the pkl."
        )

    pmin = np.array([param_min[p] for p in parameters], dtype=np.float64)
    pmax = np.array([param_max[p] for p in parameters], dtype=np.float64)
    if np.any(pmax <= pmin):
        raise SystemExit(
            "Every parameters_max must exceed parameters_min. Got "
            + ", ".join(f"{p}=[{lo},{hi}]"
                        for p, lo, hi in zip(parameters, pmin, pmax)
                        if hi <= lo)
        )

    blob: dict[str, np.ndarray] = {
        "schema_version": np.int64(1),
        "source_pkl_sha256": np.array(_sha256(pkl_path)),
        "parameters": np.array(parameters),
        "n_parameters": np.int64(n_params_int),
        "modes": np.asarray(modes, dtype=np.float64),
        "n_modes": np.int64(n_modes_int),
        "n_layers": np.int64(n_layers_int),
        "n_hidden": np.asarray(n_hidden, dtype=np.int64),
        "architecture": np.asarray(architecture, dtype=np.int64),
        "parameters_mean": np.asarray(parameters_mean_, dtype=np.float32),
        "parameters_std": np.asarray(parameters_std_, dtype=np.float32),
        "features_mean": np.asarray(features_mean_, dtype=np.float32),
        "features_std": np.asarray(features_std_, dtype=np.float32),
        "parameters_min": pmin,
        "parameters_max": pmax,
    }
    for i in range(n_layers_int):
        blob[f"W{i}"] = np.asarray(W_[i], dtype=np.float32)
        blob[f"b{i}"] = np.asarray(b_[i], dtype=np.float32)
    for i in range(n_layers_int - 1):
        blob[f"alpha{i}"] = np.asarray(alphas_[i], dtype=np.float32)
        blob[f"beta{i}"] = np.asarray(betas_[i], dtype=np.float32)

    np.savez(out_path, **blob)
    total = sum(a.nbytes for a in blob.values() if hasattr(a, "nbytes")) / 1e6
    print(f"wrote {out_path}  ({total:.1f} MB)")
    print(
        f"  parameters: {parameters}\n"
        f"  n_layers:   {n_layers_int}\n"
        f"  architecture: {architecture}\n"
        f"  k-modes: {n_modes_int} in [{modes.min():.2e}, {modes.max():.2e}]\n"
        f"  prior box:"
    )
    for p, lo, hi in zip(parameters, pmin, pmax):
        print(f"    {p:<12s}: [{lo:g}, {hi:g}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--pkl", required=True, type=Path,
                    help="path to the trained camb_*_emulator.pkl")
    ap.add_argument("--out", required=True, type=Path,
                    help="destination .npz path")
    ap.add_argument("--param-min", default=None,
                    help="comma-separated 'name=value' list; overrides "
                         "LHS_params_summary.txt if both present")
    ap.add_argument("--param-max", default=None,
                    help="same format as --param-min, for upper bounds")
    ap.add_argument("--lhs-summary", type=Path, default=None,
                    help="path to LHS_params_summary.txt (default: next to pkl)")
    ap.add_argument("--z-range", type=str, default="0.0,2.0",
                    help="comma-separated 'zmin,zmax' since z is not "
                         "in LHS summary (default 0.0,2.0 matching "
                         "camb_pipeline_training.ini)")
    args = ap.parse_args()

    cli_min = _parse_kv_bounds(args.param_min)
    cli_max = _parse_kv_bounds(args.param_max)

    lhs_path = args.lhs_summary or (args.pkl.parent / "LHS_params_summary.txt")
    file_min, file_max = _read_lhs_summary(lhs_path)

    # CLI wins over file. z comes from --z-range if not in either.
    z_lo_s, z_hi_s = args.z_range.split(",")
    z_lo, z_hi = float(z_lo_s), float(z_hi_s)

    pmin = {**file_min, **cli_min}
    pmax = {**file_max, **cli_max}
    pmin.setdefault("z", z_lo)
    pmax.setdefault("z", z_hi)

    export(args.pkl, args.out, pmin, pmax)


if __name__ == "__main__":
    main()
