"""Numpy-only inference path for CosmoPower-trained emulators.

Consumes .npz artifacts produced by `export_cosmopower_numpy.py` and
reproduces `cosmopower_NN.forward_pass_np` line-for-line without
requiring TensorFlow. Intended for environments where TF is not
available (e.g. the CosmoSIS runtime in y3cl_je).

Public API mirrors the minimum subset of `cosmopower_NN` that
downstream consumers (the `cp_camb` CosmoSIS module) actually use:

    from cp_numpy import CosmoPowerNumpyNN
    emu = CosmoPowerNumpyNN("camb_linear_emulator.npz")
    P_log10 = emu.predictions_np(
        {"h0": np.array([0.67]), "omega_m": np.array([0.3]), ...}
    )  # shape (n_samples, n_modes)
    # Convenience for CosmoSIS: one param set, many redshifts:
    P_grid = emu.predictions_np_at_z(
        {"h0": 0.67, "omega_m": 0.3, ..., "mnu": 0.06},
        z_array=np.linspace(0, 1, 50),
    )  # shape (n_z, n_modes)

Activation function matches CosmoPower exactly:
    f(x) = (beta + (1 - beta) * sigmoid(alpha * x)) * x
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    # 0.5*(1 + tanh(x/2)) is stable for x in both large-positive and
    # large-negative limits, unlike 1/(1+exp(-x)) which overflows.
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class CosmoPowerNumpyNN:
    """Load an exported CosmoPower emulator and run inference in numpy."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        d = np.load(path, allow_pickle=False)

        schema = int(d["schema_version"])
        if schema != 1:
            raise RuntimeError(
                f"{path}: unsupported schema_version={schema}; "
                f"expected 1. Re-export with the matching exporter."
            )

        self.path = str(path)
        self.parameters = [str(s) for s in d["parameters"]]
        self.n_parameters = int(d["n_parameters"])
        self.modes = d["modes"].astype(np.float64)
        self.n_modes = int(d["n_modes"])
        self.n_layers = int(d["n_layers"])
        self.architecture = d["architecture"].astype(np.int64).tolist()

        self._pmu = d["parameters_mean"].astype(np.float32)
        self._pstd = d["parameters_std"].astype(np.float32)
        self._fmu = d["features_mean"].astype(np.float32)
        self._fstd = d["features_std"].astype(np.float32)

        self.parameters_min = d["parameters_min"].astype(np.float64)
        self.parameters_max = d["parameters_max"].astype(np.float64)

        self._W = [d[f"W{i}"].astype(np.float32) for i in range(self.n_layers)]
        self._b = [d[f"b{i}"].astype(np.float32) for i in range(self.n_layers)]
        self._alpha = [d[f"alpha{i}"].astype(np.float32)
                       for i in range(self.n_layers - 1)]
        self._beta = [d[f"beta{i}"].astype(np.float32)
                      for i in range(self.n_layers - 1)]

        # Track which params have tripped the prior-box clip. One warning
        # per name per process — keeps MCMC logs clean.
        self._warned: set[str] = set()

    # ------------------------------------------------------------------

    def _clip_to_prior(self, arr: np.ndarray) -> np.ndarray:
        """In-place-by-copy clip to the LHS training box. Warn once per
        parameter that trips the wall."""
        lo = self.parameters_min.astype(arr.dtype)
        hi = self.parameters_max.astype(arr.dtype)
        below = arr < lo
        above = arr > hi
        hit = below | above
        if hit.any():
            for j, name in enumerate(self.parameters):
                if hit[:, j].any() and name not in self._warned:
                    self._warned.add(name)
                    log.warning(
                        "CosmoPowerNumpyNN(%s): %s outside training box "
                        "[%g, %g]; clipping. (1 warning per parameter)",
                        Path(self.path).name, name, lo[j], hi[j],
                    )
            arr = np.minimum(np.maximum(arr, lo), hi)
        return arr

    def _dict_to_ordered_arr(self, params: dict[str, np.ndarray]) -> np.ndarray:
        """Stack params into shape (n_samples, n_parameters) in self.parameters order."""
        cols = []
        for name in self.parameters:
            if name not in params:
                raise KeyError(
                    f"missing parameter {name!r}. Expected keys: {self.parameters}"
                )
            cols.append(np.atleast_1d(np.asarray(params[name], dtype=np.float32)))
        # Broadcast scalars against arrays
        n = max(c.size for c in cols)
        stacked = np.empty((n, len(cols)), dtype=np.float32)
        for j, c in enumerate(cols):
            if c.size == 1:
                stacked[:, j] = c.item()
            elif c.size == n:
                stacked[:, j] = c
            else:
                raise ValueError(
                    f"parameter {self.parameters[j]!r} has size {c.size}, "
                    f"expected 1 or {n}"
                )
        return stacked

    # ------------------------------------------------------------------

    def predictions_np(self, params: dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass through the NN.

        Returns
        -------
        array of shape (n_samples, n_modes), in log10 P(k) units (the
        CosmoPower convention for our camb emulators).
        """
        arr = self._dict_to_ordered_arr(params)
        arr = self._clip_to_prior(arr)

        z = (arr - self._pmu) / self._pstd
        for i in range(self.n_layers - 1):
            a = z @ self._W[i] + self._b[i]
            g = _stable_sigmoid(self._alpha[i] * a)
            z = (self._beta[i] + (1.0 - self._beta[i]) * g) * a
        z = z @ self._W[-1] + self._b[-1]
        return (z * self._fstd + self._fmu).astype(np.float64)

    def predictions_np_at_z(
        self,
        params: dict[str, float],
        z_array: np.ndarray,
    ) -> np.ndarray:
        """Evaluate at a single cosmology across a redshift grid.

        Parameters
        ----------
        params : dict with all emulator parameter names EXCEPT `z`
                 (or including z; it'll be overwritten).
        z_array : 1-D array of redshifts to evaluate at.

        Returns
        -------
        array of shape (n_z, n_modes), in log10 P(k) units.
        """
        z_array = np.atleast_1d(np.asarray(z_array, dtype=np.float32))
        nz = z_array.size
        tiled = {k: np.full(nz, float(v), dtype=np.float32)
                 for k, v in params.items() if k != "z"}
        tiled["z"] = z_array
        return self.predictions_np(tiled)


__all__ = ["CosmoPowerNumpyNN"]
