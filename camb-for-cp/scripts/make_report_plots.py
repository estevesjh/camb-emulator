"""
Generate report plots for the v2 linear emulators.

Outputs (into report/plots/):
  err_cdf.pdf      — |dP/P| CDF for both emulators
  err_vs_k.pdf     — median and 95th pct |dP/P| vs k
  examples.pdf     — 4 held-out cosmologies, truth vs predicted, residuals

Usage:
  $JESTEVES_COSMOPOWER_PY scripts/make_report_plots.py
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from cosmopower import cosmopower_NN

DATA_DIR = "./training_data_v2c"
PLOT_DIR = "./report/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

PARAMS = ["h0", "omega_m", "omega_b", "n_s", "log1e10As", "mnu"]

EMULATORS = [
    ("linear_v2c",      "camb_linear_v2c_emulator",      r"$P^{\rm mm}(k)$"),
    ("linear_nonu_v2c", "camb_linear_nonu_v2c_emulator", r"$P^{\rm cb}(k)$"),
]


def load_test(spectra):
    params = np.load(os.path.join(DATA_DIR, f"camb_{spectra}_params_test.npy"))
    logp   = np.load(os.path.join(DATA_DIR, f"camb_{spectra}_logpower_test.npy"))
    return params, logp


def predict(cp_nn, params_arr, batch=50000):
    out = []
    for i in range(0, len(params_arr), batch):
        p = {n: params_arr[i:i+batch, j] for j, n in enumerate(PARAMS)}
        out.append(cp_nn.predictions_np(p))
    return np.concatenate(out, axis=0)


def frac_error(log_pred, log_true):
    r = log_pred - log_true
    return np.abs(np.power(10.0, np.clip(r, -30, 30)) - 1.0)


def main():
    k = np.load(os.path.join(DATA_DIR, "camb_linear_v2c_modes.npy"))

    results = {}
    for spectra, ckpt, label in EMULATORS:
        print(f"Loading {ckpt} ...")
        cp_nn = cosmopower_NN(restore=True, restore_filename=ckpt)
        params, logp_true = load_test(spectra)
        logp_pred = predict(cp_nn, params)
        fe = frac_error(logp_pred, logp_true)
        results[spectra] = {
            "label": label, "params": params, "k": k,
            "logp_true": logp_true, "logp_pred": logp_pred, "fe": fe,
        }
        print(f"  n_test={len(params)}  median|dP/P|={np.median(fe)*100:.3f}%")

    # ---- 1) CDF of |dP/P| ----
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    colors = {"linear_v2c": "C0", "linear_nonu_v2c": "C3"}
    for spectra, r in results.items():
        flat = r["fe"].ravel()
        xs = np.sort(flat)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        idx = np.linspace(0, len(xs) - 1, 4000).astype(int)
        ax.semilogx(xs[idx] * 100, ys[idx],
                    color=colors[spectra], lw=1.8, label=r["label"])
    for t, ls in [(0.01, ":"), (0.05, "--")]:
        ax.axvline(t * 100, color="gray", ls=ls, lw=0.8)
    ax.set_xlabel(r"$|P_{\rm pred}/P_{\rm true} - 1|$ [%]")
    ax.set_ylabel(r"CDF (per-$k$, per-cosmology)")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(0, 1.01)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "err_cdf.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- 2) |dP/P| vs k ----
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for spectra, r in results.items():
        med = np.median(r["fe"], axis=0) * 100
        p95 = np.percentile(r["fe"], 95, axis=0) * 100
        ax.loglog(k, med, color=colors[spectra], lw=1.8,
                  label=f"{r['label']} median")
        ax.loglog(k, p95, color=colors[spectra], lw=1.2, ls="--",
                  label=f"{r['label']} 95th")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(r"$|P_{\rm pred}/P_{\rm true} - 1|$ [%]")
    ax.set_xlim(k.min(), k.max())
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "err_vs_k.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- 3) example spectra ----
    r0 = results["linear_v2c"]
    params = r0["params"]

    # pick 4 cosmologies spanning (omega_m, mnu)
    om = params[:, PARAMS.index("omega_m")]
    mnu = params[:, PARAMS.index("mnu")]
    picks = [
        ("low $\\Omega_m$, low $m_\\nu$",
         np.argmin(0.7 * (om - om.min()) + 0.3 * (mnu - mnu.min()))),
        ("high $\\Omega_m$, low $m_\\nu$",
         np.argmin(0.7 * (om.max() - om) + 0.3 * (mnu - mnu.min()))),
        ("low $\\Omega_m$, high $m_\\nu$",
         np.argmin(0.7 * (om - om.min()) + 0.3 * (mnu.max() - mnu))),
        ("high $\\Omega_m$, high $m_\\nu$",
         np.argmin(0.7 * (om.max() - om) + 0.3 * (mnu.max() - mnu))),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(11.0, 4.6),
                             sharex="col",
                             gridspec_kw={"height_ratios": [2.4, 1.0]})
    for col, (title, idx) in enumerate(picks):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        for spectra, r in results.items():
            pk_true = 10 ** r["logp_true"][idx]
            pk_pred = 10 ** r["logp_pred"][idx]
            c = colors[spectra]
            ax_top.loglog(k, pk_true, color=c, lw=1.4, ls="-",
                          label=f"{r['label']} truth" if col == 0 else None)
            ax_top.loglog(k, pk_pred, color=c, lw=1.0, ls="--",
                          label=f"{r['label']} emu" if col == 0 else None)
            ax_bot.semilogx(k, (pk_pred / pk_true - 1.0) * 100,
                            color=c, lw=1.2)
        ax_top.set_title(title, fontsize=9)
        ax_top.grid(True, which="both", alpha=0.25)
        ax_bot.grid(True, which="both", alpha=0.25)
        ax_bot.axhline(0, color="gray", lw=0.6)
        ax_bot.set_xlim(k.min(), k.max())
        ax_bot.set_ylim(-2.5, 2.5)
        ax_bot.set_xlabel(r"$k$ [$h$/Mpc]")
        if col == 0:
            ax_top.set_ylabel(r"$P(k)$ [$({\rm Mpc}/h)^3$]")
            ax_bot.set_ylabel(r"$P_{\rm pred}/P_{\rm true} - 1$ [%]")
    axes[0, 0].legend(loc="lower left", fontsize=7, frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "examples.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote plots to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
