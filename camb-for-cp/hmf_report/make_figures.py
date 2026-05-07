"""Generate the figures that accompany report.tex.

Two inputs: the save_dirs produced by
    cosmosis cosmosis-models/smoke_hmf_nonu_camb.ini              # (P(k) shape only)
    cosmosis cosmosis-models/smoke_hmf_nonu_cp.ini
    cosmosis cosmosis-models/smoke_full_lzbins_camb.ini           # (number counts)
    cosmosis cosmosis-models/smoke_full_lzbins_cp.ini

Run from the directory that contains the save_dirs, then
    python .../validations/hmf_report/make_figures.py

Writes PDFs into `figs/` next to this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

LAMBDA_BIN_LABELS = [r"[20, 30)", r"[30, 45)", r"[45, 60)", r"[60, $\infty$)"]
# DES Y1 cluster z bins (see y1_mock/mock.ini, y1_mock_emcee.ini).
Z_BIN_LABELS = [r"[0.20, 0.35)", r"[0.35, 0.50)", r"[0.50, 0.65)"]
Z_BIN_CENTERS = np.array([0.275, 0.425, 0.575])


def _load_mf(save_dir):
    mf = save_dir / "mass_function"
    m = np.loadtxt(mf / "m_h.txt")
    z = np.loadtxt(mf / "z.txt")
    dn = np.loadtxt(mf / "dndlnmh.txt")
    if dn.shape == (z.size, m.size):
        dn = dn.T
    return m, z, dn


def _load_pk(save_dir, section):
    d = save_dir / section
    k = np.loadtxt(d / "k_h.txt")
    z = np.loadtxt(d / "z.txt")
    p = np.loadtxt(d / "p_k.txt")
    if p.shape == (z.size, k.size):
        p = p.T
    return k, z, p


def _load_growth(save_dir):
    g = save_dir / "growth_parameters"
    z = np.loadtxt(g / "z.txt")
    d = np.loadtxt(g / "d_z.txt")
    return z, d / np.interp(0.0, z, d)


def _load_nc_grid(save_dir):
    """Full-integral NumCountsFullScalarIntegrand output shaped (n_z, n_lambda)."""
    vals = np.loadtxt(save_dir / "numcountsfullscalarintegrand" / "vals.txt")
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    return vals


def main():
    cwd = Path.cwd()
    camb_hmf = cwd / "smoke_hmf_nonu_camb_output"
    emu_hmf  = cwd / "smoke_hmf_nonu_cp_output"
    camb_nc  = cwd / "smoke_full_lzbins_camb_output"
    emu_nc   = cwd / "smoke_full_lzbins_cp_output"
    for d in (camb_hmf, emu_hmf, camb_nc, emu_nc):
        if not d.is_dir():
            raise SystemExit(
                f"missing {d} — rerun the smoke inis first "
                f"(see the docstring)."
            )

    # ================================================================
    # Figure 1: Number counts N_i(lambda_bin, z_bin) — CAMB vs emulator
    # ================================================================
    N_camb = _load_nc_grid(camb_nc)       # shape (n_z, n_lambda)
    N_emu  = _load_nc_grid(emu_nc)
    rel_nc = (N_emu - N_camb) / N_camb

    n_lam = N_camb.shape[1]
    n_z = N_camb.shape[0]
    x = np.arange(n_lam)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_z))

    # Residual-only figure: (N_emu - N_camb) / N_camb per (lambda_bin, z_bin)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j in range(n_z):
        ax.plot(x, rel_nc[j, :] * 100.0, "-o", color=colors[j],
                label=Z_BIN_LABELS[j], lw=2, markersize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.axhline(-0.5, ls=":", color="k", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(LAMBDA_BIN_LABELS)
    ax.set_xlabel(r"$\lambda$ bin")
    ax.set_ylabel(r"$(N_{\rm emu} - N_{\rm CAMB})/N_{\rm CAMB}$ [\%]")
    ax.set_title(r"Number-counts residual per $(\lambda,\,z)$ bin")
    ax.legend(fontsize=9, title="$z$ bin")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "number_counts_lzbins.pdf")
    plt.close(fig)

    # ================================================================
    # Figure 2: residual heatmap across (z_bin, λ_bin)
    # ================================================================
    fig, ax = plt.subplots(figsize=(6, 4))
    vmax = float(np.nanmax(np.abs(rel_nc)))
    im = ax.pcolormesh(x, np.arange(n_z), rel_nc * 100.0,
                       cmap="RdBu_r", vmin=-vmax * 100, vmax=vmax * 100,
                       shading="nearest")
    for j in range(n_z):
        for i in range(n_lam):
            ax.text(i, j, f"{rel_nc[j,i]*100:+.2f}%", ha="center", va="center",
                    fontsize=9,
                    color="black" if abs(rel_nc[j,i]) < vmax * 0.55 else "white")
    ax.set_xticks(x); ax.set_xticklabels(LAMBDA_BIN_LABELS)
    ax.set_yticks(np.arange(n_z)); ax.set_yticklabels(Z_BIN_LABELS)
    ax.set_xlabel(r"$\lambda$ bin")
    ax.set_ylabel(r"$z$ bin")
    ax.set_title(r"$(N_{\rm emu}-N_{\rm CAMB})/N_{\rm CAMB}$ [\%]")
    fig.colorbar(im, ax=ax, label="percent")
    fig.tight_layout()
    fig.savefig(FIGS / "number_counts_heatmap.pdf")
    plt.close(fig)

    # ================================================================
    # Figure 3: HMF cluster-band percentile curves vs z
    # (from smoke_hmf pair — they use a slightly different cosmology but
    # serve as a first-principles decomposition of where the N_i error
    # comes from: HMF shape, P(k, z=0), and growth.)
    # ================================================================
    m, z_mf, dn_camb_ = _load_mf(camb_hmf)
    _, _, dn_emu_ = _load_mf(emu_hmf)
    rel = (dn_emu_ - dn_camb_) / dn_camb_
    cluster = (m >= 1e13) & (m <= 1e15)
    rel_cl = rel[cluster, :]
    p50 = np.array([np.nanpercentile(np.abs(rel_cl[:, j]), 50) for j in range(z_mf.size)])
    p95 = np.array([np.nanpercentile(np.abs(rel_cl[:, j]), 95) for j in range(z_mf.size)])
    p99 = np.array([np.nanpercentile(np.abs(rel_cl[:, j]), 99) for j in range(z_mf.size)])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z_mf, p50, label="p50", lw=2)
    ax.plot(z_mf, p95, label="p95", lw=2)
    ax.plot(z_mf, p99, label="p99", lw=2)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$|\Delta\,\mathrm{d}n/\mathrm{d}\ln M_h| / "
                  r"(\mathrm{d}n/\mathrm{d}\ln M_h)_{\rm CAMB}$")
    ax.set_yscale("log")
    ax.set_title(r"HMF error vs.\ $z$ (cluster mass band $10^{13}$--$10^{15}\,M_\odot/h$)")
    ax.axhline(1e-2, ls=":", color="k", alpha=0.5)
    ax.axhline(1e-3, ls=":", color="k", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "hmf_err_vs_z.pdf")
    plt.close(fig)

    # ================================================================
    # Figure 4: P(k, z=0) emulator accuracy
    # ================================================================
    k_c, z_pk_c, p_c = _load_pk(camb_hmf, "cdm_baryon_power_lin")
    k_e, z_pk_e, p_e = _load_pk(emu_hmf, "cdm_baryon_power_lin")
    iz0_c = int(np.argmin(np.abs(z_pk_c)))
    iz0_e = int(np.argmin(np.abs(z_pk_e)))
    p_emu_on_camb = np.exp(np.interp(np.log(k_c), np.log(k_e),
                                     np.log(p_e[:, iz0_e])))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.loglog(k_c, p_c[:, iz0_c], lw=2, label="CAMB")
    ax1.loglog(k_e, p_e[:, iz0_e], lw=1, ls="--", label="cp\\_camb emulator")
    ax1.set_ylabel(r"$P_{\rm cb}(k,\,z=0)\;[({\rm Mpc}/h)^3]$")
    ax1.legend()
    ax1.set_title(r"CDM+baryon linear $P(k,z{=}0)$")
    ax2.semilogx(k_c, (p_emu_on_camb - p_c[:, iz0_c]) / p_c[:, iz0_c], lw=2)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axhline(1e-2, ls=":", color="k", alpha=0.5)
    ax2.axhline(-1e-2, ls=":", color="k", alpha=0.5)
    ax2.set_xlabel(r"$k\;[h/\mathrm{Mpc}]$")
    ax2.set_ylabel(r"$(P_{\rm emu} - P_{\rm CAMB})/P_{\rm CAMB}$")
    fig.tight_layout()
    fig.savefig(FIGS / "pk_z0_comparison.pdf")
    plt.close(fig)

    # ================================================================
    # Figure 5: Growth factor D(z)
    # ================================================================
    z_gc, d_camb = _load_growth(camb_hmf)
    z_ge, d_emu  = _load_growth(emu_hmf)
    d_emu_on_camb_z = np.interp(z_gc, z_ge, d_emu)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(z_gc, d_camb, lw=2, label=r"$D_{\rm CAMB}(z)$")
    ax1.plot(z_ge, d_emu, lw=1, ls="--",
             label=r"$D_{\rm growth\_factor}(z)$ (used by emulator pipeline)")
    ax1.set_ylabel(r"$D(z)\;[D(0)=1]$")
    ax1.legend()
    ax1.set_title(r"Growth factor used to rescale $P(k,z{=}0)\to P(k,z)$")
    ax2.plot(z_gc, d_emu_on_camb_z / d_camb - 1.0, lw=2)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axhline(1e-3, ls=":", color="k", alpha=0.4)
    ax2.axhline(-1e-3, ls=":", color="k", alpha=0.4)
    ax2.set_xlabel(r"$z$")
    ax2.set_ylabel(r"$D_{\rm emu}/D_{\rm CAMB} - 1$")
    fig.tight_layout()
    fig.savefig(FIGS / "growth_factor.pdf")
    plt.close(fig)

    # ================================================================
    # Figure 6: Scale-dependent effective D(k, z) from P_cb(k, z)/P_cb(k, 0)
    # ================================================================
    k_probes = np.array([1e-3, 1e-2, 1e-1, 3e-1])
    fig, ax = plt.subplots(figsize=(6, 4))
    for kp in k_probes:
        ikc = int(np.argmin(np.abs(np.log(k_c) - np.log(kp))))
        ike = int(np.argmin(np.abs(np.log(k_e) - np.log(kp))))
        rc = p_c[ikc, :] / p_c[ikc, iz0_c]
        re = p_e[ike, :] / p_e[ike, iz0_e]
        line, = ax.plot(z_pk_c, np.sqrt(rc), lw=2,
                        label=rf"CAMB, $k={kp:.0e}$")
        ax.plot(z_pk_e, np.sqrt(re), lw=1, ls="--", color=line.get_color(),
                label=rf"emu, $k={kp:.0e}$")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\sqrt{P(k,z)/P(k,0)}$")
    ax.set_title(r"Effective $D(k,z)$ from $P_{\rm cb}$: CAMB vs emulator")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "growth_scale_dep.pdf")
    plt.close(fig)

    # ================================================================
    # Numerical summary
    # ================================================================
    summary = FIGS.parent / "summary.txt"
    with summary.open("w") as fh:
        fh.write("# numbers consumed by report.tex\n")
        fh.write("\n# --- N_i(lambda, z) from the full-integral pipeline ---\n")
        fh.write("# CAMB reference\n")
        fh.write("# z_bin \\ lambda_bin: " + " ".join(LAMBDA_BIN_LABELS) + "\n")
        for j, zb in enumerate(Z_BIN_LABELS):
            fh.write(f"{zb}: " + " ".join(f"{v:10.3e}" for v in N_camb[j, :]) + "\n")
        fh.write("# cp_camb emulator\n")
        for j, zb in enumerate(Z_BIN_LABELS):
            fh.write(f"{zb}: " + " ".join(f"{v:10.3e}" for v in N_emu[j, :]) + "\n")
        fh.write("# (emu - camb) / camb\n")
        for j, zb in enumerate(Z_BIN_LABELS):
            fh.write(f"{zb}: " + " ".join(f"{v:+10.3e}" for v in rel_nc[j, :]) + "\n")
        worst_idx = np.unravel_index(np.argmax(np.abs(rel_nc)), rel_nc.shape)
        fh.write(f"worst (z_bin,lambda_bin) = {worst_idx}  rel = "
                 f"{rel_nc[worst_idx]:+.3e}\n")
        fh.write("\n# --- decomposition (from smoke_hmf_nonu pair) ---\n")
        fh.write(f"HMF cluster-band p99 at z=0: {p99[0]:.3e}\n")
        fh.write(f"HMF cluster-band p99 at z=1: {p99[-1]:.3e}\n")
        fh.write(f"max |P_emu - P_camb|/P_camb at z=0: "
                 f"{np.nanmax(np.abs((p_emu_on_camb - p_c[:, iz0_c]) / p_c[:, iz0_c])):.3e}\n")
        fh.write(f"D_emu/D_camb - 1 at z=1: "
                 f"{d_emu_on_camb_z[-1] / d_camb[-1] - 1:.3e}\n")
    print("wrote figures to", FIGS)
    print("wrote summary to", summary)


if __name__ == "__main__":
    main()
