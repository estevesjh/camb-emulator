"""Cross-check: does the emulator's P(k, z=0) residual correlate with
(omega_m, log1e10As) the same way the N_i residual does?

Writes:
  figs/scan_pk_residual_k.pdf       — |dP/P| over k for all 100 samples
  figs/scan_pk_residual_vs_om.pdf   — dP/P at fixed k vs. omega_m
  figs/scan_pk_residual_plane.pdf   — dP/P at k=0.1 over (Om, lnAs) plane
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
ROOT = Path("/pscratch/sd/j/jesteves/cosmology_scan")


def _load(which, i, section="cdm_baryon_power_lin"):
    p = ROOT / f"cosmology_scan_{which}_output_extracted/cosmology_scan_{which}_output/block_{i}/{section}"
    k = np.loadtxt(p / "k_h.txt")
    z = np.loadtxt(p / "z.txt")
    pk = np.loadtxt(p / "p_k.txt")   # (n_z, n_k)
    return k, z, pk


def main():
    cosmo = np.loadtxt(
        "/global/common/software/des/jesteves/y3_cluster_cpp/cosmosis-models/cosmology_scan.list",
        comments="#")
    omega_m, log1e10As = cosmo[:, 0], cosmo[:, 1]
    n = len(cosmo)

    # CAMB k grid is (150,) dense; use it as the reference.
    k_ref, _, _ = _load("camb", 0)

    # Stack rel-error at z=0 on the CAMB k grid.
    rel_pk0 = np.zeros((n, k_ref.size))    # (n_samp, n_k)
    for i in range(n):
        kc, zc, pc = _load("camb", i)
        ke, ze, pe = _load("cp", i)
        iz0c = np.argmin(np.abs(zc))
        iz0e = np.argmin(np.abs(ze))
        pe_on_c = np.exp(np.interp(np.log(k_ref), np.log(ke),
                                    np.log(pe[iz0e])))
        rel_pk0[i] = (pe_on_c - pc[iz0c]) / pc[iz0c]

    # === Figure 1: rel-error envelope vs k ===
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i in range(n):
        ax.plot(k_ref, rel_pk0[i] * 100, color="k", alpha=0.08, lw=0.6)
    p50 = np.median(rel_pk0, axis=0) * 100
    p05 = np.percentile(rel_pk0, 5, axis=0) * 100
    p95 = np.percentile(rel_pk0, 95, axis=0) * 100
    ax.fill_between(k_ref, p05, p95, alpha=0.25, color="C0", label="5--95\\%")
    ax.plot(k_ref, p50, color="C0", lw=2, label="median")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, ls=":", color="k", alpha=0.4)
    ax.axhline(-1, ls=":", color="k", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\;[h/\mathrm{Mpc}]$")
    ax.set_ylabel(r"$(P_{\rm emu} - P_{\rm CAMB})/P_{\rm CAMB}$ at $z=0$ [\%]")
    ax.set_title(r"$P_{\rm cb}(k, z=0)$ emulator error: 100 cosmology samples")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "scan_pk_residual_k.pdf")
    plt.close(fig)

    # === Figure 2: rel-error at four k-values vs omega_m ===
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    k_probes = [1e-3, 1e-2, 1e-1, 5e-1]
    for ax, kp in zip(axes.flatten(), k_probes):
        ik = np.argmin(np.abs(np.log(k_ref/kp)))
        sc = ax.scatter(omega_m, rel_pk0[:, ik] * 100, c=log1e10As,
                        cmap="viridis", s=35, edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(r"$\Omega_m$")
        ax.set_ylabel(r"$\Delta P/P$ [\%]")
        ax.set_title(rf"$k={k_ref[ik]:.3g}\,h/\mathrm{{Mpc}}$")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$\log(10^{10} A_s)$")
    fig.suptitle(r"Emulator $P(k, z=0)$ error vs $\Omega_m$", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "scan_pk_residual_vs_om.pdf")
    plt.close(fig)

    # === Figure 3: rel-error at k=0.1 over the (Om, lnAs) plane ===
    kp = 0.1
    ik = np.argmin(np.abs(np.log(k_ref/kp)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(omega_m, log1e10As, c=rel_pk0[:, ik] * 100,
                    cmap="RdBu_r", s=80,
                    vmin=-np.nanmax(np.abs(rel_pk0[:, ik])) * 100,
                    vmax=+np.nanmax(np.abs(rel_pk0[:, ik])) * 100,
                    edgecolor="k", linewidth=0.3)
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$\log(10^{10} A_s)$")
    ax.set_title(rf"$\Delta P/P$ at $k={k_ref[ik]:.2g}\,h/\mathrm{{Mpc}}$, $z=0$ [\%]")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("percent")
    fig.tight_layout()
    fig.savefig(FIGS / "scan_pk_residual_plane.pdf")
    plt.close(fig)

    # === Figure 4: growth D(z) rel-error vs omega_m ===
    # Compute D_emu(z)/D_camb(z) - 1 at several redshifts, for all samples.
    z_probes = [0.3, 0.5, 0.7, 1.0]
    growth_rel = np.zeros((n, len(z_probes)))   # (n_samp, n_z_probe)
    for i in range(n):
        gd_c = ROOT / f"cosmology_scan_camb_output_extracted/cosmology_scan_camb_output/block_{i}/growth_parameters"
        gd_e = ROOT / f"cosmology_scan_cp_output_extracted/cosmology_scan_cp_output/block_{i}/growth_parameters"
        zc = np.loadtxt(gd_c / "z.txt"); Dc = np.loadtxt(gd_c / "d_z.txt")
        ze = np.loadtxt(gd_e / "z.txt"); De = np.loadtxt(gd_e / "d_z.txt")
        Dc = Dc / np.interp(0.0, zc, Dc)
        De = De / np.interp(0.0, ze, De)
        for j, zt in enumerate(z_probes):
            Dc_t = np.interp(zt, zc, Dc)
            De_t = np.interp(zt, ze, De)
            growth_rel[i, j] = De_t / Dc_t - 1.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(z_probes)))
    for j, zt in enumerate(z_probes):
        ax.scatter(omega_m, growth_rel[:, j] * 100, color=colors[j],
                   s=28, edgecolor="k", linewidth=0.2,
                   label=rf"$z={zt}$")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.1, ls=":", color="k", alpha=0.4)
    ax.axhline(-0.1, ls=":", color="k", alpha=0.4)
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$D_{\rm growth\_factor}(z)/D_{\rm CAMB}(z) - 1$ [\%]")
    ax.set_title(r"Growth-factor relative error vs $\Omega_m$")
    ax.legend(fontsize=9, title="redshift")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "scan_growth_vs_om.pdf")
    plt.close(fig)

    # === Numerical correlations ===
    summary = HERE / "scan_pk_summary.txt"
    with summary.open("w") as fh:
        fh.write(f"n_samples = {n}\n")
        fh.write("\n# correlation coefficients of dP/P vs cosmology params:\n")
        fh.write(f"{'k (h/Mpc)':>10s}  {'median dP/P [%]':>18s}  "
                 f"{'range [%]':>18s}  {'rho(Omega_m)':>14s}  "
                 f"{'rho(log1e10As)':>16s}\n")
        for kp in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]:
            ik = np.argmin(np.abs(np.log(k_ref/kp)))
            vals = rel_pk0[:, ik] * 100
            rho_om = np.corrcoef(vals, omega_m)[0, 1]
            rho_as = np.corrcoef(vals, log1e10As)[0, 1]
            rng = f"[{vals.min():+.2f}, {vals.max():+.2f}]"
            fh.write(f"{k_ref[ik]:>10.3g}  {np.median(vals):>+18.3f}  "
                     f"{rng:>18s}  {rho_om:>+14.3f}  {rho_as:>+16.3f}\n")
    print("wrote", FIGS / "scan_pk_residual_k.pdf")
    print("wrote", FIGS / "scan_pk_residual_vs_om.pdf")
    print("wrote", FIGS / "scan_pk_residual_plane.pdf")
    print("wrote", summary)


if __name__ == "__main__":
    main()
