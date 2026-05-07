"""Planck-centered cosmology scan analysis.

LHS of 100 samples in a 7-10 sigma box around Planck 2018:
    Omega_m   in [0.240, 0.390]
    log1e10As in [2.947, 3.141]

Figures:
  figs/planck_scan_nc_box.pdf           boxplot of dN/N per (lambda, z) bin
  figs/planck_scan_nc_plane.pdf         mean |dN/N| over (Om, lnAs)
  figs/planck_scan_pk_k.pdf             dP/P vs k, all samples
  figs/planck_scan_pk_vs_om.pdf         dP/P at 4 k values vs Omega_m
  figs/planck_scan_growth_vs_om.pdf     D_emu/D_camb - 1 vs Omega_m
  scan_summary_planck.txt               machine-readable stats
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
ROOT = Path("/pscratch/sd/j/jesteves/cosmology_scan_planck")

LAMBDA = [r"[20, 30)", r"[30, 45)", r"[45, 60)", r"[60, $\infty$)"]
Z = [r"[0.20, 0.35)", r"[0.35, 0.50)", r"[0.50, 0.65)"]


def _block_dir(which, i):
    return (ROOT /
            f"cosmology_scan_planck_{which}_output_extracted/"
            f"cosmology_scan_planck_{which}_output/block_{i}")


def _load_vals(which, i):
    return np.loadtxt(_block_dir(which, i) / "numcountsfullscalarintegrand" / "vals.txt")


def _load_pk(which, i):
    p = _block_dir(which, i) / "cdm_baryon_power_lin"
    return (np.loadtxt(p / "k_h.txt"),
            np.loadtxt(p / "z.txt"),
            np.loadtxt(p / "p_k.txt"))


def _load_growth(which, i):
    p = _block_dir(which, i) / "growth_parameters"
    return (np.loadtxt(p / "z.txt"),
            np.loadtxt(p / "d_z.txt"))


def main():
    cosmo_list = np.loadtxt(
        "/global/common/software/des/jesteves/y3_cluster_cpp/cosmosis-models/cosmology_scan_planck.list",
        comments="#")

    # Enumerate blocks that actually landed on disk (some CUBA samples
    # may have failed silently — we keep the ones that produced vals.txt).
    camb_root = (ROOT /
                 "cosmology_scan_planck_camb_output_extracted/"
                 "cosmology_scan_planck_camb_output")
    valid = []
    for d in sorted(camb_root.glob("block_*"),
                    key=lambda p: int(p.name.split("_")[1])):
        i = int(d.name.split("_")[1])
        if (d / "numcountsfullscalarintegrand" / "vals.txt").is_file():
            cp_valspath = (ROOT /
                           "cosmology_scan_planck_cp_output_extracted/"
                           f"cosmology_scan_planck_cp_output/block_{i}/"
                           "numcountsfullscalarintegrand/vals.txt")
            if cp_valspath.is_file():
                valid.append(i)
    print(f"valid samples: {len(valid)} / {len(cosmo_list)}")
    cosmo = cosmo_list[valid]
    om, lAs = cosmo[:, 0], cosmo[:, 1]

    N_c = np.array([_load_vals("camb", i) for i in valid])  # (n, nz, nl)
    N_e = np.array([_load_vals("cp", i) for i in valid])
    rel = (N_e - N_c) / N_c                                  # (n, nz, nl)
    n, nz, nl = rel.shape

    # --- Fig 1: boxplot per (lambda, z) bin over the Planck scan ---
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, nz))
    data, positions, colors = [], [], []
    for j in range(nz):
        for i in range(nl):
            data.append(rel[:, j, i] * 100.0)
            positions.append(i + (j - (nz - 1) / 2) * 0.22)
            colors.append(cmap[j])
    bp = ax.boxplot(data, positions=positions, widths=0.18,
                    patch_artist=True, showfliers=True,
                    medianprops={"color": "k", "lw": 1.2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for j in range(nz):
        ax.plot([], [], color=cmap[j], lw=8, alpha=0.7,
                label=f"$z$ {Z[j]}")
    ax.set_xticks(np.arange(nl)); ax.set_xticklabels(LAMBDA)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.axhline(-0.5, ls=":", color="k", alpha=0.4)
    ax.set_xlabel(r"$\lambda$ bin")
    ax.set_ylabel(r"$(N_{\rm emu} - N_{\rm CAMB})/N_{\rm CAMB}$ [\%]")
    ax.set_title(
        rf"Planck-centered scan ({n} samples) $\Omega_m \in [{om.min():.2f}, {om.max():.2f}]$, "
        rf"$\log(10^{{10}}A_s) \in [{lAs.min():.2f}, {lAs.max():.2f}]$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "planck_scan_nc_box.pdf")
    plt.close(fig)

    # --- Fig 2: mean |rel N| over the cosmology plane ---
    mean_abs = np.abs(rel).mean(axis=(1, 2)) * 100
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(om, lAs, c=mean_abs, s=80, cmap="Reds",
                    edgecolor="k", linewidth=0.3)
    ax.axvline(0.315, color="k", ls=":", alpha=0.5)
    ax.axhline(3.044, color="k", ls=":", alpha=0.5)
    ax.scatter([0.315], [3.044], marker="*", s=180, color="gold",
               edgecolor="k", linewidth=0.6, zorder=5, label="Planck 2018")
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$\log(10^{10} A_s)$")
    ax.set_title(
        r"Mean $|\Delta N/N|$ over 12 $(\lambda,z)$ bins [\%]")
    fig.colorbar(sc, ax=ax, label="percent")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "planck_scan_nc_plane.pdf")
    plt.close(fig)

    # --- Fig 3: P(k, z=0) rel error vs k, all samples ---
    k_ref, _, _ = _load_pk("camb", valid[0])
    rel_pk0 = np.zeros((n, k_ref.size))
    for ii, i in enumerate(valid):
        kc, zc, pc = _load_pk("camb", i)
        ke, ze, pe = _load_pk("cp", i)
        i0c = np.argmin(np.abs(zc)); i0e = np.argmin(np.abs(ze))
        pe_on_c = np.exp(np.interp(np.log(k_ref), np.log(ke),
                                    np.log(pe[i0e])))
        rel_pk0[ii] = (pe_on_c - pc[i0c]) / pc[i0c]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for ii in range(n):
        ax.plot(k_ref, rel_pk0[ii] * 100, color="k", alpha=0.08, lw=0.6)
    p05 = np.percentile(rel_pk0, 5, axis=0) * 100
    p95 = np.percentile(rel_pk0, 95, axis=0) * 100
    p50 = np.median(rel_pk0, axis=0) * 100
    ax.fill_between(k_ref, p05, p95, alpha=0.25, color="C0",
                    label="5--95\\%")
    ax.plot(k_ref, p50, color="C0", lw=2, label="median")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, ls=":", color="k", alpha=0.4)
    ax.axhline(-1, ls=":", color="k", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\;[h/\mathrm{Mpc}]$")
    ax.set_ylabel(r"$(P_{\rm emu}-P_{\rm CAMB})/P_{\rm CAMB}$ at $z=0$ [\%]")
    ax.set_title(r"Planck-centered scan: $P_{\rm cb}(k, z=0)$ emulator error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "planck_scan_pk_k.pdf")
    plt.close(fig)

    # --- Fig 4: dP/P at 4 k-values vs omega_m ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    k_probes = [1e-3, 1e-2, 1e-1, 5e-1]
    for ax, kp in zip(axes.flatten(), k_probes):
        ik = np.argmin(np.abs(np.log(k_ref / kp)))
        sc = ax.scatter(om, rel_pk0[:, ik] * 100, c=lAs, cmap="viridis",
                        s=35, edgecolor="k", linewidth=0.3)
        ax.axvline(0.315, color="k", ls=":", alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(r"$\Omega_m$")
        ax.set_ylabel(r"$\Delta P/P$ [\%]")
        ax.set_title(rf"$k={k_ref[ik]:.3g}\,h/\mathrm{{Mpc}}$")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label=r"$\log(10^{10}A_s)$")
    fig.suptitle(r"Planck-centered scan: emulator $\Delta P/P$ vs $\Omega_m$")
    fig.tight_layout()
    fig.savefig(FIGS / "planck_scan_pk_vs_om.pdf")
    plt.close(fig)

    # --- Fig 5: growth residual vs omega_m ---
    z_probes = [0.3, 0.5, 0.7, 1.0]
    growth_rel = np.zeros((n, len(z_probes)))
    for ii, i in enumerate(valid):
        zc, Dc = _load_growth("camb", i)
        ze, De = _load_growth("cp", i)
        Dc = Dc / np.interp(0.0, zc, Dc)
        De = De / np.interp(0.0, ze, De)
        for j, zt in enumerate(z_probes):
            growth_rel[ii, j] = np.interp(zt, ze, De) / np.interp(zt, zc, Dc) - 1.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(z_probes)))
    for j, zt in enumerate(z_probes):
        ax.scatter(om, growth_rel[:, j] * 100, color=colors[j],
                   s=28, edgecolor="k", linewidth=0.2,
                   label=rf"$z={zt}$")
    ax.axvline(0.315, color="k", ls=":", alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.1, ls=":", color="k", alpha=0.4)
    ax.axhline(-0.1, ls=":", color="k", alpha=0.4)
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$D_{\rm growth\_factor}/D_{\rm CAMB}-1$ [\%]")
    ax.set_title(r"Planck-centered scan: growth-factor residual vs $\Omega_m$")
    ax.legend(fontsize=9, title="redshift")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "planck_scan_growth_vs_om.pdf")
    plt.close(fig)

    # --- Summary ---
    summary = HERE / "scan_summary_planck.txt"
    with summary.open("w") as fh:
        fh.write(f"n_valid = {n} / {len(cosmo_list)} planned\n")
        fh.write(f"Omega_m   range sampled = [{om.min():.4f}, {om.max():.4f}]\n")
        fh.write(f"log1e10As range sampled = [{lAs.min():.4f}, {lAs.max():.4f}]\n")
        fh.write("# Planck 2018 center = (0.3153, 3.044)\n\n")
        fh.write("# per-bin |rel N| percentiles over the scan (%):\n")
        for j in range(nz):
            for i in range(nl):
                v = np.abs(rel[:, j, i]) * 100
                fh.write(f"z={Z[j]:<16s} lambda={LAMBDA[i]:<12s} "
                         f"median={np.median(v):.3f}  p95={np.percentile(v,95):.3f}  "
                         f"max={v.max():.3f}\n")
        w = np.unravel_index(np.argmax(np.abs(rel)), rel.shape)
        fh.write(f"\nworst: sample {valid[w[0]]} (Om={om[w[0]]:.4f}, "
                 f"lnAs={lAs[w[0]]:.4f}), z_bin={w[1]}, lambda_bin={w[2]}, "
                 f"rel={rel[w]*100:+.3f}%\n")
        fh.write("\n# P(k,z=0) rel-error summary over the scan (%):\n")
        fh.write(f"{'k':>10s}  {'med':>10s}  {'p5':>10s}  {'p95':>10s}  "
                 f"{'rho(Om)':>10s}  {'rho(lnAs)':>10s}\n")
        for kp in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]:
            ik = np.argmin(np.abs(np.log(k_ref/kp)))
            v = rel_pk0[:, ik] * 100
            fh.write(f"{k_ref[ik]:>10.3g}  {np.median(v):>+10.3f}  "
                     f"{np.percentile(v, 5):>+10.3f}  {np.percentile(v, 95):>+10.3f}  "
                     f"{np.corrcoef(v, om)[0,1]:>+10.3f}  {np.corrcoef(v, lAs)[0,1]:>+10.3f}\n")
        fh.write("\n# growth rel-error summary over the scan (%):\n")
        for j, zt in enumerate(z_probes):
            v = growth_rel[:, j] * 100
            fh.write(f"z={zt:.1f}: median={np.median(v):+.4f}  "
                     f"range=[{v.min():+.4f}, {v.max():+.4f}]  "
                     f"rho(Omega_m)={np.corrcoef(v, om)[0,1]:+.3f}\n")
    print("wrote figures to", FIGS)
    print("wrote summary to", summary)


if __name__ == "__main__":
    main()
