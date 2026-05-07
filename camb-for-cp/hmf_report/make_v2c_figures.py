"""Regenerate the validation figures using the retrained v2c emulator.

v2c = v2 base training set + appended 100k Planck-centered LHS samples.
Same architecture; denser coverage in the Y1 prior region.

Inputs:
  /pscratch/.../smoke_full_lzbins_cp_v2c_output/              (fiducial)
  /pscratch/.../cosmology_scan_cp_v2c_output/                 (100 LHS)
  plus CAMB reference runs already on disk.

Writes:
  figs/number_counts_lzbins.pdf           fiducial residual (overwrites v2)
  figs/number_counts_heatmap.pdf          fiducial heatmap (overwrites v2)
  figs/scan_precision.pdf                 boxplot (overwrites v2)
  figs/scan_precision_heatmap.pdf         median/p95/max heatmap
  figs/scan_scatter_vs_cosmo.pdf          rel-err vs Om and lnAs
  figs/scan_meanerr_over_plane.pdf        mean |rel| over (Om, lnAs)
  figs/scan_pk_residual_k.pdf             dP/P vs k envelope
  figs/scan_pk_residual_vs_om.pdf         dP/P at 4 k vs Omega_m
  figs/scan_growth_vs_om.pdf              D rel-err vs Omega_m
  scan_summary.txt / v2c summary
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
Z_BIN_LABELS = [r"[0.20, 0.35)", r"[0.35, 0.50)", r"[0.50, 0.65)"]


def _load_vals(path):
    v = np.loadtxt(path)
    return v.reshape(-1) if v.ndim == 1 else v


def _load_mf(save_dir):
    mf = save_dir / "mass_function"
    m = np.loadtxt(mf / "m_h.txt")
    z = np.loadtxt(mf / "z.txt")
    dn = np.loadtxt(mf / "dndlnmh.txt")
    if dn.shape == (z.size, m.size):
        dn = dn.T
    return m, z, dn


def _load_pk(save_dir, section="cdm_baryon_power_lin"):
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


# =========================================================================
# 1. Fiducial (lzbins) figures
# =========================================================================
def fiducial():
    cwd = Path("/pscratch/sd/j/jesteves/lzbins_compare")
    camb = cwd / "smoke_full_lzbins_camb_output"
    emu  = cwd / "smoke_full_lzbins_cp_v2c_output"

    N_camb = _load_vals(camb / "numcountsfullscalarintegrand" / "vals.txt")
    N_emu  = _load_vals(emu  / "numcountsfullscalarintegrand" / "vals.txt")
    rel_nc = (N_emu - N_camb) / N_camb

    n_z, n_lam = N_camb.shape
    x = np.arange(n_lam)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_z))

    # residual-only figure (matches old number_counts_lzbins.pdf)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j in range(n_z):
        ax.plot(x, rel_nc[j, :] * 100.0, "-o", color=colors[j],
                label=Z_BIN_LABELS[j], lw=2, markersize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.axhline(-0.5, ls=":", color="k", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(LAMBDA_BIN_LABELS)
    ax.set_xlabel(r"$\lambda$ bin")
    ax.set_ylabel(r"$(N_{\rm emu} - N_{\rm CAMB})/N_{\rm CAMB}$ [\%]")
    ax.set_title(r"Number-counts residual at the fiducial cosmology (v2c)")
    ax.legend(fontsize=9, title="$z$ bin")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "number_counts_lzbins.pdf")
    plt.close(fig)

    # heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    vmax = float(np.nanmax(np.abs(rel_nc)))
    im = ax.pcolormesh(x, np.arange(n_z), rel_nc * 100.0,
                       cmap="RdBu_r", vmin=-vmax*100, vmax=vmax*100,
                       shading="nearest")
    for j in range(n_z):
        for i in range(n_lam):
            ax.text(i, j, f"{rel_nc[j,i]*100:+.2f}%", ha="center", va="center",
                    fontsize=9,
                    color="black" if abs(rel_nc[j,i]) < vmax*0.55 else "white")
    ax.set_xticks(x); ax.set_xticklabels(LAMBDA_BIN_LABELS)
    ax.set_yticks(np.arange(n_z)); ax.set_yticklabels(Z_BIN_LABELS)
    ax.set_xlabel(r"$\lambda$ bin"); ax.set_ylabel(r"$z$ bin")
    ax.set_title(r"$(N_{\rm emu}-N_{\rm CAMB})/N_{\rm CAMB}$ [\%] (v2c)")
    fig.colorbar(im, ax=ax, label="percent")
    fig.tight_layout()
    fig.savefig(FIGS / "number_counts_heatmap.pdf")
    plt.close(fig)
    return rel_nc


# =========================================================================
# 2. HMF / P(k,0) / D(z) fiducial diagnostics are unchanged from v2 paths
#    (smoke_hmf_nonu_*); skip regeneration here.
# =========================================================================


# =========================================================================
# 3. Scan analysis: cosmology_scan_camb vs cosmology_scan_cp_v2c
# =========================================================================
def scan():
    scan_root = Path("/pscratch/sd/j/jesteves/cosmology_scan")
    camb_root = (scan_root /
                 "cosmology_scan_camb_output_extracted" /
                 "cosmology_scan_camb_output")
    emu_root = (scan_root /
                "cosmology_scan_cp_v2c_output_extracted" /
                "cosmology_scan_cp_v2c_output")

    blocks = sorted(
        camb_root.glob("block_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    valid = []
    for b in blocks:
        i = int(b.name.split("_")[1])
        if ((b / "numcountsfullscalarintegrand" / "vals.txt").is_file()
            and (emu_root / f"block_{i}" / "numcountsfullscalarintegrand" / "vals.txt").is_file()):
            valid.append(i)
    print(f"valid scan samples: {len(valid)}")

    cosmo = np.loadtxt(
        "/global/common/software/des/jesteves/y3_cluster_cpp/cosmosis-models/cosmology_scan.list",
        comments="#")[valid]
    om, lAs = cosmo[:, 0], cosmo[:, 1]

    N_c = np.array([_load_vals(camb_root / f"block_{i}" / "numcountsfullscalarintegrand" / "vals.txt") for i in valid])
    N_e = np.array([_load_vals(emu_root  / f"block_{i}" / "numcountsfullscalarintegrand" / "vals.txt") for i in valid])
    rel = (N_e - N_c) / N_c
    n, nz, nl = rel.shape

    # --- boxplot ---
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
                label=f"$z$ {Z_BIN_LABELS[j]}")
    ax.set_xticks(np.arange(nl)); ax.set_xticklabels(LAMBDA_BIN_LABELS)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.axhline(-0.5, ls=":", color="k", alpha=0.4)
    ax.set_xlabel(r"$\lambda$ bin")
    ax.set_ylabel(r"$(N_{\rm emu}-N_{\rm CAMB})/N_{\rm CAMB}$ [\%]")
    ax.set_title(
        r"Cosmology-scan precision over 100 samples (retrained v2c emulator)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "scan_precision.pdf")
    plt.close(fig)

    # --- median/p95/max heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    stats = [
        ("median", np.nanmedian(np.abs(rel), axis=0)),
        ("p95",    np.nanpercentile(np.abs(rel), 95, axis=0)),
        ("max",    np.nanmax(np.abs(rel), axis=0)),
    ]
    vmax = float(np.nanmax(stats[-1][1])) * 100
    for ax, (label, arr) in zip(axes, stats):
        im = ax.pcolormesh(np.arange(nl), np.arange(nz), arr * 100,
                           cmap="Reds", vmin=0, vmax=vmax, shading="nearest")
        for j in range(nz):
            for i in range(nl):
                ax.text(i, j, f"{arr[j,i]*100:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if arr[j,i]*100 > vmax*0.55 else "k")
        ax.set_xticks(np.arange(nl)); ax.set_xticklabels(LAMBDA_BIN_LABELS, fontsize=8)
        ax.set_yticks(np.arange(nz)); ax.set_yticklabels(Z_BIN_LABELS, fontsize=8)
        ax.set_title(rf"$|{{\Delta N}}/N|_{{\rm {label}}}$ [\%]")
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(FIGS / "scan_precision_heatmap.pdf")
    plt.close(fig)

    # --- scatter vs cosmo at worst bin ---
    worst_idx = np.unravel_index(np.argmax(np.abs(rel).mean(axis=0)), rel.shape[1:])
    rel_worst = rel[:, worst_idx[0], worst_idx[1]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, x, label in [(ax1, om, r"$\Omega_m$"), (ax2, lAs, r"$\log(10^{10}A_s)$")]:
        sc = ax.scatter(x, rel_worst * 100, c=lAs if ax is ax1 else om,
                        cmap="viridis", s=30, edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(0.5, ls=":", color="k", alpha=0.4)
        ax.axhline(-0.5, ls=":", color="k", alpha=0.4)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$(N_{\rm emu}-N_{\rm CAMB})/N_{\rm CAMB}$ [\%]")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label=(r"$\log(10^{10}A_s)$" if ax is ax1 else r"$\Omega_m$"))
    z_lbl = Z_BIN_LABELS[worst_idx[0]]
    l_lbl = LAMBDA_BIN_LABELS[worst_idx[1]]
    fig.suptitle(
        rf"Residual vs.\ cosmology (bin with largest mean: $z${z_lbl}, $\lambda${l_lbl})")
    fig.tight_layout()
    fig.savefig(FIGS / "scan_scatter_vs_cosmo.pdf")
    plt.close(fig)

    # --- mean |rel| over plane ---
    mean_abs = np.abs(rel).mean(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(om, lAs, c=mean_abs*100, s=60, cmap="Reds",
                    edgecolor="k", linewidth=0.3)
    ax.set_xlabel(r"$\Omega_m$"); ax.set_ylabel(r"$\log(10^{10}A_s)$")
    ax.set_title(r"Mean $|\Delta N/N|$ over 12 $(\lambda,z)$ bins [\%]")
    fig.colorbar(sc, ax=ax, label="percent")
    fig.tight_layout()
    fig.savefig(FIGS / "scan_meanerr_over_plane.pdf")
    plt.close(fig)

    # --- P(k,0) envelope + vs Om ---
    k_ref = np.loadtxt(camb_root / f"block_{valid[0]}" / "cdm_baryon_power_lin" / "k_h.txt")
    rel_pk0 = np.zeros((n, k_ref.size))
    for ii, i in enumerate(valid):
        # _load_pk returns p in shape (n_k, n_z) after transpose
        kc, zc, pc = _load_pk(camb_root / f"block_{i}")
        ke, ze, pe = _load_pk(emu_root  / f"block_{i}")
        i0c = int(np.argmin(np.abs(zc))); i0e = int(np.argmin(np.abs(ze)))
        pc_z0 = pc[:, i0c]
        pe_z0 = pe[:, i0e]
        pe_on_c = np.exp(np.interp(np.log(k_ref), np.log(ke), np.log(pe_z0)))
        rel_pk0[ii] = (pe_on_c - pc_z0) / pc_z0

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for ii in range(n):
        ax.plot(k_ref, rel_pk0[ii]*100, color="k", alpha=0.08, lw=0.6)
    p05 = np.percentile(rel_pk0, 5, axis=0)*100
    p95 = np.percentile(rel_pk0, 95, axis=0)*100
    p50 = np.median(rel_pk0, axis=0)*100
    ax.fill_between(k_ref, p05, p95, alpha=0.25, color="C0", label="5--95\\%")
    ax.plot(k_ref, p50, color="C0", lw=2, label="median")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, ls=":", color="k", alpha=0.4); ax.axhline(-1, ls=":", color="k", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\;[h/\mathrm{Mpc}]$")
    ax.set_ylabel(r"$(P_{\rm emu}-P_{\rm CAMB})/P_{\rm CAMB}$ at $z=0$ [\%]")
    ax.set_title(r"$P_{\rm cb}(k, z=0)$ emulator error over 100 samples (v2c)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "scan_pk_residual_k.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, kp in zip(axes.flatten(), [1e-3, 1e-2, 1e-1, 5e-1]):
        ik = np.argmin(np.abs(np.log(k_ref/kp)))
        sc = ax.scatter(om, rel_pk0[:, ik]*100, c=lAs, cmap="viridis",
                        s=35, edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(r"$\Omega_m$"); ax.set_ylabel(r"$\Delta P/P$ [\%]")
        ax.set_title(rf"$k={k_ref[ik]:.3g}\,h/\mathrm{{Mpc}}$")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label=r"$\log(10^{10}A_s)$")
    fig.suptitle(r"v2c: emulator $P(k, z=0)$ error vs $\Omega_m$")
    fig.tight_layout()
    fig.savefig(FIGS / "scan_pk_residual_vs_om.pdf")
    plt.close(fig)

    # --- growth residual vs Om ---
    z_probes = [0.3, 0.5, 0.7, 1.0]
    growth_rel = np.zeros((n, len(z_probes)))
    for ii, i in enumerate(valid):
        zc, Dc = _load_growth(camb_root / f"block_{i}")
        ze, De = _load_growth(emu_root  / f"block_{i}")
        for jj, zt in enumerate(z_probes):
            growth_rel[ii, jj] = np.interp(zt, ze, De) / np.interp(zt, zc, Dc) - 1.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    col = plt.cm.plasma(np.linspace(0.1, 0.85, len(z_probes)))
    for jj, zt in enumerate(z_probes):
        ax.scatter(om, growth_rel[:, jj]*100, color=col[jj], s=28,
                   edgecolor="k", linewidth=0.2, label=rf"$z={zt}$")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.1, ls=":", color="k", alpha=0.4); ax.axhline(-0.1, ls=":", color="k", alpha=0.4)
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$D_{\rm growth\_factor}/D_{\rm CAMB}-1$ [\%]")
    ax.set_title(r"Growth-factor residual vs $\Omega_m$ (v2c scan)")
    ax.legend(fontsize=9, title="redshift")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "scan_growth_vs_om.pdf")
    plt.close(fig)

    # --- summary ---
    summary = HERE / "scan_summary.txt"
    with summary.open("w") as fh:
        fh.write(f"# v2c retrained emulator summary  n_samples={n}\n")
        fh.write(f"omega_m   sampled = [{om.min():.4f}, {om.max():.4f}]\n")
        fh.write(f"log1e10As sampled = [{lAs.min():.4f}, {lAs.max():.4f}]\n\n")
        fh.write("# per-bin |rel N| percentiles over the scan (%):\n")
        for j in range(nz):
            for i in range(nl):
                v = np.abs(rel[:, j, i]) * 100
                fh.write(f"z={Z_BIN_LABELS[j]:<16s} lambda={LAMBDA_BIN_LABELS[i]:<12s} "
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
        fh.write("\n# growth rel-error summary (%):\n")
        for jj, zt in enumerate(z_probes):
            v = growth_rel[:, jj] * 100
            fh.write(f"z={zt:.1f}: median={np.median(v):+.4f}  "
                     f"range=[{v.min():+.4f}, {v.max():+.4f}]  "
                     f"rho(Omega_m)={np.corrcoef(v, om)[0,1]:+.3f}\n")


def main():
    fiducial()
    scan()
    print("done")


if __name__ == "__main__":
    main()
