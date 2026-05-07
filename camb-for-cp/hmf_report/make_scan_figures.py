"""Analyze the 100-sample (omega_m, log1e10As) cosmology scan.

Inputs (produced by the `list` sampler with `save =` set):
  cosmology_scan_camb_output/block_<i>.tgz
  cosmology_scan_cp_output/block_<i>.tgz
and the sample list at cosmosis-models/cosmology_scan.list.

Expects each tgz to have been pre-extracted to
    {save_dir}_extracted/{save_dir}/block_<i>/numcountsfullscalarintegrand/vals.txt

Writes:
  scan_summary.txt            (machine-readable)
  figs/scan_precision.pdf     (per-bin distribution of rel errors)
  figs/scan_precision_heatmap.pdf (median/p95/max across the 100 scan
                                   samples, per (lambda, z) bin)
  figs/scan_scatter_vs_cosmo.pdf (rel error vs omega_m and log1e10As)
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


def _load_all(extracted_root: Path, inner_name: str):
    """Return (n_samples, n_z, n_lambda) stack of N_i(lambda, z)."""
    inner = extracted_root / inner_name
    blocks = sorted(inner.glob("block_*"), key=lambda p: int(p.name.split("_")[1]))
    stacks = []
    for b in blocks:
        vals = np.loadtxt(b / "numcountsfullscalarintegrand" / "vals.txt")
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        stacks.append(vals)
    return np.array(stacks)


def main():
    root = Path("/pscratch/sd/j/jesteves/cosmology_scan")
    N_camb = _load_all(root / "cosmology_scan_camb_output_extracted",
                       "cosmology_scan_camb_output")   # (n_samp, n_z, n_lambda)
    N_cp   = _load_all(root / "cosmology_scan_cp_output_extracted",
                       "cosmology_scan_cp_output")
    print(f"CAMB samples: {N_camb.shape}")
    print(f"CP   samples: {N_cp.shape}")
    assert N_camb.shape == N_cp.shape, "sample count mismatch"

    cosmo_list = np.loadtxt(
        "/global/common/software/des/jesteves/y3_cluster_cpp/cosmosis-models/cosmology_scan.list",
        comments="#")
    omega_m, log1e10As = cosmo_list[:, 0], cosmo_list[:, 1]

    rel = (N_cp - N_camb) / N_camb          # (n_samp, n_z, n_lambda)
    n_samp, n_z, n_lam = rel.shape

    # ---- Figure 1: violin/box of rel error per (lambda, z) bin ----
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    positions = []
    labels = []
    colors = []
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_z))
    for j in range(n_z):
        for i in range(n_lam):
            data.append(rel[:, j, i] * 100.0)
            positions.append(i + (j - (n_z - 1) / 2) * 0.22)
            labels.append("")
            colors.append(cmap[j])
    bp = ax.boxplot(data, positions=positions, widths=0.18,
                    patch_artist=True, showfliers=True,
                    medianprops={"color": "k", "lw": 1.2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for j in range(n_z):
        ax.plot([], [], color=cmap[j], lw=8, alpha=0.7,
                label=f"$z$ {Z_BIN_LABELS[j]}")
    ax.set_xticks(np.arange(n_lam))
    ax.set_xticklabels(LAMBDA_BIN_LABELS)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.axhline(-0.5, ls=":", color="k", alpha=0.4)
    ax.set_xlabel(r"$\lambda$ bin")
    ax.set_ylabel(r"$(N_{\rm emu} - N_{\rm CAMB})/N_{\rm CAMB}$ [\%]")
    ax.set_title(
        r"Cosmology-scan precision over 100 $(\Omega_m,\,\log(10^{10}A_s))$ samples")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "scan_precision.pdf")
    plt.close(fig)

    # ---- Figure 2: (median, p95, max) heatmap over the 100 samples ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    stats = [
        ("median", np.nanmedian(np.abs(rel), axis=0)),
        ("p95",    np.nanpercentile(np.abs(rel), 95, axis=0)),
        ("max",    np.nanmax(np.abs(rel), axis=0)),
    ]
    vmax = float(np.nanmax(stats[-1][1])) * 100
    for ax, (label, arr) in zip(axes, stats):
        im = ax.pcolormesh(np.arange(n_lam), np.arange(n_z), arr * 100,
                           cmap="Reds", vmin=0, vmax=vmax, shading="nearest")
        for j in range(n_z):
            for i in range(n_lam):
                ax.text(i, j, f"{arr[j, i]*100:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if arr[j, i] * 100 > vmax * 0.55 else "k")
        ax.set_xticks(np.arange(n_lam))
        ax.set_xticklabels(LAMBDA_BIN_LABELS, fontsize=8)
        ax.set_yticks(np.arange(n_z))
        ax.set_yticklabels(Z_BIN_LABELS, fontsize=8)
        ax.set_title(
            rf"$|{{\Delta N}}/N|_{{\rm {label}}}$ over 100 samples [\%]")
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(FIGS / "scan_precision_heatmap.pdf")
    plt.close(fig)

    # ---- Figure 3: rel error scatter vs cosmology params ----
    # Collapse to the worst (z, lambda) bin for the scatter.
    worst_idx = np.unravel_index(np.argmax(np.abs(rel).mean(axis=0)), rel.shape[1:])
    rel_worst = rel[:, worst_idx[0], worst_idx[1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, x, label in [
        (ax1, omega_m, r"$\Omega_m$"),
        (ax2, log1e10As, r"$\log(10^{10} A_s)$"),
    ]:
        sc = ax.scatter(x, rel_worst * 100, c=log1e10As if ax is ax1 else omega_m,
                        cmap="viridis", s=30, edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(0.5, ls=":", color="k", alpha=0.4)
        ax.axhline(-0.5, ls=":", color="k", alpha=0.4)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$(N_{\rm emu} - N_{\rm CAMB})/N_{\rm CAMB}$ [\%]")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$\log(10^{10} A_s)$" if ax is ax1 else r"$\Omega_m$")
    z_lbl = Z_BIN_LABELS[worst_idx[0]]
    l_lbl = LAMBDA_BIN_LABELS[worst_idx[1]]
    fig.suptitle(
        rf"Residual vs.\ cosmology (bin with largest mean $|\Delta N/N|$: "
        rf"$z\in${z_lbl}, $\lambda\in${l_lbl})")
    fig.tight_layout()
    fig.savefig(FIGS / "scan_scatter_vs_cosmo.pdf")
    plt.close(fig)

    # ---- Figure 4: rel error heatmap (median bin) vs (omega_m, log1e10As) ----
    # Show the mean |rel| across (lambda, z) for each sample.
    mean_abs_rel = np.abs(rel).mean(axis=(1, 2))  # (n_samp,)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(omega_m, log1e10As, c=mean_abs_rel * 100, s=60,
                    cmap="Reds", edgecolor="k", linewidth=0.3)
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$\log(10^{10} A_s)$")
    ax.set_title(
        r"Mean $|\Delta N/N|$ over 12 $(\lambda,z)$ bins [\%]")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("percent")
    fig.tight_layout()
    fig.savefig(FIGS / "scan_meanerr_over_plane.pdf")
    plt.close(fig)

    # ---- Summary ----
    summary = HERE / "scan_summary.txt"
    with summary.open("w") as fh:
        fh.write(f"n_samples = {n_samp}\n")
        fh.write(f"omega_m   range = [{omega_m.min():.4f}, {omega_m.max():.4f}]\n")
        fh.write(f"log1e10As range = [{log1e10As.min():.4f}, {log1e10As.max():.4f}]\n")
        fh.write("\n# |rel| percentiles per (z_bin, lambda_bin) over 100 samples\n")
        fh.write("# (in percent)\n")
        for j in range(n_z):
            for i in range(n_lam):
                vals = np.abs(rel[:, j, i]) * 100
                fh.write(
                    f"z={Z_BIN_LABELS[j]:<16s} lambda={LAMBDA_BIN_LABELS[i]:<12s} "
                    f"median={np.median(vals):.3f}  "
                    f"p95={np.percentile(vals, 95):.3f}  "
                    f"max={np.max(vals):.3f}\n")
        worst_sample_bin = np.unravel_index(np.argmax(np.abs(rel)), rel.shape)
        fh.write(
            f"\nglobal worst: sample {worst_sample_bin[0]} "
            f"(omega_m={omega_m[worst_sample_bin[0]]:.4f}, "
            f"log1e10As={log1e10As[worst_sample_bin[0]]:.4f}), "
            f"z_bin={worst_sample_bin[1]}, lambda_bin={worst_sample_bin[2]}, "
            f"rel={rel[worst_sample_bin]*100:+.3f}%\n")
    print("wrote figures to", FIGS)
    print("wrote summary to", summary)


if __name__ == "__main__":
    main()
