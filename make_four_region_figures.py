#!/usr/bin/env python3
"""Final qualitative figures: AMDI versus Learned AMDI.

Evaluation only. No policy training is performed.

This submission-ready version produces the reconstruction/refinement figure with:
  1. six panel axes of identical size,
  2. a dedicated grayscale intensity colorbar on the far left,
  3. a dedicated refinement-level colorbar on the far right,
  4. spacer columns so the side colorbars do not overlap with panel labels,
     tick labels, or axis labels,
  5. bold panel identifiers without redundant panel titles, and
  6. enlarged, embedded journal-scale typography and marks.

Run from the Learned AMDI repository:
    python make_four_region_figures.py \
        --checkpoint checkpoints/selected/best_policy.pt \
        --device cpu \
        --outdir IMG

The figure uses only the display names "AMDI" and "Learned AMDI".
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if not (ROOT / "learned_amdi").exists():
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from learned_amdi.config import RewardConfig
from learned_amdi.datasets import noisy_case
from learned_amdi.evaluation import load_policy
from learned_amdi.runner import collect_trajectory, run_deterministic_trajectory
from learned_amdi.training import build_backend, make_reference


def panel(ax, label):
    ax.text(
        0.025,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16.0,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.8),
        zorder=20,
    )


def level_map(backend):
    n = backend.shape[0]
    result = np.zeros((n, n), dtype=float)
    for level, tr in backend.active_leaf_cells():
        width = n // (2**level)
        r0 = int(tr[0] * width)
        c0 = int(tr[1] * width)
        result[r0 : r0 + width, c0 : c0 + width] = level
    return result


def save(fig, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_image_axis(ax, label):
    ax.set_xlabel(r"$x$ [-]")
    ax.set_ylabel(r"$y$ [-]")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_box_aspect(1)
    panel(ax, label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/selected/best_policy.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--outdir", default="IMG")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--noise-seed", type=int, default=137)
    args = p.parse_args()

    # Sized for reduction to a journal text width of approximately 6.5--7 in.
    # The final printed text is therefore about 9--10 pt rather than the
    # undersized 6--7 pt produced by the earlier figure sources.
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 14.0,
        "axes.labelsize": 15.5,
        "xtick.labelsize": 13.0,
        "ytick.labelsize": 13.0,
        "legend.fontsize": 12.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 7.0,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if args.device == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = args.device
    device = torch.device(dev)

    actor, critic, scales, cfg = load_policy(args.checkpoint, device)
    n_steps = int(cfg["training"]["n_steps"])
    sigma = float(cfg["data"]["noise_sigma"])
    rcfg = RewardConfig(**cfg["reward"])

    truth, noisy = noisy_case("four_region", args.n, 0, sigma, args.noise_seed)

    tmp = build_backend(noisy, cfg)
    reference = make_reference(noisy, tmp, n_steps)

    # AMDI
    b_amdi = build_backend(noisy, cfg)
    diag_amdi = run_deterministic_trajectory(
        b_amdi, reference, n_steps, "baseline"
    )
    u_amdi = np.clip(b_amdi.current_field(), 0, 1)
    l_amdi = level_map(b_amdi)

    # Learned AMDI
    b_learn = build_backend(noisy, cfg)
    _, diag_learn = collect_trajectory(
        b_learn, reference, actor, critic, scales, rcfg, n_steps, device, True
    )
    u_learn = np.clip(b_learn.current_field(), 0, 1)
    l_learn = level_map(b_learn)

    # ================================================================
    # Reconstruction + refinement
    # ================================================================
    # Layout: [left colorbar] [left spacer] [panel 1] [panel 2] [panel 3]
    #         [right spacer] [right colorbar]
    # This guarantees identical panel sizes while keeping the colorbars clear
    # of the panel y-labels and tick labels.
    fig = plt.figure(figsize=(11.0, 6.35))
    gs = GridSpec(
        2,
        7,
        figure=fig,
        width_ratios=[0.11, 0.10, 1.0, 1.0, 1.0, 0.10, 0.11],
        height_ratios=[1.0, 1.0],
        left=0.055,
        right=0.975,
        bottom=0.105,
        top=0.97,
        wspace=0.26,
        hspace=0.28,
    )

    # panel axes: all six are placed in the same three central columns
    ax_a = fig.add_subplot(gs[0, 2])
    ax_b = fig.add_subplot(gs[0, 3])
    ax_c = fig.add_subplot(gs[0, 4])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[1, 3])
    ax_f = fig.add_subplot(gs[1, 4])

    # side colorbar axes
    cax_left = fig.add_subplot(gs[:, 0])
    cax_right = fig.add_subplot(gs[:, 6])

    # grayscale / intensity panels
    im_gray = ax_a.imshow(
        truth,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(0, 1, 1, 0),
        interpolation="nearest",
        aspect="equal",
    )
    ax_b.imshow(
        noisy,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(0, 1, 1, 0),
        interpolation="nearest",
        aspect="equal",
    )
    ax_c.imshow(
        u_amdi,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(0, 1, 1, 0),
        interpolation="nearest",
        aspect="equal",
    )
    ax_d.imshow(
        u_learn,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(0, 1, 1, 0),
        interpolation="nearest",
        aspect="equal",
    )

    # refinement panels
    vmax_level = int(max(b_amdi.max_level, b_learn.max_level))
    ax_e_im = ax_e.imshow(
        l_amdi,
        cmap="viridis",
        vmin=0,
        vmax=vmax_level,
        extent=(0, 1, 1, 0),
        interpolation="nearest",
        aspect="equal",
    )
    ax_f_im = ax_f.imshow(
        l_learn,
        cmap="viridis",
        vmin=0,
        vmax=vmax_level,
        extent=(0, 1, 1, 0),
        interpolation="nearest",
        aspect="equal",
    )

    # style all six panel axes identically
    style_image_axis(ax_a, "(a)")
    style_image_axis(ax_b, "(b)")
    style_image_axis(ax_c, "(c)")
    style_image_axis(ax_d, "(d)")
    style_image_axis(ax_e, "(e)")
    style_image_axis(ax_f, "(f)")

    # left intensity colorbar (shared by clean/noisy/AMDI/Learned AMDI)
    cbar_left = fig.colorbar(im_gray, cax=cax_left, orientation="vertical")
    cbar_left.set_ticks(np.linspace(0.0, 1.0, 6))
    cbar_left.ax.yaxis.set_ticks_position("left")
    cbar_left.ax.yaxis.set_label_position("left")
    cbar_left.ax.tick_params(pad=2)
    cbar_left.set_label("Normalized intensity [-]", rotation=90, labelpad=8)

    # right refinement colorbar (shared by AMDI / Learned AMDI refinement)
    cbar_right = fig.colorbar(ax_f_im, cax=cax_right, orientation="vertical")
    cbar_right.set_ticks(np.arange(0, vmax_level + 1))
    cbar_right.ax.yaxis.set_ticks_position("right")
    cbar_right.ax.yaxis.set_label_position("right")
    cbar_right.ax.tick_params(pad=2)
    cbar_right.set_label("Leaf refinement level [-]", rotation=90, labelpad=10)

    save(fig, Path(args.outdir), "Fig_four_region_reconstruction_and_refinement")

    # ================================================================
    # Outer-iteration trajectory diagnostics
    # ================================================================
    steps = np.arange(1, n_steps + 1)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.45))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.80, wspace=0.68)

    fields = [
        ("reference_error", r"$E_{\rm ref}$ [-]"),
        ("C_rel", r"$C_{\rm rel}$ [-]"),
        ("switching", r"$\sigma_{\rm sw}$ [-]"),
    ]

    handles = labels = None
    for j, (ax, (key, ylabel)) in enumerate(zip(axes, fields)):
        ax.plot(steps, [d[key] for d in diag_amdi], "s-", label="AMDI")
        ax.plot(steps, [d[key] for d in diag_learn], "o-", label="Learned AMDI")
        ax.set_xlabel("outer iteration [count]")
        ax.set_ylabel(ylabel, labelpad=4)
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.20)
        ax.margins(y=0.10)
        panel(ax, f"({chr(97 + j)})")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.98),
        ncol=2,
        frameon=False,
    )

    save(fig, Path(args.outdir), "Fig_four_region_trajectory_diagnostics")

    print("Evaluation-only figures written to:", Path(args.outdir).resolve())


if __name__ == "__main__":
    main()
