#!/usr/bin/env python3
"""Final publication figures for Learned AMDI.

This script reads publication_summary_FINAL.json and the archived
Section 5 transfer-control summary.
No training and no numerical simulation are performed.

Display names used in every figure:
    AMDI
    Learned AMDI
    VAMPyR/MRCPP

"Haar" is intentionally omitted from plot legends. The basis can be stated
once in the manuscript caption/method section instead of cluttering the plot.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def F(x):
    return float(x)


def panel(ax, label):
    # Keep panel labels INSIDE the axes so they never collide with neighbours.
    ax.text(
        0.025, 0.975, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12.5, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8),
        zorder=20,
    )


def save(fig, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="results/publication_summary_FINAL/publication_summary_FINAL.json",
    )
    parser.add_argument(
        "--transfer-summary",
        default="results/12_section5_controls/transfer_summary.csv",
        help="Archived four-method resolution-transfer summary.",
    )
    parser.add_argument("--outdir", default="IMG")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)

    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 11.0,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 7.0,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ================================================================
    # 1. Accuracy--representation Pareto family
    # ================================================================
    rows = sorted(summary["pareto"], key=lambda r: F(r["C_rel_mean"]))
    crel = np.array([F(r["C_rel_mean"]) for r in rows])
    eref = np.array([F(r["reference_error_mean"]) for r in rows])
    rmse = np.array([F(r["RMSE_mean"]) for r in rows])
    ssim = np.array([F(r["SSIM_mean"]) for r in rows])
    lam  = np.array([F(r["lambda_occ"]) for r in rows])

    # Wider figure + large wspace: y-label of panel (b) cannot enter panel (a).
    fig, axes = plt.subplots(1, 3, figsize=(9.20, 2.90))
    fig.subplots_adjust(
        left=0.070, right=0.99, bottom=0.20, top=0.94, wspace=0.62
    )

    quantities = [
        (eref, r"$E_{\rm ref}$ [-]"),
        (rmse, "RMSE [-]"),
        (ssim, "SSIM [-]"),
    ]
    for j, (ax, (vals, ylabel)) in enumerate(zip(axes, quantities)):
        ax.plot(crel, vals, "o-")
        ax.set_xlabel(r"$C_{\rm rel}$ [-]")
        ax.set_ylabel(ylabel, labelpad=4)
        ax.grid(True, alpha=0.20)
        ax.margins(x=0.08, y=0.12)
        panel(ax, f"({chr(97+j)})")


    # Only identify operating points discussed explicitly in the manuscript.
    for x, y, w in zip(crel, eref, lam):
        if np.isclose(w, 0.25):
            axes[0].annotate(
                r"$\lambda_{\rm occ}=0.25$", (x, y),
                xytext=(12, -12), textcoords="offset points", fontsize=9.0,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.5),
            )
        elif np.isclose(w, 0.15):
            axes[0].annotate(
                r"$\lambda_{\rm occ}=0.15$", (x, y),
                xytext=(10, 6), textcoords="offset points", fontsize=9.0,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.5),
            )

    save(fig, outdir, "Fig_pareto_accuracy_complexity")

    # ================================================================
    # 2. Resolution transfer
    # ================================================================
    with Path(args.transfer_summary).open(newline="", encoding="utf-8") as stream:
        transfer = list(csv.DictReader(stream))

    method_map = {
        "AMDI": "Deterministic AMDI",
        "Learned AMDI": "Learned AMDI",
        "Top-K: matched validation occupancy": r"Top-$K$ control",
        "Threshold: matched validation occupancy": "Threshold control",
    }
    styles = {
        "AMDI": dict(color="#4D4D4D", marker="s", linestyle="--"),
        "Learned AMDI": dict(color="#0072B2", marker="o", linestyle="-"),
        "Top-K: matched validation occupancy": dict(
            color="#D55E00", marker="^", linestyle="-."
        ),
        "Threshold: matched validation occupancy": dict(
            color="#009E73", marker="D", linestyle=":"
        ),
    }
    ns = sorted({int(r["n"]) for r in transfer})

    fields = [
        ("RMSE", "RMSE [-]"),
        ("SSIM", "SSIM [-]"),
        ("reference_error", r"$E_{\rm ref}$ [-]"),
        ("C_rel", r"$C_{\rm rel}$ [-]"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.65))
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.105, top=0.845,
        wspace=0.31, hspace=0.42
    )
    axes = axes.ravel()

    handles = labels = None
    for j, (ax, (key, ylabel)) in enumerate(zip(axes, fields)):
        for raw_name, display_name in method_map.items():
            ys = []
            for n in ns:
                row = next(
                    r for r in transfer
                    if int(r["n"]) == n and r["method"] == raw_name
                )
                ys.append(F(row[key]))
            ax.plot(
                ns,
                ys,
                label=display_name,
                markerfacecolor="white",
                markeredgewidth=1.35,
                linewidth=2.15,
                markersize=7.5,
                **styles[raw_name],
            )

        ax.set_xlabel(r"Image resolution $N$")
        ax.set_ylabel(ylabel, labelpad=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(v) for v in ns])
        ax.grid(True, color="0.82", linewidth=0.7, alpha=0.65)
        ax.margins(x=0.08, y=0.13)
        panel(ax, f"({chr(97+j)})")

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.545, 0.975), ncol=2, frameon=False,
        columnspacing=1.8, handlelength=2.7, handletextpad=0.65
    )
    save(fig, outdir, "Fig_resolution_transfer")

    # ================================================================
    # 3. Training-seed robustness + reward ablation
    # ================================================================
    seeds = sorted(
        summary["training_seed_summary"],
        key=lambda r: int(r["training_seed"])
    )
    ablation = summary["reward_ablation"]

    fig, axes = plt.subplots(1, 3, figsize=(9.35, 3.00))
    fig.subplots_adjust(
        left=0.070, right=0.99, bottom=0.20, top=0.90, wspace=0.66
    )

    # (a) Seed operating points
    ax = axes[0]
    offsets = {
        20260811: (7, 5),
        20260817: (7, 5),
        20260823: (7, -15),
    }
    for row in seeds:
        seed = int(row["training_seed"])
        x = F(row["C_rel_mean"])
        y = F(row["reference_error_mean"])
        ax.plot(x, y, "o")
        ax.annotate(
            str(seed)[-2:], (x, y),
            xytext=offsets[seed], textcoords="offset points", fontsize=9.0,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.4),
        )
    ax.set_xlabel(r"$C_{\rm rel}$ [-]")
    ax.set_ylabel(r"$E_{\rm ref}$ [-]")
    ax.grid(True, alpha=0.20)
    ax.margins(x=0.18, y=0.20)
    panel(ax, "(a)")

    # (b) Categorical seed comparison -- no misleading connecting lines
    ax = axes[1]
    pos = np.arange(len(seeds), dtype=float)
    suffix = [str(int(r["training_seed"]))[-2:] for r in seeds]
    rm = np.array([F(r["RMSE_mean"]) for r in seeds])
    ss = np.array([F(r["SSIM_mean"]) for r in seeds])

    ax.plot(pos - 0.07, rm/rm.mean(), "o", linestyle="none", label="RMSE / mean")
    ax.plot(pos + 0.07, ss/ss.mean(), "s", linestyle="none", label="SSIM / mean")
    ax.axhline(1.0, linewidth=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(suffix)
    ax.set_xlabel("training seed suffix [-]")
    ax.set_ylabel("normalized metric [-]", labelpad=3)
    ax.grid(True, alpha=0.20)
    ax.margins(x=0.20, y=0.18)
    ax.legend(
        frameon=False, loc="upper center",
        bbox_to_anchor=(0.52, 0.98), ncol=1
    )
    panel(ax, "(b)")

    # (c) Reward ablation
    ax = axes[2]
    cfg = {
        "full":          (8, -15, "Full reward"),
        "no_switching":  (8,  7,  "No switching"),
        "no_occupancy": (-62, 8,  "No occupancy"),
    }
    for row in ablation:
        key = row["ablation"]
        x = F(row["C_rel_mean"])
        y = F(row["reference_error_mean"])
        ax.plot(x, y, "o")
        dx, dy, txt = cfg[key]
        ax.annotate(
            txt, (x, y), xytext=(dx, dy),
            textcoords="offset points", fontsize=9.0,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.4),
        )
    ax.set_xlabel(r"$C_{\rm rel}$ [-]")
    ax.set_ylabel(r"$E_{\rm ref}$ [-]", labelpad=4)
    ax.grid(True, alpha=0.20)
    ax.margins(x=0.14, y=0.17)
    panel(ax, "(c)")

    save(fig, outdir, "Fig_robustness_ablation")

    # ================================================================
    # 4. VAMPyR/MRCPP localization cross-check
    # ================================================================
    vr = summary["vampyr_localization"]
    regions = [r["region"] for r in vr]
    xpos = np.arange(len(regions), dtype=float)
    width = 0.24

    amdi = np.array([F(r["amdi_mean_level"]) for r in vr])
    learned = np.array([F(r["learned_mean_level"]) for r in vr])
    vampyr = np.array([F(r["vampyr_mean_effective_level"]) for r in vr])

    fig, ax = plt.subplots(figsize=(6.5, 3.45))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.19, top=0.78)

    # NO "Haar" in the legend.
    b1 = ax.bar(xpos-width, amdi, width=width, label="AMDI")
    b2 = ax.bar(xpos, learned, width=width, label="Learned AMDI")
    b3 = ax.bar(xpos+width, vampyr, width=width, label="VAMPyR/MRCPP")

    ax.set_xticks(xpos)
    ax.set_xticklabels(regions)
    ax.set_xlabel("region [-]")
    ax.set_ylabel("mean refinement/effective level [-]")
    ax.grid(True, axis="y", alpha=0.20)
    ax.set_ylim(0, max(amdi.max(), learned.max(), vampyr.max())*1.18)

    ax.legend(
        frameon=False, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, 1.28)
    )

    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9.5)

    save(fig, outdir, "Fig_vampyr_localization")

    print("Final metric figures written to:", outdir.resolve())


if __name__ == "__main__":
    main()
