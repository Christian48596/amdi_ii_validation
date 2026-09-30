#!/usr/bin/env python3
"""Regenerate the matched PPO/control frontiers from archived result files.

This script performs no simulation and no policy training.  It reads the
four-image PPO summary and the control frontiers produced by experiment 12.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def panel(ax, label: str) -> None:
    ax.text(
        0.025,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.8),
        zorder=20,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppo-summary",
        type=Path,
        default=Path("results/publication_summary_FINAL/publication_summary_FINAL.json"),
    )
    parser.add_argument(
        "--control-frontiers",
        type=Path,
        default=Path("results/12_section5_controls/pareto_control_frontiers.csv"),
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("results/12_section5_controls/pareto_baselines.csv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("IMG"))
    args = parser.parse_args()

    archived = json.loads(args.ppo_summary.read_text(encoding="utf-8"))
    controls = pd.read_csv(args.control_frontiers)
    baselines = pd.read_csv(args.baselines)

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 12.0,
            "axes.labelsize": 12.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "lines.linewidth": 2.1,
            "lines.markersize": 6.5,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    learned = pd.DataFrame(
        {
            "C_rel": [float(r["C_rel_mean"]) for r in archived["pareto"]],
            "reference_error": [float(r["reference_error_mean"]) for r in archived["pareto"]],
            "RMSE": [float(r["RMSE_mean"]) for r in archived["pareto"]],
            "SSIM": [float(r["SSIM_mean"]) for r in archived["pareto"]],
        }
    )

    series = [
        ("Learned AMDI", learned, "#1f77b4", "o", 7.0),
        (
            "Top-$K$ observed detail",
            controls[controls["mode"] == "topk"],
            "#ff7f0e",
            "s",
            5.8,
        ),
        (
            "Observed-detail threshold",
            controls[controls["mode"] == "threshold"],
            "#2ca02c",
            "D",
            5.8,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.45))
    fields = (
        ("reference_error", r"$E_{\mathrm{ref}}$ [-]"),
        ("RMSE", "RMSE [-]"),
        ("SSIM", "SSIM [-]"),
    )

    for index, (ax, (field, ylabel)) in enumerate(zip(axes, fields)):
        for label, data, color, marker, marker_size in series:
            ordered = data.sort_values("C_rel")
            ax.plot(
                ordered["C_rel"],
                ordered[field],
                color=color,
                marker=marker,
                markersize=marker_size,
                markeredgewidth=0.6,
                label=label,
            )

        baseline_styles = (
            ("AMDI", "#d62728", "s", 9.0, "#d62728"),
            ("Retain initial tree", "#9467bd", "^", 9.0, "none"),
        )
        for method, color, marker, marker_size, facecolor in baseline_styles:
            row = baselines[baselines["method"] == method].iloc[0]
            ax.plot(
                float(row["C_rel"]),
                float(row[field]),
                linestyle="none",
                marker=marker,
                markersize=marker_size,
                markerfacecolor=facecolor,
                markeredgecolor=color,
                markeredgewidth=1.5,
                label=method,
                zorder=10,
            )

        ax.set_xlabel(r"Terminal occupancy $C_{\mathrm{rel}}$ [-]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.margins(x=0.04, y=0.08)
        panel(ax, f"({chr(97 + index)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=1.35,
        handletextpad=0.55,
    )
    fig.subplots_adjust(top=0.73, bottom=0.22, left=0.075, right=0.985, wspace=0.39)

    args.outdir.mkdir(parents=True, exist_ok=True)
    pdf = args.outdir / "Fig_section5_matched_frontiers.pdf"
    png = args.outdir / "Fig_section5_matched_frontiers.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
