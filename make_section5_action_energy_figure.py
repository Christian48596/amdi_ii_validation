#!/usr/bin/env python3
"""Create the Section 5 action/energy audit figure from experiment 11."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def panel(ax, label: str) -> None:
    """Place a bold journal-style panel identifier inside an axis."""
    ax.text(
        0.025,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.8),
        zorder=20,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("results/11_section5_diagnostics"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("IMG"))
    args = parser.parse_args()

    step_file = args.diagnostics_dir / "step_diagnostics.csv"
    rows = pd.read_csv(step_file)
    rows = rows[(rows["dataset"] == "holdout") & (rows["n"] == 32)].copy()
    if rows.empty:
        raise RuntimeError(f"no 32x32 holdout rows in {step_file}")

    methods = ["AMDI", "Learned AMDI", "Learned: decision 1 only"]
    labels = {
        "AMDI": "Deterministic AMDI",
        "Learned AMDI": "Learned AMDI",
        "Learned: decision 1 only": "Learned: decision 1 only",
    }
    colors = {
        "AMDI": "#d62728",
        "Learned AMDI": "#1f77b4",
        "Learned: decision 1 only": "#9467bd",
    }
    markers = {"AMDI": "s", "Learned AMDI": "o", "Learned: decision 1 only": "^"}

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 11.0,
            "axes.labelsize": 11.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "lines.linewidth": 2.0,
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
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    ax_err, ax_dof, ax_energy, ax_delta = axs.ravel()

    for method in methods:
        g = rows[rows["method"] == method].groupby("step", as_index=False).mean(numeric_only=True)
        ls = "--" if method == "Learned: decision 1 only" else "-"
        ax_err.plot(
            g["step"],
            g["reference_error"],
            marker=markers[method],
            color=colors[method],
            linestyle=ls,
            linewidth=2.0,
            markersize=6.5,
            label=labels[method],
        )
        ax_dof.plot(
            g["step"],
            g["active_after"],
            marker=markers[method],
            color=colors[method],
            linestyle=ls,
            linewidth=2.0,
            markersize=6.5,
            label=labels[method],
        )

    for ax in (ax_err, ax_dof):
        ax.set_xticks(range(1, 7))
        ax.set_xlabel("Adaptation decision")
        ax.grid(alpha=0.22)
    ax_err.set_ylabel(r"Mean $E_{\mathrm{ref}}$")
    ax_err.legend(frameon=False)
    ax_dof.set_ylabel("Mean active degrees of freedom")
    panel(ax_err, "(a)")
    panel(ax_dof, "(b)")

    learned = rows[rows["method"] == "Learned AMDI"].groupby("step", as_index=False).mean(numeric_only=True)
    energy_series = [
        ("energy_before_propagation", "Before propagation", "#7f7f7f", "o"),
        ("energy_after_propagation", "After propagation", "#2ca02c", "s"),
        ("energy_after_adaptation", "After adaptation", "#ff7f0e", "^"),
    ]
    for key, label, color, marker in energy_series:
        ax_energy.plot(
            learned["step"],
            learned[key],
            marker=marker,
            color=color,
            linewidth=2.0,
            markersize=6.5,
            label=label,
        )
    ax_energy.set_xticks(range(1, 7))
    ax_energy.set_xlabel("Adaptation decision")
    ax_energy.set_ylabel(r"Mean AMDI energy $\mathcal{E}$")
    ax_energy.grid(alpha=0.22)
    ax_energy.legend(frameon=False)
    panel(ax_energy, "(c)")

    step_values = np.arange(1, 7)
    width = 0.34
    delta_by_method = {}
    counts_by_method = {}
    for method in ("AMDI", "Learned AMDI"):
        grouped = rows[rows["method"] == method].groupby("step")
        delta_by_method[method] = grouped["adaptation_energy_change"].mean().reindex(step_values).to_numpy()
        counts_by_method[method] = grouped["adaptation_energy_change"].apply(
            lambda values: int((values > 1.0e-12).sum())
        ).reindex(step_values).to_numpy()
    scale = 1.0e5
    ax_delta.bar(
        step_values - width / 2,
        scale * delta_by_method["AMDI"],
        width,
        color=colors["AMDI"],
        label="Deterministic AMDI",
    )
    bars = ax_delta.bar(
        step_values + width / 2,
        scale * delta_by_method["Learned AMDI"],
        width,
        color=colors["Learned AMDI"],
        label="Learned AMDI",
    )
    for bar, count in zip(bars, counts_by_method["Learned AMDI"]):
        height = bar.get_height()
        offset = 0.08 if height >= 0 else -0.08
        va = "bottom" if height >= 0 else "top"
        ax_delta.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{count}/9",
            ha="center",
            va=va,
            fontsize=9.5,
        )
    ax_delta.axhline(0.0, color="black", linewidth=0.7)
    ax_delta.set_xticks(step_values)
    ax_delta.set_xlabel("Adaptation decision")
    ax_delta.set_ylabel(r"Mean $10^{5}\,\Delta\mathcal{E}_{\mathrm{adapt}}$")
    ax_delta.set_ylim(-0.30, 3.80)
    ax_delta.grid(axis="y", alpha=0.22)
    ax_delta.legend(frameon=False, loc="upper right")
    panel(ax_delta, "(d)")
    args.outdir.mkdir(parents=True, exist_ok=True)
    pdf = args.outdir / "Fig_section5_action_energy_audit.pdf"
    png = args.outdir / "Fig_section5_action_energy_audit.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
