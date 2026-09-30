#!/usr/bin/env python3
"""Generate all six Learned-AMDI manuscript figures in one house style.

The script is evaluation/plotting only: it performs no training and no new
parameter selection.  Five figures are reconstructed from archived result
files.  By default, the four-region figure is assembled from six archived
panels produced by the selected checkpoint; ``--recompute-four-region``
instead reproduces those panels from the checkpoint and publication
configuration using deterministic actions.

Run from the repository root, for example:

    python make_all_manuscript_figures.py --device cpu --outdir IMG

Outputs are written as vector PDF and 600-dpi PNG pairs with the exact file
names referenced by the manuscript.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
if not (ROOT / "learned_amdi").exists():
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


# Okabe--Ito-inspired, colorblind-safe palette.  Line style and marker shape
# also identify every method when printed in grayscale.
GRAY = "#4D4D4D"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"

METHOD_STYLE = {
    "AMDI": dict(color=GRAY, marker="s", linestyle="--"),
    "Learned AMDI": dict(color=BLUE, marker="o", linestyle="-"),
    "Learned: decision 1 only": dict(
        color=PURPLE, marker="^", linestyle=(0, (3, 1.5))
    ),
    "Top-K: matched validation occupancy": dict(
        color=ORANGE, marker="^", linestyle="-."
    ),
    "Threshold: matched validation occupancy": dict(
        color=GREEN, marker="D", linestyle=":"
    ),
}


def configure_style() -> None:
    """Apply the common SIAM-oriented typography and mark sizing."""
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 12.0,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 10.5,
            "lines.linewidth": 2.15,
            "lines.markersize": 7.5,
            "axes.linewidth": 1.05,
            "xtick.major.width": 1.05,
            "ytick.major.width": 1.05,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel(ax, label: str, fontsize: float = 14.0) -> None:
    """Place a bold panel identifier inside the top-left of an axis."""
    ax.text(
        0.025,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.8),
        zorder=30,
    )


def finish_axis(ax, grid_axis: str = "both") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="0.82", linewidth=0.7, alpha=0.65)
    ax.tick_params(direction="out")


def plot_method(
    ax,
    x,
    y,
    method: str,
    label: str,
    *,
    markersize: float = 7.5,
    linewidth: float = 2.15,
    zorder: int = 5,
):
    return ax.plot(
        x,
        y,
        label=label,
        markerfacecolor="white",
        markeredgewidth=1.35,
        markersize=markersize,
        linewidth=linewidth,
        zorder=zorder,
        **METHOD_STYLE[method],
    )


def save_figure(fig, outdir: Path, stem: str) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{stem}.pdf"
    png = outdir / f"{stem}.png"
    temporary_pdf = outdir / f".{stem}.tmp.pdf"
    temporary_png = outdir / f".{stem}.tmp.png"
    temporary_pdf.unlink(missing_ok=True)
    temporary_png.unlink(missing_ok=True)
    fig.savefig(temporary_png, dpi=600, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(temporary_pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    gc.collect()

    # Fail loudly rather than leaving a truncated publication asset.
    with Image.open(temporary_png) as image:
        image.verify()
    pdf_bytes = temporary_pdf.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-") or b"%%EOF" not in pdf_bytes[-1024:]:
        raise RuntimeError(f"Invalid PDF output: {temporary_pdf}")
    temporary_png.replace(png)
    temporary_pdf.replace(pdf)
    temporary_png.unlink(missing_ok=True)
    temporary_pdf.unlink(missing_ok=True)
    return pdf, png


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def level_map(backend) -> np.ndarray:
    n = backend.shape[0]
    result = np.zeros((n, n), dtype=float)
    for level, translation in backend.active_leaf_cells():
        width = n // (2**level)
        row = int(translation[0] * width)
        col = int(translation[1] * width)
        result[row : row + width, col : col + width] = level
    return result


def style_image_axis(ax, label: str, show_y_labels: bool) -> None:
    ax.set_xlabel(r"$x$ [-]")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    if show_y_labels:
        ax.set_ylabel(r"$y$ [-]")
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=13.0)
    panel(ax, label, fontsize=16.0)


def make_four_region_figure(args, outdir: Path) -> tuple[Path, Path]:
    asset_names = (
        "four_region_clean.png",
        "four_region_noisy.png",
        "four_region_amdi.png",
        "four_region_learned.png",
        "four_region_amdi_levels.png",
        "four_region_learned_levels.png",
    )
    asset_paths = [args.four_region_assets / name for name in asset_names]
    use_assets = not args.recompute_four_region and all(
        path.is_file() for path in asset_paths
    )

    if use_assets:
        # These six panels are archived outputs of the selected checkpoint.
        # Reading them changes only typography, axes, and layout.
        panel_images = [mpimg.imread(path) for path in asset_paths]
        maximum_level = 5
    else:
        try:
            import torch

            from learned_amdi.config import RewardConfig
            from learned_amdi.datasets import noisy_case
            from learned_amdi.evaluation import load_policy
            from learned_amdi.runner import (
                collect_trajectory,
                run_deterministic_trajectory,
            )
            from learned_amdi.training import build_backend, make_reference
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch and the Learned-AMDI package are required to recompute "
                "the four-region panels. Install requirements.txt or provide "
                f"the archived panels under {args.four_region_assets}."
            ) from exc

        device = torch.device(args.device)
        actor, critic, scales, cfg = load_policy(args.checkpoint, device)
        n_steps = int(cfg["training"]["n_steps"])
        sigma = float(cfg["data"]["noise_sigma"])
        reward_cfg = RewardConfig(**cfg["reward"])

        truth, noisy = noisy_case(
            "four_region", args.n, 0, sigma, args.noise_seed
        )
        reference = make_reference(noisy, build_backend(noisy, cfg), n_steps)

        deterministic_backend = build_backend(noisy, cfg)
        run_deterministic_trajectory(
            deterministic_backend, reference, n_steps, "baseline"
        )
        deterministic_field = np.clip(
            deterministic_backend.current_field(), 0, 1
        )
        deterministic_levels = level_map(deterministic_backend)

        learned_backend = build_backend(noisy, cfg)
        collect_trajectory(
            learned_backend,
            reference,
            actor,
            critic,
            scales,
            reward_cfg,
            n_steps,
            device,
            True,
        )
        learned_field = np.clip(learned_backend.current_field(), 0, 1)
        learned_levels = level_map(learned_backend)
        maximum_level = int(
            max(deterministic_backend.max_level, learned_backend.max_level)
        )
        panel_images = [
            truth,
            noisy,
            deterministic_field,
            learned_field,
            deterministic_levels,
            learned_levels,
        ]

    fig = plt.figure(figsize=(11.0, 6.35))
    grid = GridSpec(
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
    axes = [
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[0, 4]),
        fig.add_subplot(grid[1, 2]),
        fig.add_subplot(grid[1, 3]),
        fig.add_subplot(grid[1, 4]),
    ]
    cax_intensity = fig.add_subplot(grid[:, 0])
    cax_level = fig.add_subplot(grid[:, 6])

    if use_assets:
        for ax, image in zip(axes, panel_images):
            ax.imshow(
                image,
                extent=(0, 1, 1, 0),
                interpolation="nearest",
                aspect="equal",
            )
        image_handle = ScalarMappable(norm=Normalize(0, 1), cmap="gray")
        image_handle.set_array([])
        level_handle = ScalarMappable(
            norm=Normalize(0, maximum_level), cmap="viridis"
        )
        level_handle.set_array([])
    else:
        image_handle = None
        for ax, image in zip(axes[:4], panel_images[:4]):
            image_handle = ax.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=(0, 1, 1, 0),
                interpolation="nearest",
                aspect="equal",
            )
        level_handle = axes[4].imshow(
            panel_images[4],
            cmap="viridis",
            vmin=0,
            vmax=maximum_level,
            extent=(0, 1, 1, 0),
            interpolation="nearest",
            aspect="equal",
        )
        axes[5].imshow(
            panel_images[5],
            cmap="viridis",
            vmin=0,
            vmax=maximum_level,
            extent=(0, 1, 1, 0),
            interpolation="nearest",
            aspect="equal",
        )

    for index, ax in enumerate(axes):
        style_image_axis(
            ax,
            f"({chr(97 + index)})",
            show_y_labels=index in (0, 3),
        )

    intensity_bar = fig.colorbar(
        image_handle, cax=cax_intensity, orientation="vertical"
    )
    intensity_bar.set_ticks(np.linspace(0.0, 1.0, 6))
    intensity_bar.ax.yaxis.set_ticks_position("left")
    intensity_bar.ax.yaxis.set_label_position("left")
    intensity_bar.ax.tick_params(labelsize=13.0, pad=2)
    intensity_bar.set_label(
        "Normalized intensity [-]", rotation=90, labelpad=8, fontsize=15.5
    )

    level_bar = fig.colorbar(level_handle, cax=cax_level, orientation="vertical")
    level_bar.set_ticks(np.arange(0, maximum_level + 1))
    level_bar.ax.yaxis.set_ticks_position("right")
    level_bar.ax.yaxis.set_label_position("right")
    level_bar.ax.tick_params(labelsize=13.0, pad=2)
    level_bar.set_label(
        "Leaf refinement level [-]", rotation=90, labelpad=10, fontsize=15.5
    )

    return save_figure(
        fig, outdir, "Fig_four_region_reconstruction_and_refinement"
    )


def make_action_energy_figure(
    diagnostics_dir: Path, outdir: Path
) -> tuple[Path, Path]:
    source = diagnostics_dir / "step_diagnostics.csv"
    rows = pd.read_csv(source)
    require_columns(
        rows,
        {
            "dataset",
            "n",
            "method",
            "step",
            "reference_error",
            "active_after",
            "energy_before_propagation",
            "energy_after_propagation",
            "energy_after_adaptation",
            "adaptation_energy_change",
        },
        source,
    )
    rows = rows[(rows["dataset"] == "holdout") & (rows["n"] == 32)].copy()
    if rows.empty:
        raise RuntimeError(f"No 32x32 holdout rows in {source}")

    methods = ["AMDI", "Learned AMDI", "Learned: decision 1 only"]
    display = {
        "AMDI": "Deterministic AMDI",
        "Learned AMDI": "Learned AMDI",
        "Learned: decision 1 only": "Learned: decision 1 only",
    }

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.15))
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.10, top=0.855,
        wspace=0.32, hspace=0.42
    )
    ax_error, ax_dof, ax_energy, ax_delta = axes.ravel()

    for method in methods:
        group = (
            rows[rows["method"] == method]
            .groupby("step", as_index=False)
            .mean(numeric_only=True)
        )
        plot_method(
            ax_error,
            group["step"],
            group["reference_error"],
            method,
            display[method],
        )
        plot_method(
            ax_dof,
            group["step"],
            group["active_after"],
            method,
            display[method],
        )

    for ax in (ax_error, ax_dof):
        ax.set_xticks(range(1, 7))
        ax.set_xlabel("Adaptation decision")
        ax.margins(y=0.13)
        finish_axis(ax)
    ax_error.set_ylabel(r"Mean $E_{\mathrm{ref}}$ [-]")
    ax_dof.set_ylabel("Mean active degrees of freedom")
    panel(ax_error, "(a)")
    panel(ax_dof, "(b)")

    learned = (
        rows[rows["method"] == "Learned AMDI"]
        .groupby("step", as_index=False)
        .mean(numeric_only=True)
    )
    energy_series = (
        (
            "energy_before_propagation",
            "Before propagation",
            dict(color=GRAY, marker="s", linestyle="--"),
        ),
        (
            "energy_after_propagation",
            "After propagation",
            dict(color=GREEN, marker="D", linestyle=":"),
        ),
        (
            "energy_after_adaptation",
            "After adaptation",
            dict(color=ORANGE, marker="^", linestyle="-."),
        ),
    )
    for key, label, style in energy_series:
        ax_energy.plot(
            learned["step"],
            learned[key],
            label=label,
            markerfacecolor="white",
            markeredgewidth=1.35,
            markersize=7.2,
            linewidth=2.1,
            **style,
        )
    ax_energy.set_xticks(range(1, 7))
    ax_energy.set_xlabel("Adaptation decision")
    ax_energy.set_ylabel(r"Mean AMDI energy $\mathcal{E}$")
    ax_energy.margins(y=0.13)
    finish_axis(ax_energy)
    ax_energy.legend(frameon=False, loc="upper right")
    panel(ax_energy, "(c)")

    steps = np.arange(1, 7)
    width = 0.34
    means = {}
    counts = {}
    for method in ("AMDI", "Learned AMDI"):
        grouped = rows[rows["method"] == method].groupby("step")
        means[method] = (
            grouped["adaptation_energy_change"]
            .mean()
            .reindex(steps)
            .to_numpy()
        )
        counts[method] = (
            grouped["adaptation_energy_change"]
            .apply(lambda values: int((values > 1.0e-12).sum()))
            .reindex(steps)
            .to_numpy()
        )
    scale = 1.0e5
    ax_delta.bar(
        steps - width / 2,
        scale * means["AMDI"],
        width,
        color=GRAY,
        edgecolor="white",
        linewidth=0.8,
        label="Deterministic AMDI",
    )
    learned_bars = ax_delta.bar(
        steps + width / 2,
        scale * means["Learned AMDI"],
        width,
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        label="Learned AMDI",
    )
    for bar, count in zip(learned_bars, counts["Learned AMDI"]):
        height = bar.get_height()
        if height > 0.5:
            y = height - 0.12
            va = "top"
            color = "white"
            weight = "bold"
        else:
            y = height + 0.09 if height > 0.04 else 0.08
            va = "bottom"
            color = "black"
            weight = "normal"
        ax_delta.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{count}/9",
            ha="center",
            va=va,
            fontsize=10.0,
            color=color,
            fontweight=weight,
        )
    ax_delta.axhline(0.0, color="black", linewidth=0.8)
    ax_delta.set_xticks(steps)
    ax_delta.set_xlabel("Adaptation decision")
    ax_delta.set_ylabel(r"Mean $10^{5}\,\Delta\mathcal{E}_{\mathrm{adapt}}$")
    ax_delta.set_ylim(-0.30, 3.80)
    finish_axis(ax_delta, grid_axis="y")
    ax_delta.legend(frameon=False, loc="upper right")
    panel(ax_delta, "(d)")

    handles, labels = ax_error.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.545, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.6,
        handletextpad=0.6,
    )
    return save_figure(fig, outdir, "Fig_section5_action_energy_audit")


def make_matched_frontiers_figure(
    summary: dict,
    control_frontiers_file: Path,
    baselines_file: Path,
    outdir: Path,
) -> tuple[Path, Path]:
    controls = pd.read_csv(control_frontiers_file)
    baselines = pd.read_csv(baselines_file)
    require_columns(
        controls,
        {"mode", "C_rel", "reference_error", "RMSE", "SSIM"},
        control_frontiers_file,
    )
    require_columns(
        baselines,
        {"method", "C_rel", "reference_error", "RMSE", "SSIM"},
        baselines_file,
    )
    learned = pd.DataFrame(
        {
            "lambda_occ": [float(row["lambda_occ"]) for row in summary["pareto"]],
            "C_rel": [float(row["C_rel_mean"]) for row in summary["pareto"]],
            "reference_error": [
                float(row["reference_error_mean"]) for row in summary["pareto"]
            ],
            "RMSE": [float(row["RMSE_mean"]) for row in summary["pareto"]],
            "SSIM": [float(row["SSIM_mean"]) for row in summary["pareto"]],
        }
    )

    series = (
        (
            "Learned AMDI",
            learned,
            "Learned AMDI",
            7.5,
        ),
        (
            r"Top-$K$ observed detail",
            controls[controls["mode"] == "topk"],
            "Top-K: matched validation occupancy",
            5.8,
        ),
        (
            "Observed-detail threshold",
            controls[controls["mode"] == "threshold"],
            "Threshold: matched validation occupancy",
            5.8,
        ),
    )
    fields = (
        ("reference_error", r"$E_{\mathrm{ref}}$ [-]"),
        ("RMSE", "RMSE [-]"),
        ("SSIM", "SSIM [-]"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.55, 3.60))
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.215, top=0.75, wspace=0.42
    )
    for index, (ax, (field, ylabel)) in enumerate(zip(axes, fields)):
        for label, frame, style_name, marker_size in series:
            ordered = frame.sort_values("C_rel")
            plot_method(
                ax,
                ordered["C_rel"],
                ordered[field],
                style_name,
                label,
                markersize=marker_size,
                linewidth=2.15,
            )

        amdi = baselines[baselines["method"] == "AMDI"].iloc[0]
        retain = baselines[baselines["method"] == "Retain initial tree"].iloc[0]
        ax.plot(
            float(amdi["C_rel"]),
            float(amdi[field]),
            linestyle="none",
            marker="s",
            markersize=7.8,
            markerfacecolor="white",
            markeredgecolor=GRAY,
            markeredgewidth=1.5,
            label="Deterministic AMDI",
            zorder=12,
        )
        # The two baselines coincide on this subset.  A larger open triangle
        # surrounds the square without displacing either reported coordinate.
        ax.plot(
            float(retain["C_rel"]),
            float(retain[field]),
            linestyle="none",
            marker="^",
            markersize=12.0,
            markerfacecolor="none",
            markeredgecolor=PURPLE,
            markeredgewidth=1.5,
            label="Retain initial tree",
            zorder=13,
        )
        ax.set_xlabel(r"Terminal occupancy $C_{\mathrm{rel}}$ [-]")
        ax.set_ylabel(ylabel, labelpad=4)
        ax.margins(x=0.05, y=0.10)
        finish_axis(ax)
        panel(ax, f"({chr(97 + index)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.515, 0.99),
        ncol=3,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.6,
        handletextpad=0.55,
    )
    return save_figure(fig, outdir, "Fig_section5_matched_frontiers")


def make_resolution_transfer_figure(
    transfer_file: Path, outdir: Path
) -> tuple[Path, Path]:
    transfer = pd.read_csv(transfer_file)
    require_columns(
        transfer,
        {"n", "method", "RMSE", "SSIM", "reference_error", "C_rel"},
        transfer_file,
    )
    method_map = {
        "AMDI": "Deterministic AMDI",
        "Learned AMDI": "Learned AMDI",
        "Top-K: matched validation occupancy": r"Top-$K$ control",
        "Threshold: matched validation occupancy": "Threshold control",
    }
    selected = transfer[transfer["method"].isin(method_map)].copy()
    resolutions = sorted(int(value) for value in selected["n"].unique())
    expected = len(method_map) * len(resolutions)
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} transfer rows, found {len(selected)} in {transfer_file}"
        )

    fields = (
        ("RMSE", "RMSE [-]"),
        ("SSIM", "SSIM [-]"),
        ("reference_error", r"$E_{\mathrm{ref}}$ [-]"),
        ("C_rel", r"$C_{\mathrm{rel}}$ [-]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.65))
    fig.subplots_adjust(
        left=0.105, right=0.985, bottom=0.105, top=0.845,
        wspace=0.31, hspace=0.42
    )
    axes = axes.ravel()
    for index, (ax, (field, ylabel)) in enumerate(zip(axes, fields)):
        for method, label in method_map.items():
            group = selected[selected["method"] == method].sort_values("n")
            plot_method(
                ax,
                group["n"],
                group[field],
                method,
                label,
            )
        ax.set_xlabel(r"Image resolution $N$")
        ax.set_ylabel(ylabel, labelpad=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(resolutions)
        ax.set_xticklabels([str(value) for value in resolutions])
        ax.margins(x=0.08, y=0.13)
        finish_axis(ax)
        panel(ax, f"({chr(97 + index)})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.545, 0.975),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.7,
        handletextpad=0.65,
    )
    return save_figure(fig, outdir, "Fig_resolution_transfer")


def make_robustness_figure(summary: dict, outdir: Path) -> tuple[Path, Path]:
    seeds = sorted(
        summary["training_seed_summary"], key=lambda row: int(row["training_seed"])
    )
    ablations = summary["reward_ablation"]

    fig, axes = plt.subplots(1, 3, figsize=(9.55, 3.45))
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.22, top=0.92, wspace=0.48
    )

    ax = axes[0]
    seed_markers = ("o", "s", "^")
    for row, marker in zip(seeds, seed_markers):
        x = float(row["C_rel_mean"])
        y = float(row["reference_error_mean"])
        suffix = str(int(row["training_seed"]))[-2:]
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=marker,
            markersize=8.5,
            markerfacecolor="white",
            markeredgecolor=BLUE,
            markeredgewidth=1.6,
        )
        ax.annotate(
            suffix,
            (x, y),
            xytext=(7, 6 if suffix != "23" else -15),
            textcoords="offset points",
            fontsize=10.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.4),
        )
    ax.set_xlabel(r"$C_{\mathrm{rel}}$ [-]")
    ax.set_ylabel(r"$E_{\mathrm{ref}}$ [-]")
    ax.margins(x=0.18, y=0.20)
    finish_axis(ax)
    panel(ax, "(a)")

    ax = axes[1]
    positions = np.arange(len(seeds), dtype=float)
    suffixes = [str(int(row["training_seed"]))[-2:] for row in seeds]
    rmse = np.array([float(row["RMSE_mean"]) for row in seeds])
    ssim = np.array([float(row["SSIM_mean"]) for row in seeds])
    ax.plot(
        positions - 0.07,
        rmse / rmse.mean(),
        linestyle="none",
        marker="o",
        markersize=8.0,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        markeredgewidth=1.5,
        label="RMSE / mean",
    )
    ax.plot(
        positions + 0.07,
        ssim / ssim.mean(),
        linestyle="none",
        marker="s",
        markersize=7.8,
        markerfacecolor="white",
        markeredgecolor=ORANGE,
        markeredgewidth=1.5,
        label="SSIM / mean",
    )
    ax.axhline(1.0, color=GRAY, linewidth=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(suffixes)
    ax.set_xlabel("Training-seed suffix")
    ax.set_ylabel("Normalized metric [-]")
    ax.margins(x=0.20, y=0.18)
    finish_axis(ax)
    ax.legend(frameon=False, loc="upper center")
    panel(ax, "(b)")

    ax = axes[2]
    ablation_style = {
        "full": (BLUE, "o", "Full reward", (8, -16)),
        "no_switching": (GREEN, "D", "No switching", (8, 7)),
        "no_occupancy": (ORANGE, "^", "No occupancy", (-70, 8)),
    }
    for row in ablations:
        key = row["ablation"]
        color, marker, label, offset = ablation_style[key]
        x = float(row["C_rel_mean"])
        y = float(row["reference_error_mean"])
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=marker,
            markersize=8.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
        )
        ax.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=10.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.4),
        )
    ax.set_xlabel(r"$C_{\mathrm{rel}}$ [-]")
    ax.set_ylabel(r"$E_{\mathrm{ref}}$ [-]")
    ax.margins(x=0.16, y=0.18)
    finish_axis(ax)
    panel(ax, "(c)")

    return save_figure(fig, outdir, "Fig_robustness_ablation")


def make_vampyr_figure(summary: dict, outdir: Path) -> tuple[Path, Path]:
    rows = summary["vampyr_localization"]
    regions = [row["region"] for row in rows]
    positions = np.arange(len(regions), dtype=float)
    width = 0.24
    deterministic = np.array([float(row["amdi_mean_level"]) for row in rows])
    learned = np.array([float(row["learned_mean_level"]) for row in rows])
    vampyr = np.array(
        [float(row["vampyr_mean_effective_level"]) for row in rows]
    )

    fig, ax = plt.subplots(figsize=(7.35, 3.70))
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.20, top=0.78)
    bars_deterministic = ax.bar(
        positions - width,
        deterministic,
        width=width,
        color=GRAY,
        edgecolor="white",
        linewidth=0.8,
        label="Deterministic AMDI",
    )
    bars_learned = ax.bar(
        positions,
        learned,
        width=width,
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        label="Learned AMDI",
    )
    bars_vampyr = ax.bar(
        positions + width,
        vampyr,
        width=width,
        color=GREEN,
        edgecolor="white",
        linewidth=0.8,
        hatch="//",
        label="VAMPyR/MRCPP",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(regions)
    ax.set_xlabel("Region")
    ax.set_ylabel("Mean refinement/effective level [-]")
    ax.set_ylim(0, max(deterministic.max(), learned.max(), vampyr.max()) * 1.18)
    finish_axis(ax, grid_axis="y")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        columnspacing=1.6,
    )
    for bars in (bars_deterministic, bars_learned, bars_vampyr):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=10.5)

    return save_figure(fig, outdir, "Fig_vampyr_localization")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/publication_summary_FINAL/publication_summary_FINAL.json"),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("results/11_section5_diagnostics"),
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
    parser.add_argument(
        "--transfer-summary",
        type=Path,
        default=Path("results/12_section5_controls/transfer_summary.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/selected/best_policy.pt"),
    )
    parser.add_argument(
        "--four-region-assets",
        type=Path,
        default=Path("figure_assets"),
        help="Archived selected-checkpoint panels for style-only regeneration.",
    )
    parser.add_argument(
        "--recompute-four-region",
        action="store_true",
        help="Recompute the four-region panels from the checkpoint instead of using archived panels.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", type=Path, default=Path("IMG"))
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--noise-seed", type=int, default=137)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    expected_keys = {
        "pareto",
        "training_seed_summary",
        "reward_ablation",
        "vampyr_localization",
    }
    missing = expected_keys.difference(summary)
    if missing:
        raise ValueError(f"{args.summary} is missing keys: {sorted(missing)}")

    outputs = []
    outputs.extend(make_action_energy_figure(args.diagnostics_dir, args.outdir))
    outputs.extend(
        make_matched_frontiers_figure(
            summary, args.control_frontiers, args.baselines, args.outdir
        )
    )
    outputs.extend(
        make_resolution_transfer_figure(args.transfer_summary, args.outdir)
    )
    outputs.extend(make_robustness_figure(summary, args.outdir))
    outputs.extend(make_vampyr_figure(summary, args.outdir))
    # Render the large raster-panel figure last to keep peak image memory away
    # from the vector-first figures above.
    outputs.extend(make_four_region_figure(args, args.outdir))

    # Some network/overlay filesystems can retain a transient rename sidecar.
    # It is never a publication output and is removed after all writers close.
    for temporary in args.outdir.glob(".*.tmp.*"):
        temporary.unlink(missing_ok=True)

    if len(outputs) != 12 or not all(path.is_file() for path in outputs):
        raise RuntimeError("Figure generation did not produce all 12 expected files")
    print("Generated six manuscript figures as PDF/PNG pairs:")
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
