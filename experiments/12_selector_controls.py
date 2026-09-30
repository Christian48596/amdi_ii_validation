#!/usr/bin/env python3
"""Validation-tuned retain/top-K/threshold controls for Learned AMDI.

Tune on image seeds 100--103 with noise seeds 20000+seed.  Freeze choices
before evaluating nine holdout cases.  The prespecified budget/threshold grid
is also evaluated on the same four-case noise convention as the PPO Pareto plot.
No policy training or modification of the AMDI numerical core is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learned_amdi.backend import REFINE
from learned_amdi.config import load_json
from learned_amdi.training import build_backend
from amdi.plotting import save_publication_figure
from section5_helpers import (
    average, prepare_case, publication_cases, run_case, selected_policy, write_csv,
)


# Prespecified integers. All choices between these candidates use validation
# data only; no held-out test value is used to select K.
K_GRID = [0, 4, 8, 12, 16, 20, 28, 32, 36, 40, 44, 48, 56, 72, 96, 128, 160, 200]
THRESHOLD_QUANTILES = [0, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .975, .99, 1.0]


def threshold_grid(validation_cases, cfg):
    details = []
    for prepared in validation_cases:
        b = build_backend(prepared.noisy, cfg)
        # Masks and observed-image details are evaluated at the first actual
        # post-propagation decision, like the learned actor's q feature.
        b.propagate_for_adaptation()
        details.extend(
            float(np.linalg.norm(rec.coeff_block))
            for rec in b.candidate_records() if rec.mask[REFINE]
        )
    if not details:
        raise RuntimeError("no eligible validation refinements")
    values = [float(np.nextafter(0.0, -np.inf))]
    values += [float(np.quantile(details, q)) for q in THRESHOLD_QUANTILES]
    values += [float(np.nextafter(max(details), np.inf))]
    return sorted(set(values))


def evaluate_grid(cases, cfg, mode, parameters, label_prefix, *, collect_runs=False):
    aggregate_rows, all_runs = [], []
    for param in parameters:
        print(f"  {label_prefix} parameter={param:.8g}", flush=True)
        run_rows = [
            run_case(p, cfg, mode, label=label_prefix, parameter=param)[1]
            for p in cases
        ]
        aggregate_rows.append({"control": label_prefix, "mode": mode,
                               "parameter": param, **average(run_rows)})
        if collect_runs:
            all_runs.extend(run_rows)
    return aggregate_rows, all_runs


def choose(grid, target_occupancy):
    # These are *two separate, validation-only* rules.  The first optimizes
    # the same complete trajectory reward as PPO. The second matches the
    # learned policy's mean validation terminal occupancy before test.
    by_return = max(grid, key=lambda r: (r["return"], -r["C_rel"]))
    by_occupancy = min(
        grid,
        key=lambda r: (abs(r["C_rel"] - target_occupancy), r["reference_error"], -r["return"]),
    )
    return by_return, by_occupancy


def evaluate_named(cases, cfg, variants, actor, scales, device):
    rows = []
    for prepared in cases:
        spec = prepared.spec
        print(f"[{spec.dataset}] {spec.case_id}, {spec.n}x{spec.n}", flush=True)
        for name, mode, param in variants:
            rows.append(run_case(
                prepared, cfg, mode, label=name, parameter=param,
                actor=actor, scales=scales, device=device,
            )[1])
    return rows


def summarize(rows):
    result = []
    for dataset, n, method in sorted({(r["dataset"], r["n"], r["method"]) for r in rows}):
        rr = [r for r in rows if (r["dataset"], r["n"], r["method"]) == (dataset, n, method)]
        result.append({"dataset": dataset, "n": n, **average(rr, method=method)})
    return result


def paired_differences(rows):
    learned = {(r["dataset"], r["n"], r["case_id"]): r for r in rows if r["method"] == "Learned AMDI"}
    out = []
    for row in rows:
        if row["method"] == "Learned AMDI":
            continue
        key = (row["dataset"], row["n"], row["case_id"])
        policy = learned[key]
        out.append({
            "dataset": row["dataset"], "n": row["n"],
            "case_id": row["case_id"], "method": row["method"],
            "delta_E_ref_control_minus_learned": row["reference_error"] - policy["reference_error"],
            "delta_C_rel_control_minus_learned": row["C_rel"] - policy["C_rel"],
            "delta_RMSE_control_minus_learned": row["RMSE"] - policy["RMSE"],
            "delta_SSIM_control_minus_learned": row["SSIM"] - policy["SSIM"],
        })
    return out


def plot_frontier(frontier, baseline, out):
    archived = json.loads((ROOT / "results/publication_summary_FINAL/publication_summary_FINAL.json").read_text())
    learned = archived["pareto"]
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 12.0,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.1,
        "lines.markersize": 6.5,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axs = plt.subplots(1, 3, figsize=(9.4, 3.45))
    series = {
        "Learned AMDI": [
            {"C_rel": r["C_rel_mean"], "reference_error": r["reference_error_mean"],
             "RMSE": r["RMSE_mean"], "SSIM": r["SSIM_mean"]} for r in learned
        ],
        "Top-K observed detail": [r for r in frontier if r["mode"] == "topk"],
        "Observed-detail threshold": [r for r in frontier if r["mode"] == "threshold"],
    }
    ykeys = ("reference_error", "RMSE", "SSIM")
    ylabels = (r"$E_{\mathrm{ref}}$ [-]", "RMSE [-]", "SSIM [-]")
    line_styles = {
        "Learned AMDI": ("#1f77b4", "o", 7.0),
        "Top-K observed detail": ("#ff7f0e", "s", 5.8),
        "Observed-detail threshold": ("#2ca02c", "D", 5.8),
    }
    for index, (ax, ykey, ylabel) in enumerate(zip(axs, ykeys, ylabels)):
        for label, rr in series.items():
            ordered = sorted(rr, key=lambda r: r["C_rel"])
            color, marker, marker_size = line_styles[label]
            ax.plot(
                [r["C_rel"] for r in ordered], [r[ykey] for r in ordered],
                color=color, marker=marker, markersize=marker_size,
                markeredgewidth=0.6, label=label,
            )
        for marker, label, color, facecolor in (
            ("s", "AMDI", "#d62728", "#d62728"),
            ("^", "Retain initial tree", "#9467bd", "none"),
        ):
            r = next(x for x in baseline if x["method"] == label)
            ax.plot(
                r["C_rel"], r[ykey], marker=marker, markersize=9,
                markerfacecolor=facecolor, markeredgecolor=color,
                markeredgewidth=1.5, linestyle="none", label=label, zorder=10,
            )
        ax.set(xlabel=r"Terminal occupancy $C_{\mathrm{rel}}$ [-]", ylabel=ylabel)
        ax.grid(alpha=.22)
        ax.margins(x=.04, y=.08)
        ax.text(
            .025, .975, f"({chr(97 + index)})", transform=ax.transAxes,
            ha="left", va="top", fontsize=13, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.86, pad=.8),
            zorder=20,
        )
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=3, frameon=False, columnspacing=1.35, handletextpad=.55)
    fig.subplots_adjust(top=.73, bottom=.22, left=.075, right=.985, wspace=.39)
    save_publication_figure(fig, out, "Fig_section5_matched_frontiers")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/publication.json"))
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/selected/best_policy.pt"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    ap.add_argument("--outdir", default="results/12_section5_controls")
    ap.add_argument("--skip-frontier", action="store_true", help="omit full grid evaluation on the four-image Pareto subset")
    ap.add_argument("--include-transfer", action="store_true", help="run frozen controls on the 32/64/128 transfer set")
    args = ap.parse_args()
    cfg = load_json(args.config)
    device = torch.device(args.device)
    actor, scales = selected_policy(Path(args.checkpoint), cfg, device)
    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    validation = [prepare_case(s, cfg) for s in publication_cases(cfg, "validation")]
    learned_validation = [
        run_case(p, cfg, "learned", label="Learned AMDI", actor=actor, scales=scales, device=device)[1]
        for p in validation
    ]
    target_occ = average(learned_validation)["C_rel"]
    thresholds = threshold_grid(validation, cfg)
    k_grid = K_GRID
    print(f"Validation occupancy target: {target_occ:.8g}; {len(k_grid)} K values and {len(thresholds)} thresholds")
    grid_k, _ = evaluate_grid(validation, cfg, "topk", k_grid, "Top-K observed detail")
    grid_t, _ = evaluate_grid(validation, cfg, "threshold", thresholds, "Observed-detail threshold")
    val_grid = grid_k + grid_t
    write_csv(out / "validation_grid.csv", val_grid)
    k_reward, k_match = choose(grid_k, target_occ)
    t_reward, t_match = choose(grid_t, target_occ)
    selected = {
        "validation_image_seeds": cfg["data"]["validation_image_seeds"],
        "validation_noise_seed_rule": "20000 + image_seed",
        "target_learned_validation_C_rel": target_occ,
        "topk_best_validation_return": k_reward,
        "topk_matched_validation_occupancy": k_match,
        "threshold_best_validation_return": t_reward,
        "threshold_matched_validation_occupancy": t_match,
        "note": "All four parameter choices were frozen before the holdout and Pareto evaluations.",
    }
    (out / "selected_on_validation.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    # Chosen controls and full nine-case Table 1 dataset share image/noise draws.
    variants = [
        ("AMDI", "amdi", None),
        ("Learned AMDI", "learned", None),
        ("Retain initial tree", "retain", None),
        ("Top-K: validation return", "topk", int(k_reward["parameter"])),
        ("Top-K: matched validation occupancy", "topk", int(k_match["parameter"])),
        ("Threshold: validation return", "threshold", float(t_reward["parameter"])),
        ("Threshold: matched validation occupancy", "threshold", float(t_match["parameter"])),
    ]
    holdout = [prepare_case(s, cfg) for s in publication_cases(cfg, "holdout")]
    holdout_runs = evaluate_named(holdout, cfg, variants, actor, scales, device)
    write_csv(out / "holdout_runs.csv", holdout_runs)
    write_csv(out / "holdout_summary.csv", summarize(holdout_runs))
    write_csv(out / "holdout_paired_differences.csv", paired_differences(holdout_runs))

    if not args.skip_frontier:
        pareto_cases = [prepare_case(s, cfg) for s in publication_cases(cfg, "pareto")]
        frontier, frontier_runs = [], []
        for mode, parameters, label in (
            ("topk", k_grid, "Top-K observed detail"),
            ("threshold", thresholds, "Observed-detail threshold"),
        ):
            aggregates, runs = evaluate_grid(pareto_cases, cfg, mode, parameters, label, collect_runs=True)
            frontier += aggregates
            frontier_runs += runs
        write_csv(out / "pareto_control_runs.csv", frontier_runs)
        write_csv(out / "pareto_control_frontiers.csv", frontier)
        base_runs = evaluate_named(pareto_cases, cfg, [
            ("AMDI", "amdi", None), ("Retain initial tree", "retain", None),
        ], actor, scales, device)
        base = summarize(base_runs)
        write_csv(out / "pareto_baselines.csv", base)
        plot_frontier(frontier, base, out)

    if args.include_transfer:
        transfer_cases = [prepare_case(s, cfg) for s in publication_cases(cfg, "transfer")]
        transfer_variants = [v for v in variants if v[0] in {
            "AMDI", "Learned AMDI", "Retain initial tree",
            "Top-K: matched validation occupancy", "Threshold: matched validation occupancy",
        }]
        transfer = evaluate_named(transfer_cases, cfg, transfer_variants, actor, scales, device)
        write_csv(out / "transfer_runs.csv", transfer)
        write_csv(out / "transfer_summary.csv", summarize(transfer))
        write_csv(out / "transfer_paired_differences.csv", paired_differences(transfer))

    print(f"Wrote Section 5 controls to: {out}")
    print("Selected controls (validation only):")
    for key in ("topk_best_validation_return", "topk_matched_validation_occupancy",
                "threshold_best_validation_return", "threshold_matched_validation_occupancy"):
        row = selected[key]
        print(f"  {key}: parameter={row['parameter']:.8g}, "
              f"C_rel={row['C_rel']:.6f}, E_ref={row['reference_error']:.6f}")


if __name__ == "__main__":
    main()
