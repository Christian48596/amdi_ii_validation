#!/usr/bin/env python3
"""Export accepted actions, deterministic candidate scores, and energy traces.

Uses the publication checkpoint and the exact manuscript image/noise seeds.
It does not retrain a policy or change the AMDI implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learned_amdi.config import load_json
from section5_helpers import average, prepare_case, publication_cases, run_case, selected_policy, write_csv


def compare_archive(summary: list[dict]) -> list[str]:
    archived = json.loads((ROOT / "results/publication_summary_FINAL/publication_summary_FINAL.json").read_text())
    expected = {row["method"]: row for row in archived["holdout"]}
    errors = []
    for method in ("AMDI", "Learned AMDI"):
        found = next(row for row in summary if row["method"] == method)
        original = expected[method]
        for current_name, original_name in (
            ("reference_error", "reference_error_mean"),
            ("C_rel", "C_rel_mean"),
            ("RMSE", "RMSE_mean"),
            ("SSIM", "SSIM_mean"),
            ("switching_mean", "switching_mean"),
        ):
            if not np.isclose(found[current_name], original[original_name], atol=2e-5, rtol=1e-5):
                errors.append(
                    f"{method} {current_name}: new={found[current_name]:.8g}, "
                    f"archived={original[original_name]:.8g}"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/publication.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/selected/best_policy.pt"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--outdir", default="results/11_section5_diagnostics")
    parser.add_argument("--include-transfer", action="store_true", help="also evaluate 32/64/128 transfer cases")
    parser.add_argument("--max-cases", type=int, default=None, help="smoke check only; final run must use all nine cases")
    args = parser.parse_args()

    cfg = load_json(args.config)
    device = torch.device(args.device)
    actor, scales = selected_policy(Path(args.checkpoint), cfg, device)
    out = Path(args.outdir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    steps, terminals, candidates = [], [], []

    datasets = ["holdout"] + (["transfer"] if args.include_transfer else [])
    for dataset in datasets:
        cases = publication_cases(cfg, dataset)
        if dataset == "holdout" and args.max_cases is not None:
            cases = cases[: args.max_cases]
        for spec in cases:
            print(f"[{dataset}] {spec.case_id}, {spec.n}x{spec.n}, noise={spec.noise_seed}", flush=True)
            prepared = prepare_case(spec, cfg)
            variants = [("amdi", "AMDI"), ("learned", "Learned AMDI")]
            if dataset == "holdout":
                variants.append(("learned_first_only", "Learned: decision 1 only"))
            for mode, label in variants:
                s, t, c = run_case(
                    prepared, cfg, mode, label=label, actor=actor, scales=scales,
                    device=device, save_candidates=(mode == "amdi"),
                )
                steps.extend(s)
                terminals.append(t)
                candidates.extend(c)

    write_csv(out / "step_diagnostics.csv", steps)
    write_csv(out / "deterministic_candidates.csv", candidates)
    write_csv(out / "terminal_metrics.csv", terminals)
    summary = []
    for dataset, n, method in sorted({(r["dataset"], r["n"], r["method"]) for r in terminals}):
        group = [r for r in terminals if (r["dataset"], r["n"], r["method"]) == (dataset, n, method)]
        row = {"dataset": dataset, "n": n, **average(group, method=method)}
        group_steps = [r for r in steps if r["dataset"] == dataset and r["n"] == n and r["method"] == method]
        row.update({
            "total_executed_refine": sum(int(r["executed_refine"]) for r in group_steps),
            "total_executed_coarsen_groups": sum(int(r["executed_coarsen_groups"]) for r in group_steps),
            "adaptation_energy_increases": sum(r["adaptation_energy_change"] > 1e-12 for r in group_steps),
        })
        summary.append(row)
    write_csv(out / "summary.csv", summary)

    by_step = []
    for dataset, n, method, step in sorted({
        (r["dataset"], r["n"], r["method"], r["step"]) for r in steps
    }):
        group = [r for r in steps if (r["dataset"], r["n"], r["method"], r["step"])
                 == (dataset, n, method, step)]
        by_step.append({
            "dataset": dataset, "n": n, "method": method, "step": step,
            "n_cases": len(group),
            "mean_active_dofs": float(np.mean([r["active_after"] for r in group])),
            "mean_C_rel": float(np.mean([r["C_rel"] for r in group])),
            "mean_reference_error": float(np.mean([r["reference_error"] for r in group])),
            "proposed_refine_total": "" if method == "AMDI" else sum(int(r["proposed_refine"]) for r in group),
            "proposed_coarsen_total": "" if method == "AMDI" else sum(int(r["proposed_coarsen"]) for r in group),
            "executed_refine_total": sum(int(r["executed_refine"]) for r in group),
            "executed_coarsen_groups_total": sum(int(r["executed_coarsen_groups"]) for r in group),
            "deterministic_retained_cases": sum(r["deterministic_choice"] == "retain" for r in group),
            "deterministic_refined_cases": sum(r["deterministic_choice"] == "refine" for r in group),
            "deterministic_coarsened_cases": sum(r["deterministic_choice"] == "coarsen" for r in group),
            "adaptation_energy_increases": sum(r["adaptation_energy_change"] > 1e-12 for r in group),
        })
    write_csv(out / "action_energy_by_step.csv", by_step)

    archive_errors = []
    if args.max_cases is None:
        archive_errors = compare_archive([row for row in summary if row["dataset"] == "holdout"])
    status = {
        "archived_table_1_check": (
            "SKIPPED" if args.max_cases is not None else
            ("PASS" if not archive_errors else "FAIL")
        ),
        "mismatches": archive_errors,
    }
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps(status, indent=2))
    if archive_errors:
        raise SystemExit("New evaluation differs from the archived Table 1; inspect status.json")


if __name__ == "__main__":
    main()
