#!/usr/bin/env python3
"""Reproduce the numerical data used in the Learned AMDI manuscript.

No training is performed.  The script evaluates the archived checkpoints on
exactly the fixed image/noise seeds used for the publication results and
writes a fresh JSON/CSV bundle to ``reproduced_results/`` by default.

VAMPyR/MRCPP localization is intentionally kept separate because it requires
the optional ``vampyr`` dependency; see ``experiments/10_vampyr_localization_crosscheck.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from learned_amdi.config import load_json
from learned_amdi.evaluation import evaluate_case
from learned_amdi.io import ensure_dir, write_csv, write_json

PARETO_WEIGHTS = [0.03, 0.08, 0.15, 0.25, 0.30, 0.60]
ROBUSTNESS_SEEDS = [20260811, 20260817, 20260823]


def aggregate(rows: list[dict], method: str) -> dict:
    rr = [r for r in rows if r["method"] == method]
    return {
        "method": method,
        "n_cases": len(rr),
        "RMSE_mean": float(np.mean([r["RMSE"] for r in rr])),
        "RMSE_std": float(np.std([r["RMSE"] for r in rr], ddof=1)) if len(rr) > 1 else 0.0,
        "SSIM_mean": float(np.mean([r["SSIM"] for r in rr])),
        "C_rel_mean": float(np.mean([r["C_rel"] for r in rr])),
        "reference_error_mean": float(np.mean([r["reference_error"] for r in rr])),
        "switching_mean": float(np.mean([r["switching_mean"] for r in rr])),
    }


def find_learned(rows: list[dict]) -> dict:
    return next(r for r in rows if r["method"] == "Learned AMDI")


def close(a, b, atol=2e-12, rtol=2e-10):
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol))


def compare_rows(actual, archived, keys, id_keys=()):
    failures=[]
    for ar in actual:
        match=None
        for rr in archived:
            if all(str(ar[k]) == str(rr[k]) for k in id_keys):
                match=rr; break
        if match is None:
            failures.append(f"missing archived row for {[(k, ar[k]) for k in id_keys]}")
            continue
        for k in keys:
            if not close(ar[k], match[k]):
                failures.append(f"{id_keys}: {k}: reproduced={ar[k]} archived={match[k]}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    ap.add_argument("--outdir", default="reproduced_results")
    args = ap.parse_args()
    device = torch.device(args.device)
    out = ensure_dir(ROOT / args.outdir)

    cfg = load_json(ROOT / "configs" / "publication.json")
    n = int(cfg["data"]["train_n"])
    sigma = float(cfg["data"]["noise_sigma"])
    test_seeds = [int(x) for x in cfg["data"]["test_image_seeds"]]
    selected = ROOT / "checkpoints" / "selected" / "best_policy.pt"

    # 1. Nine-case holdout.
    print("[1/5] Reproducing nine-case holdout...", flush=True)
    holdout_runs=[]
    for seed in test_seeds:
        holdout_runs += evaluate_case(
            selected, kind="random_multiscale", n=n, image_seed=seed,
            noise_sigma=sigma, noise_seed=30000 + seed, device=device,
        )
    holdout_runs += evaluate_case(
        selected, kind="four_region", n=n, image_seed=0,
        noise_sigma=sigma, noise_seed=137, device=device,
    )
    holdout_summary = [aggregate(holdout_runs, m) for m in
                       ["AMDI", "Learned AMDI", "Uniform AMDI reference"]]

    # 2. Pareto family, fixed four-image subset.
    print("[2/5] Reproducing six-point Pareto family...", flush=True)
    pareto=[]
    for w in PARETO_WEIGHTS:
        tag=f"occ_{w:.4g}".replace(".", "p")
        ck=ROOT / "checkpoints" / "pareto" / tag / "best_policy.pt"
        rows=[]
        for seed in test_seeds[:4]:
            rows.append(find_learned(evaluate_case(
                ck, kind="random_multiscale", n=n, image_seed=seed,
                noise_sigma=sigma, noise_seed=40000 + seed, device=device,
            )))
        pareto.append({
            "lambda_occ": w,
            "reference_error_mean": float(np.mean([r["reference_error"] for r in rows])),
            "C_rel_mean": float(np.mean([r["C_rel"] for r in rows])),
            "RMSE_mean": float(np.mean([r["RMSE"] for r in rows])),
            "SSIM_mean": float(np.mean([r["SSIM"] for r in rows])),
        })

    # 3. Resolution transfer, fixed four-image subset, no retraining.
    print("[3/5] Reproducing 32/64/128 resolution transfer...", flush=True)
    transfer=[]
    for nn in [32,64,128]:
        rows=[]
        for seed in test_seeds[:4]:
            rows += evaluate_case(
                selected, kind="random_multiscale", n=nn, image_seed=seed,
                noise_sigma=sigma, noise_seed=50000 + seed + nn, device=device,
            )
        for method in ["AMDI", "Learned AMDI", "Uniform AMDI reference"]:
            rr=[r for r in rows if r["method"] == method]
            transfer.append({
                "n": nn,
                "method": method,
                "RMSE_mean": float(np.mean([r["RMSE"] for r in rr])),
                "SSIM_mean": float(np.mean([r["SSIM"] for r in rr])),
                "C_rel_mean": float(np.mean([r["C_rel"] for r in rr])),
                "reference_error_mean": float(np.mean([r["reference_error"] for r in rr])),
            })

    # 4. Reward ablation, fixed four-image subset.
    print("[4/5] Reproducing reward ablation...", flush=True)
    ablation=[]
    for name in ["full", "no_occupancy", "no_switching"]:
        ck=ROOT / "checkpoints" / "ablation" / name / "best_policy.pt"
        rows=[]
        for seed in test_seeds[:4]:
            rows.append(find_learned(evaluate_case(
                ck, kind="random_multiscale", n=n, image_seed=seed,
                noise_sigma=sigma, noise_seed=60000 + seed, device=device,
            )))
        ablation.append({
            "ablation": name,
            "reference_error_mean": float(np.mean([r["reference_error"] for r in rows])),
            "C_rel_mean": float(np.mean([r["C_rel"] for r in rows])),
            "switching_mean": float(np.mean([r["switching_mean"] for r in rows])),
            "RMSE_mean": float(np.mean([r["RMSE"] for r in rows])),
        })

    # 5. Three independent training seeds, fixed four-image subset.
    print("[5/5] Reproducing three-seed robustness...", flush=True)
    seed_summary=[]
    for tseed in ROBUSTNESS_SEEDS:
        ck=ROOT / "checkpoints" / "robustness" / str(tseed) / "best_policy.pt"
        rows=[]
        for seed in test_seeds[:4]:
            rows.append(find_learned(evaluate_case(
                ck, kind="random_multiscale", n=n, image_seed=seed,
                noise_sigma=sigma, noise_seed=70000 + seed, device=device,
            )))
        seed_summary.append({
            "training_seed": tseed,
            "RMSE_mean": float(np.mean([r["RMSE"] for r in rows])),
            "SSIM_mean": float(np.mean([r["SSIM"] for r in rows])),
            "C_rel_mean": float(np.mean([r["C_rel"] for r in rows])),
            "reference_error_mean": float(np.mean([r["reference_error"] for r in rows])),
            "switching_mean": float(np.mean([r["switching_mean"] for r in rows])),
        })

    reproduced={
        "holdout":holdout_summary,
        "pareto":pareto,
        "resolution_transfer":transfer,
        "reward_ablation":ablation,
        "training_seed_summary":seed_summary,
    }
    write_json(out / "reproduced_summary.json", reproduced)
    write_csv(out / "holdout_summary.csv", holdout_summary)
    write_csv(out / "pareto.csv", pareto)
    write_csv(out / "resolution_transfer.csv", transfer)
    write_csv(out / "reward_ablation.csv", ablation)
    write_csv(out / "training_seed_summary.csv", seed_summary)

    archived=json.loads((ROOT / "results" / "publication_summary_FINAL" /
                         "publication_summary_FINAL.json").read_text())
    failures=[]
    failures += compare_rows(holdout_summary, archived["holdout"],
        ["RMSE_mean","RMSE_std","SSIM_mean","C_rel_mean","reference_error_mean","switching_mean"],
        ["method"])
    failures += compare_rows(pareto, archived["pareto"],
        ["reference_error_mean","C_rel_mean","RMSE_mean","SSIM_mean"], ["lambda_occ"])
    # Compare only AMDI/Learned AMDI transfer rows; these are manuscript curves.
    transfer_main=[r for r in transfer if r["method"] in {"AMDI","Learned AMDI"}]
    archived_transfer=[r for r in archived["resolution_transfer"] if r["method"] in {"AMDI","Learned AMDI"}]
    failures += compare_rows(transfer_main, archived_transfer,
        ["RMSE_mean","SSIM_mean","C_rel_mean","reference_error_mean"], ["n","method"])
    failures += compare_rows(ablation, archived["reward_ablation"],
        ["reference_error_mean","C_rel_mean","switching_mean","RMSE_mean"], ["ablation"])
    failures += compare_rows(seed_summary, archived["training_seed_summary"],
        ["RMSE_mean","SSIM_mean","C_rel_mean","reference_error_mean","switching_mean"], ["training_seed"])

    print("\nReproduction output:", out.resolve())
    if failures:
        print("\nARCHIVED RESULT COMPARISON: FAILED")
        for f in failures: print(" -", f)
        raise SystemExit(2)
    print("ARCHIVED RESULT COMPARISON: PASS")
    print("All manuscript numerical summaries reproduced within floating-point tolerance.")


if __name__ == "__main__":
    main()
