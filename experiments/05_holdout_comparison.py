#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learned_amdi.evaluation import evaluate_case, load_policy
from learned_amdi.io import ensure_dir, write_csv, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "selected" / "best_policy.pt"),
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-cases", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    _, _, _, cfg = load_policy(args.checkpoint, device)
    seeds = list(cfg["data"]["test_image_seeds"])
    if args.max_cases is not None:
        seeds = seeds[: args.max_cases]

    n = int(cfg["data"]["train_n"])
    sigma = float(cfg["data"]["noise_sigma"])
    rows = []

    for seed in seeds:
        rows += evaluate_case(
            args.checkpoint,
            kind="random_multiscale",
            n=n,
            image_seed=int(seed),
            noise_sigma=sigma,
            noise_seed=30000 + int(seed),
            device=device,
        )

    # Mandatory deterministic AMDI four-region holdout.
    rows += evaluate_case(
        args.checkpoint,
        kind="four_region",
        n=n,
        image_seed=0,
        noise_sigma=sigma,
        noise_seed=137,
        device=device,
    )

    out = ensure_dir(ROOT / "results" / "05_holdout_comparison")
    write_csv(out / "holdout_runs.csv", rows)

    summary = []
    for method in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == method]
        summary.append(
            {
                "method": method,
                "n_cases": len(rr),
                "RMSE_mean": float(np.mean([r["RMSE"] for r in rr])),
                "RMSE_std": float(np.std([r["RMSE"] for r in rr], ddof=1)) if len(rr) > 1 else 0.0,
                "SSIM_mean": float(np.mean([r["SSIM"] for r in rr])),
                "C_rel_mean": float(np.mean([r["C_rel"] for r in rr])),
                "reference_error_mean": float(np.mean([r["reference_error"] for r in rr])),
                "switching_mean": float(np.mean([r["switching_mean"] for r in rr])),
            }
        )

    write_csv(out / "holdout_summary.csv", summary)
    assessment = {
        "protocol": "all policy/hyperparameters frozen before test",
        "methods": [r["method"] for r in summary],
    }
    write_json(out / "assessment.json", assessment)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
