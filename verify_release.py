#!/usr/bin/env python3
"""Fast integrity and manuscript-number checks for the Learned AMDI release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent

EXPECTED_SHA256 = {
    "checkpoints/selected/best_policy.pt": "0ae7790bc63b4895ccf60b1abce11b06b836bdf9460f944c57da40e51127d980",
    "checkpoints/pareto/occ_0p03/best_policy.pt": "2a03896d3ce876cf9560935e9ab3691611a905b75b9e461a68517b6a99bb23d8",
    "checkpoints/pareto/occ_0p08/best_policy.pt": "703e158aa028ae778c986edd115e796b909e3572773e87160f8f28a136b860e5",
    "checkpoints/pareto/occ_0p15/best_policy.pt": "0ae7790bc63b4895ccf60b1abce11b06b836bdf9460f944c57da40e51127d980",
    "checkpoints/pareto/occ_0p25/best_policy.pt": "424be7b868456ed708a1a4d19e62abebf3f2fe0de73ebf94c2138a13eec81429",
    "checkpoints/pareto/occ_0p3/best_policy.pt": "30599f848d784847d1c1eb0fee7b9b9a68e0e304eaf639d9aff77b07a524a21e",
    "checkpoints/pareto/occ_0p6/best_policy.pt": "e095d6449b94e5bf993fb4f058bd3260aeb71cdd1c3c9a42d4a4d4cf4ed45eb4",
    "checkpoints/robustness/20260811/best_policy.pt": "0ae7790bc63b4895ccf60b1abce11b06b836bdf9460f944c57da40e51127d980",
    "checkpoints/robustness/20260817/best_policy.pt": "deee2c02a968c1eecbbe716afbec17a9405f9a51d11a48765541481cbf1882a5",
    "checkpoints/robustness/20260823/best_policy.pt": "60b04b090d52d916a760faf0b7c0875fe19fbcec473a8a1b537c7c48d6f1d30a",
    "checkpoints/ablation/full/best_policy.pt": "0ae7790bc63b4895ccf60b1abce11b06b836bdf9460f944c57da40e51127d980",
    "checkpoints/ablation/no_occupancy/best_policy.pt": "b60227ae46f3140a06708bc9f482410c692e476d323af40932aa75e2ff4fc862",
    "checkpoints/ablation/no_switching/best_policy.pt": "280e88378181a1253ee32bebc1bc2711c362b9323d9e127e03ba9f437f05e32b",
}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_close(actual, expected, name, atol=5e-13):
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=5e-12):
        raise AssertionError(f"{name}: {actual} != {expected}")


def main():
    # Checkpoint integrity.
    for rel, expected in EXPECTED_SHA256.items():
        p=ROOT/rel
        if not p.exists():
            raise FileNotFoundError(rel)
        got=sha256(p)
        if got != expected:
            raise AssertionError(f"SHA-256 mismatch for {rel}: {got}")

    cfg=json.loads((ROOT/"configs/publication.json").read_text())
    assert cfg["training"]["seed"] == 20260811
    assert cfg["training"]["n_steps"] == 6
    assert cfg["ppo"]["updates"] == 400
    assert cfg["ppo"]["trajectories_per_update"] == 16
    assert_close(cfg["ppo"]["epsilon_ppo"], 0.10, "epsilon_ppo")
    assert_close(cfg["ppo"]["learning_rate"], 5e-5, "learning_rate")
    assert_close(cfg["ppo"]["c_h"], 0.02, "entropy coefficient")
    assert_close(cfg["reward"]["lambda_occ"], 0.15, "selected lambda_occ")

    summary=json.loads((ROOT/"results/publication_summary_FINAL/publication_summary_FINAL.json").read_text())
    hold={r["method"]:r for r in summary["holdout"]}
    assert set(hold) == {"AMDI", "Learned AMDI", "Uniform AMDI reference"}
    assert_close(hold["AMDI"]["reference_error_mean"], 0.17496419044413442, "AMDI holdout E_ref")
    assert_close(hold["AMDI"]["C_rel_mean"], 0.13736979166666666, "AMDI holdout C_rel")
    assert_close(hold["Learned AMDI"]["reference_error_mean"], 0.13656908890512015, "Learned holdout E_ref")
    assert_close(hold["Learned AMDI"]["C_rel_mean"], 0.2666015625, "Learned holdout C_rel")
    assert_close(hold["Learned AMDI"]["RMSE_mean"], 0.05306480249629621, "Learned holdout RMSE")
    assert_close(hold["Learned AMDI"]["SSIM_mean"], 0.7989935641859282, "Learned holdout SSIM")

    weights=[round(float(r["lambda_occ"]),2) for r in summary["pareto"]]
    assert weights == [0.03,0.08,0.15,0.25,0.30,0.60]
    p25=next(r for r in summary["pareto"] if np.isclose(float(r["lambda_occ"]),0.25))
    assert_close(p25["RMSE_mean"],0.05140622133375324,"Pareto lambda=0.25 RMSE")
    assert_close(p25["SSIM_mean"],0.8201866554586911,"Pareto lambda=0.25 SSIM")

    pa=summary["protocol_alignment"]
    assert pa["pass"] is True
    assert_close(pa["final_relative_difference"],0.0,"protocol final difference")
    assert int(pa["tree_distance"]) == 0
    ref=summary["reference_check"]
    assert ref["pass"] is True and ref["basis_size"] == 1024 and ref["energy_monotone"] is True

    required_figs=[
        "Fig_four_region_reconstruction_and_refinement.pdf",
        "Fig_four_region_trajectory_diagnostics.pdf",
        "Fig_pareto_accuracy_complexity.pdf",
        "Fig_resolution_transfer.pdf",
        "Fig_robustness_ablation.pdf",
        "Fig_vampyr_localization.pdf",
    ]
    for f in required_figs:
        if not (ROOT/"IMG"/f).exists():
            raise FileNotFoundError(f"IMG/{f}")

    print("Learned AMDI release verification: PASS")
    print("- checkpoint hashes: PASS")
    print("- frozen publication configuration: PASS")
    print("- manuscript numerical values: PASS")
    print("- deterministic protocol/reference checks: PASS")
    print("- publication figures present: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Learned AMDI release verification: FAIL\n{exc}", file=sys.stderr)
        raise
