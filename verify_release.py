#!/usr/bin/env python3
"""Fast integrity and manuscript-number checks for the Learned AMDI release."""
from __future__ import annotations

import csv
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

REQUIRED_RESULT_FILES = (
    "results/11_section5_diagnostics/action_energy_by_step.csv",
    "results/11_section5_diagnostics/deterministic_candidates.csv",
    "results/11_section5_diagnostics/status.json",
    "results/11_section5_diagnostics/step_diagnostics.csv",
    "results/11_section5_diagnostics/summary.csv",
    "results/11_section5_diagnostics/terminal_metrics.csv",
    "results/12_section5_controls/holdout_paired_differences.csv",
    "results/12_section5_controls/holdout_runs.csv",
    "results/12_section5_controls/holdout_summary.csv",
    "results/12_section5_controls/pareto_baselines.csv",
    "results/12_section5_controls/pareto_control_frontiers.csv",
    "results/12_section5_controls/pareto_control_runs.csv",
    "results/12_section5_controls/selected_on_validation.json",
    "results/12_section5_controls/transfer_paired_differences.csv",
    "results/12_section5_controls/transfer_runs.csv",
    "results/12_section5_controls/transfer_summary.csv",
    "results/12_section5_controls/validation_grid.csv",
)

FIGURE_STEMS = (
    "Fig_four_region_reconstruction_and_refinement",
    "Fig_section5_action_energy_audit",
    "Fig_section5_matched_frontiers",
    "Fig_resolution_transfer",
    "Fig_robustness_ablation",
    "Fig_vampyr_localization",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual, expected, name, atol=5e-13):
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=5e-12):
        raise AssertionError(f"{name}: {actual} != {expected}")


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_for(rows, **criteria):
    return next(
        row for row in rows
        if all(str(row[key]) == str(value) for key, value in criteria.items())
    )


def verify_figures() -> None:
    expected = {f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("pdf", "png")}
    actual = {path.name for path in (ROOT / "IMG").iterdir() if path.is_file()}
    if actual != expected:
        raise AssertionError(
            f"IMG contents differ from the six required figure pairs: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for name in sorted(expected):
        data = (ROOT / "IMG" / name).read_bytes()
        if name.endswith(".png"):
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise AssertionError(f"invalid PNG signature: IMG/{name}")
            if not data.endswith(b"IEND\xaeB\x60\x82"):
                raise AssertionError(f"truncated PNG: IMG/{name}")
        else:
            if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
                raise AssertionError(f"invalid PDF: IMG/{name}")


def verify_manifest() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        raise FileNotFoundError("MANIFEST.sha256")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        relative = relative.removeprefix("./")
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest entry missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"manifest mismatch: {relative}")
        listed.add(relative)
    for required in REQUIRED_RESULT_FILES:
        if required not in listed:
            raise AssertionError(f"new result absent from manifest: {required}")
    for stem in FIGURE_STEMS:
        for suffix in ("pdf", "png"):
            relative = f"IMG/{stem}.{suffix}"
            if relative not in listed:
                raise AssertionError(f"figure absent from manifest: {relative}")


def main() -> None:
    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(relative)
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"SHA-256 mismatch for {relative}: {actual}")

    config = json.loads((ROOT / "configs/publication.json").read_text(encoding="utf-8"))
    assert config["training"]["seed"] == 20260811
    assert config["training"]["n_steps"] == 6
    assert config["ppo"]["updates"] == 400
    assert config["ppo"]["trajectories_per_update"] == 16
    assert_close(config["ppo"]["epsilon_ppo"], 0.10, "epsilon_ppo")
    assert_close(config["ppo"]["learning_rate"], 5e-5, "learning_rate")
    assert_close(config["ppo"]["c_h"], 0.02, "entropy coefficient")
    assert_close(config["reward"]["lambda_occ"], 0.15, "selected lambda_occ")

    summary = json.loads(
        (ROOT / "results/publication_summary_FINAL/publication_summary_FINAL.json")
        .read_text(encoding="utf-8")
    )
    holdout = {row["method"]: row for row in summary["holdout"]}
    assert set(holdout) == {"AMDI", "Learned AMDI", "Uniform AMDI reference"}
    assert_close(holdout["AMDI"]["reference_error_mean"], 0.17496419044413442, "AMDI E_ref")
    assert_close(holdout["AMDI"]["C_rel_mean"], 0.13736979166666666, "AMDI C_rel")
    assert_close(holdout["Learned AMDI"]["reference_error_mean"], 0.13656908890512015, "Learned E_ref")
    assert_close(holdout["Learned AMDI"]["C_rel_mean"], 0.2666015625, "Learned C_rel")
    assert_close(holdout["Learned AMDI"]["RMSE_mean"], 0.05306480249629621, "Learned RMSE")
    assert_close(holdout["Learned AMDI"]["SSIM_mean"], 0.7989935641859282, "Learned SSIM")

    weights = [round(float(row["lambda_occ"]), 2) for row in summary["pareto"]]
    assert weights == [0.03, 0.08, 0.15, 0.25, 0.30, 0.60]
    selected_pareto = next(
        row for row in summary["pareto"]
        if np.isclose(float(row["lambda_occ"]), 0.25)
    )
    assert_close(selected_pareto["RMSE_mean"], 0.05140622133375324, "Pareto 0.25 RMSE")
    assert_close(selected_pareto["SSIM_mean"], 0.8201866554586911, "Pareto 0.25 SSIM")

    for relative in REQUIRED_RESULT_FILES:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    status = json.loads(
        (ROOT / "results/11_section5_diagnostics/status.json").read_text(encoding="utf-8")
    )
    assert status["archived_table_1_check"] == "PASS"
    assert status["mismatches"] == []

    diagnostic = read_csv("results/11_section5_diagnostics/summary.csv")
    deterministic = row_for(diagnostic, dataset="holdout", n="32", method="AMDI")
    learned = row_for(diagnostic, dataset="holdout", n="32", method="Learned AMDI")
    decision_one = row_for(
        diagnostic, dataset="holdout", n="32", method="Learned: decision 1 only"
    )
    assert int(float(deterministic["total_executed_refine"])) == 0
    assert int(float(deterministic["total_executed_coarsen_groups"])) == 4
    assert int(float(learned["total_executed_refine"])) == 393
    assert int(float(learned["total_executed_coarsen_groups"])) == 0
    assert int(float(learned["adaptation_energy_increases"])) == 15
    assert_close(decision_one["reference_error"], 0.13741794847368277, "decision-1 E_ref")

    action_rows = read_csv("results/11_section5_diagnostics/action_energy_by_step.csv")
    learned_steps = {
        int(row["step"]): int(float(row["executed_refine_total"]))
        for row in action_rows
        if row["dataset"] == "holdout" and row["n"] == "32"
        and row["method"] == "Learned AMDI"
    }
    assert learned_steps == {1: 383, 2: 10, 3: 0, 4: 0, 5: 0, 6: 0}

    candidates = read_csv("results/11_section5_diagnostics/deterministic_candidates.csv")
    selected_candidates = [
        row for row in candidates
        if row["dataset"] == "holdout" and row["n"] == "32" and row["selected"] == "1"
    ]
    assert len(selected_candidates) == 54
    assert not any(row["candidate_type"] == "refine" for row in selected_candidates)

    controls = read_csv("results/12_section5_controls/holdout_summary.csv")
    threshold = row_for(
        controls, dataset="holdout", n="32",
        method="Threshold: matched validation occupancy",
    )
    topk = row_for(
        controls, dataset="holdout", n="32",
        method="Top-K: matched validation occupancy",
    )
    retain = row_for(controls, dataset="holdout", n="32", method="Retain initial tree")
    assert_close(threshold["reference_error"], 0.13792153610325308, "threshold E_ref")
    assert_close(threshold["C_rel"], 0.2604166666666667, "threshold C_rel")
    assert_close(topk["reference_error"], 0.14174842461449016, "top-K E_ref")
    assert_close(topk["C_rel"], 0.244140625, "top-K C_rel")
    assert_close(retain["reference_error"], 0.17493927881362292, "retain E_ref")

    selection = json.loads(
        (ROOT / "results/12_section5_controls/selected_on_validation.json")
        .read_text(encoding="utf-8")
    )
    assert selection["validation_image_seeds"] == [100, 101, 102, 103]
    assert selection["topk_matched_validation_occupancy"]["parameter"] == 36
    assert_close(
        selection["threshold_matched_validation_occupancy"]["parameter"],
        0.006125713758860572,
        "selected threshold",
    )

    protocol = summary["protocol_alignment"]
    assert protocol["pass"] is True
    assert_close(protocol["final_relative_difference"], 0.0, "protocol final difference")
    assert int(protocol["tree_distance"]) == 0
    reference = summary["reference_check"]
    assert (
        reference["pass"] is True
        and reference["basis_size"] == 1024
        and reference["energy_monotone"] is True
    )

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "family-names: Beck" in citation
    assert "given-names: Joakim Henrik" in citation
    assert "family-names: Frediani" not in citation
    assert (ROOT / "LICENSE").is_file()

    verify_figures()
    verify_manifest()

    print("Learned AMDI release verification: PASS")
    print("- checkpoint hashes and frozen configuration: PASS")
    print("- manuscript and Section 5 numerical values: PASS")
    print("- deterministic, control, and reference records: PASS")
    print("- six publication figure pairs: PASS")
    print("- citation, license, and SHA-256 manifest: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Learned AMDI release verification: FAIL\n{exc}", file=sys.stderr)
        raise
