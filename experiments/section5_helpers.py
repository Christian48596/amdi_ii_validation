"""Shared, read-only evaluations for the Section 5 diagnostic scripts.

The existing AMDI backend, checkpoint loader, feature builder, and image metrics
are used without changing the production algorithms.  All new result files are
written by the command-line scripts in this directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from amdi.energy import total_energy
from amdi.haar import coefficients_vector, transfer_coefficients
from amdi.metrics import all_metrics
from amdi.solver import candidate_trees_from_data
from learned_amdi.backend import REFINE, RETAIN
from learned_amdi.datasets import noisy_case
from learned_amdi.evaluation import load_policy
from learned_amdi.features import build_feature_matrix, normalized_error, switching_measure
from learned_amdi.training import build_backend, make_reference


@dataclass(frozen=True)
class Case:
    dataset: str
    kind: str
    n: int
    image_seed: int
    noise_seed: int

    @property
    def case_id(self) -> str:
        return "four_region" if self.kind == "four_region" else f"random_{self.image_seed}"


@dataclass
class PreparedCase:
    spec: Case
    truth: np.ndarray
    noisy: np.ndarray
    reference: list[np.ndarray]


def publication_cases(cfg: dict, dataset: str) -> list[Case]:
    n = int(cfg["data"]["train_n"])
    if dataset == "holdout":
        cases = [
            Case(dataset, "random_multiscale", n, int(seed), 30000 + int(seed))
            for seed in cfg["data"]["test_image_seeds"]
        ]
        return cases + [Case(dataset, "four_region", n, 0, 137)]
    if dataset == "validation":
        return [
            Case(dataset, "random_multiscale", n, int(seed), 20000 + int(seed))
            for seed in cfg["data"]["validation_image_seeds"]
        ]
    if dataset == "pareto":
        return [
            Case(dataset, "random_multiscale", n, int(seed), 40000 + int(seed))
            for seed in cfg["data"]["test_image_seeds"][:4]
        ]
    if dataset == "transfer":
        return [
            Case(dataset, "random_multiscale", nn, int(seed), 50000 + int(seed) + nn)
            for nn in (32, 64, 128)
            for seed in cfg["data"]["test_image_seeds"][:4]
        ]
    raise ValueError(f"unknown dataset: {dataset}")


def prepare_case(spec: Case, cfg: dict) -> PreparedCase:
    sigma = float(cfg["data"]["noise_sigma"])
    truth, noisy = noisy_case(spec.kind, spec.n, spec.image_seed, sigma, spec.noise_seed)
    ref = make_reference(noisy, build_backend(noisy, cfg), int(cfg["training"]["n_steps"]))
    return PreparedCase(spec, truth, noisy, ref)


def selected_policy(checkpoint: Path, cfg: dict, device: torch.device):
    actor, _critic, scales, checkpoint_cfg = load_policy(checkpoint, device)
    for key in ("amdi", "data", "network", "training", "reward"):
        if checkpoint_cfg[key] != cfg[key]:
            raise ValueError(f"checkpoint and configs/publication.json differ in {key}")
    actor.eval()
    return actor, scales


def classify_tree_change(before, after) -> str:
    old, new = set(before.refined), set(after.refined)
    if old == new:
        return "retain"
    if old < new:
        return "refine"
    if new < old:
        return "coarsen"
    return "mixed"


def _candidate_scores(backend, old_tree) -> tuple[list[dict], int, set]:
    candidates = candidate_trees_from_data(
        backend.tree, backend.coeffs, backend.full_data,
        refine_fraction=backend.refine_fraction,
        coarsen_fraction=backend.coarsen_fraction,
    )
    rows = []
    for i, candidate in enumerate(candidates):
        coeffs = transfer_coefficients(backend.coeffs, old_tree, candidate)
        energy, _ = total_energy(
            coefficients_vector(coeffs, candidate), candidate,
            backend.full_data, backend.eparams, backend.gparams,
        )
        penalty = 0.5 * backend.zeta * float(old_tree.distance(candidate) ** 2)
        rows.append({
            "candidate_id": i,
            "candidate_type": classify_tree_change(old_tree, candidate),
            "candidate_active_dofs": candidate.basis_size(),
            "candidate_energy": float(energy),
            "tree_change_penalty": float(penalty),
            "candidate_score": float(energy + penalty),
        })
    winner = int(np.argmin([r["candidate_score"] for r in rows]))
    return rows, winner, set(candidates[winner].refined)


def _actor_actions(backend, actor, scales, step: int, n_steps: int, device):
    records = backend.candidate_records()
    if not records:
        raise RuntimeError("empty actor candidate set")
    x_np = build_feature_matrix(
        backend.current_field(), records, backend.active_leaf_cells(),
        backend.max_level, step, n_steps, scales,
    )
    masks_np = np.stack([rec.mask for rec in records])
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    masks = torch.tensor(masks_np, dtype=torch.bool, device=device)
    with torch.no_grad():
        logits = actor(x)
        actions = torch.argmax(
            logits.masked_fill(~masks, torch.finfo(logits.dtype).min), dim=-1
        ).cpu().tolist()
    return {rec.cell: int(action) for rec, action in zip(records, actions)}


def _control_actions(backend, mode: str, step: int, parameter: float | int | None):
    all_records = backend.candidate_records()
    actions = {rec.cell: RETAIN for rec in all_records}
    if step != 1 or mode == "retain":
        return actions
    records = [rec for rec in all_records if rec.mask[REFINE]]
    if mode == "topk":
        eligible = sorted(records, key=lambda r: (-float(np.linalg.norm(r.coeff_block)), r.cell))
        selected = eligible[: max(0, int(parameter))]
    elif mode == "threshold":
        # parameter is the raw observed-image Haar-detail norm, not q/s_q.
        selected = [rec for rec in records if float(np.linalg.norm(rec.coeff_block)) > float(parameter)]
    else:
        raise ValueError(mode)
    actions.update({rec.cell: REFINE for rec in selected})
    return actions


def run_case(
    prepared: PreparedCase,
    cfg: dict,
    mode: str,
    *,
    label: str | None = None,
    actor=None,
    scales=None,
    device: torch.device | None = None,
    parameter: float | int | None = None,
    save_candidates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Run exactly six propagate-then-adapt steps and return step/terminal rows."""
    if mode not in {"amdi", "learned", "learned_first_only", "retain", "topk", "threshold"}:
        raise ValueError(mode)
    if mode.startswith("learned") and (actor is None or scales is None or device is None):
        raise ValueError("a loaded actor, scales, and device are required")
    b = build_backend(prepared.noisy, cfg)
    spec = prepared.spec
    n_steps = int(cfg["training"]["n_steps"])
    eps = float(cfg["reward"]["epsilon_num"])
    gamma = float(cfg["ppo"]["gamma"])
    name = label or mode
    steps: list[dict[str, Any]] = []
    candidates_out: list[dict[str, Any]] = []
    discounted_return = 0.0

    for idx in range(n_steps):
        step = idx + 1
        before_indices = b.active_dof_indices()
        before_tree = b.tree.copy()
        energy_before = b.energy()
        prop_diag = b.propagate_for_adaptation()
        energy_after_prop = b.energy()
        choice = ""
        n_candidates = 0

        if mode == "amdi":
            candidate_rows, winner, winning_tree = _candidate_scores(b, before_tree)
            choice = candidate_rows[winner]["candidate_type"]
            n_candidates = len(candidate_rows)
            if save_candidates:
                for row in candidate_rows:
                    candidates_out.append({
                        "dataset": spec.dataset, "case_id": spec.case_id,
                        "n": spec.n, "image_seed": spec.image_seed,
                        "noise_seed": spec.noise_seed, "step": step,
                        **row, "selected": int(row["candidate_id"] == winner),
                    })
            b.deterministic_adapt(prop_diag)
            if b.tree.refined != winning_tree:
                raise AssertionError("candidate choice mismatch")
            proposed_refine = proposed_retain = proposed_coarsen = ""
            executed_refine = len(set(b.tree.refined) - set(before_tree.refined))
            executed_coarsen_groups = len(set(before_tree.refined) - set(b.tree.refined))
        else:
            if mode == "learned" or (mode == "learned_first_only" and step == 1):
                action_map = _actor_actions(b, actor, scales, step, n_steps, device)
            elif mode == "learned_first_only":
                action_map = _control_actions(b, "retain", step, None)
            else:
                action_map = _control_actions(b, mode, step, parameter)
            diag = b.learned_adapt(action_map, prop_diag)
            proposed_refine = diag.proposed_refine
            proposed_retain = diag.proposed_retain
            proposed_coarsen = diag.proposed_coarsen
            executed_refine = diag.executed_refine
            executed_coarsen_groups = diag.executed_coarsen_groups

        after_indices = b.active_dof_indices()
        switch = switching_measure(before_indices, after_indices, b.n_full)
        err = normalized_error(b.current_field(), prepared.reference[step], eps)
        energy_after_adapt = b.energy()
        reward = -(
            float(cfg["reward"]["lambda_err"]) * err
            + float(cfg["reward"]["lambda_occ"]) * b.occupancy
            + float(cfg["reward"]["lambda_sw"]) * switch
        )
        discounted_return += gamma**idx * reward
        steps.append({
            "dataset": spec.dataset, "case_id": spec.case_id,
            "kind": spec.kind, "n": spec.n, "image_seed": spec.image_seed,
            "noise_seed": spec.noise_seed, "method": name, "step": step,
            "active_before": len(before_indices), "active_after": b.n_active,
            "C_rel": b.occupancy, "reference_error": err,
            "switching": switch, "reward": reward,
            "energy_before_propagation": energy_before,
            "energy_after_propagation": energy_after_prop,
            "energy_after_adaptation": energy_after_adapt,
            "adaptation_energy_change": energy_after_adapt - energy_after_prop,
            "fixed_tree_safeguard_accepted": int(prop_diag.safeguard_accepted),
            "deterministic_candidate_count": n_candidates,
            "deterministic_choice": choice,
            "proposed_refine": proposed_refine,
            "proposed_retain": proposed_retain,
            "proposed_coarsen": proposed_coarsen,
            "executed_refine": executed_refine,
            "executed_coarsen_groups": executed_coarsen_groups,
            "changed_refined_cells": len(set(b.tree.refined) ^ set(before_tree.refined)),
        })

    quality = all_metrics(
        np.clip(b.current_field(), 0.0, 1.0), prepared.truth,
        active=b.n_active, full=b.n_full,
    )
    terminal: dict[str, Any] = {
        "dataset": spec.dataset, "case_id": spec.case_id,
        "kind": spec.kind, "n": spec.n,
        "image_seed": spec.image_seed, "noise_seed": spec.noise_seed,
        "method": name, "parameter": "" if parameter is None else parameter,
        **quality, "active_dofs": b.n_active,
        "reference_error": steps[-1]["reference_error"],
        "switching_mean": float(np.mean([r["switching"] for r in steps])),
        "return": float(discounted_return),
        "total_executed_refine": sum(int(r["executed_refine"]) for r in steps),
        "total_executed_coarsen_groups": sum(int(r["executed_coarsen_groups"]) for r in steps),
        "n_adaptation_energy_increases": sum(r["adaptation_energy_change"] > 1e-12 for r in steps),
    }
    return steps, terminal, candidates_out


def average(rows: list[dict], *, method: str | None = None) -> dict[str, float]:
    if not rows:
        raise ValueError("empty set of trajectories")
    fields = ("reference_error", "C_rel", "RMSE", "SSIM", "switching_mean", "return", "active_dofs", "total_executed_refine", "total_executed_coarsen_groups")
    result = {field: float(np.mean([float(r[field]) for r in rows])) for field in fields}
    result["n_cases"] = len(rows)
    if method is not None:
        result["method"] = method
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
