from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from amdi.metrics import all_metrics
from .backend import HaarAMDIBackend
from .config import FeatureScales, RewardConfig
from .features import build_feature_matrix, normalized_error, switching_measure
from .networks import SharedLocalActor, VariableSetCritic, masked_categorical
from .rollout import RolloutStep, Trajectory


def collect_trajectory(
    backend: HaarAMDIBackend,
    reference: Sequence[np.ndarray],
    actor: SharedLocalActor,
    critic: VariableSetCritic,
    scales: FeatureScales,
    reward_cfg: RewardConfig,
    n_steps: int,
    device: torch.device,
    deterministic: bool = False,
) -> tuple[Trajectory, list[dict]]:
    if len(reference) != n_steps + 1:
        raise ValueError("reference must contain n_steps+1 images")
    backend.reset()
    traj = Trajectory()
    diagnostics: list[dict] = []
    actor.eval(); critic.eval()
    for n in range(n_steps):
        # deterministic AMDI ordering: propagate on M^n first, then make the adaptive
        # decision from the propagated field and update M^n -> M^{n+1}.
        before_dofs = backend.active_dof_indices()
        step_diag = backend.propagate_for_adaptation()

        records = backend.candidate_records()
        if not records:
            raise RuntimeError("empty AMDI candidate set")
        before_leaves = backend.active_leaf_cells()
        decision_step = n + 1
        x_np = build_feature_matrix(
            backend.current_field(), records, before_leaves, backend.max_level,
            decision_step, n_steps, scales,
        )
        masks_np = np.stack([r.mask for r in records])
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        masks = torch.tensor(masks_np, dtype=torch.bool, device=device)
        occ_before = backend.occupancy
        time_fraction = decision_step / float(max(n_steps, 1))
        with torch.no_grad():
            logits = actor(x)
            dist = masked_categorical(logits, masks)
            if deterministic:
                actions = torch.argmax(
                    logits.masked_fill(~masks, torch.finfo(logits.dtype).min),
                    dim=-1,
                )
            else:
                actions = dist.sample()
            old_logp = dist.log_prob(actions).sum()
            value = critic(
                x,
                torch.tensor(occ_before, dtype=torch.float32, device=device),
                torch.tensor(time_fraction, dtype=torch.float32, device=device),
            )

        action_map = {r.cell: int(a.item()) for r, a in zip(records, actions)}
        step_diag = backend.learned_adapt(action_map, step_diag)
        after_dofs = backend.active_dof_indices()
        sw = switching_measure(before_dofs, after_dofs, backend.n_full)
        err = normalized_error(
            backend.current_field(), reference[n + 1], reward_cfg.epsilon_num
        )
        reward = -(
            reward_cfg.lambda_err * err
            + reward_cfg.lambda_occ * backend.occupancy
            + reward_cfg.lambda_sw * sw
        )
        traj.steps.append(
            RolloutStep(
                features=x.detach().cpu(),
                masks=masks.detach().cpu(),
                actions=actions.detach().cpu(),
                old_joint_log_prob=old_logp.detach().cpu(),
                old_value=value.detach().cpu(),
                reward=float(reward),
                occupancy_before=float(occ_before),
                time_fraction=float(time_fraction),
            )
        )
        counts = np.bincount(actions.cpu().numpy(), minlength=3)
        diagnostics.append({
            "step": n + 1,
            "reference_error": float(err),
            "C_rel": float(backend.occupancy),
            "switching": float(sw),
            "reward": float(reward),
            "proposed_coarsen": int(counts[0]),
            "proposed_retain": int(counts[1]),
            "proposed_refine": int(counts[2]),
            "executed_coarsen_groups": int(step_diag.executed_coarsen_groups),
            "executed_refine": int(step_diag.executed_refine),
            "energy": float(step_diag.energy),
        })
    return traj, diagnostics


def run_deterministic_trajectory(
    backend: HaarAMDIBackend,
    reference: Sequence[np.ndarray],
    n_steps: int,
    mode: str,
    epsilon_num: float = 1e-12,
) -> list[dict]:
    backend.reset()
    out=[]
    prev = backend.active_dof_indices()
    for n in range(n_steps):
        if mode == "baseline":
            diag = backend.deterministic_baseline_step()
        elif mode == "control":
            diag = backend.deterministic_control_step()
        else:
            raise ValueError(mode)
        cur = backend.active_dof_indices()
        out.append({
            "step": n+1,
            "reference_error": normalized_error(backend.current_field(), reference[n+1], epsilon_num),
            "C_rel": backend.occupancy,
            "switching": switching_measure(prev, cur, backend.n_full),
            "energy": float(diag.energy),
        })
        prev=cur
    return out


def final_quality(image: np.ndarray, truth: np.ndarray, n_active: int) -> dict:
    return all_metrics(np.clip(image,0,1), truth, active=n_active, full=truth.size)
