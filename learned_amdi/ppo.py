from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .config import PPOConfig
from .networks import SharedLocalActor, VariableSetCritic, masked_categorical
from .rollout import Trajectory


def compute_gae(traj: Trajectory, gamma: float, lambda_gae: float) -> None:
    if not traj.steps:
        return
    rewards = np.asarray([s.reward for s in traj.steps], float)
    values = np.asarray([float(s.old_value.item()) for s in traj.steps], float)
    next_values = np.r_[values[1:], 0.0]
    deltas = rewards + gamma * next_values - values
    adv = np.zeros_like(deltas)
    running = 0.0
    for i in range(len(deltas) - 1, -1, -1):
        running = deltas[i] + gamma * lambda_gae * running
        adv[i] = running
    targets = adv + values
    for step, a, target in zip(traj.steps, adv, targets):
        step.advantage = float(a)
        step.value_target = float(target)


class PPOTrainer:
    def __init__(self, actor: SharedLocalActor, critic: VariableSetCritic, cfg: PPOConfig, device: torch.device):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.learning_rate,
        )

    def _step_loss(self, step, adv_mean: float, adv_std: float):
        x = step.features.to(self.device)
        masks = step.masks.to(self.device)
        actions = step.actions.to(self.device)
        old_logp = step.old_joint_log_prob.to(self.device)
        dist = masked_categorical(self.actor(x), masks)
        new_logp = dist.log_prob(actions).sum()
        # Joint-action ratio exactly matching the frozen Section-IV definition.
        ratio = torch.exp(new_logp - old_logp)
        adv = torch.tensor((step.advantage - adv_mean) / adv_std, dtype=torch.float32, device=self.device)
        surrogate = ratio * adv
        surrogate_clip = torch.clamp(ratio, 1-self.cfg.epsilon_ppo, 1+self.cfg.epsilon_ppo) * adv
        policy_loss = -torch.minimum(surrogate, surrogate_clip)

        occ = torch.tensor(step.occupancy_before, dtype=torch.float32, device=self.device)
        tf = torch.tensor(step.time_fraction, dtype=torch.float32, device=self.device)
        value = self.critic(x, occ, tf)
        target = torch.tensor(step.value_target, dtype=torch.float32, device=self.device)
        value_loss = (value - target).pow(2)
        entropy = dist.entropy().mean()
        total = policy_loss + self.cfg.c_v * value_loss - self.cfg.c_h * entropy
        return total, policy_loss.detach(), value_loss.detach(), entropy.detach(), (new_logp-old_logp).detach()

    def update(self, trajectories: Iterable[Trajectory]) -> dict[str, float]:
        trajectories = list(trajectories)
        for traj in trajectories:
            compute_gae(traj, self.cfg.gamma, self.cfg.lambda_gae)
        steps = [s for t in trajectories for s in t.steps]
        if not steps:
            raise ValueError("empty rollout batch")
        av = np.asarray([s.advantage for s in steps], float)
        amean = float(av.mean())
        astd = float(av.std() + 1e-8)
        sums = dict(loss=0.0, policy=0.0, value=0.0, entropy=0.0, abs_log_ratio=0.0)
        count = 0
        for _ in range(self.cfg.epochs_per_update):
            for idx in np.random.permutation(len(steps)):
                self.opt.zero_grad(set_to_none=True)
                total, pl, vl, ent, dlr = self._step_loss(steps[int(idx)], amean, astd)
                total.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.cfg.max_grad_norm)
                self.opt.step()
                sums["loss"] += float(total.detach())
                sums["policy"] += float(pl)
                sums["value"] += float(vl)
                sums["entropy"] += float(ent)
                sums["abs_log_ratio"] += float(torch.abs(dlr))
                count += 1
        return {k: v/max(count,1) for k,v in sums.items()}
