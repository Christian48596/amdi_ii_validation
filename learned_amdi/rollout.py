from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RolloutStep:
    features: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    old_joint_log_prob: torch.Tensor
    old_value: torch.Tensor
    reward: float
    occupancy_before: float
    time_fraction: float
    advantage: float = 0.0
    value_target: float = 0.0


@dataclass
class Trajectory:
    steps: list[RolloutStep] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return float(sum(s.reward for s in self.steps))
