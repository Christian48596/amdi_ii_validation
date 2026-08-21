from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureScales:
    s_q: float
    s_g: float
    s_v: float
    epsilon_feat: float = 1.0e-12


@dataclass(frozen=True)
class RewardConfig:
    lambda_err: float = 1.0
    lambda_occ: float = 0.15
    lambda_sw: float = 0.02
    epsilon_num: float = 1.0e-12


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.0
    lambda_gae: float = 0.95
    epsilon_ppo: float = 0.2
    c_v: float = 0.5
    c_h: float = 0.01
    learning_rate: float = 3.0e-4
    epochs_per_update: int = 4
    trajectories_per_update: int = 4
    updates: int = 50
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class NetworkConfig:
    actor_hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (64, 64)
    activation: str = "tanh"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
