from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(name)


def _mlp(input_dim: int, hidden: Sequence[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = input_dim
    for width in hidden:
        layers += [nn.Linear(d, int(width)), _activation(activation)]
        d = int(width)
    return nn.Sequential(*layers)


class SharedLocalActor(nn.Module):
    def __init__(self, hidden=(64, 64), activation="tanh") -> None:
        super().__init__()
        self.encoder = _mlp(6, hidden, activation)
        last = int(hidden[-1]) if hidden else 6
        self.head = nn.Linear(last, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class VariableSetCritic(nn.Module):
    def __init__(self, hidden=(64, 64), activation="tanh") -> None:
        super().__init__()
        self.encoder = _mlp(14, hidden, activation)  # mean6 + max6 + occ + time
        last = int(hidden[-1]) if hidden else 14
        self.head = nn.Linear(last, 1)

    def descriptor(self, x: torch.Tensor, occupancy: torch.Tensor, time_fraction: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 6 or x.shape[0] == 0:
            raise ValueError("critic expects [n_candidates,6] with n_candidates>0")
        return torch.cat(
            [x.mean(0), x.max(0).values, occupancy.reshape(1), time_fraction.reshape(1)]
        )

    def forward(self, x, occupancy, time_fraction):
        return self.head(self.encoder(self.descriptor(x, occupancy, time_fraction))).squeeze(-1)


def masked_categorical(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    masks = masks.bool()
    if logits.shape != masks.shape:
        raise ValueError("logits/masks shape mismatch")
    if torch.any(masks.sum(-1) == 0):
        raise ValueError("each candidate must retain at least one admissible action")
    return Categorical(logits=logits.masked_fill(~masks, torch.finfo(logits.dtype).min))
