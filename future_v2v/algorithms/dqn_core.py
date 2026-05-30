from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    priority: float = 1.0
    duration: int = 1


class ReplayBuffer:
    def __init__(self, capacity: int, prioritized: bool = True) -> None:
        self.capacity = capacity
        self.prioritized = prioritized
        self.items: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if not self.prioritized:
            return random.sample(list(self.items), batch_size)
        priorities = np.array([max(1e-3, item.priority) for item in self.items], dtype=float)
        probs = priorities / priorities.sum()
        idx = np.random.choice(len(self.items), size=batch_size, replace=False, p=probs)
        materialized = list(self.items)
        return [materialized[int(i)] for i in idx]


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_count: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
