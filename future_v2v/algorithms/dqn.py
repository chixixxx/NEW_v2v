from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from future_v2v.algorithms.baselines import TimingPolicy, teacher_policies
from future_v2v.config import TrainingConfig
from future_v2v.envs.timing_env import MATCH, FutureV2VTimingEnv


@dataclass(frozen=True)
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    priority: float = 1.0


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
    def __init__(self, obs_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DQNTimingAgent:
    name = "dqn_adaptive_timing"

    def __init__(
        self,
        obs_dim: int,
        training_config: TrainingConfig,
        device: str | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.config = training_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(obs_dim, training_config.hidden_dim).to(self.device)
        self.target = QNetwork(obs_dim, training_config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=training_config.learning_rate)
        self.replay = ReplayBuffer(training_config.replay_capacity, prioritized=training_config.prioritized_replay)
        self.global_step = 0

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray, epsilon: float = 0.0) -> int:
        _ = env
        if random.random() < epsilon:
            return int(random.choice([0, 1]))
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.online(tensor), dim=1).item())

    def train(
        self,
        env_factory: Callable[[], FutureV2VTimingEnv],
        episodes: int,
        seed_start: int = 10_000,
    ) -> list[dict[str, float | int]]:
        self._teacher_prefill(env_factory, seed_start)
        history: list[dict[str, float | int]] = []
        for episode in range(episodes):
            env: FutureV2VTimingEnv = env_factory()
            obs, _ = env.reset(seed=seed_start + 1000 + episode)
            terminated = False
            truncated = False
            episode_reward = 0.0
            loss_values: list[float] = []
            while not (terminated or truncated):
                epsilon = self._epsilon()
                action = self.act(env, obs, epsilon=epsilon)
                next_obs, reward, terminated, truncated, _info = env.step(action)
                priority = 1.0 + abs(reward) / 20.0 + (0.5 if action == MATCH else 0.0)
                self.replay.add(Transition(obs, action, reward, next_obs, terminated or truncated, priority))
                obs = next_obs
                episode_reward += reward
                self.global_step += 1
                if len(self.replay) >= self.config.min_replay_size:
                    loss_values.append(self._optimize_step())
                if self.global_step % self.config.target_update_interval == 0:
                    self.target.load_state_dict(self.online.state_dict())
            metrics = env.episode_metrics(policy_name=self.name, seed=seed_start + 1000 + episode)
            history.append(
                {
                    "episode": episode,
                    "reward": float(episode_reward),
                    "loss": float(np.mean(loss_values)) if loss_values else 0.0,
                    "epsilon": self._epsilon(),
                    "future_v2v_score": metrics.future_v2v_score,
                    "platform_profit": metrics.platform_profit,
                    "service_rate": metrics.service_rate,
                    "dispatch_epoch_count": metrics.dispatch_epoch_count,
                }
            )
        return history

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "model": self.online.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, training_config: TrainingConfig, device: str | None = None) -> DQNTimingAgent:
        payload = torch.load(path, map_location=device or "cpu")
        agent = cls(obs_dim=int(payload["obs_dim"]), training_config=training_config, device=device)
        agent.online.load_state_dict(payload["model"])
        agent.target.load_state_dict(payload["model"])
        return agent

    def _teacher_prefill(self, env_factory: Callable[[], FutureV2VTimingEnv], seed_start: int) -> None:
        policies = teacher_policies()
        for episode in range(self.config.teacher_prefill_episodes):
            policy: TimingPolicy = policies[episode % len(policies)]
            env: FutureV2VTimingEnv = env_factory()
            obs, _ = env.reset(seed=seed_start + episode)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = policy.act(env, obs)
                next_obs, reward, terminated, truncated, _info = env.step(action)
                priority = 1.5 + abs(reward) / 20.0 + (0.5 if action == MATCH else 0.0)
                self.replay.add(Transition(obs, action, reward, next_obs, terminated or truncated, priority))
                obs = next_obs

    def _optimize_step(self) -> float:
        batch = self.replay.sample(self.config.batch_size)
        obs = torch.as_tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([item.next_obs for item in batch]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        q_values = self.online(obs).gather(1, actions).squeeze(1)
        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = torch.argmax(self.online(next_obs), dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = torch.max(self.target(next_obs), dim=1).values
            target = rewards + self.config.gamma * (1.0 - done) * next_q
        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def _epsilon(self) -> float:
        fraction = min(1.0, self.global_step / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * math.exp(-4.0 * fraction)
