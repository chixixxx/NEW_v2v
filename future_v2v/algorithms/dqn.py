from __future__ import annotations

import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from future_v2v.algorithms.baselines import TimingPolicy, teacher_policies
from future_v2v.config import EnvironmentConfig, ScaleConfig, TrainingConfig
from future_v2v.envs.timing_env import ACTION_COUNT, MATCH_FULL, MATCH_TOP_BATCH, WAIT, FutureV2VTimingEnv
from future_v2v.progress import progress


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
    def __init__(self, obs_dim: int, hidden_dim: int, action_count: int = ACTION_COUNT) -> None:
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


class DQNTimingAgent:
    name = "dqn_adaptive_timing_legacy"

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
        self.last_q_values: list[float] | None = None
        self.validation_history: list[dict[str, float | int]] = []
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_validation_score = float("-inf")
        self._validation_scenarios: list[dict[str, object]] | None = None

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray, epsilon: float = 0.0) -> int:
        _ = env
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online(tensor)
            self.last_q_values = [float(value) for value in q_values.squeeze(0).detach().cpu().tolist()]
            if random.random() < epsilon:
                return int(random.choice(list(range(ACTION_COUNT))))
            return int(torch.argmax(q_values, dim=1).item())

    def train(
        self,
        env_factory: Callable[[], FutureV2VTimingEnv],
        episodes: int,
        seed_start: int = 10_000,
        rollout_workers: int = 1,
        env_config: EnvironmentConfig | None = None,
        scale_config: ScaleConfig | None = None,
    ) -> list[dict[str, float | int]]:
        self._teacher_prefill(env_factory, seed_start)
        if rollout_workers > 1:
            if env_config is None or scale_config is None:
                raise ValueError("parallel rollout training requires env_config and scale_config")
            return self._train_parallel_rollouts(
                episodes=episodes,
                seed_start=seed_start,
                rollout_workers=rollout_workers,
                env_config=env_config,
                scale_config=scale_config,
            )
        history: list[dict[str, float | int]] = []
        for episode in progress(range(episodes), desc="train DQN episodes", total=episodes, unit="episode"):
            episode_started_at = time.perf_counter()
            env: FutureV2VTimingEnv = env_factory()
            obs, _ = env.reset(seed=seed_start + 1000 + episode)
            terminated = False
            truncated = False
            episode_reward = 0.0
            step_count = 0
            action_counts = [0 for _ in range(ACTION_COUNT)]
            loss_values: list[float] = []
            while not (terminated or truncated):
                epsilon = self._epsilon()
                action = self.act(env, obs, epsilon=epsilon)
                action_counts[action] += 1
                next_obs, reward, terminated, truncated, _info = env.step(action)
                priority = 1.0 + abs(reward) / 20.0 + (0.5 if action in (MATCH_TOP_BATCH, MATCH_FULL) else 0.0)
                self.replay.add(Transition(obs, action, reward, next_obs, terminated or truncated, priority))
                obs = next_obs
                episode_reward += reward
                step_count += 1
                self.global_step += 1
                if len(self.replay) >= self.config.min_replay_size:
                    loss_values.append(self._optimize_step())
                if self.global_step % self.config.target_update_interval == 0:
                    self.target.load_state_dict(self.online.state_dict())
            metrics = env.episode_metrics(policy_name=self.name, seed=seed_start + 1000 + episode)
            elapsed = max(1e-9, time.perf_counter() - episode_started_at)
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
                    "wait_count": action_counts[0],
                    "top_batch_count": action_counts[1],
                    "full_match_count": action_counts[2],
                    "steps_per_sec": step_count / elapsed,
                    "episodes_per_sec": 1.0 / elapsed,
                    "rollout_worker_count": 1,
                    "replay_size": len(self.replay),
                }
            )
            self._maybe_validate(
                env_config=env_config,
                scale_config=scale_config,
                seed_start=seed_start,
                episode=episode,
                force=episode == episodes - 1,
            )
        self._restore_best_checkpoint()
        return history

    def _train_parallel_rollouts(
        self,
        *,
        episodes: int,
        seed_start: int,
        rollout_workers: int,
        env_config: EnvironmentConfig,
        scale_config: ScaleConfig,
    ) -> list[dict[str, float | int]]:
        history: list[dict[str, float | int]] = []
        episode_ids = list(range(episodes))
        with ProcessPoolExecutor(max_workers=rollout_workers) as executor:
            for offset in progress(
                range(0, episodes, rollout_workers),
                desc="train parallel rollout batches",
                total=math.ceil(episodes / rollout_workers),
                unit="batch",
            ):
                batch_ids = episode_ids[offset : offset + rollout_workers]
                state_dict = {key: value.detach().cpu() for key, value in self.online.state_dict().items()}
                epsilon = self._epsilon()
                started_at = time.perf_counter()
                futures = [
                    executor.submit(
                        _run_dqn_rollout_task,
                        env_config,
                        scale_config,
                        self.config,
                        state_dict,
                        seed_start + 1000 + episode,
                        epsilon,
                    )
                    for episode in batch_ids
                ]
                for episode, future in zip(batch_ids, futures):
                    result = future.result()
                    transitions: list[Transition] = result["transitions"]
                    loss_values: list[float] = []
                    for transition in transitions:
                        self.replay.add(transition)
                        self.global_step += 1
                        if len(self.replay) >= self.config.min_replay_size:
                            loss_values.append(self._optimize_step())
                        if self.global_step % self.config.target_update_interval == 0:
                            self.target.load_state_dict(self.online.state_dict())
                    elapsed = max(1e-9, time.perf_counter() - started_at)
                    steps = len(transitions)
                    metrics = result["metrics"]
                    action_counts = result["action_counts"]
                    history.append(
                        {
                            "episode": episode,
                            "reward": float(result["episode_reward"]),
                            "loss": float(np.mean(loss_values)) if loss_values else 0.0,
                            "epsilon": epsilon,
                            "future_v2v_score": metrics.future_v2v_score,
                            "platform_profit": metrics.platform_profit,
                            "service_rate": metrics.service_rate,
                            "dispatch_epoch_count": metrics.dispatch_epoch_count,
                            "wait_count": int(action_counts[0]),
                            "top_batch_count": int(action_counts[1]),
                            "full_match_count": int(action_counts[2]),
                            "steps_per_sec": steps / elapsed,
                            "episodes_per_sec": max(1, len(batch_ids)) / elapsed,
                            "rollout_worker_count": rollout_workers,
                            "replay_size": len(self.replay),
                        }
                    )
                self._maybe_validate(
                    env_config=env_config,
                    scale_config=scale_config,
                    seed_start=seed_start,
                    episode=batch_ids[-1],
                    force=batch_ids[-1] == episode_ids[-1],
                )
        self._restore_best_checkpoint()
        return sorted(history, key=lambda row: int(row["episode"]))

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

    def _maybe_validate(
        self,
        *,
        env_config: EnvironmentConfig | None,
        scale_config: ScaleConfig | None,
        seed_start: int,
        episode: int,
        force: bool = False,
    ) -> None:
        if env_config is None or scale_config is None or self.config.validation_episodes <= 0:
            return
        interval = max(1, int(self.config.validation_interval_episodes))
        if not force and (episode + 1) % interval != 0:
            return
        scores = []
        profits = []
        dispatch_counts = []
        bucket_scores: dict[str, list[float]] = {}
        scenarios = self._validation_manifest(env_config, scale_config, seed_start)
        for idx, scenario in enumerate(scenarios):
            env = FutureV2VTimingEnv(env_config, scale_config, seed=seed_start + 50_000 + idx)
            if scenario.get("day") and scenario.get("start_tick_day") != "":
                obs, _ = env.reset_to_tlc_window(
                    seed=int(scenario["seed"]),
                    day=str(scenario["day"]),
                    start_tick_day=int(scenario["start_tick_day"]),
                    scenario_id=str(scenario["scenario_id"]),
                )
            else:
                obs, _ = env.reset(seed=int(scenario["seed"]))
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = self.act(env, obs, epsilon=0.0)
                env.set_action_q_values(self.last_q_values)
                obs, _reward, terminated, truncated, _info = env.step(action)
            metrics = env.episode_metrics(policy_name=self.name, seed=seed_start + 50_000 + idx)
            scores.append(metrics.future_v2v_score)
            profits.append(metrics.platform_profit)
            dispatch_counts.append(metrics.dispatch_epoch_count)
            bucket = str(scenario.get("time_of_day_bucket", "unknown"))
            bucket_scores.setdefault(bucket, []).append(metrics.future_v2v_score)
        bucket_means = {
            bucket: float(np.mean(values))
            for bucket, values in bucket_scores.items()
            if values
        }
        row = {
            "episode": episode,
            "future_v2v_score_mean": float(np.mean(scores)),
            "platform_profit_mean": float(np.mean(profits)),
            "dispatch_epoch_count_mean": float(np.mean(dispatch_counts)),
            "validation_bucket_min_score_mean": float(min(bucket_means.values())) if bucket_means else 0.0,
        }
        for bucket in self.config.validation_time_buckets:
            row[f"validation_{bucket}_score_mean"] = bucket_means.get(bucket, 0.0)
        mean_score = float(row["future_v2v_score_mean"])
        worst_bucket_score = float(row["validation_bucket_min_score_mean"])
        off_peak_score = float(row.get("validation_off_peak_score_mean", 0.0))
        checkpoint_selection_score, off_peak_floor_penalty = self._checkpoint_selection_score(
            mean_score=mean_score,
            worst_bucket_score=worst_bucket_score,
            off_peak_score=off_peak_score,
        )
        row["validation_off_peak_floor_penalty"] = off_peak_floor_penalty
        row["checkpoint_selection_score"] = checkpoint_selection_score
        self.validation_history.append(row)
        score = checkpoint_selection_score
        if score > self.best_validation_score:
            self.best_validation_score = score
            self.best_state_dict = {key: value.detach().cpu().clone() for key, value in self.online.state_dict().items()}

    def _checkpoint_selection_score(
        self,
        *,
        mean_score: float,
        worst_bucket_score: float,
        off_peak_score: float,
    ) -> tuple[float, float]:
        worst_weight = float(np.clip(self.config.validation_worst_bucket_weight, 0.0, 1.0))
        off_peak_floor_penalty = max(0.0, float(self.config.validation_off_peak_score_floor) - off_peak_score)
        score = (1.0 - worst_weight) * mean_score + worst_weight * worst_bucket_score - off_peak_floor_penalty
        return float(score), float(off_peak_floor_penalty)

    def _validation_manifest(
        self,
        env_config: EnvironmentConfig,
        scale_config: ScaleConfig,
        seed_start: int,
    ) -> list[dict[str, object]]:
        if self._validation_scenarios is not None:
            return self._validation_scenarios
        env = FutureV2VTimingEnv(env_config, scale_config, seed=seed_start + 50_000)
        generator = getattr(env, "generator", None)
        buckets = list(self.config.validation_time_buckets) or ["unknown"]
        total = max(1, int(self.config.validation_episodes))
        per_bucket = max(1, math.ceil(total / len(buckets)))
        rows: list[dict[str, object]] = []
        if hasattr(generator, "manifest_row"):
            bucket_counts = {bucket: 0 for bucket in buckets}
            attempts = 0
            max_attempts = max(200, total * 80)
            while len(rows) < total and attempts < max_attempts:
                row = generator.manifest_row(
                    seed=seed_start + 60_000 + attempts,
                    scenario_id=f"val_{len(rows):03d}",
                )
                bucket = str(row.get("time_of_day_bucket", ""))
                if bucket in bucket_counts and bucket_counts[bucket] < per_bucket:
                    row["scenario_id"] = f"val_{len(rows):03d}_{bucket}"
                    rows.append(dict(row))
                    bucket_counts[bucket] += 1
                attempts += 1
            attempts = 0
            while len(rows) < total and attempts < max_attempts:
                row = generator.manifest_row(
                    seed=seed_start + 70_000 + attempts,
                    scenario_id=f"val_{len(rows):03d}",
                )
                rows.append(dict(row))
                attempts += 1
        while len(rows) < total:
            idx = len(rows)
            rows.append(
                {
                    "scenario_id": f"val_{idx:03d}",
                    "seed": seed_start + 50_000 + idx,
                    "day": "",
                    "start_tick_day": "",
                    "time_of_day_bucket": "unknown",
                }
            )
        self._validation_scenarios = rows[:total]
        return self._validation_scenarios

    def _restore_best_checkpoint(self) -> None:
        if self.best_state_dict is None:
            return
        self.online.load_state_dict(self.best_state_dict)
        self.target.load_state_dict(self.best_state_dict)

    @classmethod
    def load(cls, path: Path, training_config: TrainingConfig, device: str | None = None) -> DQNTimingAgent:
        payload = torch.load(path, map_location=device or "cpu")
        agent = cls(obs_dim=int(payload["obs_dim"]), training_config=training_config, device=device)
        agent.online.load_state_dict(payload["model"])
        agent.target.load_state_dict(payload["model"])
        return agent

    def _teacher_prefill(self, env_factory: Callable[[], FutureV2VTimingEnv], seed_start: int) -> None:
        policies = teacher_policies()
        for episode in progress(
            range(self.config.teacher_prefill_episodes),
            desc="teacher replay prefill",
            total=self.config.teacher_prefill_episodes,
            unit="episode",
        ):
            policy: TimingPolicy = policies[episode % len(policies)]
            env: FutureV2VTimingEnv = env_factory()
            obs, _ = env.reset(seed=seed_start + episode)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = policy.act(env, obs)
                wait_opportunity = env.estimate_wait_opportunity() if action == WAIT else 0.0
                next_obs, reward, terminated, truncated, _info = env.step(action)
                priority = 1.5 + abs(reward) / 20.0 + (0.5 if action in (MATCH_TOP_BATCH, MATCH_FULL) else 0.0)
                if action == WAIT and wait_opportunity > 0.0:
                    priority += 0.35
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
            durations = torch.as_tensor([max(1, item.duration) for item in batch], dtype=torch.float32, device=self.device)
            target = rewards + torch.pow(torch.as_tensor(self.config.gamma, dtype=torch.float32, device=self.device), durations) * (1.0 - done) * next_q
        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def _epsilon(self) -> float:
        fraction = min(1.0, self.global_step / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * math.exp(-4.0 * fraction)

def _run_dqn_rollout_task(
    env_config: EnvironmentConfig,
    scale_config: ScaleConfig,
    training_config: TrainingConfig,
    state_dict: dict[str, torch.Tensor],
    seed: int,
    epsilon: float,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    env = FutureV2VTimingEnv(env_config, scale_config, seed=seed)
    obs, _ = env.reset(seed=seed)
    agent = DQNTimingAgent(obs_dim=len(obs), training_config=training_config, device="cpu")
    agent.online.load_state_dict(state_dict)
    transitions: list[Transition] = []
    terminated = False
    truncated = False
    episode_reward = 0.0
    action_counts = [0 for _ in range(ACTION_COUNT)]
    while not (terminated or truncated):
        action = agent.act(env, obs, epsilon=epsilon)
        action_counts[action] += 1
        next_obs, reward, terminated, truncated, _info = env.step(action)
        priority = 1.0 + abs(reward) / 20.0 + (0.5 if action in (MATCH_TOP_BATCH, MATCH_FULL) else 0.0)
        transitions.append(Transition(obs, action, reward, next_obs, terminated or truncated, priority))
        obs = next_obs
        episode_reward += reward
    return {
        "transitions": transitions,
        "episode_reward": episode_reward,
        "metrics": env.episode_metrics(policy_name=DQNTimingAgent.name, seed=seed),
        "action_counts": action_counts,
    }
