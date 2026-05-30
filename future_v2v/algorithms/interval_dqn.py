from __future__ import annotations

import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from future_v2v.algorithms.dqn_core import QNetwork, ReplayBuffer, Transition
from future_v2v.algorithms.reward_shaping import compute_pbrs_potential, shape_reward
from future_v2v.config import EnvironmentConfig, ScaleConfig, TrainingConfig
from future_v2v.envs.timing_env import MATCH_FULL, WAIT, FutureV2VTimingEnv
from future_v2v.progress import progress


INTERVAL_ACTION_COUNT = 4
INTERVAL_ACTION_NAMES = {
    0: "dispatch_now",
    1: "delay_1_then_dispatch",
    2: "delay_2_then_dispatch",
    3: "delay_3_then_dispatch",
}


class AdaptiveIntervalDQNAgent:
    name = "adaptive_interval_dqn"

    def __init__(
        self,
        obs_dim: int,
        training_config: TrainingConfig,
        device: str | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.config = training_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(obs_dim, training_config.hidden_dim, action_count=INTERVAL_ACTION_COUNT).to(self.device)
        self.target = QNetwork(obs_dim, training_config.hidden_dim, action_count=INTERVAL_ACTION_COUNT).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=training_config.learning_rate)
        self.replay = ReplayBuffer(training_config.replay_capacity, prioritized=training_config.prioritized_replay)
        self.global_step = 0
        self.last_q_values: list[float] | None = None
        self.validation_history: list[dict[str, float | int]] = []
        self.interval_trace: list[dict[str, object]] = []
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_validation_score = float("-inf")
        self._validation_scenarios: list[dict[str, object]] | None = None

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online(tensor)
            self.last_q_values = [float(value) for value in q_values.squeeze(0).detach().cpu().tolist()]
            if random.random() < epsilon:
                return int(random.choice(list(range(INTERVAL_ACTION_COUNT))))
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
                raise ValueError("parallel interval training requires env_config and scale_config")
            return self._train_parallel_rollouts(
                episodes=episodes,
                seed_start=seed_start,
                rollout_workers=rollout_workers,
                env_config=env_config,
                scale_config=scale_config,
            )
        history: list[dict[str, float | int]] = []
        for episode in progress(range(episodes), desc="train interval DQN episodes", total=episodes, unit="episode"):
            env = env_factory()
            obs, _ = env.reset(seed=seed_start + 1000 + episode)
            row = self._run_training_episode(env, obs, episode=episode, seed=seed_start + 1000 + episode)
            history.append(row)
            self._maybe_validate(env_config=env_config, scale_config=scale_config, seed_start=seed_start, episode=episode)
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
                desc="train interval rollout batches",
                total=math.ceil(episodes / rollout_workers),
                unit="batch",
            ):
                batch_ids = episode_ids[offset : offset + rollout_workers]
                state_dict = {key: value.detach().cpu() for key, value in self.online.state_dict().items()}
                epsilon = self._epsilon()
                started_at = time.perf_counter()
                futures = [
                    executor.submit(
                        _run_interval_rollout_task,
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
                    loss_values = self._consume_transitions(result["transitions"])
                    elapsed = max(1e-9, time.perf_counter() - started_at)
                    metrics = result["metrics"]
                    action_counts = result["interval_action_counts"]
                    total_actions = max(1, sum(action_counts))
                    history.append(
                        {
                            "episode": episode,
                            "reward": float(result["episode_reward"]),
                            "raw_reward": float(result.get("raw_reward", result["episode_reward"])),
                            "legacy_reward": float(result.get("legacy_reward", result["episode_reward"])),
                            "shaped_reward": float(result["episode_reward"]),
                            "potential_delta_sum": float(result.get("potential_delta_sum", 0.0)),
                            "reward_shaping_mode": self.config.reward_shaping_mode,
                            "loss": float(np.mean(loss_values)) if loss_values else 0.0,
                            "epsilon": epsilon,
                            "future_v2v_score": metrics.future_v2v_score,
                            "platform_profit": metrics.platform_profit,
                            "service_rate": metrics.service_rate,
                            "dispatch_epoch_count": metrics.dispatch_epoch_count,
                            "interval_dispatch_now_count": int(action_counts[0]),
                            "interval_delay_1_count": int(action_counts[1]),
                            "interval_delay_2_count": int(action_counts[2]),
                            "interval_delay_3_count": int(action_counts[3]),
                            "interval_delay_2_share": float(action_counts[2] / total_actions),
                            "steps_per_sec": int(result["base_step_count"]) / elapsed,
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

    def _run_training_episode(
        self,
        env: FutureV2VTimingEnv,
        obs: np.ndarray,
        *,
        episode: int,
        seed: int,
    ) -> dict[str, float | int]:
        started_at = time.perf_counter()
        terminated = False
        truncated = False
        episode_reward = 0.0
        raw_reward_sum = 0.0
        legacy_reward_sum = 0.0
        potential_delta_sum = 0.0
        base_step_count = 0
        interval_action_counts = [0 for _ in range(INTERVAL_ACTION_COUNT)]
        loss_values: list[float] = []
        while not (terminated or truncated):
            epsilon = self._epsilon()
            action = self.act(obs, epsilon=epsilon)
            interval_action_counts[action] += 1
            next_obs, reward, terminated, truncated, duration, trace_row = execute_interval_action(
                env,
                action,
                training_config=self.config,
            )
            self.interval_trace.append(trace_row)
            priority = 1.0 + abs(reward) / 20.0 + (0.25 if action == 0 else 0.15)
            self.replay.add(Transition(obs, action, reward, next_obs, terminated or truncated, priority, duration))
            loss_values.extend(self._optimize_after_transition())
            obs = next_obs
            episode_reward += reward
            raw_reward_sum += float(trace_row.get("raw_reward", reward))
            legacy_reward_sum += float(trace_row.get("legacy_reward", reward))
            potential_delta_sum += float(trace_row.get("pbrs_delta", 0.0))
            base_step_count += duration
        metrics = env.episode_metrics(policy_name=self.name, seed=seed)
        elapsed = max(1e-9, time.perf_counter() - started_at)
        total_actions = max(1, sum(interval_action_counts))
        return {
            "episode": episode,
            "reward": float(episode_reward),
            "raw_reward": float(raw_reward_sum),
            "legacy_reward": float(legacy_reward_sum),
            "shaped_reward": float(episode_reward),
            "potential_delta_sum": float(potential_delta_sum),
            "reward_shaping_mode": self.config.reward_shaping_mode,
            "loss": float(np.mean(loss_values)) if loss_values else 0.0,
            "epsilon": self._epsilon(),
            "future_v2v_score": metrics.future_v2v_score,
            "platform_profit": metrics.platform_profit,
            "service_rate": metrics.service_rate,
            "dispatch_epoch_count": metrics.dispatch_epoch_count,
            "interval_dispatch_now_count": interval_action_counts[0],
            "interval_delay_1_count": interval_action_counts[1],
            "interval_delay_2_count": interval_action_counts[2],
            "interval_delay_3_count": interval_action_counts[3],
            "interval_delay_2_share": float(interval_action_counts[2] / total_actions),
            "steps_per_sec": base_step_count / elapsed,
            "episodes_per_sec": 1.0 / elapsed,
            "rollout_worker_count": 1,
            "replay_size": len(self.replay),
        }

    def _consume_transitions(self, transitions: list[Transition]) -> list[float]:
        loss_values: list[float] = []
        for transition in transitions:
            self.replay.add(transition)
            self.global_step += 1
            if len(self.replay) >= self.config.min_replay_size:
                loss_values.append(self._optimize_step())
            if self.global_step % self.config.target_update_interval == 0:
                self.target.load_state_dict(self.online.state_dict())
        return loss_values

    def _optimize_after_transition(self) -> list[float]:
        self.global_step += 1
        losses = []
        if len(self.replay) >= self.config.min_replay_size:
            losses.append(self._optimize_step())
        if self.global_step % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return losses

    def _optimize_step(self) -> float:
        batch = self.replay.sample(self.config.batch_size)
        obs = torch.as_tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([item.next_obs for item in batch]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        durations = torch.as_tensor([max(1, item.duration) for item in batch], dtype=torch.float32, device=self.device)
        q_values = self.online(obs).gather(1, actions).squeeze(1)
        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = torch.argmax(self.online(next_obs), dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = torch.max(self.target(next_obs), dim=1).values
            discount = torch.pow(torch.as_tensor(self.config.gamma, dtype=torch.float32, device=self.device), durations)
            target = rewards + discount * (1.0 - done) * next_q
        loss = torch.nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def run_eval_episode(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> list[dict[str, object]]:
        self.interval_trace = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = self.act(obs, epsilon=0.0)
            obs, _reward, terminated, truncated, _duration, trace_row = execute_interval_action(
                env,
                action,
                q_values=self.last_q_values,
                training_config=self.config,
            )
            self.interval_trace.append(trace_row)
        return self.interval_trace

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "model": self.online.state_dict(),
                "config": self.config.__dict__,
                "action_count": INTERVAL_ACTION_COUNT,
                "observation_profile": self.config.observation_profile,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, training_config: TrainingConfig, device: str | None = None) -> AdaptiveIntervalDQNAgent:
        payload = torch.load(path, map_location=device or "cpu")
        payload_profile = payload.get("observation_profile")
        if payload_profile and payload_profile != training_config.observation_profile:
            raise ValueError(
                "checkpoint observation_profile mismatch: "
                f"checkpoint={payload_profile!r}, current={training_config.observation_profile!r}"
            )
        agent = cls(obs_dim=int(payload["obs_dim"]), training_config=training_config, device=device)
        agent.online.load_state_dict(payload["model"])
        agent.target.load_state_dict(payload["model"])
        return agent

    def _teacher_prefill(self, env_factory: Callable[[], FutureV2VTimingEnv], seed_start: int) -> None:
        for episode in progress(
            range(self.config.teacher_prefill_episodes),
            desc="interval teacher replay prefill",
            total=self.config.teacher_prefill_episodes,
            unit="episode",
        ):
            env = env_factory()
            obs, _ = env.reset(seed=seed_start + episode)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = interval_teacher_action(env)
                priority_bonus = interval_teacher_priority_bonus(env, action)
                next_obs, reward, terminated, truncated, duration, _trace = execute_interval_action(
                    env,
                    action,
                    training_config=self.config,
                )
                priority = 1.4 + abs(reward) / 20.0 + priority_bonus
                self.replay.add(Transition(obs, action, reward, next_obs, terminated or truncated, priority, duration))
                obs = next_obs

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
            self.run_eval_episode(env, obs)
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
            "future_v2v_score_mean": float(np.mean(scores)) if scores else 0.0,
            "platform_profit_mean": float(np.mean(profits)) if profits else 0.0,
            "dispatch_epoch_count_mean": float(np.mean(dispatch_counts)) if dispatch_counts else 0.0,
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
                rows.append(
                    dict(
                        generator.manifest_row(
                            seed=seed_start + 70_000 + attempts,
                            scenario_id=f"val_{len(rows):03d}",
                        )
                    )
                )
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

    def _epsilon(self) -> float:
        fraction = min(1.0, self.global_step / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * math.exp(-4.0 * fraction)


def execute_interval_action(
    env: FutureV2VTimingEnv,
    interval_action: int,
    *,
    q_values: list[float] | None = None,
    training_config: TrainingConfig | None = None,
) -> tuple[np.ndarray, float, bool, bool, int, dict[str, object]]:
    delay = int(np.clip(interval_action, 0, INTERVAL_ACTION_COUNT - 1))
    start_tick = env.current_tick
    start_snapshot = env.snapshot()
    start_orders = len(start_snapshot.active_orders)
    state_action_features = env.state_action_features(start_snapshot)
    potential_start = compute_pbrs_potential(env, training_config, start_snapshot) if training_config else 0.0
    if training_config and start_tick == 0:
        env.training_initial_potential = potential_start
    raw_reward_sum = 0.0
    legacy_reward_sum = 0.0
    duration = 0
    terminated = False
    truncated = False
    obs = env._observation()
    for _ in range(delay):
        obs, reward, terminated, truncated, info = env.step(WAIT)
        raw_reward_sum += float(info.get("base_reward", reward))
        legacy_reward_sum += float(info.get("legacy_reward", reward))
        duration += 1
        if terminated or truncated:
            break
    final_dispatch_executed = False
    if not (terminated or truncated):
        env.set_action_q_values(q_values)
        action = MATCH_FULL if env.snapshot().active_orders else WAIT
        obs, reward, terminated, truncated, info = env.step(action)
        raw_reward_sum += float(info.get("base_reward", reward))
        legacy_reward_sum += float(info.get("legacy_reward", reward))
        duration += 1
        final_dispatch_executed = action == MATCH_FULL
    end_tick = env.current_tick
    reward_mode = training_config.reward_shaping_mode if training_config else "legacy_delta"
    potential_end_unclipped = compute_pbrs_potential(env, training_config) if training_config else 0.0
    terminal = terminated or truncated
    potential_end = 0.0 if terminal else potential_end_unclipped
    shaped = shape_reward(
        raw_reward=raw_reward_sum,
        legacy_reward=legacy_reward_sum,
        reward_mode=reward_mode,
        gamma=float(training_config.gamma) if training_config else 1.0,
        duration=duration,
        potential_start=potential_start,
        potential_end=potential_end,
        potential_end_unclipped=potential_end_unclipped,
        initial_potential=float(getattr(env, "training_initial_potential", potential_start)),
        terminal=terminal,
        terminal_mode=training_config.pbrs_terminal_mode if training_config else "finite_horizon_correction",
    )
    total_reward = shaped.reward
    pbrs_delta = shaped.pbrs_delta
    trace_row = {
        "start_tick": start_tick,
        "end_tick": end_tick,
        "interval_action": interval_action,
        "interval_action_name": INTERVAL_ACTION_NAMES.get(interval_action, "unknown"),
        "delay_ticks": delay,
        "duration_ticks": duration,
        "final_dispatch_executed": final_dispatch_executed,
        "active_orders_at_start": start_orders,
        "reward": total_reward,
        "raw_reward": raw_reward_sum,
        "legacy_reward": legacy_reward_sum,
        "shaped_reward": total_reward,
        "reward_shaping_mode": reward_mode,
        "potential_start": potential_start,
        "potential_end": potential_end,
        "potential_end_unclipped": potential_end_unclipped,
        "pbrs_delta": pbrs_delta,
        "finite_horizon_terminal_correction": shaped.terminal_correction,
    }
    trace_row.update(state_action_features)
    if q_values is not None:
        for idx, value in enumerate(q_values):
            trace_row[f"q_interval_{idx}"] = float(value)
    return obs, float(total_reward), terminated, truncated, max(1, duration), trace_row

def interval_teacher_action(env: FutureV2VTimingEnv) -> int:
    snapshot = env.snapshot()
    if not snapshot.active_orders:
        return 1
    from future_v2v.algorithms.baselines import HandcraftedDeadlineRulePolicy

    near_deadline_share = sum(
        1
        for order in snapshot.active_orders
        if order.max_wait_ticks - order.waiting_ticks(env.current_tick) <= 1
    ) / max(1, len(snapshot.active_orders))
    waiting_ratios = [order.waiting_ratio(env.current_tick) for order in snapshot.active_orders]
    flex_values = [vehicle.time_flexibility_ticks(env.current_tick) for vehicle in snapshot.active_vehicles]
    mean_flex = float(np.mean(flex_values)) if flex_values else 0.0
    strong_rule_action = HandcraftedDeadlineRulePolicy().act(env, env._observation())
    if strong_rule_action == MATCH_FULL:
        return 0
    if near_deadline_share >= 0.12 or (waiting_ratios and max(waiting_ratios) >= 0.82) or mean_flex <= 3.0:
        return 0
    opportunity = env.estimate_wait_opportunity(snapshot)
    pressure = len(snapshot.active_orders) / max(1, len(snapshot.active_vehicles))
    edge_coverage = len({edge.order_id for edge in snapshot.candidate_edges}) / max(1, len(snapshot.active_orders))
    if opportunity >= 260.0 and pressure <= 0.75 and near_deadline_share < 0.06:
        return 2
    if opportunity >= 60.0 and near_deadline_share < 0.10:
        return 1
    if pressure <= 0.28 and edge_coverage >= 0.65 and near_deadline_share < 0.05:
        return 3
    return 0 if pressure >= 0.95 else 1


def interval_teacher_priority_bonus(env: FutureV2VTimingEnv, action: int) -> float:
    snapshot = env.snapshot()
    if not snapshot.active_orders:
        return 0.0
    near_deadline_share = sum(
        1
        for order in snapshot.active_orders
        if order.max_wait_ticks - order.waiting_ticks(env.current_tick) <= 1
    ) / max(1, len(snapshot.active_orders))
    waiting_ratios = [order.waiting_ratio(env.current_tick) for order in snapshot.active_orders]
    max_waiting_ratio = max(waiting_ratios) if waiting_ratios else 0.0
    if action == 0 and (near_deadline_share >= 0.10 or max_waiting_ratio >= 0.80):
        return 1.2
    if action == 0:
        return 0.45
    if action == 1:
        return 0.25
    return 0.35


def _run_interval_rollout_task(
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
    agent = AdaptiveIntervalDQNAgent(obs_dim=len(obs), training_config=training_config, device="cpu")
    agent.online.load_state_dict(state_dict)
    transitions: list[Transition] = []
    terminated = False
    truncated = False
    episode_reward = 0.0
    raw_reward_sum = 0.0
    legacy_reward_sum = 0.0
    potential_delta_sum = 0.0
    base_step_count = 0
    interval_action_counts = [0 for _ in range(INTERVAL_ACTION_COUNT)]
    while not (terminated or truncated):
        action = agent.act(obs, epsilon=epsilon)
        interval_action_counts[action] += 1
        next_obs, reward, terminated, truncated, duration, trace = execute_interval_action(
            env,
            action,
            training_config=training_config,
        )
        priority = 1.0 + abs(reward) / 20.0 + (0.25 if action == 0 else 0.15)
        transitions.append(Transition(obs, action, reward, next_obs, terminated or truncated, priority, duration))
        obs = next_obs
        episode_reward += reward
        raw_reward_sum += float(trace.get("raw_reward", reward))
        legacy_reward_sum += float(trace.get("legacy_reward", reward))
        potential_delta_sum += float(trace.get("pbrs_delta", 0.0))
        base_step_count += duration
    return {
        "transitions": transitions,
        "episode_reward": episode_reward,
        "raw_reward": raw_reward_sum,
        "legacy_reward": legacy_reward_sum,
        "potential_delta_sum": potential_delta_sum,
        "metrics": env.episode_metrics(policy_name=AdaptiveIntervalDQNAgent.name, seed=seed),
        "interval_action_counts": interval_action_counts,
        "base_step_count": base_step_count,
    }
