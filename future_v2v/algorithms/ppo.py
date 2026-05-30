from __future__ import annotations

import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from future_v2v.algorithms.baselines import HandcraftedDeadlineRulePolicy
from future_v2v.algorithms.reward_shaping import compute_pbrs_potential, shape_reward
from future_v2v.config import EnvironmentConfig, ScaleConfig, TrainingConfig
from future_v2v.envs.timing_env import MATCH_FULL, WAIT, FutureV2VTimingEnv
from future_v2v.progress import progress


BINARY_ACTION_COUNT = 2
BINARY_ACTION_NAMES = {
    WAIT: "wait",
    MATCH_FULL: "match_full",
}


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_count: int = BINARY_ACTION_COUNT) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_count)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class AdaptiveTimingPPOAgent:
    name = "adaptive_timing_ppo"

    def __init__(
        self,
        obs_dim: int,
        training_config: TrainingConfig,
        device: str | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.config = training_config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ActorCriticNetwork(obs_dim, training_config.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=training_config.learning_rate)
        self.validation_history: list[dict[str, float | int]] = []
        self.interval_trace: list[dict[str, object]] = []
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_validation_score = float("-inf")
        self._validation_scenarios: list[dict[str, object]] | None = None

    def act(
        self,
        obs: np.ndarray,
        *,
        deterministic: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float, float, list[float], float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            dist = Categorical(logits=logits)
            if deterministic:
                action_tensor = torch.argmax(logits, dim=-1)
            elif rng is not None:
                probs_np = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
                action_tensor = torch.as_tensor([int(rng.choice(len(probs_np), p=probs_np))], device=self.device)
            else:
                action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
            entropy = dist.entropy()
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
        return (
            int(action_tensor.item()),
            float(log_prob.item()),
            float(value.item()),
            [float(item) for item in probs],
            float(entropy.item()),
        )

    def train(
        self,
        env_factory: Callable[[], FutureV2VTimingEnv],
        episodes: int,
        seed_start: int = 10_000,
        rollout_workers: int = 1,
        env_config: EnvironmentConfig | None = None,
        scale_config: ScaleConfig | None = None,
    ) -> list[dict[str, float | int]]:
        self._teacher_warm_start(env_factory, seed_start)
        if env_config is None or scale_config is None:
            env = env_factory()
            env_config = env.env_config
            scale_config = env.scale_config
        history: list[dict[str, float | int]] = []
        update_size = max(1, int(self.config.ppo_rollout_episodes_per_update))
        worker_count = max(1, int(rollout_workers))
        episode_ids = list(range(episodes))
        for offset in progress(
            range(0, episodes, update_size),
            desc="train PPO rollout batches",
            total=math.ceil(episodes / update_size),
            unit="batch",
        ):
            batch_ids = episode_ids[offset : offset + update_size]
            started_at = time.perf_counter()
            rollouts = self._collect_rollout_batch(
                batch_ids=batch_ids,
                seed_start=seed_start,
                worker_count=worker_count,
                env_config=env_config,
                scale_config=scale_config,
            )
            update_stats = self._update_from_rollouts(rollouts)
            elapsed = max(1e-9, time.perf_counter() - started_at)
            for result in rollouts:
                metrics = result["metrics"]
                counts = result["action_counts"]
                total_actions = max(1, int(counts["wait"]) + int(counts["match_full"]))
                interval_stats = result["interval_stats"]
                history.append(
                    {
                        "episode": int(result["episode"]),
                        "reward": float(result["shaped_return"]),
                        "raw_reward": float(result["raw_return"]),
                        "legacy_reward": float(result["legacy_return"]),
                        "shaped_reward": float(result["shaped_return"]),
                        "potential_delta_sum": float(result["potential_delta_sum"]),
                        "reward_shaping_mode": self.config.reward_shaping_mode,
                        "loss": float(update_stats["loss"]),
                        "policy_loss": float(update_stats["policy_loss"]),
                        "value_loss": float(update_stats["value_loss"]),
                        "entropy": float(update_stats["entropy"]),
                        "future_v2v_score": metrics.future_v2v_score,
                        "platform_profit": metrics.platform_profit,
                        "service_rate": metrics.service_rate,
                        "dispatch_epoch_count": metrics.dispatch_epoch_count,
                        "wait_action_count": int(counts["wait"]),
                        "match_full_action_count": int(counts["match_full"]),
                        "match_full_action_share": float(counts["match_full"] / total_actions),
                        "wait_action_share": float(counts["wait"] / total_actions),
                        "interval_dispatch_now_count": int(interval_stats["dispatch_now_count"]),
                        "interval_delay_1_count": int(interval_stats["delay_1_count"]),
                        "interval_delay_2_count": int(interval_stats["delay_2_count"]),
                        "interval_delay_3_count": int(interval_stats["delay_3_count"]),
                        "interval_delay_4_plus_count": int(interval_stats["delay_4_plus_count"]),
                        "interval_mean_action_interval": float(interval_stats["mean_interval"]),
                        "interval_max_action_share": float(interval_stats["max_interval_share"]),
                        "steps_per_sec": int(result["step_count"]) / elapsed,
                        "episodes_per_sec": max(1, len(batch_ids)) / elapsed,
                        "rollout_worker_count": worker_count,
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

    def _collect_rollout_batch(
        self,
        *,
        batch_ids: list[int],
        seed_start: int,
        worker_count: int,
        env_config: EnvironmentConfig,
        scale_config: ScaleConfig,
    ) -> list[dict[str, object]]:
        state_dict = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
        if worker_count <= 1 or len(batch_ids) <= 1:
            rows = []
            for episode in batch_ids:
                env = FutureV2VTimingEnv(env_config, scale_config, seed=seed_start + 1000 + episode)
                rows.append(
                    self._collect_episode(
                        env,
                        seed=seed_start + 1000 + episode,
                        episode=episode,
                        deterministic=False,
                    )
                )
            return rows
        with ProcessPoolExecutor(max_workers=min(worker_count, len(batch_ids))) as executor:
            futures = [
                executor.submit(
                    _run_ppo_rollout_task,
                    env_config,
                    scale_config,
                    self.config,
                    state_dict,
                    self.obs_dim,
                    seed_start + 1000 + episode,
                    episode,
                )
                for episode in batch_ids
            ]
            return [future.result() for future in futures]

    def _collect_episode(
        self,
        env: FutureV2VTimingEnv,
        *,
        seed: int,
        episode: int,
        deterministic: bool,
        reset: bool = True,
        initial_obs: np.ndarray | None = None,
        action_seed: int | None = None,
    ) -> dict[str, object]:
        if reset:
            obs, _ = env.reset(seed=seed)
        elif initial_obs is not None:
            obs = initial_obs
        else:
            obs = env._observation()
        initial_snapshot = env.snapshot()
        env.training_initial_potential = compute_pbrs_potential(env, self.config, initial_snapshot)
        terminated = False
        truncated = False
        steps: list[dict[str, object]] = []
        trace_rows: list[dict[str, object]] = []
        shaped_return = 0.0
        raw_return = 0.0
        legacy_return = 0.0
        potential_delta_sum = 0.0
        wait_streak = 0
        action_rng = np.random.default_rng(action_seed) if action_seed is not None else None
        while not (terminated or truncated):
            before_snapshot = env.snapshot()
            start_tick = env.current_tick
            features = env.state_action_features(before_snapshot)
            potential_start = compute_pbrs_potential(env, self.config, before_snapshot)
            action, log_prob, value, probs, entropy = self.act(obs, deterministic=deterministic, rng=action_rng)
            env.set_action_q_values(probs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            terminal = terminated or truncated
            potential_end_unclipped = compute_pbrs_potential(env, self.config)
            potential_end = 0.0 if terminal else potential_end_unclipped
            raw_reward = float(info.get("base_reward", reward))
            legacy_reward = float(info.get("legacy_reward", reward))
            shaped = shape_reward(
                raw_reward=raw_reward,
                legacy_reward=legacy_reward,
                reward_mode=self.config.reward_shaping_mode,
                gamma=float(self.config.gamma),
                duration=1,
                potential_start=potential_start,
                potential_end=potential_end,
                potential_end_unclipped=potential_end_unclipped,
                initial_potential=float(env.training_initial_potential),
                terminal=terminal,
                terminal_mode=self.config.pbrs_terminal_mode,
            )
            interval_action_name = ""
            delay_ticks: int | str = ""
            duration_ticks: int | str = ""
            final_dispatch_executed = False
            if action == MATCH_FULL:
                delay_ticks = wait_streak
                duration_ticks = wait_streak + 1
                interval_action_name = _interval_action_name(wait_streak)
                final_dispatch_executed = True
                wait_streak = 0
            else:
                wait_streak += 1
            trace_row: dict[str, object] = {
                "start_tick": start_tick,
                "end_tick": env.current_tick,
                "binary_action": action,
                "binary_action_name": BINARY_ACTION_NAMES[action],
                "interval_action": "",
                "interval_action_name": interval_action_name,
                "delay_ticks": delay_ticks,
                "duration_ticks": duration_ticks,
                "final_dispatch_executed": final_dispatch_executed,
                "current_wait_streak": wait_streak,
                "reward": shaped.reward,
                "raw_reward": raw_reward,
                "legacy_reward": legacy_reward,
                "shaped_reward": shaped.reward,
                "reward_shaping_mode": self.config.reward_shaping_mode,
                "potential_start": shaped.potential_start,
                "potential_end": shaped.potential_end,
                "potential_end_unclipped": shaped.potential_end_unclipped,
                "pbrs_delta": shaped.pbrs_delta,
                "finite_horizon_terminal_correction": shaped.terminal_correction,
                "match_probability": probs[MATCH_FULL],
                "wait_probability": probs[WAIT],
                "value_estimate": value,
                "log_prob": log_prob,
                "entropy": entropy,
            }
            trace_row.update(features)
            trace_rows.append(trace_row)
            steps.append(
                {
                    "obs": obs.astype(np.float32, copy=True),
                    "action": int(action),
                    "old_log_prob": float(log_prob),
                    "value": float(value),
                    "reward": float(shaped.reward),
                    "done": bool(terminal),
                }
            )
            shaped_return += shaped.reward
            raw_return += raw_reward
            legacy_return += legacy_reward
            potential_delta_sum += shaped.pbrs_delta
            obs = next_obs
        metrics = env.episode_metrics(policy_name=self.name, seed=seed)
        action_counts = {
            "wait": sum(1 for row in trace_rows if row.get("binary_action") == WAIT),
            "match_full": sum(1 for row in trace_rows if row.get("binary_action") == MATCH_FULL),
        }
        return {
            "episode": episode,
            "seed": seed,
            "steps": steps,
            "trace": trace_rows,
            "metrics": metrics,
            "action_counts": action_counts,
            "interval_stats": _interval_stats(trace_rows),
            "shaped_return": shaped_return,
            "raw_return": raw_return,
            "legacy_return": legacy_return,
            "potential_delta_sum": potential_delta_sum,
            "step_count": len(steps),
        }

    def _update_from_rollouts(self, rollouts: list[dict[str, object]]) -> dict[str, float]:
        steps_by_episode = [list(result["steps"]) for result in rollouts]
        flat_steps: list[dict[str, object]] = []
        advantages: list[float] = []
        returns: list[float] = []
        for steps in steps_by_episode:
            episode_advantages, episode_returns = self._gae_for_episode(steps)
            advantages.extend(episode_advantages)
            returns.extend(episode_returns)
            flat_steps.extend(steps)
        if not flat_steps:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        obs = torch.as_tensor(
            np.stack([step["obs"] for step in flat_steps]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor([step["action"] for step in flat_steps], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(
            [step["old_log_prob"] for step in flat_steps],
            dtype=torch.float32,
            device=self.device,
        )
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        if advantages_tensor.numel() > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        batch_size = max(1, min(int(self.config.batch_size), len(flat_steps)))
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        indices = np.arange(len(flat_steps))
        for _ in range(max(1, int(self.config.ppo_epochs))):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                idx = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long, device=self.device)
                logits, values = self.model(obs.index_select(0, idx))
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions.index_select(0, idx))
                ratio = torch.exp(log_probs - old_log_probs.index_select(0, idx))
                batch_adv = advantages_tensor.index_select(0, idx)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - float(self.config.ppo_clip_ratio),
                    1.0 + float(self.config.ppo_clip_ratio),
                )
                policy_loss = -torch.min(ratio * batch_adv, clipped_ratio * batch_adv).mean()
                value_loss = torch.nn.functional.mse_loss(values, returns_tensor.index_select(0, idx))
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss
                    + float(self.config.ppo_value_loss_coef) * value_loss
                    - float(self.config.ppo_entropy_coef) * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.config.ppo_max_grad_norm))
                self.optimizer.step()
                losses.append(float(loss.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
        return {
            "loss": _mean(losses),
            "policy_loss": _mean(policy_losses),
            "value_loss": _mean(value_losses),
            "entropy": _mean(entropies),
        }

    def _gae_for_episode(self, steps: list[dict[str, object]]) -> tuple[list[float], list[float]]:
        advantages = [0.0 for _ in steps]
        returns = [0.0 for _ in steps]
        gae = 0.0
        for idx in reversed(range(len(steps))):
            reward = float(steps[idx]["reward"])
            value = float(steps[idx]["value"])
            done = bool(steps[idx]["done"])
            next_value = 0.0 if idx == len(steps) - 1 else float(steps[idx + 1]["value"])
            nonterminal = 0.0 if done else 1.0
            delta = reward + float(self.config.gamma) * next_value * nonterminal - value
            gae = delta + float(self.config.gamma) * float(self.config.ppo_gae_lambda) * nonterminal * gae
            advantages[idx] = gae
            returns[idx] = gae + value
        return advantages, returns

    def _teacher_warm_start(self, env_factory: Callable[[], FutureV2VTimingEnv], seed_start: int) -> None:
        episodes = max(0, int(self.config.teacher_prefill_episodes))
        if episodes <= 0 or int(self.config.teacher_imitation_epochs) <= 0:
            return
        observations: list[np.ndarray] = []
        actions: list[int] = []
        weights: list[float] = []
        for episode in progress(range(episodes), desc="PPO teacher warm start", total=episodes, unit="episode"):
            env = env_factory()
            obs, _ = env.reset(seed=seed_start + episode)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                snapshot = env.snapshot()
                action, weight = binary_teacher_action_with_weight(env)
                observations.append(obs.astype(np.float32, copy=True))
                actions.append(action)
                weights.append(weight)
                env.set_action_q_values(None)
                obs, _reward, terminated, truncated, _info = env.step(action)
                if not snapshot.active_orders and env.current_tick > env.scale_config.horizon_ticks:
                    break
        if not observations:
            return
        obs_tensor = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        weight_tensor = torch.as_tensor(_balanced_teacher_weights(actions, weights), dtype=torch.float32, device=self.device)
        batch_size = max(1, min(int(self.config.teacher_imitation_batch_size), len(actions)))
        indices = np.arange(len(actions))
        for _ in range(max(1, int(self.config.teacher_imitation_epochs))):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                idx = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long, device=self.device)
                logits, _value = self.model(obs_tensor.index_select(0, idx))
                losses = torch.nn.functional.cross_entropy(
                    logits,
                    action_tensor.index_select(0, idx),
                    reduction="none",
                )
                loss = (losses * weight_tensor.index_select(0, idx)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.config.ppo_max_grad_norm))
                self.optimizer.step()

    def run_eval_episode(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> list[dict[str, object]]:
        result = self._collect_episode(
            env,
            seed=env.seed,
            episode=0,
            deterministic=bool(self.config.ppo_eval_deterministic),
            reset=False,
            initial_obs=obs,
            action_seed=env.seed + 910_000,
        )
        self.interval_trace = list(result["trace"])
        return self.interval_trace

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "model": self.model.state_dict(),
                "config": self.config.__dict__,
                "agent_type": "ppo",
                "action_count": BINARY_ACTION_COUNT,
                "observation_profile": self.config.observation_profile,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, training_config: TrainingConfig, device: str | None = None) -> AdaptiveTimingPPOAgent:
        payload = torch.load(path, map_location=device or "cpu")
        payload_profile = payload.get("observation_profile")
        if payload_profile and payload_profile != training_config.observation_profile:
            raise ValueError(
                "checkpoint observation_profile mismatch: "
                f"checkpoint={payload_profile!r}, current={training_config.observation_profile!r}"
            )
        agent = cls(obs_dim=int(payload["obs_dim"]), training_config=training_config, device=device)
        agent.model.load_state_dict(payload["model"])
        return agent

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
        scores: list[float] = []
        profits: list[float] = []
        dispatch_counts: list[float] = []
        bucket_scores: dict[str, list[float]] = {}
        trace_rows: list[dict[str, object]] = []
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
            trace_rows.extend(self.run_eval_episode(env, obs))
            metrics = env.episode_metrics(policy_name=self.name, seed=seed_start + 50_000 + idx)
            scores.append(metrics.future_v2v_score)
            profits.append(metrics.platform_profit)
            dispatch_counts.append(metrics.dispatch_epoch_count)
            bucket = str(scenario.get("time_of_day_bucket", "unknown"))
            bucket_scores.setdefault(bucket, []).append(metrics.future_v2v_score)
        bucket_means = {bucket: float(np.mean(values)) for bucket, values in bucket_scores.items() if values}
        action_stats = _binary_action_stats(trace_rows)
        interval_stats = _interval_stats(trace_rows)
        row = {
            "episode": episode,
            "future_v2v_score_mean": float(np.mean(scores)) if scores else 0.0,
            "platform_profit_mean": float(np.mean(profits)) if profits else 0.0,
            "dispatch_epoch_count_mean": float(np.mean(dispatch_counts)) if dispatch_counts else 0.0,
            "validation_bucket_min_score_mean": float(min(bucket_means.values())) if bucket_means else 0.0,
            "validation_wait_action_share": action_stats["wait_share"],
            "validation_match_full_action_share": action_stats["match_full_share"],
            "validation_binary_max_action_share": action_stats["max_action_share"],
            "validation_mean_interval": interval_stats["mean_interval"],
            "validation_interval_max_share": interval_stats["max_interval_share"],
        }
        for bucket in self.config.validation_time_buckets:
            row[f"validation_{bucket}_score_mean"] = bucket_means.get(bucket, 0.0)
        mean_score = float(row["future_v2v_score_mean"])
        worst_bucket_score = float(row["validation_bucket_min_score_mean"])
        off_peak_score = float(row.get("validation_off_peak_score_mean", 0.0))
        checkpoint_selection_score, off_peak_floor_penalty, action_penalty = self._checkpoint_selection_score(
            mean_score=mean_score,
            worst_bucket_score=worst_bucket_score,
            off_peak_score=off_peak_score,
            binary_max_action_share=float(action_stats["max_action_share"]),
            interval_max_share=float(interval_stats["max_interval_share"]),
            mean_interval=float(interval_stats["mean_interval"]),
        )
        row["validation_off_peak_floor_penalty"] = off_peak_floor_penalty
        row["validation_action_distribution_penalty"] = action_penalty
        row["checkpoint_selection_score"] = checkpoint_selection_score
        self.validation_history.append(row)
        if checkpoint_selection_score > self.best_validation_score:
            self.best_validation_score = checkpoint_selection_score
            self.best_state_dict = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

    def _checkpoint_selection_score(
        self,
        *,
        mean_score: float,
        worst_bucket_score: float,
        off_peak_score: float,
        binary_max_action_share: float = 0.0,
        interval_max_share: float = 0.0,
        mean_interval: float = 0.0,
    ) -> tuple[float, float, float]:
        worst_weight = float(np.clip(self.config.validation_worst_bucket_weight, 0.0, 1.0))
        off_peak_floor_penalty = max(0.0, float(self.config.validation_off_peak_score_floor) - off_peak_score)
        action_penalty = 0.0
        if binary_max_action_share > float(self.config.validation_action_max_share_cap):
            action_penalty += (
                binary_max_action_share - float(self.config.validation_action_max_share_cap)
            ) * float(self.config.validation_action_balance_penalty)
        if interval_max_share > float(self.config.validation_interval_max_share_cap):
            action_penalty += (
                interval_max_share - float(self.config.validation_interval_max_share_cap)
            ) * float(self.config.validation_action_balance_penalty)
        if mean_interval and mean_interval < float(self.config.validation_min_mean_interval):
            action_penalty += (
                float(self.config.validation_min_mean_interval) - mean_interval
            ) * float(self.config.validation_action_balance_penalty)
        if mean_interval > float(self.config.validation_max_mean_interval):
            action_penalty += (
                mean_interval - float(self.config.validation_max_mean_interval)
            ) * float(self.config.validation_action_balance_penalty)
        score = (1.0 - worst_weight) * mean_score + worst_weight * worst_bucket_score
        score -= off_peak_floor_penalty + action_penalty
        return float(score), float(off_peak_floor_penalty), float(action_penalty)

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
        self.model.load_state_dict(self.best_state_dict)


def binary_teacher_action_with_weight(env: FutureV2VTimingEnv) -> tuple[int, float]:
    snapshot = env.snapshot()
    if not snapshot.active_orders:
        return WAIT, 1.0
    waiting_ratios = [order.waiting_ratio(env.current_tick) for order in snapshot.active_orders]
    near_deadline_share = sum(
        1
        for order in snapshot.active_orders
        if order.max_wait_ticks - order.waiting_ticks(env.current_tick) <= 1
    ) / max(1, len(snapshot.active_orders))
    max_wait = max(waiting_ratios) if waiting_ratios else 0.0
    flex_values = [vehicle.time_flexibility_ticks(env.current_tick) for vehicle in snapshot.active_vehicles]
    mean_flex = float(np.mean(flex_values)) if flex_values else 0.0
    if near_deadline_share >= 0.12 or max_wait >= 0.82 or mean_flex <= 3.0:
        return MATCH_FULL, 2.5
    strong_rule_action = HandcraftedDeadlineRulePolicy().act(env, env._observation())
    if strong_rule_action == MATCH_FULL:
        return MATCH_FULL, 1.6
    opportunity = env.estimate_wait_opportunity(snapshot)
    recently_dispatched = bool(env.dispatch_ticks and env.current_tick - env.dispatch_ticks[-1] <= 1)
    pressure = len(snapshot.active_orders) / max(1, len(snapshot.active_vehicles))
    if recently_dispatched and opportunity >= -20.0 and near_deadline_share < 0.10:
        return WAIT, 1.5
    if opportunity >= 60.0 and pressure <= 0.85 and near_deadline_share < 0.10:
        return WAIT, 1.4
    if pressure >= 0.95 or near_deadline_share >= 0.08:
        return MATCH_FULL, 1.3
    return WAIT, 1.0


def _balanced_teacher_weights(actions: list[int], weights: list[float]) -> list[float]:
    counts = {WAIT: max(1, actions.count(WAIT)), MATCH_FULL: max(1, actions.count(MATCH_FULL))}
    total = max(1, len(actions))
    class_weights = {
        WAIT: total / (2.0 * counts[WAIT]),
        MATCH_FULL: total / (2.0 * counts[MATCH_FULL]),
    }
    return [float(weight) * class_weights[action] for action, weight in zip(actions, weights)]


def _interval_action_name(delay_ticks: int) -> str:
    if delay_ticks <= 0:
        return "dispatch_now"
    if delay_ticks <= 3:
        return f"delay_{delay_ticks}_then_dispatch"
    return "delay_4_plus_then_dispatch"


def _interval_stats(trace_rows: list[dict[str, object]]) -> dict[str, float | int]:
    delays = [
        int(float(row.get("delay_ticks", 0)))
        for row in trace_rows
        if row.get("final_dispatch_executed") is True and row.get("delay_ticks") != ""
    ]
    counts = {
        "dispatch_now_count": sum(1 for delay in delays if delay == 0),
        "delay_1_count": sum(1 for delay in delays if delay == 1),
        "delay_2_count": sum(1 for delay in delays if delay == 2),
        "delay_3_count": sum(1 for delay in delays if delay == 3),
        "delay_4_plus_count": sum(1 for delay in delays if delay >= 4),
    }
    total = max(1, len(delays))
    max_share = max(counts.values(), default=0) / total if delays else 0.0
    mean_interval = float(np.mean([delay + 1 for delay in delays])) if delays else 0.0
    counts["max_interval_share"] = float(max_share)
    counts["mean_interval"] = float(mean_interval)
    return counts


def _binary_action_stats(trace_rows: list[dict[str, object]]) -> dict[str, float]:
    wait_count = sum(1 for row in trace_rows if row.get("binary_action") == WAIT)
    match_count = sum(1 for row in trace_rows if row.get("binary_action") == MATCH_FULL)
    total = max(1, wait_count + match_count)
    return {
        "wait_share": wait_count / total,
        "match_full_share": match_count / total,
        "max_action_share": max(wait_count, match_count) / total,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _run_ppo_rollout_task(
    env_config: EnvironmentConfig,
    scale_config: ScaleConfig,
    training_config: TrainingConfig,
    state_dict: dict[str, torch.Tensor],
    obs_dim: int,
    seed: int,
    episode: int,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    env = FutureV2VTimingEnv(env_config, scale_config, seed=seed)
    agent = AdaptiveTimingPPOAgent(obs_dim=obs_dim, training_config=training_config, device="cpu")
    agent.model.load_state_dict(state_dict)
    return agent._collect_episode(env, seed=seed, episode=episode, deterministic=False)
