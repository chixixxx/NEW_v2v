from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from future_v2v.envs.timing_env import MATCH_FULL, MATCH_TOP_BATCH, WAIT, FutureV2VTimingEnv
from future_v2v.metrics import EpisodeMetrics


class TimingPolicy(Protocol):
    name: str

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        ...


@dataclass
class FixedIntervalPolicy:
    interval: int
    match_action: int = MATCH_FULL

    @property
    def name(self) -> str:
        suffix = "full_match" if self.match_action == MATCH_FULL else "top_batch"
        return f"fixed_{self.interval}_tick_{suffix}"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        return self.match_action if env.current_tick % self.interval == 0 else WAIT


@dataclass
class QueueThresholdPolicy:
    threshold_ratio: float | None = None
    threshold_min: int | None = None
    match_action: int = MATCH_TOP_BATCH

    name: str = "queue_threshold_top_batch"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        threshold_ratio = self.threshold_ratio if self.threshold_ratio is not None else env.env_config.queue_threshold_ratio
        configured_min = self.threshold_min if self.threshold_min is not None else env.env_config.queue_threshold_min
        scale_min = max(3, int(np.ceil(env.scale_config.total_orders * 0.10)))
        threshold_min = min(int(configured_min), scale_min)
        expected_active = max(1.0, env.scale_config.total_orders / max(1, env.scale_config.horizon_ticks) * 5.2)
        threshold = max(int(threshold_min), int(np.ceil(expected_active * threshold_ratio)))
        return self.match_action if len(env.snapshot().active_orders) >= threshold else WAIT


@dataclass
class DeadlineTriggerPolicy:
    slack_threshold: int = 1
    full_match_wait_ratio: float = 0.98

    name: str = "deadline_trigger_top_batch"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        snapshot = env.snapshot()
        if not snapshot.active_orders:
            return WAIT
        waiting_ratios = [order.waiting_ratio(env.current_tick) for order in snapshot.active_orders]
        near_deadline = [
            order for order in snapshot.active_orders
            if order.max_wait_ticks - order.waiting_ticks(env.current_tick) <= self.slack_threshold
        ]
        near_deadline_share = len(near_deadline) / max(1, len(snapshot.active_orders))
        if max(waiting_ratios) >= self.full_match_wait_ratio or near_deadline_share >= 0.22:
            return MATCH_FULL
        near_deadline_threshold = max(2, int(np.ceil(len(snapshot.active_orders) * 0.06)))
        return MATCH_TOP_BATCH if len(near_deadline) >= near_deadline_threshold else WAIT


@dataclass
class SupplyDemandPressurePolicy:
    pressure_threshold: float = 0.43
    min_orders: int = 8
    min_mean_edge_profit: float = 5.2

    name: str = "supply_demand_pressure_top_batch"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        snapshot = env.snapshot()
        demand = len(snapshot.active_orders)
        supply = len(snapshot.active_vehicles)
        if demand < self.min_orders:
            return WAIT
        pressure = demand / max(1, supply)
        waiting_ratios = [order.waiting_ratio(env.current_tick) for order in snapshot.active_orders]
        deadline_pressure = bool(waiting_ratios and max(waiting_ratios) >= 0.78)
        mean_profit = float(np.mean([edge.expected_profit for edge in snapshot.candidate_edges])) if snapshot.candidate_edges else 0.0
        edge_coverage = len({edge.order_id for edge in snapshot.candidate_edges}) / max(1, demand)
        if deadline_pressure and pressure >= 0.58:
            return MATCH_FULL
        if pressure >= self.pressure_threshold or (edge_coverage >= 0.55 and mean_profit >= self.min_mean_edge_profit):
            return MATCH_TOP_BATCH
        return WAIT


@dataclass
class ShortLookaheadTimingPolicy:
    min_profit_gain: float = 8.0
    max_safe_wait_ratio: float = 0.72

    name: str = "short_lookahead_top_batch"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        snapshot = env.snapshot()
        if not snapshot.active_orders:
            return WAIT
        waiting_ratios = [order.waiting_ratio(env.current_tick) for order in snapshot.active_orders]
        if max(waiting_ratios) >= self.max_safe_wait_ratio:
            return MATCH_TOP_BATCH
        current_profit = sum(
            edge.expected_profit
            for edge in env.matcher.solve(
                env.orders,
                env.vehicles,
                env.current_tick,
                dispatch_mode="top_batch",
                capacity=env._dispatch_capacity(snapshot),
            ).matches
        )
        future_orders = [order for order in env.orders if order.arrival_tick == env.current_tick + 1]
        pressure = len(snapshot.active_orders) / max(1, len(snapshot.active_vehicles))
        if not future_orders:
            return MATCH_TOP_BATCH if current_profit > self.min_profit_gain and pressure >= 0.55 else WAIT
        future_energy = sum(order.demand_kwh for order in future_orders)
        current_energy = sum(order.demand_kwh for order in snapshot.active_orders)
        approximate_gain = 0.08 * future_energy + 0.03 * current_energy
        if approximate_gain >= self.min_profit_gain and max(waiting_ratios) < 0.62:
            return WAIT
        return MATCH_TOP_BATCH if pressure >= 0.50 or current_profit > self.min_profit_gain * 1.5 else WAIT


def default_baselines() -> list[TimingPolicy]:
    return [
        FixedIntervalPolicy(interval=1, match_action=MATCH_FULL),
        FixedIntervalPolicy(interval=2, match_action=MATCH_FULL),
        FixedIntervalPolicy(interval=1, match_action=MATCH_TOP_BATCH),
        QueueThresholdPolicy(),
        DeadlineTriggerPolicy(),
        SupplyDemandPressurePolicy(),
        ShortLookaheadTimingPolicy(),
    ]


def policy_from_name(name: str) -> TimingPolicy:
    for policy in default_baselines():
        if policy.name == name:
            return policy
    raise KeyError(f"unknown timing baseline policy: {name}")


def teacher_policies() -> list[TimingPolicy]:
    return [
        DeadlineTriggerPolicy(),
        SupplyDemandPressurePolicy(),
        ShortLookaheadTimingPolicy(),
    ]


def run_policy_episode(env: FutureV2VTimingEnv, policy: TimingPolicy, seed: int) -> EpisodeMetrics:
    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy.act(env, obs)
        env.set_action_q_values(getattr(policy, "last_q_values", None))
        obs, _reward, terminated, truncated, _info = env.step(action)
    return env.episode_metrics(policy_name=policy.name, seed=seed)
