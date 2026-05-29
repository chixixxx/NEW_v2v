from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from future_v2v.envs.timing_env import MATCH, WAIT, FutureV2VTimingEnv
from future_v2v.metrics import EpisodeMetrics


class TimingPolicy(Protocol):
    name: str

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        ...


@dataclass
class FixedIntervalPolicy:
    interval: int

    @property
    def name(self) -> str:
        return f"fixed_{self.interval}_tick_match"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        return MATCH if env.current_tick % self.interval == 0 else WAIT


@dataclass
class QueueThresholdPolicy:
    threshold: int = 8

    name: str = "queue_threshold_match"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        return MATCH if len(env.snapshot().active_orders) >= self.threshold else WAIT


@dataclass
class DeadlineTriggerPolicy:
    slack_threshold: int = 1

    name: str = "deadline_trigger_match"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        for order in env.snapshot().active_orders:
            if order.max_wait_ticks - order.waiting_ticks(env.current_tick) <= self.slack_threshold:
                return MATCH
        return WAIT


@dataclass
class SupplyDemandPressurePolicy:
    pressure_threshold: float = 1.08
    min_orders: int = 4

    name: str = "supply_demand_pressure_match"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        snapshot = env.snapshot()
        demand = len(snapshot.active_orders)
        supply = len(snapshot.active_vehicles)
        if demand < self.min_orders:
            return WAIT
        pressure = demand / max(1, supply)
        return MATCH if pressure >= self.pressure_threshold or len(snapshot.candidate_edges) >= demand else WAIT


@dataclass
class ShortLookaheadTimingPolicy:
    min_profit_gain: float = 8.0
    max_safe_wait_ratio: float = 0.72

    name: str = "short_lookahead_timing"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        snapshot = env.snapshot()
        if not snapshot.active_orders:
            return WAIT
        waiting_ratios = [order.waiting_ratio(env.current_tick) for order in snapshot.active_orders]
        if max(waiting_ratios) >= self.max_safe_wait_ratio:
            return MATCH
        current_profit = sum(edge.expected_profit for edge in env.matcher.solve(env.orders, env.vehicles, env.current_tick).matches)
        future_orders = [order for order in env.orders if order.arrival_tick == env.current_tick + 1]
        if not future_orders:
            return MATCH if current_profit > 0.0 else WAIT
        future_energy = sum(order.demand_kwh for order in future_orders)
        current_energy = sum(order.demand_kwh for order in snapshot.active_orders)
        approximate_gain = 0.08 * future_energy + 0.03 * current_energy
        return WAIT if approximate_gain >= self.min_profit_gain else MATCH


def default_baselines() -> list[TimingPolicy]:
    return [
        FixedIntervalPolicy(interval=1),
        FixedIntervalPolicy(interval=2),
        FixedIntervalPolicy(interval=3),
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
        obs, _reward, terminated, truncated, _info = env.step(action)
    return env.episode_metrics(policy_name=policy.name, seed=seed)
