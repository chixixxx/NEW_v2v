from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from future_v2v.envs.timing_env import MATCH_FULL, WAIT, FutureV2VTimingEnv
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
        return f"fixed_{self.interval}_tick_full_match"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        return MATCH_FULL if env.current_tick % self.interval == 0 else WAIT


@dataclass
class HandcraftedObservableRulePolicy:
    slack_threshold: int = 1
    urgent_full_share: float = 0.18
    high_wait_ratio: float = 0.86
    min_profit_per_order: float = 5.0
    include_future_orders: bool = False

    name: str = "handcrafted_observable_rule"

    def act(self, env: FutureV2VTimingEnv, obs: np.ndarray) -> int:
        _ = obs
        snapshot = env.snapshot()
        if not snapshot.active_orders:
            return WAIT
        waiting_ratios = [order.waiting_ratio(env.current_tick) for order in snapshot.active_orders]
        near_deadline = [
            order
            for order in snapshot.active_orders
            if order.max_wait_ticks - order.waiting_ticks(env.current_tick) <= self.slack_threshold
        ]
        near_deadline_share = len(near_deadline) / max(1, len(snapshot.active_orders))
        if near_deadline_share >= self.urgent_full_share or max(waiting_ratios) >= self.high_wait_ratio:
            return MATCH_FULL
        recently_dispatched = bool(env.dispatch_ticks and env.current_tick - env.dispatch_ticks[-1] <= 1)
        wait_opportunity = env.estimate_wait_opportunity(
            snapshot,
            include_future_orders=self.include_future_orders,
        )
        if recently_dispatched and wait_opportunity >= 0.0 and near_deadline_share < 0.08:
            return WAIT
        full_plan = env.matcher.solve(env.orders, env.vehicles, env.current_tick)
        expected_profit = sum(match.expected_profit for match in full_plan.matches)
        profit_per_order = expected_profit / max(1, len(snapshot.active_orders))
        pressure = len(snapshot.active_orders) / max(1, len(snapshot.active_vehicles))
        if profit_per_order >= self.min_profit_per_order and (pressure >= 0.50 or near_deadline_share >= 0.08):
            return MATCH_FULL
        return WAIT


@dataclass
class HandcraftedLookaheadRulePolicy(HandcraftedObservableRulePolicy):
    include_future_orders: bool = True
    name: str = "handcrafted_lookahead_rule"


@dataclass
class HandcraftedDeadlineRulePolicy(HandcraftedLookaheadRulePolicy):
    name: str = "handcrafted_deadline_rule"


def default_baselines() -> list[TimingPolicy]:
    return [
        FixedIntervalPolicy(interval=1),
        FixedIntervalPolicy(interval=2),
        FixedIntervalPolicy(interval=3),
        FixedIntervalPolicy(interval=4),
        HandcraftedObservableRulePolicy(),
        HandcraftedLookaheadRulePolicy(),
    ]


def policy_from_name(name: str) -> TimingPolicy:
    if name == "handcrafted_deadline_rule":
        return HandcraftedDeadlineRulePolicy()
    for policy in default_baselines():
        if policy.name == name:
            return policy
    raise KeyError(f"unknown timing baseline policy: {name}")


def run_policy_episode(env: FutureV2VTimingEnv, policy: TimingPolicy, seed: int) -> EpisodeMetrics:
    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy.act(env, obs)
        env.set_action_q_values(getattr(policy, "last_q_values", None))
        obs, _reward, terminated, truncated, _info = env.step(action)
    return env.episode_metrics(policy_name=policy.name, seed=seed)
