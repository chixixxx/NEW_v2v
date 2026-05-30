from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from future_v2v.config import TrainingConfig
from future_v2v.envs.timing_env import FutureV2VTimingEnv


@dataclass(frozen=True)
class RewardShapingResult:
    reward: float
    pbrs_delta: float
    potential_start: float
    potential_end: float
    potential_end_unclipped: float
    terminal_correction: float


def compute_pbrs_potential(
    env: FutureV2VTimingEnv,
    training_config: TrainingConfig | None,
    snapshot=None,
    *,
    clipped: bool = True,
) -> float:
    if training_config is None:
        return 0.0
    values = env._state_feature_values(snapshot)
    potential = (
        training_config.potential_candidate_margin_weight * values["candidate_margin_proxy_raw"]
        + training_config.potential_feasible_density_weight * values["feasible_edge_density_raw"]
        + training_config.potential_urgent_coverage_weight * values["urgent_feasible_coverage"]
        - training_config.potential_pickup_time_weight * values["mean_pickup_time_est_raw"]
        - training_config.potential_pickup_distance_weight * values["mean_pickup_distance_est_raw"]
        - training_config.potential_service_risk_weight * values["service_risk_potential_raw"]
        - training_config.potential_urgent_service_risk_weight * values["projected_urgent_service_risk"]
        - training_config.potential_near_deadline_weight * values["near_deadline_order_share_raw"]
        - training_config.potential_soc_binding_weight * values["soc_safety_binding_rate"]
    )
    if clipped and training_config.pbrs_clip > 0:
        potential = float(np.clip(potential, -training_config.pbrs_clip, training_config.pbrs_clip))
    return float(potential)


def shape_reward(
    *,
    raw_reward: float,
    legacy_reward: float,
    reward_mode: str,
    gamma: float,
    duration: int,
    potential_start: float,
    potential_end: float,
    potential_end_unclipped: float,
    initial_potential: float,
    terminal: bool,
    terminal_mode: str,
) -> RewardShapingResult:
    discount = float(gamma) ** max(1, int(duration))
    terminal_correction = 0.0
    if reward_mode == "none":
        return RewardShapingResult(
            reward=float(raw_reward),
            pbrs_delta=0.0,
            potential_start=float(potential_start),
            potential_end=float(potential_end),
            potential_end_unclipped=float(potential_end_unclipped),
            terminal_correction=0.0,
        )
    if reward_mode == "legacy_delta":
        return RewardShapingResult(
            reward=float(legacy_reward),
            pbrs_delta=0.0,
            potential_start=float(potential_start),
            potential_end=float(potential_end),
            potential_end_unclipped=float(potential_end_unclipped),
            terminal_correction=0.0,
        )
    if reward_mode != "pbrs":
        raise ValueError(f"unknown reward_shaping_mode={reward_mode!r}; expected none, legacy_delta, or pbrs")

    if terminal_mode == "finite_horizon_correction":
        if terminal:
            pbrs_delta = float(initial_potential - potential_start)
            terminal_correction = float(initial_potential)
        else:
            pbrs_delta = float(potential_end - potential_start)
    elif terminal:
        end_for_delta = 0.0 if terminal_mode == "zero_terminal_with_diagnostic" else potential_end
        pbrs_delta = float(discount * end_for_delta - potential_start)
    else:
        pbrs_delta = float(discount * potential_end - potential_start)
    return RewardShapingResult(
        reward=float(raw_reward + pbrs_delta),
        pbrs_delta=pbrs_delta,
        potential_start=float(potential_start),
        potential_end=float(potential_end),
        potential_end_unclipped=float(potential_end_unclipped),
        terminal_correction=terminal_correction,
    )
