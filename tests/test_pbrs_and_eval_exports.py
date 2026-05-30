from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

from future_v2v.algorithms.interval_dqn import AdaptiveIntervalDQNAgent, compute_pbrs_potential, execute_interval_action
from future_v2v.config import TrainingConfig
from tests.test_env_semantics import install_single_order_vehicle, make_env


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_experiment", SCRIPT_PATH)
assert SPEC is not None
run_experiment = importlib.util.module_from_spec(SPEC)
sys.modules["run_experiment"] = run_experiment
assert SPEC.loader is not None
SPEC.loader.exec_module(run_experiment)


def make_training_config(**overrides) -> TrainingConfig:
    data = {
        "gamma": 0.98,
        "learning_rate": 0.001,
        "batch_size": 4,
        "replay_capacity": 100,
        "min_replay_size": 4,
        "target_update_interval": 10,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 100,
        "teacher_prefill_episodes": 0,
        "validation_episodes": 0,
        "validation_interval_episodes": 1,
        "checkpoint_selection_metric": "future_v2v_score_mean",
        "hidden_dim": 16,
        "double_dqn": True,
        "prioritized_replay": False,
    }
    data.update(overrides)
    return TrainingConfig(**data)


def test_pbrs_zero_potential_equals_raw_reward() -> None:
    env = make_env()
    install_single_order_vehicle(env, max_wait_ticks=3)
    training = make_training_config(
        reward_shaping_mode="pbrs",
        potential_candidate_margin_weight=0.0,
        potential_feasible_density_weight=0.0,
        potential_urgent_coverage_weight=0.0,
        potential_pickup_time_weight=0.0,
        potential_pickup_distance_weight=0.0,
        potential_service_risk_weight=0.0,
        potential_urgent_service_risk_weight=0.0,
        potential_near_deadline_weight=0.0,
        potential_soc_binding_weight=0.0,
    )
    _obs, reward, _terminated, _truncated, _duration, trace = execute_interval_action(
        env,
        0,
        training_config=training,
    )
    assert reward == trace["raw_reward"]
    assert trace["pbrs_delta"] == 0.0


def test_pbrs_uses_duration_discounted_potential_delta() -> None:
    env = make_env()
    install_single_order_vehicle(env, max_wait_ticks=4)
    training = make_training_config(reward_shaping_mode="pbrs", pbrs_clip=0.0)
    start_potential = compute_pbrs_potential(env, training)
    _obs, reward, terminated, truncated, duration, trace = execute_interval_action(
        env,
        1,
        training_config=training,
    )
    end_potential = 0.0 if terminated or truncated else compute_pbrs_potential(env, training)
    expected_delta = training.gamma**duration * end_potential - start_potential
    assert abs(float(trace["pbrs_delta"]) - expected_delta) < 1e-6
    assert abs(reward - (float(trace["raw_reward"]) + expected_delta)) < 1e-6


def test_checkpoint_rejects_observation_profile_mismatch(tmp_path: Path) -> None:
    compact = make_training_config(observation_profile="compact_v2v")
    legacy = replace(compact, observation_profile="legacy_full")
    agent = AdaptiveIntervalDQNAgent(obs_dim=3, training_config=compact, device="cpu")
    path = tmp_path / "agent.pt"
    agent.save(path)
    try:
        AdaptiveIntervalDQNAgent.load(path, legacy, device="cpu")
    except ValueError as exc:
        assert "observation_profile mismatch" in str(exc)
    else:
        raise AssertionError("expected observation profile mismatch")


def test_state_action_bucket_summary_counts_actions() -> None:
    rows = [
        {
            "policy_name": "adaptive_interval_dqn",
            "interval_action_name": "dispatch_now",
            "active_orders": 20,
            "available_vehicles": 30,
            "supply_demand_ratio": 1.5,
            "near_deadline_share": 0.05,
            "candidate_density": 0.04,
            "raw_reward": 10.0,
            "shaped_reward": 12.0,
        },
        {
            "policy_name": "adaptive_interval_dqn",
            "interval_action_name": "delay_1_then_dispatch",
            "active_orders": 20,
            "available_vehicles": 30,
            "supply_demand_ratio": 1.5,
            "near_deadline_share": 0.05,
            "candidate_density": 0.04,
            "raw_reward": 20.0,
            "shaped_reward": 22.0,
        },
    ]
    out = run_experiment._state_action_bucket_summary(rows)
    assert out[0]["sample_count"] == 2
    assert out[0]["dispatch_now_share"] == 0.5
    assert out[0]["delay_1_share"] == 0.5


def test_paired_delta_summary_includes_standard_error() -> None:
    env = make_env()
    metric_a = env.episode_metrics(policy_name="fixed_1_tick_full_match", seed=1)
    metric_b = replace(metric_a, policy_name="adaptive_interval_dqn", future_v2v_score=metric_a.future_v2v_score + 10)
    rows = run_experiment._paired_policy_delta_summary([metric_a, metric_b])
    assert "score_delta_sem" in rows[0]
    assert "profit_delta_sem" in rows[0]
