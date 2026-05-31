from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from future_v2v.algorithms.interval_dqn import AdaptiveIntervalDQNAgent, compute_pbrs_potential, execute_interval_action
from future_v2v.algorithms.ppo import AdaptiveTimingPPOAgent, BINARY_ACTION_COUNT
from future_v2v.algorithms.reward_shaping import shape_reward
from future_v2v.config import TrainingConfig
from future_v2v.envs.timing_env import MATCH_FULL, WAIT
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


def test_legacy_pbrs_terminal_mode_uses_duration_discounted_potential_delta() -> None:
    env = make_env()
    install_single_order_vehicle(env, max_wait_ticks=4)
    training = make_training_config(
        reward_shaping_mode="pbrs",
        pbrs_clip=0.0,
        pbrs_terminal_mode="zero_terminal_with_diagnostic",
    )
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


def test_pbrs_terminal_correction_uses_initial_potential() -> None:
    shaped = shape_reward(
        raw_reward=10.0,
        legacy_reward=8.0,
        reward_mode="pbrs",
        gamma=0.98,
        duration=1,
        potential_start=4.0,
        potential_end=0.0,
        potential_end_unclipped=7.0,
        initial_potential=11.0,
        terminal=True,
        terminal_mode="finite_horizon_correction",
    )
    assert shaped.reward == 17.0
    assert shaped.pbrs_delta == 7.0
    assert shaped.terminal_correction == 11.0


def test_finite_horizon_pbrs_telescopes_to_zero() -> None:
    first = shape_reward(
        raw_reward=1.0,
        legacy_reward=1.0,
        reward_mode="pbrs",
        gamma=0.98,
        duration=1,
        potential_start=10.0,
        potential_end=14.0,
        potential_end_unclipped=14.0,
        initial_potential=10.0,
        terminal=False,
        terminal_mode="finite_horizon_correction",
    )
    second = shape_reward(
        raw_reward=2.0,
        legacy_reward=2.0,
        reward_mode="pbrs",
        gamma=0.98,
        duration=1,
        potential_start=14.0,
        potential_end=0.0,
        potential_end_unclipped=9.0,
        initial_potential=10.0,
        terminal=True,
        terminal_mode="finite_horizon_correction",
    )
    assert first.pbrs_delta + second.pbrs_delta == 0.0
    assert first.reward + second.reward == 3.0


def test_ppo_actor_critic_outputs_binary_actions() -> None:
    training = make_training_config(agent_type="ppo")
    agent = AdaptiveTimingPPOAgent(obs_dim=5, training_config=training, device="cpu")
    obs = np.zeros(5, dtype=np.float32)
    action, log_prob, value, probs, entropy = agent.act(obs, deterministic=True)
    assert action in (WAIT, MATCH_FULL)
    assert len(probs) == BINARY_ACTION_COUNT
    assert abs(sum(probs) - 1.0) < 1e-6
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert entropy >= 0.0


def test_ppo_checkpoint_selection_penalizes_action_collapse() -> None:
    training = make_training_config(
        agent_type="ppo",
        validation_action_balance_penalty=100.0,
        validation_action_max_share_cap=0.85,
        validation_interval_max_share_cap=0.75,
        validation_min_mean_interval=1.15,
        validation_max_mean_interval=2.45,
    )
    agent = AdaptiveTimingPPOAgent(obs_dim=3, training_config=training, device="cpu")
    score, off_peak_penalty, action_penalty = agent._checkpoint_selection_score(
        mean_score=100.0,
        worst_bucket_score=80.0,
        off_peak_score=0.0,
        binary_max_action_share=1.0,
        interval_max_share=1.0,
        mean_interval=1.0,
    )
    assert off_peak_penalty == 0.0
    assert action_penalty > 0.0
    assert score < 97.0


def test_ppo_gae_uses_reward_scale_for_value_targets() -> None:
    training = make_training_config(agent_type="ppo", ppo_reward_scale=1000.0)
    agent = AdaptiveTimingPPOAgent(obs_dim=3, training_config=training, device="cpu")
    advantages, returns = agent._gae_for_episode(
        [
            {
                "reward": 1000.0,
                "value": 0.0,
                "done": True,
            }
        ]
    )
    assert advantages == [1.0]
    assert returns == [1.0]


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


def test_eval_summary_zh_csv_uses_utf8_bom(tmp_path) -> None:
    from future_v2v.metrics import write_csv

    path = tmp_path / "eval_summary_zh.csv"
    write_csv(path, [{"策略名称": "自适应时机PPO"}], encoding="utf-8-sig")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
