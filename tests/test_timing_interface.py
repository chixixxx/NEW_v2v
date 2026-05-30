from __future__ import annotations

import numpy as np
import pytest

from future_v2v.algorithms.baselines import default_baselines, policy_from_name
from future_v2v.algorithms.interval_dqn import interval_teacher_action
from future_v2v.envs.timing_env import ACTION_COUNT, COMPACT_OBSERVATION_NAMES, MATCH_FULL, WAIT
from tests.test_env_semantics import install_single_order_vehicle, make_env


def test_baselines_use_wait_full_timing_interface() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    obs = env._observation()
    for policy in default_baselines():
        action = policy.act(env, obs)
        assert action in (WAIT, MATCH_FULL)


def test_observation_is_fixed_low_dimensional_vector() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    obs = env._observation()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (env.observation_dim,)
    assert env.observation_names == COMPACT_OBSERVATION_NAMES
    assert "estimated_full_match_friction" not in env.observation_names
    assert "mean_pickup_distance_est" in env.observation_names


def test_environment_action_count_is_wait_and_full_match() -> None:
    assert ACTION_COUNT == 2


def test_default_eval_baselines_are_concise_main_table() -> None:
    names = [policy.name for policy in default_baselines()]
    assert names == [
        "fixed_1_tick_full_match",
        "fixed_2_tick_full_match",
        "fixed_3_tick_full_match",
        "fixed_4_tick_full_match",
        "handcrafted_deadline_rule",
    ]
    assert policy_from_name("fixed_1_tick_full_match").name == "fixed_1_tick_full_match"
    with pytest.raises(KeyError):
        policy_from_name("fixed_1_tick_top_batch")


def test_interval_teacher_uses_immediate_match_for_deadline_rescue() -> None:
    env = make_env()
    install_single_order_vehicle(env, max_wait_ticks=1)
    env.current_tick = 1
    assert interval_teacher_action(env) == 0


def test_tlc_wait_windows_convert_to_three_minute_ticks() -> None:
    from future_v2v.simulation.tlc_generator import TLCManhattanScenarioGenerator

    env = make_env()
    rng = np.random.default_rng(7)
    generator = TLCManhattanScenarioGenerator(env.env_config, env.scale_config, data=None)  # type: ignore[arg-type]
    high_pressure = {generator._sample_wait_ticks(2.45, rng) for _ in range(100)}
    normal_pressure = {generator._sample_wait_ticks(2.05, rng) for _ in range(100)}
    low_pressure = {generator._sample_wait_ticks(1.75, rng) for _ in range(100)}
    assert high_pressure <= {3, 4, 5, 6}
    assert normal_pressure <= {6, 8, 10, 13, 16}
    assert low_pressure <= {12, 15, 18, 21, 25}
