from __future__ import annotations

import numpy as np

from future_v2v.algorithms.baselines import QueueThresholdPolicy, default_baselines, teacher_policies
from future_v2v.envs.timing_env import ACTION_COUNT, MATCH_FULL, MATCH_TOP_BATCH, WAIT
from tests.test_env_semantics import install_single_order_vehicle, make_env


def test_teacher_and_baselines_use_three_action_timing_interface() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    obs = env._observation()
    for policy in [*teacher_policies(), *default_baselines()]:
        action = policy.act(env, obs)
        assert action in (WAIT, MATCH_TOP_BATCH, MATCH_FULL)


def test_observation_is_fixed_low_dimensional_vector() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    obs = env._observation()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (env.observation_dim,)


def test_dqn_action_count_is_three() -> None:
    assert ACTION_COUNT == 3


def test_queue_threshold_uses_scale_aware_ratio() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    policy = QueueThresholdPolicy(threshold_ratio=1.0, threshold_min=4)
    assert policy.act(env, env._observation()) == WAIT


def test_tlc_wait_windows_convert_to_three_minute_ticks() -> None:
    from future_v2v.simulation.tlc_generator import TLCManhattanScenarioGenerator

    env = make_env()
    rng = np.random.default_rng(7)
    generator = TLCManhattanScenarioGenerator(env.env_config, env.scale_config, data=None)  # type: ignore[arg-type]
    high_pressure = {generator._sample_wait_ticks(1.6, rng) for _ in range(100)}
    normal_pressure = {generator._sample_wait_ticks(1.0, rng) for _ in range(100)}
    low_pressure = {generator._sample_wait_ticks(0.6, rng) for _ in range(100)}
    assert high_pressure <= {3, 4, 5, 6}
    assert normal_pressure <= {5, 7, 9, 12, 15}
    assert low_pressure <= {9, 12, 15, 18, 21}
