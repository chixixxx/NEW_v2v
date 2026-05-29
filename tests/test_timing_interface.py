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
