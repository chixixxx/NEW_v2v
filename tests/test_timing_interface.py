from __future__ import annotations

import numpy as np

from future_v2v.algorithms.baselines import default_baselines, teacher_policies
from future_v2v.envs.timing_env import MATCH, WAIT
from tests.test_env_semantics import install_single_order_vehicle, make_env


def test_teacher_and_baselines_use_binary_timing_actions() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    obs = env._observation()
    for policy in [*teacher_policies(), *default_baselines()]:
        action = policy.act(env, obs)
        assert action in (WAIT, MATCH)


def test_observation_is_fixed_low_dimensional_vector() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    obs = env._observation()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (env.observation_dim,)

