from __future__ import annotations

from future_v2v.config import DispatchFrictionConfig, EnvironmentConfig, ScaleConfig
from future_v2v.algorithms.interval_dqn import execute_interval_action
from future_v2v.envs.timing_env import LEGACY_OBSERVATION_NAMES, MATCH_FULL, WAIT, FutureV2VTimingEnv
from future_v2v.simulation.entities import ORDER_EXPIRED, ORDER_MATCHED, Order, Vehicle


def make_env() -> FutureV2VTimingEnv:
    env_config = EnvironmentConfig(
        zone_count=4,
        service_kwh_per_tick=5.0,
        pickup_cap_minutes=30.0,
        platform_pickup_cost_per_min=0.0,
        wait_penalty_per_order_tick=0.0,
        expired_penalty=10.0,
        cancelled_penalty=8.0,
        service_rate_target=0.6,
        urgent_service_rate_target=0.7,
        expired_rate_cap=0.2,
        cancelled_rate_cap=0.1,
        mean_pickup_soft_cap_min=10.0,
        mean_commitment_soft_cap_ticks=7.0,
        profit_scale_fallback=5.0,
        enable_stochastic_acceptance=False,
        enable_stochastic_cancellation=False,
        dispatch_friction=DispatchFrictionConfig(enabled=False),
    )
    scale = ScaleConfig(
        name="unit",
        horizon_ticks=3,
        terminal_buffer_ticks=1,
        total_orders=0,
        candidate_vehicles=0,
        vehicle_join_probability=1.0,
        fleet_probability=0.0,
        train_episodes=1,
        eval_episodes=1,
    )
    env = FutureV2VTimingEnv(env_config, scale, seed=7)
    env.reset(seed=7)
    return env


def install_single_order_vehicle(env: FutureV2VTimingEnv, max_wait_ticks: int = 2) -> None:
    order = Order(
        order_id=1,
        arrival_tick=0,
        origin_zone=0,
        destination_zone=1,
        demand_kwh=5.0,
        max_wait_ticks=max_wait_ticks,
        willingness_to_pay_per_kwh=8.0,
        cancel_sensitivity=0.0,
    )
    vehicle = Vehicle(
        vehicle_id=2,
        join_tick=0,
        leave_tick=10,
        current_zone=0,
        destination_zone=2,
        battery_capacity_kwh=80.0,
        current_soc_kwh=60.0,
        reserve_kwh=15.0,
        energy_cost_per_kwh=0.25,
        service_premium_per_kwh=1.75,
        time_cost_per_min=0.0,
        owner_accept_sensitivity=0.5,
        fleet_flag=False,
    )
    env.orders = [order]
    env.vehicles = [vehicle]
    env.orders_by_id = {order.order_id: order}
    env.vehicles_by_id = {vehicle.vehicle_id: vehicle}
    env.current_tick = 0


def test_wait_does_not_match_and_can_expire_order() -> None:
    env = make_env()
    install_single_order_vehicle(env, max_wait_ticks=0)
    _obs, reward, _terminated, _truncated, info = env.step(WAIT)
    assert env.orders[0].status == ORDER_EXPIRED
    assert info["step_result"].matched_count == 0
    assert reward < 0.0


def test_match_updates_order_vehicle_and_profit() -> None:
    env = make_env()
    install_single_order_vehicle(env, max_wait_ticks=2)
    _obs, reward, _terminated, _truncated, info = env.step(MATCH_FULL)
    assert env.orders[0].status == ORDER_MATCHED
    assert env.orders[0].matched_vehicle_id == 2
    assert env.vehicles[0].served_count == 1
    assert info["step_result"].accepted_count == 1
    assert info["step_result"].dispatch_mode == "match_full"
    assert reward > 0.0
    assert info["step_result"].total_pickup_distance_km == env.orders[0].pickup_distance_km_est
    assert info["step_result"].mean_pickup_distance_km > 0.0


def test_episode_metrics_include_pickup_distance_in_score() -> None:
    baseline = make_env()
    install_single_order_vehicle(baseline, max_wait_ticks=2)
    baseline.step(MATCH_FULL)
    baseline_metrics = baseline.episode_metrics(policy_name="unit", seed=1)

    penalized = make_env()
    penalized.env_config = penalized.env_config.__class__(
        **{
            **penalized.env_config.__dict__,
            "pickup_distance_penalty_per_km": 2.0,
        }
    )
    install_single_order_vehicle(penalized, max_wait_ticks=2)
    penalized.step(MATCH_FULL)
    metrics = penalized.episode_metrics(policy_name="unit", seed=1)
    assert metrics.total_pickup_distance_km == penalized.orders[0].pickup_distance_km_est
    assert metrics.distance_adjusted_score == metrics.future_v2v_score
    assert abs(
        baseline_metrics.future_v2v_score - metrics.future_v2v_score - 2.0 * metrics.total_pickup_distance_km
    ) < 1e-9


def test_interval_action_waits_then_dispatches_full_match() -> None:
    env = make_env()
    install_single_order_vehicle(env, max_wait_ticks=10)
    _obs, reward, terminated, truncated, duration, trace = execute_interval_action(env, 2)
    assert duration == 3
    assert not terminated
    assert not truncated
    assert trace["interval_action_name"] == "delay_2_then_dispatch"
    assert trace["final_dispatch_executed"] is True
    assert env.orders[0].status == ORDER_MATCHED
    assert reward > 0.0


def test_dispatch_friction_is_subtracted_from_match_profit() -> None:
    env = make_env()
    env.env_config = env.env_config.__class__(
        **{
            **env.env_config.__dict__,
            "observation_profile": "legacy_full",
            "dispatch_friction": DispatchFrictionConfig(
                enabled=True,
                setup_cost=7.0,
                pair_coordination_cost=0.0,
                full_mode_extra_pair_cost=0.0,
                refresh_cost=0.0,
            ),
        }
    )
    env.matcher.env_config = env.env_config
    install_single_order_vehicle(env, max_wait_ticks=2)
    _obs, _reward, _terminated, _truncated, info = env.step(MATCH_FULL)
    assert info["step_result"].platform_profit == env.orders[0].realized_profit - 7.0


def test_dispatch_friction_breakdown_is_applied() -> None:
    env = make_env()
    env.env_config = env.env_config.__class__(
        **{
            **env.env_config.__dict__,
            "observation_profile": "legacy_full",
            "dispatch_friction": DispatchFrictionConfig(
                enabled=True,
                setup_cost=3.0,
                pair_coordination_cost=2.0,
                full_mode_extra_pair_cost=5.0,
                refresh_cost=0.0,
            ),
        }
    )
    env.matcher.env_config = env.env_config
    install_single_order_vehicle(env, max_wait_ticks=2)
    _obs, _reward, _terminated, _truncated, info = env.step(MATCH_FULL)
    assert info["step_result"].dispatch_friction_cost == 10.0
    assert info["step_result"].dispatch_setup_cost == 3.0
    assert info["step_result"].dispatch_pair_coordination_cost == 2.0
    assert info["step_result"].dispatch_full_mode_extra_cost == 5.0
    assert info["step_result"].platform_profit == env.orders[0].realized_profit - 10.0


def test_full_mode_only_adds_extra_pair_friction() -> None:
    env = make_env()
    env.env_config = env.env_config.__class__(
        **{
            **env.env_config.__dict__,
            "dispatch_friction": DispatchFrictionConfig(
                enabled=True,
                setup_cost=3.0,
                pair_coordination_cost=2.0,
                full_mode_extra_pair_cost=5.0,
                refresh_cost=0.0,
            ),
        }
    )
    env.matcher.env_config = env.env_config
    install_single_order_vehicle(env, max_wait_ticks=2)
    _obs, _reward, _terminated, _truncated, info = env.step(MATCH_FULL)
    assert info["step_result"].dispatch_friction_cost == 10.0
    assert info["step_result"].dispatch_full_mode_extra_cost == 5.0


def test_refresh_friction_decays_smoothly_after_recent_dispatch() -> None:
    env = make_env()
    env.env_config = env.env_config.__class__(
        **{
            **env.env_config.__dict__,
            "dispatch_friction": DispatchFrictionConfig(
                enabled=True,
                setup_cost=0.0,
                pair_coordination_cost=0.0,
                full_mode_extra_pair_cost=0.0,
                refresh_cost=9.0,
                refresh_decay_ticks=1.0,
            ),
        }
    )
    env.matcher.env_config = env.env_config
    install_single_order_vehicle(env, max_wait_ticks=2)
    _obs, _reward, _terminated, _truncated, info = env.step(MATCH_FULL)
    assert info["step_result"].dispatch_refresh_cost == 0.0
    _obs, _reward, _terminated, _truncated, info = env.step(MATCH_FULL)
    one_tick_cost = info["step_result"].dispatch_refresh_cost
    env.step(WAIT)
    _obs, _reward, _terminated, _truncated, info = env.step(MATCH_FULL)
    assert 0.0 < info["step_result"].dispatch_refresh_cost < one_tick_cost


def test_observation_exposes_dispatch_timing_and_service_risk_features() -> None:
    env = make_env()
    env.env_config = env.env_config.__class__(
        **{
            **env.env_config.__dict__,
            "observation_profile": "legacy_full",
            "dispatch_friction": DispatchFrictionConfig(
                enabled=True,
                setup_cost=10.0,
                pair_coordination_cost=1.0,
                full_mode_extra_pair_cost=2.0,
                refresh_cost=9.0,
                refresh_decay_ticks=1.0,
            ),
        }
    )
    env.matcher.env_config = env.env_config
    install_single_order_vehicle(env, max_wait_ticks=1)
    env.current_tick = 1
    env.dispatch_ticks = [0]
    obs = env._observation()
    values = dict(zip(LEGACY_OBSERVATION_NAMES, obs))
    assert values["ticks_since_last_dispatch"] > 0.0
    assert values["estimated_full_match_friction"] > 0.0
    assert values["near_deadline_order_share"] == 1.0
    assert values["projected_service_risk"] > 0.0


def test_service_risk_delta_shaping_rewards_risk_improvement() -> None:
    env = make_env()
    orders = [
        Order(
            order_id=idx,
            arrival_tick=0,
            origin_zone=0,
            destination_zone=1,
            demand_kwh=5.0,
            max_wait_ticks=10,
            willingness_to_pay_per_kwh=8.0,
            cancel_sensitivity=0.0,
        )
        for idx in range(25)
    ]
    env.orders = orders
    env.vehicles = []
    env.orders_by_id = {order.order_id: order for order in orders}
    env.vehicles_by_id = {}
    env.current_tick = 1
    risk_now = env.service_risk_potential()
    improved_reward = env._step_reward(env.last_step_result, risk_before=risk_now + 100.0)
    worsened_reward = env._step_reward(env.last_step_result, risk_before=risk_now - 100.0)
    assert improved_reward > worsened_reward


def test_wait_opportunity_bonus_rewards_candidate_gain_without_losses() -> None:
    env = make_env()
    env._last_wait_tradeoff = {
        "candidate_profit_delta": 500.0,
        "expired_after_wait": 0,
        "cancelled_after_wait": 0,
    }
    bonus = env._wait_opportunity_bonus(wait_penalty=0.0)
    env._last_wait_tradeoff = {
        "candidate_profit_delta": 500.0,
        "expired_after_wait": 60,
        "cancelled_after_wait": 0,
    }
    loss_bonus = env._wait_opportunity_bonus(wait_penalty=0.0)
    assert bonus > 0.0
    assert loss_bonus == 0.0


def test_action_traces_record_wait_and_dispatch() -> None:
    env = make_env()
    install_single_order_vehicle(env, max_wait_ticks=2)
    env.step(WAIT)
    assert env.action_trace[-1]["action_name"] == "wait"
    assert env.wait_tradeoff_trace
    env.step(MATCH_FULL)
    assert env.dispatch_trace[-1]["dispatch_mode"] == "match_full"


def test_cancel_probability_is_flat_mid_wait_and_steeper_near_deadline() -> None:
    env = make_env()
    install_single_order_vehicle(env, max_wait_ticks=10)
    env.orders[0].cancel_sensitivity = 0.30
    env.current_tick = 5
    assert env._cancel_probability(env.orders[0]) == 0.0
    env.current_tick = 7
    mid = env._cancel_probability(env.orders[0])
    env.current_tick = 10
    late = env._cancel_probability(env.orders[0])
    assert 0.0 < mid < late
