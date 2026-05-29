from __future__ import annotations

from future_v2v.config import EnvironmentConfig, ScaleConfig
from future_v2v.envs.timing_env import MATCH_FULL, MATCH_TOP_BATCH, WAIT, FutureV2VTimingEnv
from future_v2v.simulation.entities import ORDER_EXPIRED, ORDER_MATCHED, Order, Vehicle


def make_env() -> FutureV2VTimingEnv:
    env_config = EnvironmentConfig(
        zone_count=4,
        action_space="wait_topbatch_full",
        service_kwh_per_tick=5.0,
        pickup_cap_minutes=30.0,
        platform_pickup_cost_per_min=0.0,
        dispatch_fixed_cost=0.0,
        dispatch_capacity_ratio=0.55,
        dispatch_capacity_min=1,
        dispatch_capacity_max=80,
        queue_threshold_ratio=0.35,
        queue_threshold_min=1,
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
        reservation_price_per_kwh=2.0,
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
    _obs, reward, _terminated, _truncated, info = env.step(MATCH_TOP_BATCH)
    assert env.orders[0].status == ORDER_MATCHED
    assert env.orders[0].matched_vehicle_id == 2
    assert env.vehicles[0].served_count == 1
    assert info["step_result"].accepted_count == 1
    assert info["step_result"].dispatch_mode == "match_top_batch"
    assert reward > 0.0


def test_dispatch_fixed_cost_is_subtracted_from_match_profit() -> None:
    env = make_env()
    env.env_config = env.env_config.__class__(**{**env.env_config.__dict__, "dispatch_fixed_cost": 7.0})
    env.matcher.env_config = env.env_config
    install_single_order_vehicle(env, max_wait_ticks=2)
    _obs, _reward, _terminated, _truncated, info = env.step(MATCH_FULL)
    assert info["step_result"].platform_profit == env.orders[0].realized_profit - 7.0


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
