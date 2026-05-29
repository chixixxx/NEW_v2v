from __future__ import annotations

from tests.test_env_semantics import install_single_order_vehicle, make_env


def test_matcher_rejects_energy_shortage() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    env.vehicles[0].current_soc_kwh = 16.0
    edge = env.matcher.score_edge(env.orders[0], env.vehicles[0], tick=0)
    assert not edge.feasible
    assert edge.reason == "energy_shortage"


def test_matcher_rejects_pickup_cap() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    env.env_config = env.env_config.__class__(**{**env.env_config.__dict__, "pickup_cap_minutes": 1.0})
    env.matcher.env_config = env.env_config
    env.vehicles[0].current_zone = 3
    edge = env.matcher.score_edge(env.orders[0], env.vehicles[0], tick=0)
    assert not edge.feasible
    assert edge.reason == "pickup_cap"

def test_solver_respects_one_vehicle_one_order() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    duplicate = env.orders[0].__class__(
        order_id=3,
        arrival_tick=0,
        origin_zone=0,
        destination_zone=1,
        demand_kwh=5.0,
        max_wait_ticks=2,
        willingness_to_pay_per_kwh=8.0,
        cancel_sensitivity=0.0,
    )
    env.orders.append(duplicate)
    env.orders_by_id[duplicate.order_id] = duplicate
    plan = env.matcher.solve(env.orders, env.vehicles, tick=0)
    assert len(plan.matches) == 1


def test_top_batch_respects_capacity() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    for idx in range(3):
        order = env.orders[0].__class__(
            order_id=10 + idx,
            arrival_tick=0,
            origin_zone=0,
            destination_zone=1,
            demand_kwh=5.0,
            max_wait_ticks=3,
            willingness_to_pay_per_kwh=8.0 + idx,
            cancel_sensitivity=0.0,
        )
        vehicle = env.vehicles[0].__class__(
            vehicle_id=20 + idx,
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
        env.orders.append(order)
        env.vehicles.append(vehicle)
    plan = env.matcher.solve(env.orders, env.vehicles, tick=0, dispatch_mode="top_batch", capacity=2)
    assert len(plan.matches) == 2
    assert all(edge.expected_profit > 0.0 for edge in plan.matches)


def test_full_match_keeps_all_profitable_assignment_matches() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    full = env.matcher.solve(env.orders, env.vehicles, tick=0, dispatch_mode="full", capacity=1)
    legacy = env.matcher.solve(env.orders, env.vehicles, tick=0)
    assert [(edge.order_id, edge.vehicle_id) for edge in full.matches] == [
        (edge.order_id, edge.vehicle_id) for edge in legacy.matches
    ]
