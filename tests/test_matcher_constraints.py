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

