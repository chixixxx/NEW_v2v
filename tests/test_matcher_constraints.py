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


def test_full_match_keeps_all_profitable_assignment_matches() -> None:
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
            energy_cost_per_kwh=0.25,
            service_premium_per_kwh=1.75,
            time_cost_per_min=0.0,
            owner_accept_sensitivity=0.5,
            fleet_flag=False,
        )
        env.orders.append(order)
        env.vehicles.append(vehicle)
    plan = env.matcher.solve(env.orders, env.vehicles, tick=0)
    assert len(plan.matches) == 4
    assert all(edge.expected_profit > 0.0 for edge in plan.matches)


def test_solver_signature_is_full_match_only() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    plan = env.matcher.solve(env.orders, env.vehicles, tick=0)
    assert [(edge.order_id, edge.vehicle_id) for edge in plan.matches] == [(1, 2)]


def test_matcher_uses_transfer_efficiency_and_price_components() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    edge = env.matcher.score_edge(env.orders[0], env.vehicles[0], tick=0)
    assert edge.feasible
    assert edge.delivered_kwh == env.orders[0].demand_kwh
    assert round(edge.donor_output_kwh, 6) == round(env.orders[0].demand_kwh / env.env_config.battery_health.transfer_efficiency, 6)
    assert edge.energy_loss_kwh > 0.0
    assert edge.seller_energy_cost > 0.0
    assert edge.seller_degradation_cost > 0.0
    assert edge.seller_service_premium > 0.0
    assert edge.seller_reimbursement == edge.seller_energy_cost + edge.seller_degradation_cost + edge.seller_service_premium


def test_matcher_respects_health_soc_floor() -> None:
    env = make_env()
    install_single_order_vehicle(env)
    vehicle = env.vehicles[0]
    vehicle.reserve_kwh = 5.0
    vehicle.current_soc_kwh = vehicle.battery_capacity_kwh * env.env_config.battery_health.donor_min_soc_ratio + 4.0
    edge = env.matcher.score_edge(env.orders[0], vehicle, tick=0)
    assert not edge.feasible
    assert edge.reason == "energy_shortage"
