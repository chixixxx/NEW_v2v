from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from future_v2v.config import EnvironmentConfig
from future_v2v.simulation.entities import CandidateEdge, Match, Order, Vehicle
from future_v2v.simulation.network import ZoneNetwork


@dataclass(frozen=True)
class MatchPlan:
    edges: list[CandidateEdge]
    matches: list[CandidateEdge]


class ConstrainedMatcher:
    def __init__(self, env_config: EnvironmentConfig, network: ZoneNetwork) -> None:
        self.env_config = env_config
        self.network = network

    def build_edges(self, orders: list[Order], vehicles: list[Vehicle], tick: int) -> list[CandidateEdge]:
        edges: list[CandidateEdge] = []
        active_vehicles = [vehicle for vehicle in vehicles if vehicle.is_active(tick)]
        vehicles_by_zone: dict[int, list[Vehicle]] = {}
        for vehicle in active_vehicles:
            vehicles_by_zone.setdefault(vehicle.current_zone, []).append(vehicle)
        for order in orders:
            if not order.is_active(tick):
                continue
            for vehicle in self._candidate_vehicles_for_order(order, active_vehicles, vehicles_by_zone, tick):
                edges.append(self.score_edge(order, vehicle, tick))
        return [edge for edge in edges if edge.feasible]

    def score_edge(self, order: Order, vehicle: Vehicle, tick: int) -> CandidateEdge:
        pickup_ticks = self.network.travel_ticks(vehicle.current_zone, order.origin_zone, tick)
        pickup_minutes = pickup_ticks * self.network.minutes_per_tick
        service_ticks = order.demand_kwh / max(0.1, self.env_config.service_kwh_per_tick)
        commitment_ticks = pickup_ticks + service_ticks
        revenue = order.demand_kwh * order.willingness_to_pay_per_kwh
        seller_compensation = order.demand_kwh * vehicle.reservation_price_per_kwh
        platform_cost = pickup_minutes * self.env_config.platform_pickup_cost_per_min
        time_cost = pickup_minutes * vehicle.time_cost_per_min
        immediate_profit = revenue - seller_compensation - platform_cost - time_cost
        feasible, reason = self._check_feasible(order, vehicle, tick, pickup_minutes, commitment_ticks)
        accept_probability = self._accept_probability(order, vehicle, immediate_profit, pickup_minutes)
        expected_profit = immediate_profit * accept_probability
        if expected_profit <= 0.0:
            feasible = False
            reason = "non_positive_expected_profit"
        return CandidateEdge(
            order_id=order.order_id,
            vehicle_id=vehicle.vehicle_id,
            pickup_ticks=pickup_ticks,
            pickup_minutes=pickup_minutes,
            service_ticks=service_ticks,
            total_commitment_ticks=commitment_ticks,
            revenue=revenue,
            seller_compensation=seller_compensation,
            platform_cost=platform_cost,
            immediate_profit=immediate_profit,
            accept_probability=accept_probability,
            expected_profit=expected_profit,
            feasible=feasible,
            reason=reason,
        )

    def solve(
        self,
        orders: list[Order],
        vehicles: list[Vehicle],
        tick: int,
        dispatch_mode: str = "full",
        capacity: int | None = None,
    ) -> MatchPlan:
        edges = self.build_edges(orders, vehicles, tick)
        if not edges:
            return MatchPlan(edges=[], matches=[])
        orders_by_id = {order.order_id: order for order in orders}
        active_order_ids = sorted({edge.order_id for edge in edges})
        active_vehicle_ids = sorted({edge.vehicle_id for edge in edges})
        order_index = {order_id: idx for idx, order_id in enumerate(active_order_ids)}
        vehicle_index = {vehicle_id: idx for idx, vehicle_id in enumerate(active_vehicle_ids)}
        dummy_count = len(active_order_ids)
        weights = np.zeros((len(active_order_ids), len(active_vehicle_ids) + dummy_count), dtype=float)
        edge_by_pair: dict[tuple[int, int], CandidateEdge] = {}
        for edge in edges:
            row = order_index[edge.order_id]
            col = vehicle_index[edge.vehicle_id]
            weights[row, col] = max(weights[row, col], edge.expected_profit)
            edge_by_pair[(edge.order_id, edge.vehicle_id)] = edge
        row_ind, col_ind = linear_sum_assignment(weights, maximize=True)
        matches: list[CandidateEdge] = []
        for row, col in zip(row_ind, col_ind):
            if col >= len(active_vehicle_ids) or weights[row, col] <= 0.0:
                continue
            order_id = active_order_ids[row]
            vehicle_id = active_vehicle_ids[col]
            matches.append(edge_by_pair[(order_id, vehicle_id)])
        if dispatch_mode == "top_batch":
            limit = max(0, int(capacity or 0))
            if limit <= 0:
                matches = []
            elif len(matches) > limit:
                matches = sorted(
                    matches,
                    key=lambda edge: self._adjusted_edge_value(edge, orders_by_id[edge.order_id], tick),
                    reverse=True,
                )[:limit]
        elif dispatch_mode != "full":
            raise ValueError(f"unknown dispatch_mode={dispatch_mode!r}; expected 'full' or 'top_batch'")
        return MatchPlan(edges=edges, matches=matches)

    def _adjusted_edge_value(self, edge: CandidateEdge, order: Order, tick: int) -> float:
        wait_ratio = order.waiting_ratio(tick)
        pickup_penalty = 0.06 * edge.pickup_minutes
        wait_risk_penalty = 1.4 * max(0.0, wait_ratio - 0.70)
        urgency_bonus = 2.0 if order.is_urgent() else 0.0
        deadline_bonus = 3.5 * max(0.0, wait_ratio - 0.68)
        profit_bonus = 0.10 * max(0.0, edge.expected_profit - 10.0)
        return edge.expected_profit + profit_bonus - pickup_penalty - wait_risk_penalty + urgency_bonus + deadline_bonus

    def realize_matches(
        self,
        matches: list[CandidateEdge],
        orders_by_id: dict[int, Order],
        vehicles_by_id: dict[int, Vehicle],
        tick: int,
        rng: np.random.Generator,
        stochastic_acceptance: bool,
    ) -> list[Match]:
        realized: list[Match] = []
        for edge in matches:
            order = orders_by_id[edge.order_id]
            vehicle = vehicles_by_id[edge.vehicle_id]
            if not order.is_active(tick) or not vehicle.is_active(tick):
                continue
            accepted = True
            if stochastic_acceptance:
                accepted = bool(rng.random() <= edge.accept_probability)
            realized_profit = edge.immediate_profit if accepted else 0.0
            if accepted:
                order.status = "matched"
                order.matched_vehicle_id = vehicle.vehicle_id
                order.match_tick = tick
                order.pickup_minutes = edge.pickup_minutes
                order.commitment_ticks = edge.total_commitment_ticks
                order.realized_profit = realized_profit
                vehicle.status = "busy"
                vehicle.busy_until_tick = tick + edge.total_commitment_ticks
                vehicle.current_soc_kwh -= order.demand_kwh
                vehicle.current_zone = order.destination_zone
                vehicle.served_count += 1
                vehicle.supplied_kwh += order.demand_kwh
            realized.append(
                Match(
                    order_id=edge.order_id,
                    vehicle_id=edge.vehicle_id,
                    expected_profit=edge.expected_profit,
                    realized_profit=realized_profit,
                    pickup_minutes=edge.pickup_minutes,
                    total_commitment_ticks=edge.total_commitment_ticks,
                    accepted=accepted,
                )
            )
        return realized

    def _check_feasible(
        self,
        order: Order,
        vehicle: Vehicle,
        tick: int,
        pickup_minutes: float,
        commitment_ticks: float,
    ) -> tuple[bool, str]:
        if pickup_minutes > self.env_config.pickup_cap_minutes:
            return False, "pickup_cap"
        if vehicle.available_energy_kwh() < order.demand_kwh:
            return False, "energy_shortage"
        if tick + commitment_ticks > vehicle.leave_tick:
            return False, "vehicle_time_window"
        if order.waiting_ticks(tick) > order.max_wait_ticks:
            return False, "order_expired"
        return True, "feasible"

    @staticmethod
    def _accept_probability(order: Order, vehicle: Vehicle, immediate_profit: float, pickup_minutes: float) -> float:
        margin_per_kwh = immediate_profit / max(0.1, order.demand_kwh)
        pickup_disutility = 0.025 * pickup_minutes
        raw = (margin_per_kwh - pickup_disutility) / max(0.1, vehicle.owner_accept_sensitivity)
        return float(1.0 / (1.0 + math.exp(-raw)))

    def _candidate_vehicles_for_order(
        self,
        order: Order,
        active_vehicles: list[Vehicle],
        vehicles_by_zone: dict[int, list[Vehicle]],
        tick: int,
    ) -> list[Vehicle]:
        limit = int(getattr(self.env_config, "max_candidate_vehicles_per_order", 0) or 0)
        if limit <= 0 or len(active_vehicles) <= limit:
            return active_vehicles
        candidates: list[tuple[float, float, int, Vehicle]] = []
        for zone, zone_vehicles in vehicles_by_zone.items():
            pickup_minutes = self.network.travel_minutes(zone, order.origin_zone, tick)
            if pickup_minutes > self.env_config.pickup_cap_minutes:
                continue
            for vehicle in zone_vehicles:
                if vehicle.available_energy_kwh() < order.demand_kwh:
                    continue
                candidates.append((pickup_minutes, -vehicle.available_energy_kwh(), vehicle.vehicle_id, vehicle))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in candidates[:limit]]
