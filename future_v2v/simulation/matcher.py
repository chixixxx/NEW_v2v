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
    rejected_reason_counts: dict[str, int]


class ConstrainedMatcher:
    def __init__(self, env_config: EnvironmentConfig, network: ZoneNetwork) -> None:
        self.env_config = env_config
        self.network = network

    def build_edges(
        self,
        orders: list[Order],
        vehicles: list[Vehicle],
        tick: int,
        *,
        include_infeasible: bool = False,
    ) -> list[CandidateEdge]:
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
        if include_infeasible:
            return edges
        return [edge for edge in edges if edge.feasible]

    def score_edge(self, order: Order, vehicle: Vehicle, tick: int) -> CandidateEdge:
        battery = self.env_config.battery_health
        pickup_ticks = self.network.travel_ticks(vehicle.current_zone, order.origin_zone, tick)
        pickup_minutes = pickup_ticks * self.network.minutes_per_tick
        pickup_distance_km_est = self.network.pickup_distance_km_est(vehicle.current_zone, order.origin_zone, tick)
        delivered_kwh = order.demand_kwh
        donor_output_kwh = delivered_kwh / max(1e-6, battery.transfer_efficiency)
        energy_loss_kwh = donor_output_kwh - delivered_kwh
        power_limited_kwh_per_tick = battery.max_discharge_power_kw * self.env_config.tick_minutes / 60.0
        service_kwh_per_tick = min(self.env_config.service_kwh_per_tick, max(0.1, power_limited_kwh_per_tick))
        service_ticks = donor_output_kwh / max(0.1, service_kwh_per_tick)
        commitment_ticks = pickup_ticks + service_ticks
        buyer_payment = delivered_kwh * order.willingness_to_pay_per_kwh
        seller_energy_cost = donor_output_kwh * vehicle.energy_cost_per_kwh
        seller_degradation_cost = donor_output_kwh * battery.degradation_cost_per_kwh
        seller_service_premium = donor_output_kwh * vehicle.service_premium_per_kwh
        seller_reimbursement = seller_energy_cost + seller_degradation_cost + seller_service_premium
        platform_pickup_cost = pickup_minutes * self.env_config.platform_pickup_cost_per_min
        seller_time_cost = pickup_minutes * vehicle.time_cost_per_min
        immediate_profit = buyer_payment - seller_reimbursement - platform_pickup_cost - seller_time_cost
        donor_soc_after_kwh = vehicle.current_soc_kwh - donor_output_kwh
        feasible, reason = self._check_feasible(
            order,
            vehicle,
            tick,
            pickup_minutes,
            commitment_ticks,
            service_ticks,
            donor_output_kwh,
            donor_soc_after_kwh,
        )
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
            pickup_distance_km_est=pickup_distance_km_est,
            service_ticks=service_ticks,
            total_commitment_ticks=commitment_ticks,
            delivered_kwh=delivered_kwh,
            donor_output_kwh=donor_output_kwh,
            energy_loss_kwh=energy_loss_kwh,
            buyer_payment=buyer_payment,
            seller_reimbursement=seller_reimbursement,
            seller_energy_cost=seller_energy_cost,
            seller_degradation_cost=seller_degradation_cost,
            seller_service_premium=seller_service_premium,
            platform_pickup_cost=platform_pickup_cost,
            seller_time_cost=seller_time_cost,
            immediate_profit=immediate_profit,
            accept_probability=accept_probability,
            expected_profit=expected_profit,
            feasible=feasible,
            donor_soc_after_kwh=donor_soc_after_kwh,
            reason=reason,
        )

    def solve(self, orders: list[Order], vehicles: list[Vehicle], tick: int) -> MatchPlan:
        all_edges = self.build_edges(orders, vehicles, tick, include_infeasible=True)
        rejected_reason_counts: dict[str, int] = {}
        for edge in all_edges:
            if not edge.feasible:
                rejected_reason_counts[edge.reason] = rejected_reason_counts.get(edge.reason, 0) + 1
        edges = [edge for edge in all_edges if edge.feasible]
        if not edges:
            return MatchPlan(edges=[], matches=[], rejected_reason_counts=rejected_reason_counts)
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
        return MatchPlan(edges=edges, matches=matches, rejected_reason_counts=rejected_reason_counts)

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
                order.pickup_distance_km_est = edge.pickup_distance_km_est
                order.commitment_ticks = edge.total_commitment_ticks
                order.realized_profit = realized_profit
                order.buyer_payment = edge.buyer_payment
                order.seller_reimbursement = edge.seller_reimbursement
                order.seller_energy_cost = edge.seller_energy_cost
                order.seller_degradation_cost = edge.seller_degradation_cost
                order.seller_service_premium = edge.seller_service_premium
                order.platform_pickup_cost = edge.platform_pickup_cost
                order.seller_time_cost = edge.seller_time_cost
                order.delivered_kwh = edge.delivered_kwh
                order.donor_output_kwh = edge.donor_output_kwh
                order.energy_loss_kwh = edge.energy_loss_kwh
                order.donor_soc_after_kwh = edge.donor_soc_after_kwh
                vehicle.status = "busy"
                vehicle.busy_until_tick = tick + edge.total_commitment_ticks
                vehicle.current_soc_kwh -= edge.donor_output_kwh
                vehicle.current_zone = order.destination_zone
                vehicle.served_count += 1
                vehicle.supplied_kwh += edge.donor_output_kwh
                vehicle.delivered_kwh += edge.delivered_kwh
                vehicle.energy_loss_kwh += edge.energy_loss_kwh
                vehicle.seller_reimbursement += edge.seller_reimbursement
                vehicle.seller_energy_cost += edge.seller_energy_cost
                vehicle.seller_degradation_cost += edge.seller_degradation_cost
                vehicle.seller_service_premium += edge.seller_service_premium
            realized.append(
                Match(
                    order_id=edge.order_id,
                    vehicle_id=edge.vehicle_id,
                    expected_profit=edge.expected_profit,
                    realized_profit=realized_profit,
                    pickup_minutes=edge.pickup_minutes,
                    pickup_distance_km_est=edge.pickup_distance_km_est,
                    total_commitment_ticks=edge.total_commitment_ticks,
                    accepted=accepted,
                    delivered_kwh=edge.delivered_kwh,
                    donor_output_kwh=edge.donor_output_kwh,
                    energy_loss_kwh=edge.energy_loss_kwh,
                    buyer_payment=edge.buyer_payment,
                    seller_reimbursement=edge.seller_reimbursement,
                    seller_energy_cost=edge.seller_energy_cost,
                    seller_degradation_cost=edge.seller_degradation_cost,
                    seller_service_premium=edge.seller_service_premium,
                    platform_pickup_cost=edge.platform_pickup_cost,
                    seller_time_cost=edge.seller_time_cost,
                    donor_soc_after_kwh=edge.donor_soc_after_kwh,
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
        service_ticks: float,
        donor_output_kwh: float,
        donor_soc_after_kwh: float,
    ) -> tuple[bool, str]:
        battery = self.env_config.battery_health
        if pickup_minutes > self.env_config.pickup_cap_minutes:
            return False, "pickup_cap"
        if donor_output_kwh > vehicle.available_energy_with_health_kwh(battery.donor_min_soc_ratio):
            return False, "energy_shortage"
        if donor_soc_after_kwh + 1e-9 < vehicle.health_floor_kwh(battery.donor_min_soc_ratio):
            return False, "battery_health_floor"
        service_power_kw = (donor_output_kwh / max(1e-6, service_ticks)) * (60.0 / max(1e-6, self.env_config.tick_minutes))
        if service_power_kw > battery.max_discharge_power_kw + 1e-9:
            return False, "discharge_power_cap"
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
                donor_output_kwh = order.demand_kwh / max(1e-6, self.env_config.battery_health.transfer_efficiency)
                if vehicle.available_energy_with_health_kwh(self.env_config.battery_health.donor_min_soc_ratio) < donor_output_kwh:
                    continue
                available_energy = vehicle.available_energy_with_health_kwh(
                    self.env_config.battery_health.donor_min_soc_ratio
                )
                candidates.append((pickup_minutes, -available_energy, vehicle.vehicle_id, vehicle))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in candidates[:limit]]
