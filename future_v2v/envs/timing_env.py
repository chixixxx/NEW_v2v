from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from future_v2v.config import EnvironmentConfig, ScaleConfig
from future_v2v.data.tlc_manhattan import load_tlc_manhattan_data
from future_v2v.metrics import EpisodeMetrics, constrained_profit_score
from future_v2v.simulation.entities import (
    ORDER_CANCELLED,
    ORDER_EXPIRED,
    ORDER_MATCHED,
    ORDER_PENDING,
    CandidateEdge,
    Order,
    StepResult,
    Vehicle,
)
from future_v2v.simulation.generator import ScenarioGenerator
from future_v2v.simulation.matcher import ConstrainedMatcher
from future_v2v.simulation.network import TLCManhattanZoneNetwork, ZoneNetwork
from future_v2v.simulation.tlc_generator import TLCManhattanScenarioGenerator

WAIT = 0
MATCH = 1

OBSERVATION_NAMES = (
    "active_order_count",
    "total_energy_demand",
    "mean_waiting_ratio",
    "near_deadline_order_count",
    "cancel_risk_mean",
    "mean_willingness_to_pay",
    "available_vehicle_count",
    "total_available_energy",
    "mean_energy_surplus",
    "mean_time_flexibility",
    "fleet_available_ratio",
    "resource_scarcity_index",
    "supply_demand_imbalance",
    "top_shortage_zone_pressure",
    "future_hotspot_pressure",
    "mean_pickup_time_est",
    "feasible_edge_density",
    "best_candidate_profit",
    "mean_candidate_profit",
    "urgent_feasible_coverage",
    "energy_feasible_coverage",
)


@dataclass
class EnvironmentSnapshot:
    active_orders: list[Order]
    active_vehicles: list[Vehicle]
    candidate_edges: list[CandidateEdge]


class FutureV2VTimingEnv:
    """A compact Gymnasium-style environment for dynamic V2V matching timing."""

    def __init__(self, env_config: EnvironmentConfig, scale_config: ScaleConfig, seed: int = 0) -> None:
        self.env_config = env_config
        self.scale_config = scale_config
        self.network, self.generator = self._build_network_and_generator()
        self.matcher = ConstrainedMatcher(env_config, self.network)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.current_tick = 0
        self.orders: list[Order] = []
        self.vehicles: list[Vehicle] = []
        self.orders_by_id: dict[int, Order] = {}
        self.vehicles_by_id: dict[int, Vehicle] = {}
        self.platform_profit = 0.0
        self.rejected_matches = 0
        self.dispatch_ticks: list[int] = []
        self.last_step_result = StepResult()

    @property
    def observation_dim(self) -> int:
        return len(OBSERVATION_NAMES)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        scenario = self.generator.generate(self.seed)
        if hasattr(self.network, "set_episode_context"):
            self.network.set_episode_context(start_tick_day=scenario.start_tick_day)
        self.orders = scenario.orders
        self.vehicles = scenario.vehicles
        self.orders_by_id = {order.order_id: order for order in self.orders}
        self.vehicles_by_id = {vehicle.vehicle_id: vehicle for vehicle in self.vehicles}
        self.current_tick = 0
        self.platform_profit = 0.0
        self.rejected_matches = 0
        self.dispatch_ticks = []
        self.last_step_result = StepResult()
        obs = self._observation()
        return obs, {"seed": self.seed, "observation_names": OBSERVATION_NAMES}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if action not in (WAIT, MATCH):
            raise ValueError(f"invalid action {action}; expected WAIT=0 or MATCH=1")
        self._refresh_vehicle_status()
        result = StepResult(dispatch_executed=action == MATCH)
        if action == MATCH:
            plan = self.matcher.solve(self.orders, self.vehicles, self.current_tick)
            realized = self.matcher.realize_matches(
                plan.matches,
                self.orders_by_id,
                self.vehicles_by_id,
                self.current_tick,
                self.rng,
                stochastic_acceptance=self.env_config.enable_stochastic_acceptance,
            )
            accepted = [match for match in realized if match.accepted]
            result.matched_count = len(realized)
            result.accepted_count = len(accepted)
            result.rejected_count = len(realized) - len(accepted)
            result.platform_profit = float(sum(match.realized_profit for match in accepted))
            if accepted:
                result.mean_pickup_minutes = float(np.mean([match.pickup_minutes for match in accepted]))
                result.mean_commitment_ticks = float(np.mean([match.total_commitment_ticks for match in accepted]))
            self.rejected_matches += result.rejected_count
            self.platform_profit += result.platform_profit
            self.dispatch_ticks.append(self.current_tick)
        self.current_tick += 1
        expired, cancelled = self._advance_order_lifecycle()
        result.expired_count = expired
        result.cancelled_count = cancelled
        self.last_step_result = result
        reward = self._step_reward(result)
        terminated = self.current_tick >= self.scale_config.horizon_ticks + self.scale_config.terminal_buffer_ticks
        truncated = False
        obs = self._observation()
        info = {
            "tick": self.current_tick,
            "step_result": result,
            "platform_profit": self.platform_profit,
        }
        return obs, reward, terminated, truncated, info

    def snapshot(self) -> EnvironmentSnapshot:
        active_orders = [order for order in self.orders if order.is_active(self.current_tick)]
        active_vehicles = [vehicle for vehicle in self.vehicles if vehicle.is_active(self.current_tick)]
        candidate_edges = self.matcher.build_edges(active_orders, active_vehicles, self.current_tick)
        return EnvironmentSnapshot(active_orders=active_orders, active_vehicles=active_vehicles, candidate_edges=candidate_edges)

    def episode_metrics(self, policy_name: str, seed: int) -> EpisodeMetrics:
        served = [order for order in self.orders if order.status == ORDER_MATCHED]
        urgent = [order for order in self.orders if order.is_urgent()]
        urgent_served = [order for order in urgent if order.status == ORDER_MATCHED]
        expired = [order for order in self.orders if order.status == ORDER_EXPIRED]
        cancelled = [order for order in self.orders if order.status == ORDER_CANCELLED]
        fleet = [vehicle for vehicle in self.vehicles if vehicle.fleet_flag]
        private = [vehicle for vehicle in self.vehicles if not vehicle.fleet_flag]
        used_energy = sum(vehicle.supplied_kwh for vehicle in self.vehicles)
        initial_available_energy = sum(max(0.0, vehicle.battery_capacity_kwh * 0.88 - vehicle.reserve_kwh) for vehicle in self.vehicles)
        service_rate = len(served) / max(1, len(self.orders))
        urgent_service_rate = len(urgent_served) / max(1, len(urgent))
        expired_rate = len(expired) / max(1, len(self.orders))
        cancelled_rate = len(cancelled) / max(1, len(self.orders))
        pickup_values = [order.pickup_minutes for order in served]
        commitment_values = [order.commitment_ticks for order in served]
        wait_values = [max(0, int(order.match_tick or 0) - order.arrival_tick) for order in served]
        mean_pickup = float(np.mean(pickup_values)) if pickup_values else 0.0
        mean_commitment = float(np.mean(commitment_values)) if commitment_values else 0.0
        profit_scale = self._profit_scale()
        score = constrained_profit_score(
            platform_profit=self.platform_profit,
            total_orders=len(self.orders),
            served_orders=len(served),
            service_rate=service_rate,
            urgent_service_rate=urgent_service_rate,
            expired_rate=expired_rate,
            cancelled_rate=cancelled_rate,
            mean_pickup_time=mean_pickup,
            mean_commitment_ticks=mean_commitment,
            profit_scale=profit_scale,
            env_config=self.env_config,
        )
        return EpisodeMetrics(
            policy_name=policy_name,
            seed=seed,
            future_v2v_score=score,
            platform_profit=self.platform_profit,
            total_orders=len(self.orders),
            served_orders=len(served),
            urgent_orders=len(urgent),
            urgent_served_orders=len(urgent_served),
            expired_orders=len(expired),
            cancelled_orders=len(cancelled),
            rejected_matches=self.rejected_matches,
            dispatch_epoch_count=len(self.dispatch_ticks),
            mean_wait_before_match=float(np.mean(wait_values)) if wait_values else 0.0,
            mean_batch_interval=self._mean_batch_interval(),
            mean_pickup_time=mean_pickup,
            mean_commitment_ticks=mean_commitment,
            profit_per_served_order=self.platform_profit / max(1, len(served)),
            fleet_utilization=self._vehicle_utilization(fleet),
            private_utilization=self._vehicle_utilization(private),
            energy_utilization=used_energy / max(1.0, initial_available_energy),
        )

    def env_health_row(self, seed: int) -> dict[str, float | int]:
        obs, _ = self.reset(seed=seed)
        _ = obs
        supply_demand_ratios = []
        feasible_densities = []
        active_orders = []
        active_vehicles = []
        for _tick in range(self.scale_config.horizon_ticks):
            snapshot = self.snapshot()
            order_count = len(snapshot.active_orders)
            vehicle_count = len(snapshot.active_vehicles)
            supply_demand_ratios.append(vehicle_count / max(1, order_count))
            feasible_densities.append(len(snapshot.candidate_edges) / max(1, order_count * max(1, vehicle_count)))
            active_orders.append(order_count)
            active_vehicles.append(vehicle_count)
            self.current_tick += 1
            self._advance_order_lifecycle(apply_cancellation=False)
        return {
            "seed": seed,
            "mean_active_orders": float(np.mean(active_orders)),
            "mean_active_vehicles": float(np.mean(active_vehicles)),
            "mean_supply_demand_ratio": float(np.mean(supply_demand_ratios)),
            "mean_feasible_edge_density": float(np.mean(feasible_densities)),
            "min_supply_demand_ratio": float(np.min(supply_demand_ratios)),
            "max_active_orders": int(np.max(active_orders)),
            "max_active_vehicles": int(np.max(active_vehicles)),
        }

    def _observation(self) -> np.ndarray:
        snapshot = self.snapshot()
        active_orders = snapshot.active_orders
        active_vehicles = snapshot.active_vehicles
        edges = snapshot.candidate_edges
        order_count = len(active_orders)
        vehicle_count = len(active_vehicles)
        total_demand = sum(order.demand_kwh for order in active_orders)
        total_energy = sum(vehicle.available_energy_kwh() for vehicle in active_vehicles)
        waiting_ratios = [order.waiting_ratio(self.current_tick) for order in active_orders]
        near_deadline = sum(1 for order in active_orders if order.max_wait_ticks - order.waiting_ticks(self.current_tick) <= 1)
        cancel_risk = [self._cancel_probability(order) for order in active_orders]
        vehicle_flex = [vehicle.time_flexibility_ticks(self.current_tick) for vehicle in active_vehicles]
        fleet_count = sum(1 for vehicle in active_vehicles if vehicle.fleet_flag)
        order_zones = [order.origin_zone for order in active_orders]
        vehicle_zones = [vehicle.current_zone for vehicle in active_vehicles]
        shortage = self.network.shortage_pressure_by_zone(order_zones, vehicle_zones)
        profits = [edge.expected_profit for edge in edges]
        pickups = [edge.pickup_minutes for edge in edges]
        urgent_orders = [order.order_id for order in active_orders if order.is_urgent()]
        urgent_covered = {edge.order_id for edge in edges if edge.order_id in urgent_orders}
        energy_covered = {edge.order_id for edge in edges}
        raw = np.array(
            [
                order_count / 100.0,
                total_demand / 1000.0,
                float(np.mean(waiting_ratios)) if waiting_ratios else 0.0,
                near_deadline / 50.0,
                float(np.mean(cancel_risk)) if cancel_risk else 0.0,
                (float(np.mean([order.willingness_to_pay_per_kwh for order in active_orders])) / 12.0) if active_orders else 0.0,
                vehicle_count / 100.0,
                total_energy / 2000.0,
                (total_energy / max(1.0, vehicle_count)) / 80.0,
                (float(np.mean(vehicle_flex)) / 48.0) if vehicle_flex else 0.0,
                fleet_count / max(1, vehicle_count),
                total_demand / max(1.0, total_energy),
                (order_count - vehicle_count) / 100.0,
                float(np.max(shortage)) / 10.0,
                float(np.mean(shortage)) / 5.0,
                (float(np.mean(pickups)) / 30.0) if pickups else 0.0,
                len(edges) / max(1, order_count * max(1, vehicle_count)),
                (float(np.max(profits)) / 50.0) if profits else 0.0,
                (float(np.mean(profits)) / 30.0) if profits else 0.0,
                len(urgent_covered) / max(1, len(urgent_orders)),
                len(energy_covered) / max(1, order_count),
            ],
            dtype=np.float32,
        )
        return np.clip(raw, -10.0, 10.0)

    def _advance_order_lifecycle(self, apply_cancellation: bool = True) -> tuple[int, int]:
        expired = 0
        cancelled = 0
        for order in self.orders:
            if order.status != ORDER_PENDING or order.arrival_tick > self.current_tick:
                continue
            if order.waiting_ticks(self.current_tick) > order.max_wait_ticks:
                order.status = ORDER_EXPIRED
                expired += 1
                continue
            if apply_cancellation and self.env_config.enable_stochastic_cancellation:
                if self.rng.random() <= self._cancel_probability(order):
                    order.status = ORDER_CANCELLED
                    cancelled += 1
        return expired, cancelled

    def _refresh_vehicle_status(self) -> None:
        for vehicle in self.vehicles:
            vehicle.refresh_status(self.current_tick)

    def _cancel_probability(self, order: Order) -> float:
        if order.status != ORDER_PENDING:
            return 0.0
        waiting_ratio = order.waiting_ratio(self.current_tick)
        if waiting_ratio <= 0.45:
            return 0.0
        return float(np.clip((waiting_ratio - 0.45) * order.cancel_sensitivity, 0.0, 0.22))

    def _step_reward(self, result: StepResult) -> float:
        wait_penalty = self.env_config.wait_penalty_per_order_tick * len(
            [order for order in self.orders if order.is_active(self.current_tick)]
        )
        return (
            result.platform_profit
            - self.env_config.expired_penalty * result.expired_count
            - self.env_config.cancelled_penalty * result.cancelled_count
            - wait_penalty
        )

    def _profit_scale(self) -> float:
        positive = [order.realized_profit for order in self.orders if order.realized_profit > 0.0]
        return float(np.mean(positive)) if positive else self.env_config.profit_scale_fallback

    def _vehicle_utilization(self, vehicles: list[Vehicle]) -> float:
        if not vehicles:
            return 0.0
        return sum(1 for vehicle in vehicles if vehicle.served_count > 0) / len(vehicles)

    def _mean_batch_interval(self) -> float:
        if len(self.dispatch_ticks) <= 1:
            return float(self.scale_config.horizon_ticks)
        return float(np.mean(np.diff(self.dispatch_ticks)))

    def _build_network_and_generator(self):
        if self.env_config.scenario_source == "tlc_manhattan":
            try:
                data = load_tlc_manhattan_data(self.env_config)
                network = TLCManhattanZoneNetwork(data=data)
                generator = TLCManhattanScenarioGenerator(self.env_config, self.scale_config, data)
                return network, generator
            except FileNotFoundError:
                if not (self.env_config.allow_synthetic_smoke_fallback and self.scale_config.name == "smoke"):
                    raise
        network = ZoneNetwork(self.env_config.zone_count)
        generator = ScenarioGenerator(self.env_config, self.scale_config)
        return network, generator
