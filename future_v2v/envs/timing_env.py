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
MATCH_FULL = 1
ACTION_COUNT = 2
ACTION_NAMES = {
    WAIT: "wait",
    MATCH_FULL: "match_full",
}

LEGACY_OBSERVATION_NAMES = (
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
    "mean_available_energy_with_health",
    "soc_safety_binding_rate",
    "mean_candidate_platform_margin",
    "energy_loss_rate_estimate",
    "ticks_since_last_dispatch",
    "estimated_full_match_friction",
    "time_bucket_off_peak",
    "time_bucket_morning_peak",
    "time_bucket_midday",
    "time_bucket_evening_peak",
    "near_deadline_order_share",
    "projected_service_risk",
    "projected_urgent_service_risk",
)

COMPACT_OBSERVATION_NAMES = (
    "ticks_since_last_dispatch",
    "time_bucket_off_peak",
    "time_bucket_morning_peak",
    "time_bucket_midday",
    "time_bucket_evening_peak",
    "active_order_count",
    "total_energy_demand",
    "mean_waiting_ratio",
    "near_deadline_order_share",
    "cancel_risk_mean",
    "mean_willingness_to_pay",
    "available_vehicle_count",
    "mean_available_energy_with_health",
    "total_available_energy",
    "mean_time_flexibility",
    "fleet_available_ratio",
    "soc_safety_binding_rate",
    "supply_demand_imbalance",
    "top_shortage_zone_pressure",
    "future_hotspot_pressure",
    "mean_pickup_time_est",
    "mean_pickup_distance_est",
    "feasible_edge_density",
    "mean_candidate_platform_margin",
    "best_candidate_profit",
    "urgent_feasible_coverage",
    "energy_feasible_coverage",
    "projected_service_risk",
    "projected_urgent_service_risk",
)

OBSERVATION_NAMES = COMPACT_OBSERVATION_NAMES


@dataclass
class EnvironmentSnapshot:
    active_orders: list[Order]
    active_vehicles: list[Vehicle]
    candidate_edges: list[CandidateEdge]


class FutureV2VTimingEnv:
    """A compact Gymnasium-style environment for dynamic V2V matching timing."""

    def __init__(
        self,
        env_config: EnvironmentConfig,
        scale_config: ScaleConfig,
        seed: int = 0,
        scenario_phase: str = "train",
    ) -> None:
        self.env_config = env_config
        self.scale_config = scale_config
        self.scenario_phase = scenario_phase
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
        self.action_trace: list[dict[str, object]] = []
        self.dispatch_trace: list[dict[str, object]] = []
        self.wait_tradeoff_trace: list[dict[str, object]] = []
        self._last_wait_tradeoff: dict[str, object] | None = None
        self._next_action_q_values: tuple[float, ...] | None = None
        self.scenario_id = ""
        self.scenario_day = ""
        self.scenario_start_tick_day = 0
        self.initial_reward_potential = 0.0
        self.training_initial_potential = 0.0

    @property
    def observation_dim(self) -> int:
        return len(self.observation_names)

    @property
    def observation_names(self) -> tuple[str, ...]:
        if self.env_config.observation_profile == "legacy_full":
            return LEGACY_OBSERVATION_NAMES
        if self.env_config.observation_profile == "compact_v2v":
            return COMPACT_OBSERVATION_NAMES
        raise ValueError(
            f"unknown observation_profile={self.env_config.observation_profile!r}; "
            "expected legacy_full or compact_v2v"
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        scenario = self.generator.generate(self.seed)
        return self._reset_with_scenario(scenario)

    def reset_to_tlc_window(
        self,
        *,
        seed: int,
        day: str,
        start_tick_day: int,
        scenario_id: str = "",
    ) -> tuple[np.ndarray, dict[str, object]]:
        if not isinstance(self.generator, TLCManhattanScenarioGenerator):
            return self.reset(seed=seed)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        scenario = self.generator.generate_from_window(
            seed=self.seed,
            day=day,
            start_tick_day=int(start_tick_day),
            scenario_id=scenario_id,
        )
        return self._reset_with_scenario(scenario)

    def _reset_with_scenario(self, scenario) -> tuple[np.ndarray, dict[str, object]]:
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
        self.action_trace = []
        self.dispatch_trace = []
        self.wait_tradeoff_trace = []
        self._last_wait_tradeoff = None
        self._next_action_q_values = None
        self.scenario_id = scenario.scenario_id
        self.scenario_day = scenario.day
        self.scenario_start_tick_day = int(scenario.start_tick_day)
        obs = self._observation()
        self.initial_reward_potential = self.state_potential_proxy()
        self.training_initial_potential = self.initial_reward_potential
        return obs, {
            "seed": self.seed,
            "observation_names": self.observation_names,
            "observation_profile": self.env_config.observation_profile,
            "scenario_phase": self.scenario_phase,
            "scenario_id": self.scenario_id,
            "day": self.scenario_day,
            "start_tick_day": self.scenario_start_tick_day,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if action not in ACTION_NAMES:
            raise ValueError(f"invalid action {action}; expected 0=WAIT, 1=MATCH_FULL")
        self._refresh_vehicle_status()
        tick = self.current_tick
        before_snapshot = self.snapshot()
        risk_before = self.service_risk_potential()
        q_values = self._next_action_q_values
        self._next_action_q_values = None
        result = StepResult(dispatch_executed=action == MATCH_FULL, dispatch_mode=ACTION_NAMES[action])
        if action == MATCH_FULL:
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
            result.dispatch_capacity = len(before_snapshot.active_orders)
            result.candidate_edge_count = len(plan.edges)
            result.gross_dispatch_profit = float(sum(match.realized_profit for match in accepted))
            self.compute_dispatch_friction(
                result,
                dispatch_mode="full",
                matched_count=result.matched_count,
                tick=tick,
            )
            result.battery_health_rejection_count = sum(
                int(plan.rejected_reason_counts.get(reason, 0))
                for reason in ("energy_shortage", "battery_health_floor", "discharge_power_cap")
            )
            result.platform_profit = float(result.gross_dispatch_profit - result.dispatch_friction_cost)
            if accepted:
                result.mean_pickup_minutes = float(np.mean([match.pickup_minutes for match in accepted]))
                result.total_pickup_distance_km = float(sum(match.pickup_distance_km_est for match in accepted))
                result.mean_pickup_distance_km = float(np.mean([match.pickup_distance_km_est for match in accepted]))
                result.mean_commitment_ticks = float(np.mean([match.total_commitment_ticks for match in accepted]))
                result.buyer_payment = float(sum(match.buyer_payment for match in accepted))
                result.seller_reimbursement = float(sum(match.seller_reimbursement for match in accepted))
                result.seller_energy_cost = float(sum(match.seller_energy_cost for match in accepted))
                result.seller_degradation_cost = float(sum(match.seller_degradation_cost for match in accepted))
                result.seller_service_premium = float(sum(match.seller_service_premium for match in accepted))
                result.platform_pickup_cost = float(sum(match.platform_pickup_cost for match in accepted))
                result.seller_time_cost = float(sum(match.seller_time_cost for match in accepted))
                result.delivered_kwh = float(sum(match.delivered_kwh for match in accepted))
                result.donor_output_kwh = float(sum(match.donor_output_kwh for match in accepted))
                result.energy_loss_kwh = float(sum(match.energy_loss_kwh for match in accepted))
                result.mean_donor_soc_after_kwh = float(np.mean([match.donor_soc_after_kwh for match in accepted]))
            self.rejected_matches += result.rejected_count
            self.platform_profit += result.platform_profit
            self.dispatch_ticks.append(self.current_tick)
            self._record_dispatch_trace(tick, action, plan_edges=len(plan.edges), result=result)
        self.current_tick += 1
        expired, cancelled = self._advance_order_lifecycle()
        result.expired_count = expired
        result.cancelled_count = cancelled
        self.last_step_result = result
        if action == WAIT:
            self._last_wait_tradeoff = self._record_wait_tradeoff_trace(
                tick,
                before_snapshot,
                expired=expired,
                cancelled=cancelled,
            )
        else:
            self._last_wait_tradeoff = None
        self._record_action_trace(tick, action, before_snapshot, result, q_values)
        base_reward = self._base_step_reward(result)
        reward = self._step_reward(result, risk_before=risk_before)
        terminated = self.current_tick >= self.scale_config.horizon_ticks + self.scale_config.terminal_buffer_ticks
        truncated = False
        obs = self._observation()
        info = {
            "tick": self.current_tick,
            "step_result": result,
            "platform_profit": self.platform_profit,
            "base_reward": base_reward,
            "legacy_reward": reward,
            "service_risk_before": float(risk_before),
            "service_risk_after": float(self.service_risk_potential()),
        }
        return obs, reward, terminated, truncated, info

    def set_action_q_values(self, values: list[float] | tuple[float, ...] | np.ndarray | None) -> None:
        if values is None:
            self._next_action_q_values = None
            return
        self._next_action_q_values = tuple(float(value) for value in values)

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
        battery = self.env_config.battery_health
        initial_available_energy = sum(
            max(
                0.0,
                vehicle.battery_capacity_kwh * 0.88
                - max(vehicle.reserve_kwh, battery.donor_min_soc_ratio * vehicle.battery_capacity_kwh),
            )
            for vehicle in self.vehicles
        )
        delivered_kwh = sum(order.delivered_kwh for order in served)
        donor_output_kwh = sum(order.donor_output_kwh for order in served)
        energy_loss_kwh = sum(order.energy_loss_kwh for order in served)
        buyer_payment = sum(order.buyer_payment for order in served)
        seller_reimbursement = sum(order.seller_reimbursement for order in served)
        seller_energy_cost = sum(order.seller_energy_cost for order in served)
        seller_degradation_cost = sum(order.seller_degradation_cost for order in served)
        seller_service_premium = sum(order.seller_service_premium for order in served)
        platform_pickup_cost = sum(order.platform_pickup_cost for order in served)
        total_pickup_distance = sum(order.pickup_distance_km_est for order in served)
        seller_time_cost = sum(order.seller_time_cost for order in served)
        donor_soc_after_values = [order.donor_soc_after_kwh for order in served if order.donor_soc_after_kwh > 0.0]
        floor_by_vehicle = {
            vehicle.vehicle_id: vehicle.health_floor_kwh(battery.donor_min_soc_ratio)
            for vehicle in self.vehicles
        }
        donor_soc_violation_count = sum(
            1
            for order in served
            if order.matched_vehicle_id is not None
            and order.donor_soc_after_kwh + 1e-9 < floor_by_vehicle.get(order.matched_vehicle_id, 0.0)
        )
        dispatch_friction_cost = sum(float(row.get("dispatch_friction_cost", 0.0)) for row in self.dispatch_trace)
        dispatch_setup_cost = sum(float(row.get("dispatch_setup_cost", 0.0)) for row in self.dispatch_trace)
        dispatch_pair_coordination_cost = sum(
            float(row.get("dispatch_pair_coordination_cost", 0.0))
            for row in self.dispatch_trace
        )
        dispatch_refresh_cost = sum(float(row.get("dispatch_refresh_cost", 0.0)) for row in self.dispatch_trace)
        dispatch_full_mode_extra_cost = sum(
            float(row.get("dispatch_full_mode_extra_cost", 0.0))
            for row in self.dispatch_trace
        )
        gross_dispatch_profit = self.platform_profit + dispatch_friction_cost
        battery_health_rejection_count = sum(
            int(row.get("battery_health_rejection_count", 0))
            for row in self.dispatch_trace
        )
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
        distance_adjusted_score = score - self.env_config.pickup_distance_penalty_per_km * total_pickup_distance
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
            total_pickup_distance_km=total_pickup_distance,
            mean_pickup_distance_km=float(np.mean([order.pickup_distance_km_est for order in served])) if served else 0.0,
            pickup_distance_per_served_order=total_pickup_distance / max(1, len(served)),
            distance_adjusted_score=distance_adjusted_score,
            unmet_kwh=sum(order.demand_kwh for order in self.orders if order.status != ORDER_MATCHED),
            delivered_kwh=delivered_kwh,
            donor_output_kwh=donor_output_kwh,
            energy_loss_kwh=energy_loss_kwh,
            buyer_payment=buyer_payment,
            seller_reimbursement=seller_reimbursement,
            seller_energy_cost=seller_energy_cost,
            seller_degradation_cost=seller_degradation_cost,
            seller_service_premium=seller_service_premium,
            platform_margin=buyer_payment - seller_reimbursement - platform_pickup_cost - seller_time_cost,
            dispatch_friction_cost=dispatch_friction_cost,
            dispatch_setup_cost=dispatch_setup_cost,
            dispatch_pair_coordination_cost=dispatch_pair_coordination_cost,
            dispatch_refresh_cost=dispatch_refresh_cost,
            dispatch_full_mode_extra_cost=dispatch_full_mode_extra_cost,
            friction_share_of_gross_profit=dispatch_friction_cost / max(1e-9, gross_dispatch_profit),
            mean_donor_soc_after=float(np.mean(donor_soc_after_values)) if donor_soc_after_values else 0.0,
            min_donor_soc_after=float(np.min(donor_soc_after_values)) if donor_soc_after_values else 0.0,
            donor_soc_violation_count=donor_soc_violation_count,
            battery_health_rejection_count=battery_health_rejection_count,
            scenario_id=self.scenario_id,
            scenario_day=self.scenario_day,
            scenario_start_tick_day=self.scenario_start_tick_day,
        )

    def env_health_row(self, seed: int) -> dict[str, float | int]:
        obs, _ = self.reset(seed=seed)
        _ = obs
        return self._env_health_current_scenario(seed)

    def env_health_row_for_window(
        self,
        *,
        seed: int,
        day: str,
        start_tick_day: int,
        scenario_id: str,
    ) -> dict[str, float | int | str]:
        obs, _ = self.reset_to_tlc_window(
            seed=seed,
            day=day,
            start_tick_day=start_tick_day,
            scenario_id=scenario_id,
        )
        _ = obs
        row = self._env_health_current_scenario(seed)
        row["scenario_id"] = scenario_id
        row["day"] = day
        row["start_tick_day"] = int(start_tick_day)
        return row

    def _env_health_current_scenario(self, seed: int) -> dict[str, float | int]:
        supply_demand_ratios = []
        feasible_densities = []
        health_feasible_density = []
        health_feasible_densities = []
        available_energy_with_health = []
        soc_binding_rates = []
        active_orders = []
        active_vehicles = []
        for _tick in range(self.scale_config.horizon_ticks):
            snapshot = self.snapshot()
            order_count = len(snapshot.active_orders)
            vehicle_count = len(snapshot.active_vehicles)
            all_edges = self.matcher.build_edges(
                snapshot.active_orders,
                snapshot.active_vehicles,
                self.current_tick,
                include_infeasible=True,
            )
            supply_demand_ratios.append(vehicle_count / max(1, order_count))
            feasible_densities.append(len(snapshot.candidate_edges) / max(1, order_count * max(1, vehicle_count)))
            health_feasible_density.append(len(snapshot.candidate_edges) / max(1, order_count * max(1, vehicle_count)))
            health_feasible_densities.append(len(snapshot.candidate_edges) / max(1, len(all_edges)))
            available_energy_with_health.append(
                sum(
                    vehicle.available_energy_with_health_kwh(self.env_config.battery_health.donor_min_soc_ratio)
                    for vehicle in snapshot.active_vehicles
                )
            )
            soc_binding_rates.append(
                sum(
                    1
                    for vehicle in snapshot.active_vehicles
                    if vehicle.health_floor_kwh(self.env_config.battery_health.donor_min_soc_ratio) > vehicle.reserve_kwh
                )
                / max(1, vehicle_count)
            )
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
            "health_feasible_edge_density": float(np.mean(health_feasible_density)),
            "health_feasible_edge_share": float(np.mean(health_feasible_densities)),
            "mean_available_energy_with_health": float(np.mean(available_energy_with_health)),
            "soc_safety_binding_rate": float(np.mean(soc_binding_rates)),
            "energy_loss_rate_estimate": 1.0 - self.env_config.battery_health.transfer_efficiency,
            "min_supply_demand_ratio": float(np.min(supply_demand_ratios)),
            "max_active_orders": int(np.max(active_orders)),
            "max_active_vehicles": int(np.max(active_vehicles)),
        }

    def _state_feature_values(self, snapshot: EnvironmentSnapshot | None = None) -> dict[str, float]:
        snapshot = snapshot or self.snapshot()
        active_orders = snapshot.active_orders
        active_vehicles = snapshot.active_vehicles
        edges = snapshot.candidate_edges
        order_count = len(active_orders)
        vehicle_count = len(active_vehicles)
        total_demand = sum(order.demand_kwh for order in active_orders)
        total_energy = sum(vehicle.available_energy_kwh() for vehicle in active_vehicles)
        total_health_energy = sum(
            vehicle.available_energy_with_health_kwh(self.env_config.battery_health.donor_min_soc_ratio)
            for vehicle in active_vehicles
        )
        waiting_ratios = [order.waiting_ratio(self.current_tick) for order in active_orders]
        near_deadline = sum(
            1
            for order in active_orders
            if order.max_wait_ticks - order.waiting_ticks(self.current_tick) <= 1
        )
        cancel_risk = [self._cancel_probability(order) for order in active_orders]
        vehicle_flex = [vehicle.time_flexibility_ticks(self.current_tick) for vehicle in active_vehicles]
        fleet_count = sum(1 for vehicle in active_vehicles if vehicle.fleet_flag)
        order_zones = [order.origin_zone for order in active_orders]
        vehicle_zones = [vehicle.current_zone for vehicle in active_vehicles]
        shortage = self.network.shortage_pressure_by_zone(order_zones, vehicle_zones)
        profits = [edge.expected_profit for edge in edges]
        margins = [edge.immediate_profit for edge in edges]
        pickups = [edge.pickup_minutes for edge in edges]
        pickup_distances = [edge.pickup_distance_km_est for edge in edges]
        urgent_orders = [order.order_id for order in active_orders if order.is_urgent()]
        urgent_covered = {edge.order_id for edge in edges if edge.order_id in urgent_orders}
        energy_covered = {edge.order_id for edge in edges}
        ticks_since_last_dispatch = (
            self.current_tick - self.dispatch_ticks[-1]
            if self.dispatch_ticks
            else self.scale_config.horizon_ticks
        )
        edge_orders = {edge.order_id for edge in edges}
        edge_vehicles = {edge.vehicle_id for edge in edges}
        estimated_full_matches = min(len(edge_orders), len(edge_vehicles), order_count)
        full_friction = self.estimate_dispatch_friction(
            dispatch_mode="full",
            matched_count=estimated_full_matches,
            ticks_since_last_dispatch=ticks_since_last_dispatch,
        )
        bucket = self._time_bucket_name()
        arrived_orders = [order for order in self.orders if order.arrival_tick <= self.current_tick]
        served_orders = [order for order in arrived_orders if order.status == ORDER_MATCHED]
        arrived_urgent = [order for order in arrived_orders if order.is_urgent()]
        served_urgent = [order for order in arrived_urgent if order.status == ORDER_MATCHED]
        projected_service_rate = len(served_orders) / max(1, len(arrived_orders))
        projected_urgent_service_rate = len(served_urgent) / max(1, len(arrived_urgent))
        service_risk = max(0.0, self.env_config.service_rate_target - projected_service_rate)
        urgent_service_risk = max(0.0, self.env_config.urgent_service_rate_target - projected_urgent_service_rate)
        soc_binding_rate = (
            sum(
                1
                for vehicle in active_vehicles
                if vehicle.health_floor_kwh(self.env_config.battery_health.donor_min_soc_ratio) > vehicle.reserve_kwh
            )
            / max(1, vehicle_count)
        )
        candidate_density = len(edges) / max(1, order_count * max(1, vehicle_count))
        top_profit_edges = sorted((max(0.0, edge.expected_profit) for edge in edges), reverse=True)[
            : max(1, estimated_full_matches)
        ]
        return {
            "active_order_count_raw": float(order_count),
            "available_vehicle_count_raw": float(vehicle_count),
            "supply_demand_ratio_raw": float(vehicle_count / max(1, order_count)),
            "near_deadline_order_share_raw": float(near_deadline / max(1, order_count)),
            "feasible_edge_density_raw": float(candidate_density),
            "mean_pickup_distance_est_raw": float(np.mean(pickup_distances)) if pickup_distances else 0.0,
            "mean_pickup_time_est_raw": float(np.mean(pickups)) if pickups else 0.0,
            "candidate_margin_proxy_raw": float(sum(top_profit_edges)),
            "service_risk_potential_raw": float(self.service_risk_potential()),
            "active_order_count": order_count / 100.0,
            "total_energy_demand": total_demand / 1000.0,
            "mean_waiting_ratio": float(np.mean(waiting_ratios)) if waiting_ratios else 0.0,
            "near_deadline_order_count": near_deadline / 50.0,
            "near_deadline_order_share": near_deadline / max(1, order_count),
            "cancel_risk_mean": float(np.mean(cancel_risk)) if cancel_risk else 0.0,
            "mean_willingness_to_pay": (
                float(np.mean([order.willingness_to_pay_per_kwh for order in active_orders])) / 12.0
                if active_orders
                else 0.0
            ),
            "available_vehicle_count": vehicle_count / 100.0,
            "total_available_energy": total_energy / 2000.0,
            "mean_energy_surplus": (total_energy / max(1.0, vehicle_count)) / 80.0,
            "mean_time_flexibility": (float(np.mean(vehicle_flex)) / 48.0) if vehicle_flex else 0.0,
            "fleet_available_ratio": fleet_count / max(1, vehicle_count),
            "resource_scarcity_index": total_demand / max(1.0, total_energy),
            "supply_demand_imbalance": (order_count - vehicle_count) / 100.0,
            "top_shortage_zone_pressure": float(np.max(shortage)) / 10.0,
            "future_hotspot_pressure": float(np.mean(shortage)) / 5.0,
            "mean_pickup_time_est": (float(np.mean(pickups)) / 30.0) if pickups else 0.0,
            "mean_pickup_distance_est": (float(np.mean(pickup_distances)) / 20.0) if pickup_distances else 0.0,
            "feasible_edge_density": candidate_density,
            "best_candidate_profit": (float(np.max(profits)) / 50.0) if profits else 0.0,
            "mean_candidate_profit": (float(np.mean(profits)) / 30.0) if profits else 0.0,
            "urgent_feasible_coverage": len(urgent_covered) / max(1, len(urgent_orders)),
            "energy_feasible_coverage": len(energy_covered) / max(1, order_count),
            "mean_available_energy_with_health": (total_health_energy / max(1.0, vehicle_count)) / 80.0,
            "soc_safety_binding_rate": soc_binding_rate,
            "mean_candidate_platform_margin": (float(np.mean(margins)) / 30.0) if margins else 0.0,
            "energy_loss_rate_estimate": (
                sum(edge.energy_loss_kwh for edge in edges) / max(1e-9, sum(edge.donor_output_kwh for edge in edges))
                if edges
                else 0.0
            ),
            "ticks_since_last_dispatch": ticks_since_last_dispatch / max(1, self.scale_config.horizon_ticks),
            "estimated_full_match_friction": full_friction / 100.0,
            "time_bucket_off_peak": 1.0 if bucket == "off_peak" else 0.0,
            "time_bucket_morning_peak": 1.0 if bucket == "morning_peak" else 0.0,
            "time_bucket_midday": 1.0 if bucket == "midday" else 0.0,
            "time_bucket_evening_peak": 1.0 if bucket == "evening_peak" else 0.0,
            "projected_service_risk": service_risk,
            "projected_urgent_service_risk": urgent_service_risk,
        }

    def _observation(self) -> np.ndarray:
        values = self._state_feature_values()
        raw = np.array([values[name] for name in self.observation_names], dtype=np.float32)
        return np.clip(raw, -10.0, 10.0)

    def state_action_features(self, snapshot: EnvironmentSnapshot | None = None) -> dict[str, float]:
        values = self._state_feature_values(snapshot)
        return {
            "active_orders": values["active_order_count_raw"],
            "available_vehicles": values["available_vehicle_count_raw"],
            "supply_demand_ratio": values["supply_demand_ratio_raw"],
            "near_deadline_share": values["near_deadline_order_share_raw"],
            "candidate_density": values["feasible_edge_density_raw"],
            "mean_pickup_distance_est": values["mean_pickup_distance_est_raw"],
            "mean_pickup_time_est": values["mean_pickup_time_est_raw"],
            "projected_service_risk": values["projected_service_risk"],
            "projected_urgent_service_risk": values["projected_urgent_service_risk"],
        }

    def state_potential_proxy(self, snapshot: EnvironmentSnapshot | None = None) -> float:
        values = self._state_feature_values(snapshot)
        return float(
            values["candidate_margin_proxy_raw"]
            + 35.0 * values["feasible_edge_density_raw"]
            + 25.0 * values["urgent_feasible_coverage"]
            - 1.2 * values["mean_pickup_time_est_raw"]
            - 1.0 * values["mean_pickup_distance_est_raw"]
            - 0.04 * values["service_risk_potential_raw"]
            - 18.0 * values["projected_urgent_service_risk"]
            - 30.0 * values["near_deadline_order_share_raw"]
            - 20.0 * values["soc_safety_binding_rate"]
        )

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
        if waiting_ratio <= 0.62:
            return 0.0
        mid_ramp = max(0.0, waiting_ratio - 0.62)
        late_ramp = max(0.0, waiting_ratio - 0.86)
        raw = mid_ramp * order.cancel_sensitivity * 0.42 + late_ramp * order.cancel_sensitivity * 1.85
        return float(np.clip(raw, 0.0, 0.16))

    def compute_dispatch_friction(
        self,
        result: StepResult,
        *,
        dispatch_mode: str,
        matched_count: int,
        tick: int,
    ) -> None:
        friction = self.env_config.dispatch_friction
        if not friction.enabled:
            return
        ticks_since_last = tick - self.dispatch_ticks[-1] if self.dispatch_ticks else float("inf")
        result.ticks_since_last_dispatch = float(ticks_since_last if self.dispatch_ticks else -1.0)
        result.dispatch_setup_cost = float(friction.setup_cost)
        result.dispatch_pair_coordination_cost = float(
            friction.pair_coordination_cost * max(0, int(matched_count))
        )
        result.dispatch_refresh_cost = float(
            self._dispatch_refresh_cost(ticks_since_last)
            if self.dispatch_ticks
            else 0.0
        )
        result.dispatch_full_mode_extra_cost = float(
            friction.full_mode_extra_pair_cost * max(0, int(matched_count))
            if dispatch_mode == "full"
            else 0.0
        )
        result.dispatch_friction_cost = float(
            result.dispatch_setup_cost
            + result.dispatch_pair_coordination_cost
            + result.dispatch_refresh_cost
            + result.dispatch_full_mode_extra_cost
        )
        result.friction_share_of_gross_profit = (
            result.dispatch_friction_cost / result.gross_dispatch_profit
            if result.gross_dispatch_profit > 0.0
            else 0.0
        )

    def estimate_dispatch_friction(
        self,
        *,
        dispatch_mode: str,
        matched_count: int,
        ticks_since_last_dispatch: float,
    ) -> float:
        friction = self.env_config.dispatch_friction
        if not friction.enabled:
            return 0.0
        pair_cost = friction.pair_coordination_cost * max(0, int(matched_count))
        full_extra = (
            friction.full_mode_extra_pair_cost * max(0, int(matched_count))
            if dispatch_mode == "full"
            else 0.0
        )
        refresh_cost = self._dispatch_refresh_cost(ticks_since_last_dispatch)
        return float(friction.setup_cost + pair_cost + full_extra + refresh_cost)

    def _dispatch_refresh_cost(self, ticks_since_last_dispatch: float) -> float:
        friction = self.env_config.dispatch_friction
        refresh_cost = 0.0
        if np.isfinite(ticks_since_last_dispatch) and friction.refresh_cost > 0.0:
            refresh_cost = friction.refresh_cost * float(
                np.exp(-max(0.0, float(ticks_since_last_dispatch)) / max(1e-6, friction.refresh_decay_ticks))
            )
        return float(refresh_cost)

    def service_risk_potential(self) -> float:
        arrived_orders = [order for order in self.orders if order.arrival_tick <= self.current_tick]
        if len(arrived_orders) < max(20, int(self.scale_config.total_orders * 0.05)):
            return 0.0
        served_orders = [order for order in arrived_orders if order.status == ORDER_MATCHED]
        arrived_urgent = [order for order in arrived_orders if order.is_urgent()]
        served_urgent = [order for order in arrived_urgent if order.status == ORDER_MATCHED]
        service_rate = len(served_orders) / max(1, len(arrived_orders))
        urgent_service_rate = len(served_urgent) / max(1, len(arrived_urgent))
        expired_rate = (
            sum(1 for order in arrived_orders if order.status == ORDER_EXPIRED) / max(1, len(arrived_orders))
        )
        cancelled_rate = (
            sum(1 for order in arrived_orders if order.status == ORDER_CANCELLED) / max(1, len(arrived_orders))
        )
        service_gap = max(0.0, self.env_config.service_rate_target - service_rate)
        urgent_gap = max(0.0, self.env_config.urgent_service_rate_target - urgent_service_rate)
        expired_gap = max(0.0, expired_rate - self.env_config.expired_rate_cap)
        cancelled_gap = max(0.0, cancelled_rate - self.env_config.cancelled_rate_cap)
        return float(
            self.env_config.profit_scale_fallback
            * len(arrived_orders)
            * (1.20 * service_gap + 1.50 * urgent_gap + expired_gap + 0.80 * cancelled_gap)
        )

    def _base_step_reward(self, result: StepResult) -> float:
        wait_penalty = self.env_config.wait_penalty_per_order_tick * len(
            [order for order in self.orders if order.is_active(self.current_tick)]
        )
        batch_bonus = 0.0
        if result.dispatch_executed and result.accepted_count > 0:
            batch_bonus = min(6.0, 0.15 * result.accepted_count)
        return (
            result.platform_profit
            - self.env_config.expired_penalty * result.expired_count
            - self.env_config.cancelled_penalty * result.cancelled_count
            - wait_penalty
            + batch_bonus
        )

    def _step_reward(self, result: StepResult, *, risk_before: float | None = None) -> float:
        wait_penalty = self.env_config.wait_penalty_per_order_tick * len(
            [order for order in self.orders if order.is_active(self.current_tick)]
        )
        risk_after = self.service_risk_potential()
        if risk_before is None:
            risk_before = risk_after
        service_delta_reward = self.env_config.service_risk_delta_weight * float(
            np.clip(
                risk_before - risk_after,
                -self.env_config.service_risk_delta_clip,
                self.env_config.service_risk_delta_clip,
            )
        )
        wait_opportunity_bonus = self._wait_opportunity_bonus(wait_penalty) if result.dispatch_mode == "wait" else 0.0
        return self._base_step_reward(result) + service_delta_reward + wait_opportunity_bonus

    def _wait_opportunity_bonus(self, wait_penalty: float) -> float:
        if not self._last_wait_tradeoff:
            return 0.0
        ticks_since_last = self.current_tick - self.dispatch_ticks[-1] if self.dispatch_ticks else float("inf")
        refresh_cost = self._dispatch_refresh_cost(ticks_since_last)
        wait_opportunity = (
            float(self._last_wait_tradeoff.get("candidate_profit_delta", 0.0))
            - self.env_config.expired_penalty * float(self._last_wait_tradeoff.get("expired_after_wait", 0.0))
            - self.env_config.cancelled_penalty * float(self._last_wait_tradeoff.get("cancelled_after_wait", 0.0))
            - refresh_cost
            - wait_penalty
        )
        bounded = float(np.clip(max(0.0, wait_opportunity), 0.0, self.env_config.wait_opportunity_cap))
        return self.env_config.wait_opportunity_weight * bounded

    def estimate_wait_opportunity(self, snapshot: EnvironmentSnapshot | None = None) -> float:
        snapshot = snapshot or self.snapshot()
        future_orders = [order for order in self.orders if order.arrival_tick == self.current_tick + 1]
        current_profit = sum(edge.expected_profit for edge in snapshot.candidate_edges)
        future_profit_proxy = 0.0
        if future_orders:
            active_vehicles = snapshot.active_vehicles
            for order in future_orders:
                for vehicle in active_vehicles:
                    if vehicle.join_tick <= self.current_tick + 1 < vehicle.leave_tick:
                        travel = self.network.travel_minutes(vehicle.current_zone, order.origin_zone, self.current_tick + 1)
                        if travel <= self.env_config.pickup_cap_minutes:
                            future_profit_proxy += max(0.0, order.demand_kwh * order.willingness_to_pay_per_kwh * 0.18)
                            break
        ticks_since_last = (self.current_tick + 1) - self.dispatch_ticks[-1] if self.dispatch_ticks else float("inf")
        refresh_cost = self._dispatch_refresh_cost(ticks_since_last)
        near_deadline = sum(
            1
            for order in snapshot.active_orders
            if order.max_wait_ticks - order.waiting_ticks(self.current_tick) <= 1
        )
        deadline_cost = self.env_config.expired_penalty * near_deadline
        return float(future_profit_proxy + 0.08 * current_profit - refresh_cost - deadline_cost)

    def _time_bucket_name(self) -> str:
        tick_day = self.scenario_start_tick_day + self.current_tick
        hour = ((tick_day * self.env_config.tick_minutes) % 1440) / 60.0
        if 6 <= hour < 10:
            return "morning_peak"
        if 10 <= hour < 15:
            return "midday"
        if 15 <= hour < 20:
            return "evening_peak"
        return "off_peak"

    def _record_action_trace(
        self,
        tick: int,
        action: int,
        snapshot: EnvironmentSnapshot,
        result: StepResult,
        q_values: tuple[float, ...] | None,
    ) -> None:
        waiting_ratios = [order.waiting_ratio(tick) for order in snapshot.active_orders]
        near_deadline = sum(1 for order in snapshot.active_orders if order.max_wait_ticks - order.waiting_ticks(tick) <= 1)
        row: dict[str, object] = {
            "tick": tick,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "active_orders": len(snapshot.active_orders),
            "active_vehicles": len(snapshot.active_vehicles),
            "candidate_edges": len(snapshot.candidate_edges),
            "near_deadline_orders": near_deadline,
            "mean_waiting_ratio": float(np.mean(waiting_ratios)) if waiting_ratios else 0.0,
            "matched_count": result.matched_count,
            "accepted_count": result.accepted_count,
            "expired_count": result.expired_count,
            "cancelled_count": result.cancelled_count,
            "gross_dispatch_profit": result.gross_dispatch_profit,
            "dispatch_friction_cost": result.dispatch_friction_cost,
            "dispatch_setup_cost": result.dispatch_setup_cost,
            "dispatch_pair_coordination_cost": result.dispatch_pair_coordination_cost,
            "dispatch_refresh_cost": result.dispatch_refresh_cost,
            "dispatch_full_mode_extra_cost": result.dispatch_full_mode_extra_cost,
            "ticks_since_last_dispatch": result.ticks_since_last_dispatch,
            "friction_share_of_gross_profit": result.friction_share_of_gross_profit,
            "buyer_payment": result.buyer_payment,
            "seller_reimbursement": result.seller_reimbursement,
            "energy_loss_kwh": result.energy_loss_kwh,
            "total_pickup_distance_km": result.total_pickup_distance_km,
            "mean_pickup_distance_km": result.mean_pickup_distance_km,
            "mean_donor_soc_after_kwh": result.mean_donor_soc_after_kwh,
            "battery_health_rejection_count": result.battery_health_rejection_count,
        }
        for idx in range(ACTION_COUNT):
            row[f"q_action_{idx}"] = "" if q_values is None or idx >= len(q_values) else float(q_values[idx])
        self.action_trace.append(row)

    def _record_dispatch_trace(self, tick: int, action: int, plan_edges: int, result: StepResult) -> None:
        profit_per_dispatch = result.platform_profit
        profit_per_accepted = result.platform_profit / max(1, result.accepted_count)
        self.dispatch_trace.append(
            {
                "tick": tick,
                "action": action,
                "dispatch_mode": ACTION_NAMES[action],
                "dispatch_capacity": result.dispatch_capacity,
                "gross_dispatch_profit": result.gross_dispatch_profit,
                "dispatch_friction_cost": result.dispatch_friction_cost,
                "dispatch_setup_cost": result.dispatch_setup_cost,
                "dispatch_pair_coordination_cost": result.dispatch_pair_coordination_cost,
                "dispatch_refresh_cost": result.dispatch_refresh_cost,
                "dispatch_full_mode_extra_cost": result.dispatch_full_mode_extra_cost,
                "ticks_since_last_dispatch": result.ticks_since_last_dispatch,
                "friction_share_of_gross_profit": result.friction_share_of_gross_profit,
                "candidate_edges": plan_edges,
                "matched_count": result.matched_count,
                "accepted_count": result.accepted_count,
                "rejected_count": result.rejected_count,
                "platform_profit": result.platform_profit,
                "buyer_payment": result.buyer_payment,
                "seller_reimbursement": result.seller_reimbursement,
                "seller_energy_cost": result.seller_energy_cost,
                "seller_degradation_cost": result.seller_degradation_cost,
                "seller_service_premium": result.seller_service_premium,
                "delivered_kwh": result.delivered_kwh,
                "donor_output_kwh": result.donor_output_kwh,
                "energy_loss_kwh": result.energy_loss_kwh,
                "total_pickup_distance_km": result.total_pickup_distance_km,
                "mean_pickup_distance_km": result.mean_pickup_distance_km,
                "mean_donor_soc_after_kwh": result.mean_donor_soc_after_kwh,
                "battery_health_rejection_count": result.battery_health_rejection_count,
                "profit_per_dispatch": profit_per_dispatch,
                "profit_per_accepted_order": profit_per_accepted,
                "mean_pickup_minutes": result.mean_pickup_minutes,
                "mean_commitment_ticks": result.mean_commitment_ticks,
            }
        )

    def _record_wait_tradeoff_trace(
        self,
        tick: int,
        before_snapshot: EnvironmentSnapshot,
        *,
        expired: int,
        cancelled: int,
    ) -> dict[str, object]:
        after_snapshot = self.snapshot()
        before_edges = {(edge.order_id, edge.vehicle_id): edge for edge in before_snapshot.candidate_edges}
        after_edges = {(edge.order_id, edge.vehicle_id): edge for edge in after_snapshot.candidate_edges}
        new_edges = [edge for key, edge in after_edges.items() if key not in before_edges]
        before_orders = {order.order_id for order in before_snapshot.active_orders}
        after_orders = {order.order_id for order in after_snapshot.active_orders}
        row = {
            "tick": tick,
            "active_orders_before": len(before_snapshot.active_orders),
            "active_orders_after": len(after_snapshot.active_orders),
            "new_active_orders": len(after_orders - before_orders),
            "candidate_edges_before": len(before_snapshot.candidate_edges),
            "candidate_edges_after": len(after_snapshot.candidate_edges),
            "new_candidate_edges": len(new_edges),
            "candidate_profit_delta": sum(edge.expected_profit for edge in after_snapshot.candidate_edges)
            - sum(edge.expected_profit for edge in before_snapshot.candidate_edges),
            "new_candidate_profit": sum(edge.expected_profit for edge in new_edges),
            "new_candidate_margin": sum(edge.immediate_profit for edge in new_edges),
            "new_candidate_energy_loss_kwh": sum(edge.energy_loss_kwh for edge in new_edges),
            "expired_after_wait": expired,
            "cancelled_after_wait": cancelled,
        }
        self.wait_tradeoff_trace.append(row)
        return row

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
                network = TLCManhattanZoneNetwork(data=data, minutes_per_tick=self.env_config.tick_minutes)
                generator = TLCManhattanScenarioGenerator(
                    self.env_config,
                    self.scale_config,
                    data,
                    phase=self.scenario_phase,
                )
                return network, generator
            except FileNotFoundError:
                if not (self.env_config.allow_synthetic_smoke_fallback and self.scale_config.name == "smoke"):
                    raise
        network = ZoneNetwork(self.env_config.zone_count, minutes_per_tick=self.env_config.tick_minutes)
        generator = ScenarioGenerator(self.env_config, self.scale_config)
        return network, generator
