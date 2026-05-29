from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from future_v2v.config import EnvironmentConfig, ScaleConfig
from future_v2v.simulation.entities import Order, Vehicle


@dataclass(frozen=True)
class Scenario:
    orders: list[Order]
    vehicles: list[Vehicle]
    source: str = "synthetic"
    day: str = ""
    start_tick_day: int = 0
    scenario_id: str = ""


class ScenarioGenerator:
    def __init__(self, env_config: EnvironmentConfig, scale_config: ScaleConfig) -> None:
        self.env_config = env_config
        self.scale_config = scale_config

    def generate(self, seed: int) -> Scenario:
        rng = np.random.default_rng(seed)
        orders = self._generate_orders(rng)
        vehicles = self._generate_vehicles(rng)
        return Scenario(orders=orders, vehicles=vehicles)

    def _arrival_ticks(self, rng: np.random.Generator, total: int) -> np.ndarray:
        horizon = self.scale_config.horizon_ticks
        centers = np.array([0.28 * horizon, 0.68 * horizon])
        modes = rng.choice(len(centers), size=total, p=np.array([0.48, 0.52]))
        ticks = rng.normal(centers[modes], horizon * 0.16, size=total)
        return np.clip(np.rint(ticks), 0, horizon - 1).astype(int)

    def _zone_weights(self, tick: int) -> np.ndarray:
        zone_count = self.env_config.zone_count
        zones = np.arange(zone_count)
        rotating_hotspot = int((tick / max(1, self.scale_config.horizon_ticks)) * zone_count) % zone_count
        base = np.ones(zone_count)
        base += 2.4 * np.exp(-np.abs(zones - rotating_hotspot) / 2.0)
        base += 1.2 * np.exp(-np.abs(zones - (zone_count - 1 - rotating_hotspot)) / 3.0)
        return base / base.sum()

    def _generate_orders(self, rng: np.random.Generator) -> list[Order]:
        ticks = self._arrival_ticks(rng, self.scale_config.total_orders)
        orders: list[Order] = []
        wait_minutes = np.array([6, 9, 12, 15, 18, 24, 30, 36])
        wait_choices = np.ceil(wait_minutes / self.env_config.tick_minutes).astype(int)
        wait_probs = np.array([0.06, 0.09, 0.12, 0.15, 0.18, 0.20, 0.14, 0.06])
        wait_probs = wait_probs / wait_probs.sum()
        for order_id, tick in enumerate(ticks):
            origin = int(rng.choice(self.env_config.zone_count, p=self._zone_weights(int(tick))))
            destination = int(rng.integers(0, self.env_config.zone_count))
            demand = float(np.clip(rng.lognormal(mean=2.05, sigma=0.38), 3.0, 22.0))
            max_wait = int(rng.choice(wait_choices, p=wait_probs))
            urgency_markup = 1.0 + max(0.0, 7 - max_wait) * 0.09
            wtp = float(np.clip(rng.normal(5.6 * urgency_markup + 0.05 * demand, 0.9), 3.2, 12.0))
            cancel_sensitivity = float(np.clip(rng.normal(0.16 + 0.03 * urgency_markup, 0.025), 0.08, 0.28))
            orders.append(
                Order(
                    order_id=order_id,
                    arrival_tick=int(tick),
                    origin_zone=origin,
                    destination_zone=destination,
                    demand_kwh=demand,
                    max_wait_ticks=max_wait,
                    willingness_to_pay_per_kwh=wtp,
                    cancel_sensitivity=cancel_sensitivity,
                )
            )
        orders.sort(key=lambda order: (order.arrival_tick, order.order_id))
        return orders

    def _generate_vehicles(self, rng: np.random.Generator) -> list[Vehicle]:
        vehicles: list[Vehicle] = []
        horizon = self.scale_config.horizon_ticks + self.scale_config.terminal_buffer_ticks
        vehicle_id = 0
        for candidate_id in range(self.scale_config.candidate_vehicles):
            if rng.random() > self.scale_config.vehicle_join_probability:
                continue
            fleet = bool(rng.random() < self.scale_config.fleet_probability)
            join_tick = int(np.clip(rng.normal(0.20 * horizon, 0.22 * horizon), 0, self.scale_config.horizon_ticks - 1))
            if fleet:
                online_duration = int(rng.integers(150, 290) / self.env_config.tick_minutes)
            else:
                online_duration = int(rng.integers(80, 190) / self.env_config.tick_minutes)
            leave_tick = min(horizon, join_tick + online_duration)
            capacity = float(rng.choice([55.0, 65.0, 75.0, 90.0, 105.0]))
            soc_ratio = self._sample_soc_ratio(rng, fleet=fleet)
            reserve = float(rng.uniform(10.0, 22.0))
            energy_cost, service_premium = self._sample_price_components(rng, fleet=fleet)
            time_cost = float(np.clip(rng.normal(0.055 if fleet else 0.075, 0.018), 0.02, 0.13))
            accept_sensitivity = float(np.clip(rng.normal(0.75 if fleet else 0.95, 0.12), 0.45, 1.35))
            current_zone = int(rng.integers(0, self.env_config.zone_count))
            destination_zone = int(rng.integers(0, self.env_config.zone_count))
            vehicles.append(
                Vehicle(
                    vehicle_id=vehicle_id,
                    join_tick=join_tick,
                    leave_tick=leave_tick,
                    current_zone=current_zone,
                    destination_zone=destination_zone,
                    battery_capacity_kwh=capacity,
                    current_soc_kwh=capacity * soc_ratio,
                    reserve_kwh=reserve,
                    energy_cost_per_kwh=energy_cost,
                    service_premium_per_kwh=service_premium,
                    time_cost_per_min=time_cost,
                    owner_accept_sensitivity=accept_sensitivity,
                    fleet_flag=fleet,
                )
            )
            vehicle_id += 1
            _ = candidate_id
        vehicles.sort(key=lambda vehicle: (vehicle.join_tick, vehicle.vehicle_id))
        return vehicles

    @staticmethod
    def _sample_soc_ratio(rng: np.random.Generator, *, fleet: bool) -> float:
        if fleet:
            raw = rng.beta(7.0, 3.0)
            return float(np.clip(0.45 + 0.48 * raw, 0.48, 0.90))
        raw = rng.beta(5.2, 3.8)
        return float(np.clip(0.38 + 0.50 * raw, 0.42, 0.84))

    @staticmethod
    def _sample_price_components(rng: np.random.Generator, *, fleet: bool) -> tuple[float, float]:
        energy_cost = float(np.clip(rng.lognormal(mean=np.log(0.22 if fleet else 0.28), sigma=0.18), 0.12, 0.55))
        service_premium = float(np.clip(rng.lognormal(mean=np.log(1.75 if fleet else 2.25), sigma=0.22), 1.0, 4.2))
        return energy_cost, service_premium
