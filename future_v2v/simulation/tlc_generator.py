from __future__ import annotations

import numpy as np
import pandas as pd

from future_v2v.config import EnvironmentConfig, ScaleConfig
from future_v2v.data.tlc_manhattan import TLCManhattanData
from future_v2v.simulation.entities import Order, Vehicle
from future_v2v.simulation.generator import Scenario


class TLCManhattanScenarioGenerator:
    WINDOW_SEARCH_ATTEMPTS = 96

    def __init__(
        self,
        env_config: EnvironmentConfig,
        scale_config: ScaleConfig,
        data: TLCManhattanData,
        phase: str = "train",
    ) -> None:
        self.env_config = env_config
        self.scale_config = scale_config
        self.data = data
        self.phase = phase

    def generate(self, seed: int) -> Scenario:
        rng = np.random.default_rng(seed)
        day, start_tick_day, rows = self._select_window(rng)
        return self.generate_from_window(
            seed=seed,
            day=day,
            start_tick_day=start_tick_day,
            scenario_id=f"seed_{seed}",
        )

    def generate_from_window(self, *, seed: int, day: str, start_tick_day: int, scenario_id: str = "") -> Scenario:
        rng = np.random.default_rng(seed)
        rows = self.data.rows_for_window(day, start_tick_day, self.scale_config.horizon_ticks)
        if rows.empty:
            raise RuntimeError(f"No TLC Manhattan rows found for manifest window day={day}, start_tick_day={start_tick_day}.")
        sampled_orders = self._sample_order_rows(rows, rng)
        orders = self._generate_orders(sampled_orders, start_tick_day, rng)
        vehicles = self._generate_vehicles(start_tick_day, rng)
        return Scenario(
            orders=orders,
            vehicles=vehicles,
            source="tlc_manhattan",
            day=day,
            start_tick_day=start_tick_day,
            scenario_id=scenario_id,
        )

    def manifest_row(self, *, seed: int, scenario_id: str | None = None) -> dict[str, float | int | str]:
        rng = np.random.default_rng(seed)
        day, start_tick_day, rows = self._select_window(rng)
        return self.manifest_row_for_window(
            seed=seed,
            day=day,
            start_tick_day=start_tick_day,
            scenario_id=scenario_id or f"eval_{seed}",
            rows=rows,
        )

    def manifest_row_for_window(
        self,
        *,
        seed: int,
        day: str,
        start_tick_day: int,
        scenario_id: str,
        rows: pd.DataFrame | None = None,
    ) -> dict[str, float | int | str]:
        window_rows = rows if rows is not None else self.data.rows_for_window(day, start_tick_day, self.scale_config.horizon_ticks)
        if window_rows.empty:
            raw_rows = 0
            mean_pressure = 0.0
            mean_price = 0.0
            mean_duration = 0.0
        else:
            raw_rows = int(len(window_rows))
            mean_pressure = float(window_rows["zone_pressure"].mean())
            mean_price = float(window_rows["willingness_to_pay_proxy"].mean())
            mean_duration = float(window_rows["duration_minutes"].mean())
        bucket = self._time_of_day_bucket(start_tick_day)
        return {
            "scenario_id": scenario_id,
            "seed": int(seed),
            "day": str(day),
            "start_tick_day": int(start_tick_day),
            "time_of_day_bucket": bucket,
            "raw_tlc_rows": raw_rows,
            "mean_zone_pressure": mean_pressure,
            "mean_wtp_proxy": mean_price,
            "mean_duration_minutes": mean_duration,
        }

    def _select_window(self, rng: np.random.Generator) -> tuple[str, int, pd.DataFrame]:
        available_days = self._eligible_days()
        horizon = self.scale_config.horizon_ticks
        latest_start = max(0, self.data.ticks_per_day - horizon)
        best: tuple[str, int, pd.DataFrame] | None = None
        min_raw_rows = self._minimum_raw_rows_for_fixed_demand()
        for _ in range(self.WINDOW_SEARCH_ATTEMPTS):
            day = str(rng.choice(available_days))
            start_tick_day = self._sample_start_tick_day(rng, latest_start)
            rows = self.data.rows_for_window(day, start_tick_day, horizon)
            if best is None or len(rows) > len(best[2]):
                best = (day, start_tick_day, rows)
            if len(rows) >= min_raw_rows:
                return day, start_tick_day, rows
        if best is None or best[2].empty:
            fallback_pool = self.data.trip_rows[self.data.trip_rows["pickup_date"].isin(available_days)]
            if fallback_pool.empty:
                raise RuntimeError("No TLC Manhattan trip rows found for any sampled episode window.")
            anchor = fallback_pool.iloc[int(rng.integers(0, len(fallback_pool)))]
            day = str(anchor.pickup_date)
            start_tick_day = int(np.clip(int(anchor.pickup_tick_day) - horizon // 2, 0, latest_start))
            rows = self.data.rows_for_window(day, start_tick_day, horizon)
            if rows.empty:
                raise RuntimeError("No TLC Manhattan trip rows found around fallback anchor row.")
            return day, start_tick_day, rows
        return best

    def _sample_start_tick_day(self, rng: np.random.Generator, latest_start: int) -> int:
        ticks_per_hour = max(1, int(round(60 / self.env_config.tick_minutes)))
        horizon = self.scale_config.horizon_ticks
        windows = [
            (6 * ticks_per_hour, 8 * ticks_per_hour, "morning_peak"),
            (10 * ticks_per_hour, 13 * ticks_per_hour, "midday"),
            (15 * ticks_per_hour, 17 * ticks_per_hour, "evening_peak"),
            (0, max(1, 6 * ticks_per_hour - horizon), "stable_off_peak"),
            (20 * ticks_per_hour, self.data.ticks_per_day - horizon, "stable_off_peak"),
        ]
        candidates = [
            (max(0, start), min(latest_start, end))
            for start, end, _label in windows
            if min(latest_start, end) >= max(0, start)
        ]
        if not candidates:
            return int(rng.integers(0, latest_start + 1))
        start, end = candidates[int(rng.integers(0, len(candidates)))]
        return int(rng.integers(start, end + 1))

    def _minimum_raw_rows_for_fixed_demand(self) -> int:
        if self.scale_config.total_orders <= 0:
            return 0
        sample_rate = max(0.01, self.env_config.demand_sample_rate)
        return max(20, int(np.ceil(self.scale_config.total_orders / sample_rate)))

    def _time_of_day_bucket(self, tick_day: int) -> str:
        hour = (tick_day * self.env_config.tick_minutes) / 60.0
        if 6 <= hour < 10:
            return "morning_peak"
        if 10 <= hour < 15:
            return "midday"
        if 15 <= hour < 20:
            return "evening_peak"
        return "off_peak"

    def _eligible_days(self) -> list[str]:
        configured = [str(day) for day in self._configured_days_for_phase()]
        if configured:
            available = set(self.data.days)
            matched = [day for day in configured if day in available]
            if matched:
                return matched
            raise ValueError(
                f"No TLC days match configured {self.phase}_days={configured}; "
                f"available days include {self.data.days[:5]}"
            )
        excluded = set(self._excluded_days_for_phase())
        if excluded:
            remaining = [day for day in self.data.days if day not in excluded]
            if remaining:
                return remaining
        return self.data.days

    def _configured_days_for_phase(self) -> list[str]:
        if self.phase == "train":
            return list(self.env_config.train_days or [])
        if self.phase in {"eval", "manifest"}:
            return list(self.env_config.eval_days or [])
        if self.phase == "validation":
            return list(self.env_config.eval_days or self.env_config.train_days or [])
        raise ValueError(f"unknown TLC scenario phase={self.phase!r}; expected train, validation, eval, or manifest")

    def _excluded_days_for_phase(self) -> list[str]:
        if self.phase == "train" and self.env_config.eval_days:
            return [str(day) for day in self.env_config.eval_days]
        if self.phase in {"eval", "manifest"} and self.env_config.train_days:
            return [str(day) for day in self.env_config.train_days]
        return []

    def _sample_order_rows(self, rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        if rows.empty:
            return rows
        target = max(0, int(self.scale_config.total_orders))
        if target == 0:
            return rows.iloc[0:0].copy()
        weights = np.clip(rows["zone_pressure"].to_numpy(dtype=float), 0.4, 2.5)
        weights = weights / weights.sum()
        if len(rows) < target:
            indices = rng.choice(rows.index.to_numpy(), size=target, replace=True, p=weights)
            sampled = rows.loc[indices].copy()
        else:
            indices = rng.choice(rows.index.to_numpy(), size=target, replace=False, p=weights)
            sampled = rows.loc[indices].copy()
        return sampled.sort_values(["pickup_tick_day", "pu_zone"]).reset_index(drop=True)

    def _generate_orders(self, rows: pd.DataFrame, start_tick_day: int, rng: np.random.Generator) -> list[Order]:
        orders: list[Order] = []
        for order_id, row in enumerate(rows.itertuples(index=False)):
            pressure = float(getattr(row, "zone_pressure"))
            arrival_tick = int(getattr(row, "pickup_tick_day") - start_tick_day)
            phase = arrival_tick / max(1, self.scale_config.horizon_ticks)
            temporal_pressure = self._temporal_pressure(pressure, phase)
            demand = float(np.clip(getattr(row, "demand_kwh_base") * rng.lognormal(0.0, 0.08), 3.0, 22.0))
            max_wait = self._sample_wait_ticks(temporal_pressure, rng)
            if phase <= 0.18 or phase >= 0.78:
                max_wait = max(2, max_wait - 1)
            elif 0.48 <= phase <= 0.68:
                max_wait += 2
            urgency_markup = 1.0 + max(0.0, 6 - max_wait) * 0.08 + 0.08 * max(0.0, pressure - 1.0)
            wtp = float(np.clip(getattr(row, "willingness_to_pay_proxy") * urgency_markup, 3.2, 12.0))
            phase_cancel_shift = 0.04 if phase <= 0.18 or phase >= 0.78 else (-0.025 if 0.48 <= phase <= 0.68 else 0.0)
            cancel_sensitivity = float(
                np.clip(
                    rng.normal(0.15 + 0.035 * temporal_pressure + 0.02 * (max_wait <= 4) + phase_cancel_shift, 0.025),
                    0.08,
                    0.34,
                )
            )
            orders.append(
                Order(
                    order_id=order_id,
                    arrival_tick=arrival_tick,
                    origin_zone=int(getattr(row, "pu_zone")),
                    destination_zone=int(getattr(row, "do_zone")),
                    demand_kwh=demand,
                    max_wait_ticks=max_wait,
                    willingness_to_pay_per_kwh=wtp,
                    cancel_sensitivity=cancel_sensitivity,
                )
            )
        orders.sort(key=lambda order: (order.arrival_tick, order.order_id))
        return orders

    def _generate_vehicles(self, start_tick_day: int, rng: np.random.Generator) -> list[Vehicle]:
        horizon = self.scale_config.horizon_ticks + self.scale_config.terminal_buffer_ticks
        start_hour = (start_tick_day * self.env_config.tick_minutes) / 60.0
        stable_off_peak = start_hour < 2.5 or start_hour >= 20.0
        supply_multiplier = 1.35 if stable_off_peak else 1.0
        candidate_count = max(
            1,
            int(self.scale_config.candidate_vehicles * max(0.05, self.env_config.supply_scale) * supply_multiplier),
        )
        supply_rows = self.data.supply_rows_for_window(start_tick_day, self.scale_config.horizon_ticks)
        vehicles: list[Vehicle] = []
        for candidate_id in range(candidate_count):
            if rng.random() > self.scale_config.vehicle_join_probability:
                continue
            source_row = self._sample_supply_row(supply_rows, rng)
            fleet = bool(rng.random() < self.scale_config.fleet_probability)
            if source_row is None:
                join_tick = int(rng.integers(0, max(1, self.scale_config.horizon_ticks)))
                current_zone = int(rng.integers(0, self.data.zone_count))
                destination_zone = int(rng.integers(0, self.data.zone_count))
            else:
                join_tick = int(np.clip(int(source_row.dropoff_tick_day) - start_tick_day, 0, self.scale_config.horizon_ticks - 1))
                current_zone = int(source_row.do_zone)
                destination_zone = int(source_row.pu_zone)
            if fleet:
                online_duration = int(rng.integers(150, 290) / self.env_config.tick_minutes)
                if stable_off_peak:
                    online_duration = int(online_duration * rng.uniform(1.10, 1.25))
            else:
                online_duration = int(rng.integers(70, 165) / self.env_config.tick_minutes)
                join_phase = join_tick / max(1, self.scale_config.horizon_ticks)
                if stable_off_peak:
                    online_duration = int(online_duration * rng.uniform(1.45, 1.75))
                elif join_phase <= 0.20 or join_phase >= 0.72:
                    online_duration = max(4, int(online_duration * rng.uniform(0.45, 0.70)))
                elif 0.45 <= join_phase <= 0.65:
                    online_duration = int(online_duration * rng.uniform(1.05, 1.30))
            leave_tick = min(horizon, join_tick + online_duration)
            if leave_tick <= join_tick:
                leave_tick = min(horizon, join_tick + 1)
            capacity = float(rng.choice([55.0, 65.0, 75.0, 90.0, 105.0]))
            soc_ratio = self._sample_soc_ratio(rng, fleet=fleet)
            reserve = float(rng.uniform(10.0, 22.0))
            energy_cost, service_premium = self._sample_price_components(rng, fleet=fleet)
            time_cost = float(np.clip(rng.normal(0.055 if fleet else 0.075, 0.018), 0.02, 0.13))
            accept_sensitivity = float(np.clip(rng.normal(0.72 if fleet else 0.95, 0.12), 0.45, 1.35))
            vehicles.append(
                Vehicle(
                    vehicle_id=len(vehicles),
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
            _ = candidate_id
        vehicles.sort(key=lambda vehicle: (vehicle.join_tick, vehicle.vehicle_id))
        return vehicles

    def _sample_wait_ticks(self, pressure: float, rng: np.random.Generator) -> int:
        # TLC Manhattan zone pressure is centered well above 1.0, so thresholds
        # are calibrated against the empirical month rather than the synthetic grid.
        if pressure >= 2.30:
            minute_choices = np.array([9, 12, 15, 18])
            probs = np.array([0.18, 0.30, 0.32, 0.20])
        elif pressure <= 1.95:
            minute_choices = np.array([36, 45, 54, 63, 75])
            probs = np.array([0.16, 0.24, 0.28, 0.20, 0.12])
        else:
            minute_choices = np.array([18, 24, 30, 39, 48])
            probs = np.array([0.14, 0.22, 0.26, 0.22, 0.16])
        choices = np.maximum(1, np.ceil(minute_choices / self.env_config.tick_minutes).astype(int))
        return int(rng.choice(choices, p=probs / probs.sum()))

    @staticmethod
    def _temporal_pressure(base_pressure: float, phase: float) -> float:
        early_deadline = 0.35 * np.exp(-((phase - 0.12) ** 2) / 0.012)
        demand_burst = 0.42 * np.exp(-((phase - 0.36) ** 2) / 0.018)
        stable_lull = -0.36 * np.exp(-((phase - 0.58) ** 2) / 0.022)
        late_departure = 0.28 * np.exp(-((phase - 0.84) ** 2) / 0.014)
        return float(np.clip(base_pressure + early_deadline + demand_burst + stable_lull + late_departure, 0.45, 2.7))

    @staticmethod
    def _sample_supply_row(rows: pd.DataFrame, rng: np.random.Generator):
        if rows.empty:
            return None
        idx = int(rng.integers(0, len(rows)))
        return rows.iloc[idx]

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
