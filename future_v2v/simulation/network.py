from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from future_v2v.data.tlc_manhattan import TLCManhattanData, TICKS_PER_DAY


@dataclass(frozen=True)
class ZoneNetwork:
    zone_count: int
    minutes_per_tick: int = 5

    def __post_init__(self) -> None:
        side = int(math.ceil(math.sqrt(self.zone_count)))
        coords = []
        for zone in range(self.zone_count):
            coords.append((zone // side, zone % side))
        object.__setattr__(self, "_coords", tuple(coords))

    def distance_units(self, origin_zone: int, destination_zone: int) -> float:
        ox, oy = self._coords[origin_zone]
        dx, dy = self._coords[destination_zone]
        return float(abs(ox - dx) + abs(oy - dy) + 1)

    def traffic_multiplier(self, tick: int) -> float:
        morning_peak = math.exp(-((tick - 10) ** 2) / 42.0)
        evening_peak = math.exp(-((tick - 32) ** 2) / 55.0)
        return 0.92 + 0.28 * morning_peak + 0.22 * evening_peak

    def travel_ticks(self, origin_zone: int, destination_zone: int, tick: int) -> float:
        base = self.distance_units(origin_zone, destination_zone) * 0.55
        return max(0.4, base * self.traffic_multiplier(tick))

    def travel_minutes(self, origin_zone: int, destination_zone: int, tick: int) -> float:
        return self.travel_ticks(origin_zone, destination_zone, tick) * self.minutes_per_tick

    def shortage_pressure_by_zone(
        self,
        active_order_zones: list[int],
        active_vehicle_zones: list[int],
    ) -> np.ndarray:
        demand = np.bincount(active_order_zones, minlength=self.zone_count).astype(float)
        supply = np.bincount(active_vehicle_zones, minlength=self.zone_count).astype(float)
        return (demand + 1.0) / (supply + 1.0)


class TLCManhattanZoneNetwork:
    def __init__(self, data: TLCManhattanData, minutes_per_tick: int = 5) -> None:
        self.data = data
        self.zone_count = data.zone_count
        self.minutes_per_tick = minutes_per_tick
        self.scenario_start_tick_day = 0

    def set_episode_context(self, *, start_tick_day: int) -> None:
        self.scenario_start_tick_day = int(start_tick_day)

    def distance_units(self, origin_zone: int, destination_zone: int) -> float:
        minutes = self.data.travel_minutes(origin_zone, destination_zone, self.scenario_start_tick_day)
        return max(1.0, minutes / max(1.0, self.minutes_per_tick))

    def traffic_multiplier(self, tick: int) -> float:
        tick_day = (self.scenario_start_tick_day + tick) % TICKS_PER_DAY
        local = self.data.travel_minutes(0, 0, tick_day)
        global_median = max(1.0, self.data.global_median_duration_minutes)
        return float(np.clip(local / global_median, 0.6, 2.2))

    def travel_ticks(self, origin_zone: int, destination_zone: int, tick: int) -> float:
        tick_day = (self.scenario_start_tick_day + tick) % TICKS_PER_DAY
        minutes = self.data.travel_minutes(origin_zone, destination_zone, tick_day)
        return max(0.4, minutes / self.minutes_per_tick)

    def travel_minutes(self, origin_zone: int, destination_zone: int, tick: int) -> float:
        return self.travel_ticks(origin_zone, destination_zone, tick) * self.minutes_per_tick

    def shortage_pressure_by_zone(
        self,
        active_order_zones: list[int],
        active_vehicle_zones: list[int],
    ) -> np.ndarray:
        demand = np.bincount(active_order_zones, minlength=self.zone_count).astype(float)
        supply = np.bincount(active_vehicle_zones, minlength=self.zone_count).astype(float)
        return (demand + 1.0) / (supply + 1.0)
