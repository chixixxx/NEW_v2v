from __future__ import annotations

from dataclasses import dataclass


ORDER_PENDING = "pending"
ORDER_MATCHED = "matched"
ORDER_EXPIRED = "expired"
ORDER_CANCELLED = "cancelled"

VEHICLE_AVAILABLE = "available"
VEHICLE_BUSY = "busy"
VEHICLE_LEFT = "left"


@dataclass
class Order:
    order_id: int
    arrival_tick: int
    origin_zone: int
    destination_zone: int
    demand_kwh: float
    max_wait_ticks: int
    willingness_to_pay_per_kwh: float
    cancel_sensitivity: float
    status: str = ORDER_PENDING
    matched_vehicle_id: int | None = None
    match_tick: int | None = None
    pickup_minutes: float = 0.0
    commitment_ticks: float = 0.0
    realized_profit: float = 0.0

    def is_active(self, tick: int) -> bool:
        return self.status == ORDER_PENDING and self.arrival_tick <= tick

    def waiting_ticks(self, tick: int) -> int:
        return max(0, tick - self.arrival_tick)

    def waiting_ratio(self, tick: int) -> float:
        return self.waiting_ticks(tick) / max(1, self.max_wait_ticks)

    def is_urgent(self) -> bool:
        return self.max_wait_ticks <= 4


@dataclass
class Vehicle:
    vehicle_id: int
    join_tick: int
    leave_tick: int
    current_zone: int
    destination_zone: int
    battery_capacity_kwh: float
    current_soc_kwh: float
    reserve_kwh: float
    reservation_price_per_kwh: float
    time_cost_per_min: float
    owner_accept_sensitivity: float
    fleet_flag: bool
    status: str = VEHICLE_AVAILABLE
    busy_until_tick: float = 0.0
    served_count: int = 0
    supplied_kwh: float = 0.0

    def available_energy_kwh(self) -> float:
        return max(0.0, self.current_soc_kwh - self.reserve_kwh)

    def is_joined(self, tick: int) -> bool:
        return self.join_tick <= tick < self.leave_tick

    def refresh_status(self, tick: int) -> None:
        if tick >= self.leave_tick:
            self.status = VEHICLE_LEFT
        elif self.status == VEHICLE_BUSY and tick >= self.busy_until_tick:
            self.status = VEHICLE_AVAILABLE

    def is_active(self, tick: int) -> bool:
        self.refresh_status(tick)
        return self.status == VEHICLE_AVAILABLE and self.is_joined(tick)

    def time_flexibility_ticks(self, tick: int) -> float:
        return max(0.0, self.leave_tick - tick)


@dataclass(frozen=True)
class CandidateEdge:
    order_id: int
    vehicle_id: int
    pickup_ticks: float
    pickup_minutes: float
    service_ticks: float
    total_commitment_ticks: float
    revenue: float
    seller_compensation: float
    platform_cost: float
    immediate_profit: float
    accept_probability: float
    expected_profit: float
    feasible: bool
    reason: str = "feasible"


@dataclass(frozen=True)
class Match:
    order_id: int
    vehicle_id: int
    expected_profit: float
    realized_profit: float
    pickup_minutes: float
    total_commitment_ticks: float
    accepted: bool


@dataclass
class StepResult:
    platform_profit: float = 0.0
    matched_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    expired_count: int = 0
    cancelled_count: int = 0
    dispatch_executed: bool = False
    mean_pickup_minutes: float = 0.0
    mean_commitment_ticks: float = 0.0

