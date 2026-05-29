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
    buyer_payment: float = 0.0
    seller_reimbursement: float = 0.0
    seller_energy_cost: float = 0.0
    seller_degradation_cost: float = 0.0
    seller_service_premium: float = 0.0
    platform_pickup_cost: float = 0.0
    seller_time_cost: float = 0.0
    delivered_kwh: float = 0.0
    donor_output_kwh: float = 0.0
    energy_loss_kwh: float = 0.0
    donor_soc_after_kwh: float = 0.0

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
    energy_cost_per_kwh: float
    service_premium_per_kwh: float
    time_cost_per_min: float
    owner_accept_sensitivity: float
    fleet_flag: bool
    status: str = VEHICLE_AVAILABLE
    busy_until_tick: float = 0.0
    served_count: int = 0
    supplied_kwh: float = 0.0
    delivered_kwh: float = 0.0
    energy_loss_kwh: float = 0.0
    seller_reimbursement: float = 0.0
    seller_energy_cost: float = 0.0
    seller_degradation_cost: float = 0.0
    seller_service_premium: float = 0.0

    @property
    def reservation_price_per_kwh(self) -> float:
        return self.energy_cost_per_kwh + self.service_premium_per_kwh

    def available_energy_kwh(self) -> float:
        return max(0.0, self.current_soc_kwh - self.reserve_kwh)

    def health_floor_kwh(self, donor_min_soc_ratio: float) -> float:
        return max(self.reserve_kwh, donor_min_soc_ratio * self.battery_capacity_kwh)

    def available_energy_with_health_kwh(self, donor_min_soc_ratio: float) -> float:
        return max(0.0, self.current_soc_kwh - self.health_floor_kwh(donor_min_soc_ratio))

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
    delivered_kwh: float
    donor_output_kwh: float
    energy_loss_kwh: float
    buyer_payment: float
    seller_reimbursement: float
    seller_energy_cost: float
    seller_degradation_cost: float
    seller_service_premium: float
    platform_pickup_cost: float
    seller_time_cost: float
    immediate_profit: float
    accept_probability: float
    expected_profit: float
    feasible: bool
    donor_soc_after_kwh: float
    reason: str = "feasible"

    @property
    def revenue(self) -> float:
        return self.buyer_payment

    @property
    def seller_compensation(self) -> float:
        return self.seller_reimbursement

    @property
    def platform_cost(self) -> float:
        return self.platform_pickup_cost


@dataclass(frozen=True)
class Match:
    order_id: int
    vehicle_id: int
    expected_profit: float
    realized_profit: float
    pickup_minutes: float
    total_commitment_ticks: float
    accepted: bool
    delivered_kwh: float = 0.0
    donor_output_kwh: float = 0.0
    energy_loss_kwh: float = 0.0
    buyer_payment: float = 0.0
    seller_reimbursement: float = 0.0
    seller_energy_cost: float = 0.0
    seller_degradation_cost: float = 0.0
    seller_service_premium: float = 0.0
    platform_pickup_cost: float = 0.0
    seller_time_cost: float = 0.0
    donor_soc_after_kwh: float = 0.0


@dataclass
class StepResult:
    platform_profit: float = 0.0
    matched_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    expired_count: int = 0
    cancelled_count: int = 0
    dispatch_executed: bool = False
    dispatch_mode: str = "wait"
    dispatch_capacity: int = 0
    candidate_edge_count: int = 0
    mean_pickup_minutes: float = 0.0
    mean_commitment_ticks: float = 0.0
    buyer_payment: float = 0.0
    seller_reimbursement: float = 0.0
    seller_energy_cost: float = 0.0
    seller_degradation_cost: float = 0.0
    seller_service_premium: float = 0.0
    platform_pickup_cost: float = 0.0
    seller_time_cost: float = 0.0
    delivered_kwh: float = 0.0
    donor_output_kwh: float = 0.0
    energy_loss_kwh: float = 0.0
    mean_donor_soc_after_kwh: float = 0.0
    battery_health_rejection_count: int = 0
