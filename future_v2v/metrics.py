from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from future_v2v.config import EnvironmentConfig


@dataclass
class EpisodeMetrics:
    policy_name: str
    seed: int
    future_v2v_score: float
    platform_profit: float
    total_orders: int
    served_orders: int
    urgent_orders: int
    urgent_served_orders: int
    expired_orders: int
    cancelled_orders: int
    rejected_matches: int
    dispatch_epoch_count: int
    mean_wait_before_match: float
    mean_batch_interval: float
    mean_pickup_time: float
    mean_commitment_ticks: float
    profit_per_served_order: float
    fleet_utilization: float
    private_utilization: float
    energy_utilization: float

    @property
    def service_rate(self) -> float:
        return self.served_orders / max(1, self.total_orders)

    @property
    def urgent_service_rate(self) -> float:
        return self.urgent_served_orders / max(1, self.urgent_orders)

    @property
    def expired_rate(self) -> float:
        return self.expired_orders / max(1, self.total_orders)

    @property
    def cancelled_rate(self) -> float:
        return self.cancelled_orders / max(1, self.total_orders)

    def to_row(self) -> dict[str, float | int | str]:
        row = asdict(self)
        row["service_rate"] = self.service_rate
        row["urgent_service_rate"] = self.urgent_service_rate
        row["expired_rate"] = self.expired_rate
        row["cancelled_rate"] = self.cancelled_rate
        return row


def constrained_profit_score(
    *,
    platform_profit: float,
    total_orders: int,
    served_orders: int,
    service_rate: float,
    urgent_service_rate: float,
    expired_rate: float,
    cancelled_rate: float,
    mean_pickup_time: float,
    mean_commitment_ticks: float,
    profit_scale: float,
    env_config: EnvironmentConfig,
) -> float:
    service_penalty = 1.20 * max(0.0, env_config.service_rate_target - service_rate)
    urgent_penalty = 1.50 * max(0.0, env_config.urgent_service_rate_target - urgent_service_rate)
    expired_penalty = 1.00 * max(0.0, expired_rate - env_config.expired_rate_cap)
    cancelled_penalty = 0.80 * max(0.0, cancelled_rate - env_config.cancelled_rate_cap)
    pickup_penalty = 0.03 * served_orders * profit_scale * max(0.0, mean_pickup_time - env_config.mean_pickup_soft_cap_min)
    commitment_penalty = (
        0.04 * served_orders * profit_scale * max(0.0, mean_commitment_ticks - env_config.mean_commitment_soft_cap_ticks)
    )
    rate_penalty = profit_scale * total_orders * (service_penalty + urgent_penalty + expired_penalty + cancelled_penalty)
    return platform_profit - rate_penalty - pickup_penalty - commitment_penalty


def summarize_metrics(metrics: list[EpisodeMetrics]) -> list[dict[str, float | str]]:
    by_policy: dict[str, list[EpisodeMetrics]] = {}
    for item in metrics:
        by_policy.setdefault(item.policy_name, []).append(item)
    rows: list[dict[str, float | str]] = []
    numeric_fields = [
        "future_v2v_score",
        "platform_profit",
        "service_rate",
        "urgent_service_rate",
        "expired_rate",
        "cancelled_rate",
        "mean_wait_before_match",
        "mean_batch_interval",
        "mean_pickup_time",
        "mean_commitment_ticks",
        "profit_per_served_order",
        "fleet_utilization",
        "private_utilization",
        "energy_utilization",
        "dispatch_epoch_count",
    ]
    for policy, values in sorted(by_policy.items()):
        row: dict[str, float | str] = {"policy_name": policy, "episodes": len(values)}
        materialized = [value.to_row() for value in values]
        for field in numeric_fields:
            arr = np.array([float(item[field]) for item in materialized], dtype=float)
            row[f"{field}_mean"] = float(arr.mean())
            row[f"{field}_std"] = float(arr.std(ddof=0))
        service_rate = float(row["service_rate_mean"])
        expired_cancelled = float(row["expired_rate_mean"]) + float(row["cancelled_rate_mean"])
        mean_batch_interval = float(row["mean_batch_interval_mean"])
        row["timing_degenerate_risk"] = bool(service_rate > 0.90 and mean_batch_interval <= 1.2)
        row["environment_target_band"] = bool(0.65 <= service_rate <= 0.82 and 0.08 <= expired_cancelled <= 0.22)
        rows.append(row)
    rows.sort(key=lambda row: float(row["future_v2v_score_mean"]), reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
