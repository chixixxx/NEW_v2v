from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    minutes_per_tick: int
    output_root: str
    run_name_template: str


@dataclass(frozen=True)
class EnvironmentConfig:
    zone_count: int
    service_kwh_per_tick: float
    pickup_cap_minutes: float
    platform_pickup_cost_per_min: float
    wait_penalty_per_order_tick: float
    expired_penalty: float
    cancelled_penalty: float
    service_rate_target: float
    urgent_service_rate_target: float
    expired_rate_cap: float
    cancelled_rate_cap: float
    mean_pickup_soft_cap_min: float
    mean_commitment_soft_cap_ticks: float
    profit_scale_fallback: float
    enable_stochastic_acceptance: bool
    enable_stochastic_cancellation: bool


@dataclass(frozen=True)
class ScaleConfig:
    name: str
    horizon_ticks: int
    terminal_buffer_ticks: int
    total_orders: int
    candidate_vehicles: int
    vehicle_join_probability: float
    fleet_probability: float
    train_episodes: int
    eval_episodes: int


@dataclass(frozen=True)
class TrainingConfig:
    gamma: float
    learning_rate: float
    batch_size: int
    replay_capacity: int
    min_replay_size: int
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    teacher_prefill_episodes: int
    hidden_dim: int
    double_dqn: bool
    prioritized_replay: bool


@dataclass(frozen=True)
class ProjectConfig:
    experiment: ExperimentConfig
    environment: EnvironmentConfig
    scales: dict[str, ScaleConfig]
    training: TrainingConfig

    def scale(self, name: str) -> ScaleConfig:
        if name not in self.scales:
            raise KeyError(f"unknown scale: {name}; available={sorted(self.scales)}")
        return self.scales[name]


def _read_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    return cls(**data)


def load_project_config(path: str | Path = "configs/default.json") -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    scales = {
        name: ScaleConfig(name=name, **scale_data)
        for name, scale_data in raw["scales"].items()
    }
    return ProjectConfig(
        experiment=_read_dataclass(ExperimentConfig, raw["experiment"]),
        environment=_read_dataclass(EnvironmentConfig, raw["environment"]),
        scales=scales,
        training=_read_dataclass(TrainingConfig, raw["training"]),
    )


def resolve_run_dir(config: ProjectConfig, scale_name: str, run_name: str | None = None) -> Path:
    name = run_name or config.experiment.run_name_template.format(scale=scale_name)
    return Path(config.experiment.output_root) / name

