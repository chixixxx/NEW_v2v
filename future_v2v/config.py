from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    minutes_per_tick: int
    output_root: str
    run_name_template: str


@dataclass(frozen=True)
class BatteryHealthConfig:
    donor_min_soc_ratio: float = 0.25
    transfer_efficiency: float = 0.90
    degradation_cost_per_kwh: float = 0.08
    max_discharge_power_kw: float = 50.0


@dataclass(frozen=True)
class DispatchFrictionConfig:
    enabled: bool = True
    setup_cost: float = 10.0
    pair_coordination_cost: float = 0.55
    full_mode_extra_pair_cost: float = 0.20
    refresh_cost: float = 4.0
    refresh_decay_ticks: float = 1.5


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
    pickup_distance_penalty_per_km: float = 0.0
    observation_profile: str = "compact_v2v"
    service_risk_delta_weight: float = 0.12
    service_risk_delta_clip: float = 35.0
    wait_opportunity_weight: float = 0.02
    wait_opportunity_cap: float = 1200.0
    scenario_source: str = "synthetic"
    allow_synthetic_smoke_fallback: bool = True
    tlc_trip_path: str = ""
    taxi_zone_lookup_path: str = ""
    taxi_zone_geo_path: str = ""
    processed_dir: str = "data/processed"
    manhattan_only: bool = True
    travel_time_source: str = "empirical_median"
    tick_minutes: int = 3
    time_bucket_minutes: int = 60
    demand_sample_rate: float = 1.0
    supply_scale: float = 1.0
    max_candidate_vehicles_per_order: int = 64
    train_days: list[str] | None = None
    eval_days: list[str] | None = None
    battery_health: BatteryHealthConfig = field(default_factory=BatteryHealthConfig)
    dispatch_friction: DispatchFrictionConfig = field(default_factory=DispatchFrictionConfig)


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
    validation_episodes: int
    validation_interval_episodes: int
    checkpoint_selection_metric: str
    hidden_dim: int
    double_dqn: bool
    prioritized_replay: bool
    agent_type: str = "ppo"
    rollout_workers: int = 1
    ppo_rollout_episodes_per_update: int = 4
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.20
    ppo_value_loss_coef: float = 0.50
    ppo_entropy_coef: float = 0.02
    ppo_gae_lambda: float = 0.95
    ppo_max_grad_norm: float = 5.0
    ppo_eval_deterministic: bool = False
    teacher_imitation_epochs: int = 3
    teacher_imitation_batch_size: int = 256
    validation_action_max_share_cap: float = 0.85
    validation_interval_max_share_cap: float = 0.75
    validation_action_balance_penalty: float = 600.0
    validation_min_mean_interval: float = 1.15
    validation_max_mean_interval: float = 2.45
    reward_shaping_mode: str = "pbrs"
    pbrs_clip: float = 250.0
    pbrs_terminal_mode: str = "finite_horizon_correction"
    observation_profile: str = "compact_v2v"
    potential_candidate_margin_weight: float = 0.10
    potential_feasible_density_weight: float = 35.0
    potential_urgent_coverage_weight: float = 25.0
    potential_pickup_time_weight: float = 1.2
    potential_pickup_distance_weight: float = 1.0
    potential_service_risk_weight: float = 0.04
    potential_urgent_service_risk_weight: float = 18.0
    potential_near_deadline_weight: float = 30.0
    potential_soc_binding_weight: float = 20.0
    validation_time_buckets: list[str] = field(
        default_factory=lambda: ["morning_peak", "midday", "evening_peak", "off_peak"]
    )
    validation_worst_bucket_weight: float = 0.15
    validation_off_peak_score_floor: float = 0.0


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
    env_raw = dict(raw["environment"])
    training_raw = dict(raw["training"])
    env_raw.setdefault("tick_minutes", raw["experiment"]["minutes_per_tick"])
    observation_profile = training_raw.get("observation_profile", env_raw.get("observation_profile", "compact_v2v"))
    env_raw["observation_profile"] = observation_profile
    training_raw["observation_profile"] = observation_profile
    env_raw["battery_health"] = BatteryHealthConfig(**dict(env_raw.get("battery_health", {})))
    env_raw["dispatch_friction"] = DispatchFrictionConfig(**dict(env_raw.get("dispatch_friction", {})))
    if int(env_raw["tick_minutes"]) != int(raw["experiment"]["minutes_per_tick"]):
        raise ValueError("environment.tick_minutes must match experiment.minutes_per_tick")
    scales = {
        name: ScaleConfig(name=name, **scale_data)
        for name, scale_data in raw["scales"].items()
    }
    return ProjectConfig(
        experiment=_read_dataclass(ExperimentConfig, raw["experiment"]),
        environment=_read_dataclass(EnvironmentConfig, env_raw),
        scales=scales,
        training=_read_dataclass(TrainingConfig, training_raw),
    )


def resolve_run_dir(config: ProjectConfig, scale_name: str, run_name: str | None = None) -> Path:
    name = run_name or config.experiment.run_name_template.format(scale=scale_name)
    return Path(config.experiment.output_root) / name
