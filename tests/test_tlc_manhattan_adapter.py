from __future__ import annotations

from pathlib import Path

import pandas as pd

from future_v2v.algorithms.dqn import DQNTimingAgent
from future_v2v.config import EnvironmentConfig, ScaleConfig, TrainingConfig, load_project_config
from future_v2v.data.tlc_manhattan import load_tlc_manhattan_data, prepare_tlc_manhattan
from future_v2v.envs.timing_env import FutureV2VTimingEnv
from future_v2v.simulation.network import TLCManhattanZoneNetwork, ZoneNetwork


def make_tlc_files(tmp_path: Path) -> tuple[Path, Path]:
    lookup = pd.DataFrame(
        [
            {"LocationID": 4, "Borough": "Manhattan", "Zone": "Alphabet City", "service_zone": "Yellow Zone"},
            {"LocationID": 107, "Borough": "Manhattan", "Zone": "Gramercy", "service_zone": "Yellow Zone"},
            {"LocationID": 225, "Borough": "Brooklyn", "Zone": "Stuyvesant Heights", "service_zone": "Boro Zone"},
        ]
    )
    lookup_path = tmp_path / "taxi_zone_lookup.csv"
    lookup.to_csv(lookup_path, index=False)
    rows = []
    for idx in range(8):
        pickup = pd.Timestamp("2025-10-03 08:00:00") + pd.Timedelta(minutes=5 * idx)
        rows.append(
            {
                "tpep_pickup_datetime": pickup,
                "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=8 + idx),
                "trip_distance": 1.2 + 0.1 * idx,
                "PULocationID": 4 if idx % 2 == 0 else 107,
                "DOLocationID": 107 if idx % 2 == 0 else 4,
                "fare_amount": 12.0 + idx,
                "tip_amount": 2.0,
                "total_amount": 16.0 + idx,
            }
        )
    rows.append(
        {
            "tpep_pickup_datetime": pd.Timestamp("2025-10-03 08:00:00"),
            "tpep_dropoff_datetime": pd.Timestamp("2025-10-03 08:10:00"),
            "trip_distance": 2.0,
            "PULocationID": 225,
            "DOLocationID": 4,
            "fare_amount": 12.0,
            "tip_amount": 1.0,
            "total_amount": 15.0,
        }
    )
    trip_path = tmp_path / "yellow_tripdata_2025-10.parquet"
    pd.DataFrame(rows).to_parquet(trip_path, index=False)
    return trip_path, lookup_path


def make_env_config(tmp_path: Path, trip_path: Path, lookup_path: Path) -> EnvironmentConfig:
    return EnvironmentConfig(
        zone_count=16,
        action_space="wait_topbatch_full",
        service_kwh_per_tick=2.7,
        pickup_cap_minutes=18.0,
        platform_pickup_cost_per_min=0.06,
        dispatch_capacity_ratio=0.55,
        dispatch_capacity_min=8,
        dispatch_capacity_max=80,
        queue_threshold_ratio=0.55,
        queue_threshold_min=24,
        wait_penalty_per_order_tick=0.008,
        expired_penalty=10.0,
        cancelled_penalty=8.0,
        service_rate_target=0.62,
        urgent_service_rate_target=0.72,
        expired_rate_cap=0.16,
        cancelled_rate_cap=0.10,
        mean_pickup_soft_cap_min=10.0,
        mean_commitment_soft_cap_ticks=7.0,
        profit_scale_fallback=5.0,
        enable_stochastic_acceptance=False,
        enable_stochastic_cancellation=False,
        scenario_source="tlc_manhattan",
        tlc_trip_path=str(trip_path),
        taxi_zone_lookup_path=str(lookup_path),
        processed_dir=str(tmp_path / "processed"),
        tick_minutes=3,
        time_bucket_minutes=60,
        demand_sample_rate=1.0,
    )


def test_prepare_tlc_manhattan_filters_and_maps_zones(tmp_path: Path) -> None:
    trip_path, lookup_path = make_tlc_files(tmp_path)
    paths = prepare_tlc_manhattan(
        trip_path=trip_path,
        zone_lookup_path=lookup_path,
        processed_dir=tmp_path / "processed",
        month="2025-10",
        tick_minutes=3,
        time_bucket_minutes=60,
    )
    rows = pd.read_parquet(paths.trip_rows)
    zones = pd.read_csv(paths.zone_lookup)
    assert len(rows) == 8
    assert int(rows.iloc[0]["pickup_tick_day"]) == 160
    assert int(rows.iloc[0]["time_bucket"]) == 8
    assert set(rows["pu_zone"].unique()) == {0, 1}
    assert len(zones) == 2
    assert paths.health_report.exists()


def test_tlc_environment_generates_reproducible_scenario(tmp_path: Path) -> None:
    trip_path, lookup_path = make_tlc_files(tmp_path)
    env_config = make_env_config(tmp_path, trip_path, lookup_path)
    prepare_tlc_manhattan(
        trip_path=trip_path,
        zone_lookup_path=lookup_path,
        processed_dir=env_config.processed_dir,
        month="2025-10",
        tick_minutes=env_config.tick_minutes,
        time_bucket_minutes=env_config.time_bucket_minutes,
    )
    data = load_tlc_manhattan_data(env_config)
    assert data.travel_minutes(0, 1, 160) > 0.0
    scale = ScaleConfig(
        name="unit",
        horizon_ticks=12,
        terminal_buffer_ticks=2,
        total_orders=6,
        candidate_vehicles=8,
        vehicle_join_probability=1.0,
        fleet_probability=0.0,
        train_episodes=1,
        eval_episodes=1,
    )
    env_a = FutureV2VTimingEnv(env_config, scale, seed=11)
    env_b = FutureV2VTimingEnv(env_config, scale, seed=11)
    env_a.reset(seed=11)
    env_b.reset(seed=11)
    assert [order.arrival_tick for order in env_a.orders] == [order.arrival_tick for order in env_b.orders]
    assert env_a.network.zone_count == 2


def test_pickup_distance_can_be_estimated_for_synthetic_and_tlc_networks(tmp_path: Path) -> None:
    synthetic = ZoneNetwork(zone_count=4, minutes_per_tick=3)
    assert synthetic.pickup_distance_km_est(0, 3, 0) > 0.0
    trip_path, lookup_path = make_tlc_files(tmp_path)
    env_config = make_env_config(tmp_path, trip_path, lookup_path)
    prepare_tlc_manhattan(
        trip_path=trip_path,
        zone_lookup_path=lookup_path,
        processed_dir=env_config.processed_dir,
        month="2025-10",
        tick_minutes=env_config.tick_minutes,
        time_bucket_minutes=env_config.time_bucket_minutes,
    )
    data = load_tlc_manhattan_data(env_config)
    network = TLCManhattanZoneNetwork(data, minutes_per_tick=3)
    network.set_episode_context(start_tick_day=160)
    assert network.pickup_distance_km_est(0, 1, 0) > 0.0


def test_tick2_diagnostic_config_preserves_four_hour_main_window() -> None:
    config = load_project_config("configs/tick2_diagnostic.json")
    main = config.scale("main")
    assert config.environment.tick_minutes == 2
    assert main.horizon_ticks == 120
    assert config.environment.service_kwh_per_tick == 1.8
    assert main.horizon_ticks * config.environment.tick_minutes == 240


def test_tlc_environment_keeps_configured_order_count_with_sparse_windows(tmp_path: Path) -> None:
    trip_path, lookup_path = make_tlc_files(tmp_path)
    env_config = make_env_config(tmp_path, trip_path, lookup_path)
    prepare_tlc_manhattan(
        trip_path=trip_path,
        zone_lookup_path=lookup_path,
        processed_dir=env_config.processed_dir,
        month="2025-10",
        tick_minutes=env_config.tick_minutes,
        time_bucket_minutes=env_config.time_bucket_minutes,
    )
    scale = ScaleConfig(
        name="unit",
        horizon_ticks=12,
        terminal_buffer_ticks=2,
        total_orders=12,
        candidate_vehicles=8,
        vehicle_join_probability=1.0,
        fleet_probability=0.0,
        train_episodes=1,
        eval_episodes=1,
    )
    env = FutureV2VTimingEnv(env_config, scale, seed=11)
    env.reset(seed=11)
    assert len(env.orders) == scale.total_orders
    assert [order.order_id for order in env.orders] == list(range(scale.total_orders))


def test_tlc_manifest_window_replays_same_scenario(tmp_path: Path) -> None:
    trip_path, lookup_path = make_tlc_files(tmp_path)
    env_config = make_env_config(tmp_path, trip_path, lookup_path)
    prepare_tlc_manhattan(
        trip_path=trip_path,
        zone_lookup_path=lookup_path,
        processed_dir=env_config.processed_dir,
        month="2025-10",
        tick_minutes=env_config.tick_minutes,
        time_bucket_minutes=env_config.time_bucket_minutes,
    )
    scale = ScaleConfig(
        name="unit",
        horizon_ticks=12,
        terminal_buffer_ticks=2,
        total_orders=12,
        candidate_vehicles=8,
        vehicle_join_probability=1.0,
        fleet_probability=0.0,
        train_episodes=1,
        eval_episodes=1,
    )
    env_a = FutureV2VTimingEnv(env_config, scale, seed=11)
    manifest = env_a.generator.manifest_row(seed=11, scenario_id="fixed_eval")
    env_a.reset_to_tlc_window(
        seed=int(manifest["seed"]),
        day=str(manifest["day"]),
        start_tick_day=int(manifest["start_tick_day"]),
        scenario_id=str(manifest["scenario_id"]),
    )
    env_b = FutureV2VTimingEnv(env_config, scale, seed=99)
    env_b.reset_to_tlc_window(
        seed=int(manifest["seed"]),
        day=str(manifest["day"]),
        start_tick_day=int(manifest["start_tick_day"]),
        scenario_id=str(manifest["scenario_id"]),
    )
    assert env_a.scenario_id == "fixed_eval"
    assert [order.arrival_tick for order in env_a.orders] == [order.arrival_tick for order in env_b.orders]
    assert [vehicle.current_soc_kwh for vehicle in env_a.vehicles] == [vehicle.current_soc_kwh for vehicle in env_b.vehicles]


def test_dqn_validation_manifest_uses_time_bucket_rows(tmp_path: Path) -> None:
    trip_path, lookup_path = make_tlc_files(tmp_path)
    env_config = make_env_config(tmp_path, trip_path, lookup_path)
    prepare_tlc_manhattan(
        trip_path=trip_path,
        zone_lookup_path=lookup_path,
        processed_dir=env_config.processed_dir,
        month="2025-10",
        tick_minutes=env_config.tick_minutes,
        time_bucket_minutes=env_config.time_bucket_minutes,
    )
    scale = ScaleConfig(
        name="unit",
        horizon_ticks=12,
        terminal_buffer_ticks=2,
        total_orders=12,
        candidate_vehicles=8,
        vehicle_join_probability=1.0,
        fleet_probability=0.0,
        train_episodes=1,
        eval_episodes=1,
    )
    training = TrainingConfig(
        gamma=0.98,
        learning_rate=0.001,
        batch_size=4,
        replay_capacity=100,
        min_replay_size=4,
        target_update_interval=10,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
        teacher_prefill_episodes=0,
        validation_episodes=4,
        validation_interval_episodes=1,
        checkpoint_selection_metric="future_v2v_score_mean",
        hidden_dim=16,
        double_dqn=True,
        prioritized_replay=False,
        validation_time_buckets=["morning_peak", "off_peak"],
        validation_worst_bucket_weight=0.15,
        validation_off_peak_score_floor=0.0,
    )
    _ = env_config, scale
    agent = DQNTimingAgent(obs_dim=1, training_config=training)
    rows = agent._validation_manifest(env_config, scale, seed_start=123)
    assert len(rows) == training.validation_episodes
    assert {str(row["time_of_day_bucket"]) for row in rows}
    assert all(str(row["scenario_id"]).startswith("val_") for row in rows)


def test_dqn_checkpoint_selection_score_balances_mean_worst_and_off_peak_floor(tmp_path: Path) -> None:
    trip_path, lookup_path = make_tlc_files(tmp_path)
    env_config = make_env_config(tmp_path, trip_path, lookup_path)
    scale = ScaleConfig(
        name="unit",
        horizon_ticks=12,
        terminal_buffer_ticks=2,
        total_orders=12,
        candidate_vehicles=8,
        vehicle_join_probability=1.0,
        fleet_probability=0.0,
        train_episodes=1,
        eval_episodes=1,
    )
    training = TrainingConfig(
        gamma=0.98,
        learning_rate=0.001,
        batch_size=4,
        replay_capacity=100,
        min_replay_size=4,
        target_update_interval=10,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
        teacher_prefill_episodes=0,
        validation_episodes=4,
        validation_interval_episodes=1,
        checkpoint_selection_metric="future_v2v_score_mean",
        hidden_dim=16,
        double_dqn=True,
        prioritized_replay=False,
        validation_worst_bucket_weight=0.15,
        validation_off_peak_score_floor=0.0,
    )
    _ = env_config, scale
    agent = DQNTimingAgent(obs_dim=1, training_config=training)
    score, penalty = agent._checkpoint_selection_score(
        mean_score=100.0,
        worst_bucket_score=20.0,
        off_peak_score=-5.0,
    )
    assert score == 83.0
    assert penalty == 5.0


def test_tlc_vehicle_soc_and_price_are_bounded(tmp_path: Path) -> None:
    trip_path, lookup_path = make_tlc_files(tmp_path)
    env_config = make_env_config(tmp_path, trip_path, lookup_path)
    prepare_tlc_manhattan(
        trip_path=trip_path,
        zone_lookup_path=lookup_path,
        processed_dir=env_config.processed_dir,
        month="2025-10",
        tick_minutes=env_config.tick_minutes,
        time_bucket_minutes=env_config.time_bucket_minutes,
    )
    scale = ScaleConfig(
        name="unit",
        horizon_ticks=12,
        terminal_buffer_ticks=2,
        total_orders=12,
        candidate_vehicles=80,
        vehicle_join_probability=1.0,
        fleet_probability=0.25,
        train_episodes=1,
        eval_episodes=1,
    )
    env = FutureV2VTimingEnv(env_config, scale, seed=13)
    env.reset(seed=13)
    soc_ratios = [vehicle.current_soc_kwh / vehicle.battery_capacity_kwh for vehicle in env.vehicles]
    prices = [vehicle.reservation_price_per_kwh for vehicle in env.vehicles]
    assert min(soc_ratios) >= 0.42
    assert max(soc_ratios) <= 0.90
    assert sum(soc_ratios) / len(soc_ratios) > 0.62
    assert min(prices) >= 1.35
    assert max(prices) <= 4.75
