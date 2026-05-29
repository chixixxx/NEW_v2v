from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from future_v2v.config import EnvironmentConfig

REQUIRED_TRIP_COLUMNS = (
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "tip_amount",
    "total_amount",
)

MINUTES_PER_DAY = 24 * 60

_CACHE: dict[tuple[str, str, str, int, int], TLCManhattanData] = {}


@dataclass(frozen=True)
class TLCProcessedPaths:
    trip_rows: Path
    zone_time_matrix: Path
    zone_lookup: Path
    health_report: Path
    health_summary: Path


@dataclass
class TLCManhattanData:
    trip_rows: pd.DataFrame
    zone_time_matrix: pd.DataFrame
    zone_lookup: pd.DataFrame
    tick_minutes: int = 3
    time_bucket_minutes: int = 60

    def __post_init__(self) -> None:
        self._time_lookup = {
            (int(row.origin_zone), int(row.destination_zone), int(row.time_bucket)): float(row.median_duration_minutes)
            for row in self.zone_time_matrix.itertuples(index=False)
        }
        self._od_lookup = {
            (int(origin), int(destination)): float(group["median_duration_minutes"].median())
            for (origin, destination), group in self.zone_time_matrix.groupby(["origin_zone", "destination_zone"])
        }
        self._origin_lookup = {
            int(origin): float(group["median_duration_minutes"].median())
            for origin, group in self.zone_time_matrix.groupby("origin_zone")
        }
        self._zone_pressure_lookup = {
            (int(zone), int(bucket)): float(group["zone_pressure"].mean())
            for (zone, bucket), group in self.trip_rows.groupby(["pu_zone", "time_bucket"])
        }

    @property
    def zone_count(self) -> int:
        return int(self.zone_lookup["zone_index"].max()) + 1

    @property
    def days(self) -> list[str]:
        return sorted(self.trip_rows["pickup_date"].unique().tolist())

    @property
    def global_median_duration_minutes(self) -> float:
        return float(self.trip_rows["duration_minutes"].median())

    @property
    def ticks_per_day(self) -> int:
        return MINUTES_PER_DAY // self.tick_minutes

    @property
    def time_bucket_ticks(self) -> int:
        return max(1, self.time_bucket_minutes // self.tick_minutes)

    def rows_for_window(self, day: str, start_tick_day: int, horizon_ticks: int) -> pd.DataFrame:
        end_tick = start_tick_day + horizon_ticks
        mask = (
            (self.trip_rows["pickup_date"] == day)
            & (self.trip_rows["pickup_tick_day"] >= start_tick_day)
            & (self.trip_rows["pickup_tick_day"] < end_tick)
        )
        return self.trip_rows.loc[mask]

    def supply_rows_for_window(self, start_tick_day: int, horizon_ticks: int) -> pd.DataFrame:
        end_tick = start_tick_day + horizon_ticks
        mask = (self.trip_rows["dropoff_tick_day"] >= start_tick_day) & (self.trip_rows["dropoff_tick_day"] < end_tick)
        return self.trip_rows.loc[mask]

    def travel_minutes(self, origin_zone: int, destination_zone: int, tick_day: int) -> float:
        bucket = int((tick_day % self.ticks_per_day) // self.time_bucket_ticks)
        exact = self._time_lookup.get((int(origin_zone), int(destination_zone), bucket))
        if exact is not None:
            return exact
        od = self._od_lookup.get((int(origin_zone), int(destination_zone)))
        if od is not None:
            return od
        origin = self._origin_lookup.get(int(origin_zone))
        if origin is not None:
            return origin
        return self.global_median_duration_minutes

    def zone_pressure(self, origin_zone: int, tick_day: int) -> float:
        bucket = int((tick_day % self.ticks_per_day) // self.time_bucket_ticks)
        pressure = self._zone_pressure_lookup.get((int(origin_zone), bucket), 1.0)
        return float(np.clip(pressure, 0.5, 2.5))


def processed_paths(env_config: EnvironmentConfig, month: str | None = None) -> TLCProcessedPaths:
    processed_dir = Path(env_config.processed_dir)
    month_suffix = month or _infer_month_from_path(env_config.tlc_trip_path) or "month"
    suffix = f"{month_suffix}_tick{env_config.tick_minutes}m"
    return TLCProcessedPaths(
        trip_rows=processed_dir / f"tlc_manhattan_{suffix}.parquet",
        zone_time_matrix=processed_dir / f"zone_time_matrix_{suffix}.parquet",
        zone_lookup=processed_dir / "manhattan_zone_lookup.csv",
        health_report=processed_dir / f"data_health_report_{suffix}.md",
        health_summary=processed_dir / f"data_health_summary_{suffix}.csv",
    )


def load_tlc_manhattan_data(env_config: EnvironmentConfig) -> TLCManhattanData:
    paths = processed_paths(env_config)
    cache_key = (
        str(paths.trip_rows.resolve()),
        str(paths.zone_time_matrix.resolve()),
        str(paths.zone_lookup.resolve()),
        env_config.tick_minutes,
        env_config.time_bucket_minutes,
    )
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    if not paths.trip_rows.exists() or not paths.zone_time_matrix.exists() or not paths.zone_lookup.exists():
        raise FileNotFoundError(
            "TLC Manhattan processed files are missing. Run "
            "`python scripts/prepare_tlc_manhattan.py --month YYYY-MM` first, "
            f"or switch scenario_source to synthetic. Missing base path: {paths.trip_rows.parent}"
        )
    data = TLCManhattanData(
        trip_rows=pd.read_parquet(paths.trip_rows),
        zone_time_matrix=pd.read_parquet(paths.zone_time_matrix),
        zone_lookup=pd.read_csv(paths.zone_lookup),
        tick_minutes=env_config.tick_minutes,
        time_bucket_minutes=env_config.time_bucket_minutes,
    )
    _CACHE[cache_key] = data
    return data


def prepare_tlc_manhattan(
    *,
    trip_path: str | Path,
    zone_lookup_path: str | Path,
    processed_dir: str | Path,
    month: str | None = None,
    manhattan_only: bool = True,
    tick_minutes: int = 3,
    time_bucket_minutes: int = 60,
) -> TLCProcessedPaths:
    if MINUTES_PER_DAY % tick_minutes != 0:
        raise ValueError(f"tick_minutes must divide 1440 exactly, got {tick_minutes}")
    if time_bucket_minutes % tick_minutes != 0:
        raise ValueError(
            f"time_bucket_minutes must be a multiple of tick_minutes, got {time_bucket_minutes} and {tick_minutes}"
        )
    trip_path = Path(trip_path)
    zone_lookup_path = Path(zone_lookup_path)
    if not trip_path.exists():
        raise FileNotFoundError(f"TLC trip parquet not found: {trip_path}")
    if not zone_lookup_path.exists():
        raise FileNotFoundError(f"TLC taxi zone lookup not found: {zone_lookup_path}")
    env_stub = EnvironmentConfig(
        zone_count=16,
        action_space="wait_topbatch_full",
        service_kwh_per_tick=2.7,
        pickup_cap_minutes=18.0,
        platform_pickup_cost_per_min=0.06,
        dispatch_fixed_cost=18.0,
        dispatch_capacity_ratio=0.70,
        dispatch_capacity_min=8,
        dispatch_capacity_max=120,
        queue_threshold_ratio=0.35,
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
        enable_stochastic_acceptance=True,
        enable_stochastic_cancellation=True,
        processed_dir=str(processed_dir),
        tlc_trip_path=str(trip_path),
        tick_minutes=tick_minutes,
        time_bucket_minutes=time_bucket_minutes,
    )
    out = processed_paths(env_stub, month=month)
    out.trip_rows.parent.mkdir(parents=True, exist_ok=True)
    lookup = pd.read_csv(zone_lookup_path)
    _validate_zone_lookup(lookup)
    manhattan_lookup = lookup[lookup["Borough"].eq("Manhattan")].copy() if manhattan_only else lookup.copy()
    manhattan_lookup = manhattan_lookup.sort_values("LocationID").reset_index(drop=True)
    manhattan_lookup["zone_index"] = np.arange(len(manhattan_lookup), dtype=int)
    zone_id_to_index = dict(zip(manhattan_lookup["LocationID"].astype(int), manhattan_lookup["zone_index"].astype(int)))
    raw = pd.read_parquet(trip_path, columns=list(REQUIRED_TRIP_COLUMNS))
    _validate_trip_columns(raw)
    filtered = _clean_trip_rows(raw, set(zone_id_to_index), month or _infer_month_from_path(str(trip_path)))
    filtered["pu_zone"] = filtered["PULocationID"].map(zone_id_to_index).astype(int)
    filtered["do_zone"] = filtered["DOLocationID"].map(zone_id_to_index).astype(int)
    filtered["pickup_date"] = filtered["tpep_pickup_datetime"].dt.strftime("%Y-%m-%d")
    filtered["pickup_tick_day"] = _tick_day(filtered["tpep_pickup_datetime"], tick_minutes)
    filtered["dropoff_tick_day"] = _tick_day(filtered["tpep_dropoff_datetime"], tick_minutes)
    time_bucket_ticks = max(1, time_bucket_minutes // tick_minutes)
    filtered["time_bucket"] = (filtered["pickup_tick_day"] // time_bucket_ticks).astype(int)
    filtered["demand_kwh_base"] = np.clip(
        3.5 + 0.38 * filtered["trip_distance"] + 0.12 * filtered["duration_minutes"],
        3.0,
        22.0,
    )
    filtered["willingness_to_pay_proxy"] = np.clip(
        filtered["total_amount"] / np.maximum(filtered["demand_kwh_base"], 0.1),
        3.2,
        12.0,
    )
    zone_tick_counts = (
        filtered.groupby(["pu_zone", "time_bucket"], as_index=False)
        .size()
        .rename(columns={"size": "zone_tick_count"})
    )
    median_count = float(zone_tick_counts["zone_tick_count"].median()) if not zone_tick_counts.empty else 1.0
    zone_tick_counts["zone_pressure"] = np.clip(zone_tick_counts["zone_tick_count"] / max(1.0, median_count), 0.5, 2.5)
    filtered = filtered.merge(zone_tick_counts, on=["pu_zone", "time_bucket"], how="left")
    filtered["zone_pressure"] = filtered["zone_pressure"].fillna(1.0)
    trip_rows = filtered[
        [
            "pickup_date",
            "pickup_tick_day",
            "dropoff_tick_day",
            "time_bucket",
            "pu_zone",
            "do_zone",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "duration_minutes",
            "fare_amount",
            "tip_amount",
            "total_amount",
            "demand_kwh_base",
            "willingness_to_pay_proxy",
            "zone_pressure",
        ]
    ].copy()
    time_matrix = (
        trip_rows.groupby(["pu_zone", "do_zone", "time_bucket"], as_index=False)
        .agg(median_duration_minutes=("duration_minutes", "median"), trip_count=("duration_minutes", "size"))
        .rename(columns={"pu_zone": "origin_zone", "do_zone": "destination_zone"})
    )
    trip_rows.to_parquet(out.trip_rows, index=False)
    time_matrix.to_parquet(out.zone_time_matrix, index=False)
    manhattan_lookup.to_csv(out.zone_lookup, index=False)
    health = _health_rows(
        raw_count=len(raw),
        filtered=trip_rows,
        lookup=manhattan_lookup,
        time_matrix=time_matrix,
        tick_minutes=tick_minutes,
        time_bucket_minutes=time_bucket_minutes,
    )
    pd.DataFrame([health]).to_csv(out.health_summary, index=False)
    out.health_report.write_text(_health_report(health, out), encoding="utf-8")
    return out


def _validate_trip_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_TRIP_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"TLC trip file is missing required columns: {missing}")


def _validate_zone_lookup(df: pd.DataFrame) -> None:
    required = {"LocationID", "Borough", "Zone"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"taxi zone lookup is missing required columns: {missing}")


def _clean_trip_rows(df: pd.DataFrame, allowed_location_ids: set[int], month: str | None) -> pd.DataFrame:
    work = df.copy()
    work["tpep_pickup_datetime"] = pd.to_datetime(work["tpep_pickup_datetime"], errors="coerce")
    work["tpep_dropoff_datetime"] = pd.to_datetime(work["tpep_dropoff_datetime"], errors="coerce")
    work["duration_minutes"] = (
        work["tpep_dropoff_datetime"] - work["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    mask = (
        work["tpep_pickup_datetime"].notna()
        & work["tpep_dropoff_datetime"].notna()
        & work["PULocationID"].isin(allowed_location_ids)
        & work["DOLocationID"].isin(allowed_location_ids)
        & work["duration_minutes"].between(1.0, 120.0)
        & work["trip_distance"].between(0.1, 35.0)
        & work["fare_amount"].gt(0.0)
        & work["total_amount"].gt(0.0)
    )
    if month:
        period = pd.Period(month, freq="M")
        start = period.start_time
        end = (period + 1).start_time
        mask &= work["tpep_pickup_datetime"].ge(start) & work["tpep_pickup_datetime"].lt(end)
    return work.loc[mask].copy()


def _tick_day(series: pd.Series, tick_minutes: int) -> pd.Series:
    return ((series.dt.hour * 60 + series.dt.minute) // tick_minutes).astype(int)


def _infer_month_from_path(path: str) -> str | None:
    match = re.search(r"(20\d{2}-\d{2})", Path(path).name)
    return match.group(1) if match else None


def _health_rows(
    *,
    raw_count: int,
    filtered: pd.DataFrame,
    lookup: pd.DataFrame,
    time_matrix: pd.DataFrame,
    tick_minutes: int,
    time_bucket_minutes: int,
) -> dict[str, float | int | str]:
    ticks_per_day = MINUTES_PER_DAY // tick_minutes
    time_bucket_ticks = max(1, time_bucket_minutes // tick_minutes)
    possible_pairs = max(1, len(lookup) * len(lookup) * (ticks_per_day // time_bucket_ticks))
    return {
        "tick_minutes": tick_minutes,
        "time_bucket_minutes": time_bucket_minutes,
        "raw_rows": raw_count,
        "manhattan_clean_rows": len(filtered),
        "retained_ratio": len(filtered) / max(1, raw_count),
        "manhattan_zone_count": len(lookup),
        "unique_days": int(filtered["pickup_date"].nunique()) if not filtered.empty else 0,
        "mean_rows_per_day": float(filtered.groupby("pickup_date").size().mean()) if not filtered.empty else 0.0,
        "time_matrix_rows": len(time_matrix),
        "time_matrix_coverage": len(time_matrix) / possible_pairs,
        "median_duration_minutes": float(filtered["duration_minutes"].median()) if not filtered.empty else 0.0,
        "median_demand_kwh_base": float(filtered["demand_kwh_base"].median()) if not filtered.empty else 0.0,
        "median_wtp_proxy": float(filtered["willingness_to_pay_proxy"].median()) if not filtered.empty else 0.0,
    }


def _health_report(health: dict[str, float | int | str], paths: TLCProcessedPaths) -> str:
    lines = [
        "# TLC Manhattan Data Health Report",
        "",
        "黄出租数据只用于校准 Future V2V 的真实时空需求、OD 热点、行程时长和价格强度；电量与车辆供给仍按 V2V 业务模型生成。",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in health.items():
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- trip rows: `{paths.trip_rows}`",
            f"- zone time matrix: `{paths.zone_time_matrix}`",
            f"- zone lookup: `{paths.zone_lookup}`",
        ]
    )
    return "\n".join(lines) + "\n"
