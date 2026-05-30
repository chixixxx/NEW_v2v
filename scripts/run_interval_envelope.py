from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_v2v.algorithms.baselines import FixedIntervalPolicy, HandcraftedDeadlineRulePolicy
from future_v2v.config import ProjectConfig, load_project_config, resolve_run_dir
from future_v2v.envs.timing_env import FutureV2VTimingEnv
from future_v2v.metrics import EpisodeMetrics, summarize_metrics, write_csv
from future_v2v.progress import progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-interval envelope diagnostics for Future V2V timing.")
    parser.add_argument("--scale", choices=["smoke", "main"], default="main")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--run-name", default="interval_envelope")
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    run_dir = resolve_run_dir(config, args.scale, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_or_build_manifest(config, args.scale, run_dir, seed=args.seed, count=args.eval_episodes)
    detail_rows, metrics = interval_envelope_rows(config, args.scale, manifest, eval_workers=args.eval_workers)
    summary_rows = summarize_metrics(metrics)
    bucket_rows = interval_bucket_envelope(detail_rows)
    diversity_rows = interval_diversity_summary(bucket_rows, detail_rows, summary_rows)
    distance_rows = pickup_distance_summary(summary_rows)
    out_dir = run_dir / "env_diagnostics"
    write_csv(out_dir / "interval_envelope_detail.csv", detail_rows)
    write_csv(out_dir / "interval_envelope_summary.csv", summary_rows)
    write_csv(out_dir / "interval_bucket_envelope.csv", bucket_rows)
    write_csv(out_dir / "interval_diversity_summary.csv", diversity_rows)
    write_csv(out_dir / "pickup_distance_summary.csv", distance_rows)
    print_summary(diversity_rows[0] if diversity_rows else {})


def envelope_policies():
    return [
        FixedIntervalPolicy(interval=1),
        FixedIntervalPolicy(interval=2),
        FixedIntervalPolicy(interval=3),
        FixedIntervalPolicy(interval=4),
        HandcraftedDeadlineRulePolicy(),
    ]


def interval_envelope_rows(
    config: ProjectConfig,
    scale_name: str,
    manifest: list[dict[str, object]],
    *,
    eval_workers: int = 1,
) -> tuple[list[dict[str, object]], list[EpisodeMetrics]]:
    rows: list[dict[str, object]] = []
    metrics: list[EpisodeMetrics] = []
    tasks = [(scenario, policy) for scenario in manifest for policy in envelope_policies()]
    if eval_workers > 1:
        with ProcessPoolExecutor(max_workers=eval_workers) as executor:
            futures = [
                executor.submit(_run_envelope_policy_task, config, scale_name, scenario, policy)
                for scenario, policy in tasks
            ]
            for future in progress(futures, desc="interval envelope", total=len(futures), unit="episode"):
                metric, row = future.result()
                metrics.append(metric)
                rows.append(row)
    else:
        for scenario, policy in progress(tasks, desc="interval envelope", total=len(tasks), unit="episode"):
            metric, row = _run_envelope_policy_task(config, scale_name, scenario, policy)
            metrics.append(metric)
            rows.append(row)
    return attach_consistent_bucket_fields(rows), metrics


def _run_envelope_policy_task(
    config: ProjectConfig,
    scale_name: str,
    scenario: dict[str, object],
    policy,
) -> tuple[EpisodeMetrics, dict[str, object]]:
    env = FutureV2VTimingEnv(config.environment, config.scale(scale_name), seed=int(scenario["seed"]))
    reset_eval_env(env, scenario)
    obs = env._observation()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy.act(env, obs)
        obs, _reward, terminated, truncated, _info = env.step(action)
    metric = env.episode_metrics(policy_name=policy.name, seed=int(scenario["seed"]))
    row = metric.to_row()
    row.update(scenario_fields(scenario))
    row["interval"] = parse_interval(policy.name)
    row["match_mode"] = parse_match_mode(policy.name)
    return metric, row


def load_or_build_manifest(
    config: ProjectConfig,
    scale_name: str,
    run_dir: Path,
    *,
    seed: int,
    count: int,
) -> list[dict[str, object]]:
    path = run_dir / "env_health" / "eval_scenario_manifest.csv"
    if path.exists():
        rows = read_csv_rows(path)
        if len(rows) >= count:
            return rows[:count]
    env = FutureV2VTimingEnv(config.environment, config.scale(scale_name), seed=seed)
    generator = getattr(env, "generator", None)
    rows = []
    if hasattr(generator, "manifest_row"):
        for idx in range(count):
            rows.append(generator.manifest_row(seed=seed + idx, scenario_id=f"eval_{idx:03d}"))
    else:
        for idx in range(count):
            rows.append(
                {
                    "scenario_id": f"eval_{idx:03d}",
                    "seed": seed + idx,
                    "day": "",
                    "start_tick_day": "",
                    "time_of_day_bucket": "unknown",
                    "raw_tlc_rows": 0,
                    "mean_zone_pressure": 0.0,
                }
            )
    write_csv(path, rows)
    return rows


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def reset_eval_env(env: FutureV2VTimingEnv, scenario: dict[str, object]) -> None:
    seed = int(scenario["seed"])
    day = str(scenario.get("day", ""))
    if day:
        env.reset_to_tlc_window(
            seed=seed,
            day=day,
            start_tick_day=int(scenario["start_tick_day"]),
            scenario_id=str(scenario.get("scenario_id", f"eval_{seed}")),
        )
        return
    env.reset(seed=seed)


def scenario_fields(scenario: dict[str, object]) -> dict[str, object]:
    pressure = to_float(scenario.get("mean_zone_pressure"))
    raw_rows = to_float(scenario.get("raw_tlc_rows"))
    return {
        "scenario_id": scenario.get("scenario_id", ""),
        "time_of_day_bucket": scenario.get("time_of_day_bucket", "unknown"),
        "mean_zone_pressure": pressure,
        "raw_tlc_rows": raw_rows,
        "zone_pressure_bucket": bucketize(pressure, low=1.9, high=2.3, labels=("low_pressure", "mid_pressure", "high_pressure")),
        "demand_volume_bucket": bucketize(raw_rows, low=9000, high=18000, labels=("low_demand", "mid_demand", "high_demand")),
    }


def attach_consistent_bucket_fields(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Use one reference policy per scenario so every policy is compared inside the same bucket."""
    reference_by_scenario: dict[str, dict[str, object]] = {}
    for row in rows:
        if str(row.get("policy_name", "")) == "fixed_1_tick_full_match":
            reference_by_scenario[str(row.get("scenario_id", ""))] = row
    for row in rows:
        reference = reference_by_scenario.get(str(row.get("scenario_id", "")), row)
        service_pressure = to_float(reference.get("expired_rate")) + to_float(reference.get("cancelled_rate"))
        distance = to_float(reference.get("pickup_distance_per_served_order"))
        row.update(
            {
        "service_risk_bucket": bucketize(service_pressure, low=0.22, high=0.32, labels=("low_risk", "mid_risk", "high_risk")),
        "pickup_distance_bucket": bucketize(distance, low=2.2, high=2.8, labels=("short_pickup", "mid_pickup", "long_pickup")),
            }
        )
    return rows


def bucketize(value: float, *, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value < low:
        return labels[0]
    if value >= high:
        return labels[2]
    return labels[1]


def interval_bucket_envelope(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    bucket_keys = [
        "scenario_id",
        "time_of_day_bucket",
        "zone_pressure_bucket",
        "demand_volume_bucket",
        "service_risk_bucket",
        "pickup_distance_bucket",
    ]
    out: list[dict[str, object]] = []
    for bucket_key in bucket_keys:
        values = sorted({str(row.get(bucket_key, "")) for row in rows if str(row.get(bucket_key, ""))})
        for value in values:
            subset = [row for row in rows if str(row.get(bucket_key, "")) == value]
            if not subset:
                continue
            by_policy = aggregate_by_policy(subset)
            best = max(by_policy, key=lambda row: float(row["future_v2v_score_mean"]))
            fixed_candidates = [row for row in by_policy if parse_interval(str(row["policy_name"])) > 0]
            best_fixed = max(
                fixed_candidates,
                key=lambda row: float(row["future_v2v_score_mean"]),
                default=best,
            )
            fixed2 = next((row for row in by_policy if str(row["policy_name"]) == "fixed_2_tick_full_match"), {})
            fixed1 = next((row for row in by_policy if str(row["policy_name"]) == "fixed_1_tick_full_match"), {})
            out.append(
                {
                    "bucket_type": bucket_key,
                    "bucket_value": value,
                    "scenario_count": len({str(row.get("scenario_id", "")) for row in subset}),
                    "best_policy": best["policy_name"],
                    "best_interval": parse_interval(str(best["policy_name"])),
                    "best_score_mean": best["future_v2v_score_mean"],
                    "best_fixed_policy": best_fixed["policy_name"],
                    "best_fixed_interval": parse_interval(str(best_fixed["policy_name"])),
                    "best_fixed_score_mean": best_fixed["future_v2v_score_mean"],
                    "fixed2_score_mean": fixed2.get("future_v2v_score_mean", 0.0),
                    "fixed1_score_mean": fixed1.get("future_v2v_score_mean", 0.0),
                    "best_delta_vs_fixed2": float(best["future_v2v_score_mean"]) - float(fixed2.get("future_v2v_score_mean", 0.0)),
                    "best_delta_vs_fixed1": float(best["future_v2v_score_mean"]) - float(fixed1.get("future_v2v_score_mean", 0.0)),
                    "best_fixed_delta_vs_fixed2": float(best_fixed["future_v2v_score_mean"]) - float(fixed2.get("future_v2v_score_mean", 0.0)),
                    "best_fixed_delta_vs_fixed1": float(best_fixed["future_v2v_score_mean"]) - float(fixed1.get("future_v2v_score_mean", 0.0)),
                    "best_service_rate": best["service_rate_mean"],
                    "best_pickup_distance_per_served_order": best["pickup_distance_per_served_order_mean"],
                }
            )
    return out


def aggregate_by_policy(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_policy: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy_name"]), []).append(row)
    out = []
    numeric = [
        "future_v2v_score",
        "platform_profit",
        "service_rate",
        "expired_rate",
        "cancelled_rate",
        "mean_batch_interval",
        "total_pickup_distance_km",
        "pickup_distance_per_served_order",
    ]
    for policy, values in by_policy.items():
        item: dict[str, object] = {"policy_name": policy}
        for field in numeric:
            item[f"{field}_mean"] = float(np.mean([to_float(row.get(field)) for row in values]))
        out.append(item)
    return out


def interval_diversity_summary(
    bucket_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    intervals = [
        int(row.get("best_fixed_interval", row.get("best_interval", 0)))
        for row in bucket_rows
        if int(row.get("best_fixed_interval", row.get("best_interval", 0))) > 0
    ]
    fixed2_dominance = float(sum(interval == 2 for interval in intervals) / max(1, len(intervals)))
    unique_intervals = sorted(set(intervals))
    adaptive_scores = [max(0.0, float(row.get("best_fixed_delta_vs_fixed2", 0.0))) for row in bucket_rows]
    fixed1 = next((row for row in summary_rows if str(row["policy_name"]) == "fixed_1_tick_full_match"), {})
    fixed2 = next((row for row in summary_rows if str(row["policy_name"]) == "fixed_2_tick_full_match"), {})
    fixed3 = next((row for row in summary_rows if str(row["policy_name"]) == "fixed_3_tick_full_match"), {})
    fixed4 = next((row for row in summary_rows if str(row["policy_name"]) == "fixed_4_tick_full_match"), {})
    return [
        {
            "bucket_count": len(bucket_rows),
            "unique_best_intervals": "|".join(str(interval) for interval in unique_intervals),
            "unique_best_interval_count": len(unique_intervals),
            "fixed2_dominance_rate": fixed2_dominance,
            "adaptive_opportunity_score": float(np.mean(adaptive_scores)) if adaptive_scores else 0.0,
            "interval_diversity_ready": bool(fixed2_dominance < 0.70 and len(unique_intervals) >= 2),
            "fixed1_score_mean": fixed1.get("future_v2v_score_mean", 0.0),
            "fixed2_score_mean": fixed2.get("future_v2v_score_mean", 0.0),
            "fixed3_score_mean": fixed3.get("future_v2v_score_mean", 0.0),
            "fixed4_score_mean": fixed4.get("future_v2v_score_mean", 0.0),
            "detail_rows": len(detail_rows),
        }
    ]


def pickup_distance_summary(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "policy_name": row["policy_name"],
            "future_v2v_score_mean": row.get("future_v2v_score_mean", 0.0),
            "service_rate_mean": row.get("service_rate_mean", 0.0),
            "platform_profit_mean": row.get("platform_profit_mean", 0.0),
            "total_pickup_distance_km_mean": row.get("total_pickup_distance_km_mean", 0.0),
            "pickup_distance_per_served_order_mean": row.get("pickup_distance_per_served_order_mean", 0.0),
            "distance_adjusted_score_mean": row.get("distance_adjusted_score_mean", 0.0),
        }
        for row in summary_rows
    ]


def parse_interval(policy_name: str) -> int:
    parts = policy_name.split("_")
    if len(parts) >= 2 and parts[0] == "fixed":
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


def parse_match_mode(policy_name: str) -> str:
    if policy_name.endswith("full_match"):
        return "full"
    return "rule"


def to_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def print_summary(row: dict[str, object]) -> None:
    keys = [
        "interval_diversity_ready",
        "unique_best_intervals",
        "fixed2_dominance_rate",
        "adaptive_opportunity_score",
        "fixed1_score_mean",
        "fixed2_score_mean",
        "fixed3_score_mean",
        "fixed4_score_mean",
    ]
    width = max(len(key) for key in keys)
    for key in keys:
        print(f"{key:<{width}}  {row.get(key, '')}")


if __name__ == "__main__":
    main()
