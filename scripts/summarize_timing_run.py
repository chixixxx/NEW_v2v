from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_v2v.config import load_project_config, resolve_run_dir


PRIMARY_POLICIES = ["adaptive_timing_ppo", "adaptive_interval_dqn"]
FIXED1_POLICY = "fixed_1_tick_full_match"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Future V2V timing run without rerunning simulation.")
    parser.add_argument("--run-name", required=True, help="Run directory name under the configured output root.")
    parser.add_argument("--scale", choices=["smoke", "main"], default="main")
    parser.add_argument("--config", default="configs/default.json")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    return next((row for row in rows if str(row.get(key, "")) == value), {})


def best_row(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    return max(rows, key=lambda row: to_float(row.get(metric)), default={})


def latest_interval_distribution(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return interval_distribution_from_counts({})
    row = rows[-1]
    counts = {
        "dispatch_now": to_float(row.get("dispatch_now_count")),
        "delay_1_then_dispatch": to_float(row.get("delay_1_count")),
        "delay_2_then_dispatch": to_float(row.get("delay_2_count")),
        "delay_3_then_dispatch": to_float(row.get("delay_3_count")),
        "delay_4_plus_then_dispatch": to_float(row.get("delay_4_plus_count")),
    }
    distribution = interval_distribution_from_counts(counts)
    decision_total = to_float(row.get("wait_action_count")) + to_float(row.get("match_full_action_count"))
    if decision_total > 0.0:
        distribution["wait_action_rate"] = to_float(row.get("wait_action_count")) / decision_total
        distribution["match_full_action_rate"] = to_float(row.get("match_full_action_count")) / decision_total
        distribution["binary_max_action_share"] = max(
            distribution["wait_action_rate"],
            distribution["match_full_action_rate"],
        )
    return distribution


def eval_interval_distribution(rows: list[dict[str, str]], policy_name: str) -> dict[str, float]:
    counts = {
        "dispatch_now": 0.0,
        "delay_1_then_dispatch": 0.0,
        "delay_2_then_dispatch": 0.0,
        "delay_3_then_dispatch": 0.0,
        "delay_4_plus_then_dispatch": 0.0,
    }
    binary_counts = {"wait": 0.0, "match_full": 0.0}
    for row in rows:
        if str(row.get("policy_name", "")) != policy_name:
            continue
        binary_action = str(row.get("binary_action_name", ""))
        if binary_action in binary_counts:
            binary_counts[binary_action] += 1.0
        action = str(row.get("interval_action_name", ""))
        final_dispatch = str(row.get("final_dispatch_executed", "True"))
        if action in counts and final_dispatch in {"True", "true", "1"}:
            counts[action] += 1.0
    distribution = interval_distribution_from_counts(counts)
    decision_total = binary_counts["wait"] + binary_counts["match_full"]
    if decision_total > 0.0:
        distribution["wait_action_rate"] = binary_counts["wait"] / decision_total
        distribution["match_full_action_rate"] = binary_counts["match_full"] / decision_total
        distribution["binary_max_action_share"] = max(binary_counts.values()) / decision_total
    return distribution


def interval_distribution_from_counts(counts: dict[str, float]) -> dict[str, float]:
    dispatch_now = counts.get("dispatch_now", 0.0)
    delay_1 = counts.get("delay_1_then_dispatch", 0.0)
    delay_2 = counts.get("delay_2_then_dispatch", 0.0)
    delay_3 = counts.get("delay_3_then_dispatch", 0.0)
    delay_4_plus = counts.get("delay_4_plus_then_dispatch", 0.0)
    total = dispatch_now + delay_1 + delay_2 + delay_3 + delay_4_plus
    if total <= 0.0:
        return {
            "wait_action_rate": 0.0,
            "match_full_action_rate": 0.0,
            "dispatch_now_rate": 0.0,
            "delay_1_rate": 0.0,
            "delay_2_rate": 0.0,
            "delay_3_rate": 0.0,
            "delay_4_plus_rate": 0.0,
            "interval_max_action_share": 0.0,
            "binary_max_action_share": 0.0,
            "interval_mean_action_interval": 0.0,
            "total_interval_actions": 0.0,
        }
    rates = [dispatch_now / total, delay_1 / total, delay_2 / total, delay_3 / total, delay_4_plus / total]
    return {
        "wait_action_rate": 0.0,
        "match_full_action_rate": 0.0,
        "dispatch_now_rate": rates[0],
        "delay_1_rate": rates[1],
        "delay_2_rate": rates[2],
        "delay_3_rate": rates[3],
        "delay_4_plus_rate": rates[4],
        "interval_max_action_share": max(rates),
        "binary_max_action_share": 0.0,
        "interval_mean_action_interval": 1.0 + (delay_1 + 2.0 * delay_2 + 3.0 * delay_3 + 4.0 * delay_4_plus) / total,
        "total_interval_actions": total,
    }


def validation_snapshot(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {
            "validation_score": 0.0,
            "off_peak_score": 0.0,
            "worst_bucket_score": 0.0,
            "off_peak_floor_penalty": 0.0,
        }
    row = best_row(rows, "checkpoint_selection_score")
    return {
        "validation_score": to_float(row.get("checkpoint_selection_score")),
        "off_peak_score": to_float(row.get("validation_off_peak_score_mean")),
        "worst_bucket_score": to_float(row.get("validation_bucket_min_score_mean")),
        "off_peak_floor_penalty": to_float(row.get("validation_off_peak_floor_penalty")),
    }


def select_primary_policy(rows: list[dict[str, str]]) -> str:
    available = {str(row.get("policy_name", "")) for row in rows}
    for policy in PRIMARY_POLICIES:
        if policy in available:
            return policy
    return PRIMARY_POLICIES[0]


def summarize_run(run_dir: Path) -> dict[str, object]:
    eval_summary = read_csv_rows(run_dir / "eval" / "eval_summary.csv")
    paired = read_csv_rows(run_dir / "eval" / "paired_policy_delta_summary.csv")
    timing = read_csv_rows(run_dir / "eval" / "timing_policy_comparison.csv")
    sensitivity_path = run_dir / "eval" / "friction_sensitivity_summary.csv"
    eval_summary_path = run_dir / "eval" / "eval_summary.csv"
    sensitivity_stale = (
        sensitivity_path.exists()
        and eval_summary_path.exists()
        and sensitivity_path.stat().st_mtime < eval_summary_path.stat().st_mtime
    )
    sensitivity = [] if sensitivity_stale else read_csv_rows(sensitivity_path)
    friction_sensitivity_run = bool(sensitivity)
    interval_trace = read_csv_rows(run_dir / "eval" / "interval_policy_trace.csv")
    train_interval_actions = read_csv_rows(run_dir / "train" / "interval_action_distribution.csv")
    validation = read_csv_rows(run_dir / "train" / "validation_history.csv")

    primary_policy = select_primary_policy(eval_summary)
    dqn = row_by(eval_summary, "policy_name", primary_policy)
    fixed1 = row_by(eval_summary, "policy_name", FIXED1_POLICY)
    best = best_row(eval_summary, "future_v2v_score_mean")
    dqn_pair = row_by(paired, "policy_name", primary_policy)
    dqn_timing = row_by(timing, "policy_name", primary_policy)
    interval_rates = eval_interval_distribution(interval_trace, primary_policy)
    if interval_rates["total_interval_actions"] <= 0.0:
        interval_rates = latest_interval_distribution(train_interval_actions)
    val = validation_snapshot(validation)

    dqn_score = to_float(dqn.get("future_v2v_score_mean"))
    fixed1_score = to_float(fixed1.get("future_v2v_score_mean"))
    best_score = to_float(best.get("future_v2v_score_mean"))
    paired_delta = to_float(dqn_pair.get("score_delta_mean"), dqn_score - fixed1_score)
    best_gap = best_score - dqn_score if dqn and best else 0.0
    mean_batch_interval = to_float(
        dqn_timing.get("mean_batch_interval_mean"),
        to_float(dqn.get("mean_batch_interval_mean")),
    )
    service_rate = to_float(dqn.get("service_rate_mean"))
    platform_profit = to_float(dqn.get("platform_profit_mean"))
    win_rate = to_float(dqn_pair.get("score_win_rate"))
    no_refresh_delta = variant_delta(sensitivity, "no_refresh_friction")
    common_fixed_delta = variant_delta(sensitivity, "common_fixed_cost")
    no_dispatch_delta = variant_delta(sensitivity, "no_dispatch_friction")

    status = diagnose_interval(
        paired_delta=paired_delta,
        best_gap=best_gap,
        mean_batch_interval=mean_batch_interval,
        service_rate=service_rate,
        platform_profit=platform_profit,
        no_refresh_delta=no_refresh_delta,
        common_fixed_delta=common_fixed_delta,
        friction_sensitivity_run=friction_sensitivity_run,
        interval_mean_action_interval=interval_rates["interval_mean_action_interval"],
        interval_max_action_share=interval_rates["interval_max_action_share"],
        binary_max_action_share=interval_rates.get("binary_max_action_share", 0.0),
        off_peak_score=val["off_peak_score"],
        off_peak_penalty=val["off_peak_floor_penalty"],
    )

    return {
        "run_dir": str(run_dir),
        "primary_policy": primary_policy,
        "dqn_score": dqn_score,
        "fixed1_score": fixed1_score,
        "best_policy": best.get("policy_name", ""),
        "best_score": best_score,
        "dqn_vs_fixed1_delta": paired_delta,
        "dqn_score_win_rate": win_rate,
        "dqn_gap_to_best": best_gap,
        "dqn_service_rate": service_rate,
        "dqn_platform_profit": platform_profit,
        "dqn_mean_batch_interval": mean_batch_interval,
        "interval_dispatch_now_rate": interval_rates["dispatch_now_rate"],
        "wait_action_rate": interval_rates["wait_action_rate"],
        "match_full_action_rate": interval_rates["match_full_action_rate"],
        "interval_delay_1_rate": interval_rates["delay_1_rate"],
        "interval_delay_2_rate": interval_rates["delay_2_rate"],
        "interval_delay_3_rate": interval_rates["delay_3_rate"],
        "interval_delay_4_plus_rate": interval_rates["delay_4_plus_rate"],
        "interval_mean_action_interval": interval_rates["interval_mean_action_interval"],
        "interval_max_action_share": interval_rates["interval_max_action_share"],
        "binary_max_action_share": interval_rates["binary_max_action_share"],
        "validation_score": val["validation_score"],
        "validation_off_peak_score": val["off_peak_score"],
        "validation_worst_bucket_score": val["worst_bucket_score"],
        "validation_off_peak_floor_penalty": val["off_peak_floor_penalty"],
        "no_refresh_delta": no_refresh_delta,
        "common_fixed_delta": common_fixed_delta,
        "no_dispatch_delta": no_dispatch_delta,
        "friction_sensitivity_run": friction_sensitivity_run,
        "friction_sensitivity_stale": sensitivity_stale,
        "diagnosis": status,
    }


def variant_delta(rows: list[dict[str, str]], variant: str) -> float:
    row = row_by(rows, "variant", variant)
    return to_float(row.get("paired_best_score_delta_mean"), to_float(row.get("best_score_delta_vs_fixed1")))


def diagnose_interval(
    *,
    paired_delta: float,
    best_gap: float,
    mean_batch_interval: float,
    service_rate: float,
    platform_profit: float,
    no_refresh_delta: float,
    common_fixed_delta: float,
    interval_mean_action_interval: float,
    interval_max_action_share: float,
    off_peak_score: float,
    off_peak_penalty: float,
    friction_sensitivity_run: bool = True,
    binary_max_action_share: float = 0.0,
) -> str:
    notes: list[str] = []
    if interval_mean_action_interval < 1.15 or mean_batch_interval < 1.15:
        notes.append("INTERVAL_TOO_SHORT: strengthen delayed-interval teacher coverage")
    if interval_mean_action_interval > 2.45 and (service_rate < 0.68 or platform_profit <= 0.0):
        notes.append("INTERVAL_TOO_LONG: strengthen immediate dispatch rescue states")
    if interval_max_action_share > 0.75:
        notes.append("SINGLE_INTERVAL_DEGENERACY: action distribution is too concentrated")
    if binary_max_action_share > 0.90:
        notes.append("BINARY_ACTION_COLLAPSE: WAIT/MATCH policy is too concentrated")
    if paired_delta > 0.0 and best_gap > 250.0:
        notes.append("POLICY_WEAK_BUT_DIRECTION_OK: tune interval teacher and checkpoint selection")
    if friction_sensitivity_run and (no_refresh_delta <= 0.0 or common_fixed_delta <= 0.0):
        notes.append("FRICTION_SENSITIVE: do not use as main result before environment review")
    if off_peak_score < 0.0 or off_peak_penalty > 0.0:
        notes.append("OFF_PEAK_RISK: checkpoint is not robust across validation buckets")
    if not notes:
        if paired_delta > 250.0 and best_gap < 180.0:
            return "CANDIDATE_READY"
        return "MINI_GATE_PASS_BUT_NOT_FORMAL"
    return "; ".join(notes)


def fmt_float(value: object, digits: int = 3) -> str:
    return f"{to_float(value):.{digits}f}"


def print_summary(summary: dict[str, object]) -> None:
    rows = [
        ("run_dir", str(summary["run_dir"])),
        ("primary_policy", str(summary["primary_policy"])),
        ("best_policy", str(summary["best_policy"])),
        ("dqn_score", fmt_float(summary["dqn_score"], 2)),
        ("fixed1_score", fmt_float(summary["fixed1_score"], 2)),
        ("dqn_vs_fixed1_delta", fmt_float(summary["dqn_vs_fixed1_delta"], 2)),
        ("dqn_score_win_rate", fmt_float(summary["dqn_score_win_rate"], 3)),
        ("dqn_gap_to_best", fmt_float(summary["dqn_gap_to_best"], 2)),
        ("dqn_service_rate", fmt_float(summary["dqn_service_rate"], 3)),
        ("dqn_mean_batch_interval", fmt_float(summary["dqn_mean_batch_interval"], 3)),
        ("wait_action_rate", fmt_float(summary["wait_action_rate"], 3)),
        ("match_full_action_rate", fmt_float(summary["match_full_action_rate"], 3)),
        ("interval_dispatch_now_rate", fmt_float(summary["interval_dispatch_now_rate"], 3)),
        ("interval_delay_1_rate", fmt_float(summary["interval_delay_1_rate"], 3)),
        ("interval_delay_2_rate", fmt_float(summary["interval_delay_2_rate"], 3)),
        ("interval_delay_3_rate", fmt_float(summary["interval_delay_3_rate"], 3)),
        ("interval_delay_4_plus_rate", fmt_float(summary["interval_delay_4_plus_rate"], 3)),
        ("interval_mean_action_interval", fmt_float(summary["interval_mean_action_interval"], 3)),
        ("interval_max_action_share", fmt_float(summary["interval_max_action_share"], 3)),
        ("binary_max_action_share", fmt_float(summary["binary_max_action_share"], 3)),
        ("validation_off_peak_score", fmt_float(summary["validation_off_peak_score"], 2)),
        ("validation_worst_bucket_score", fmt_float(summary["validation_worst_bucket_score"], 2)),
        ("no_refresh_delta", fmt_float(summary["no_refresh_delta"], 2)),
        ("common_fixed_delta", fmt_float(summary["common_fixed_delta"], 2)),
        ("no_dispatch_delta", fmt_float(summary["no_dispatch_delta"], 2)),
        ("friction_sensitivity_run", str(summary["friction_sensitivity_run"])),
        ("friction_sensitivity_stale", str(summary["friction_sensitivity_stale"])),
        ("diagnosis", str(summary["diagnosis"])),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    run_dir = resolve_run_dir(config, args.scale, args.run_name)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    print_summary(summarize_run(run_dir))


if __name__ == "__main__":
    main()
