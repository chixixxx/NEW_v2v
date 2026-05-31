from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_v2v.config import ProjectConfig, load_project_config, resolve_run_dir
from future_v2v.metrics import summarize_metrics, write_csv
from scripts.run_experiment import (
    _evaluate_policy_specs,
    _load_or_build_eval_manifest,
    _paired_policy_delta_summary,
)
from future_v2v.algorithms.baselines import default_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run candidate vehicle cap sensitivity for Future V2V.")
    parser.add_argument("--scale", choices=["smoke", "main"], default="main")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--run-name", default=None, help="Existing run directory used for manifest/checkpoint.")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--caps", default="20,32,48,64,0", help="Comma-separated caps; 0 means unlimited.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_project_config(args.config)
    run_dir = resolve_run_dir(base_config, args.scale, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    count = args.eval_episodes or base_config.scale(args.scale).eval_episodes
    manifest_rows = _load_or_build_eval_manifest(base_config, args.scale, run_dir, seed=args.seed, count=count)
    rows: list[dict[str, object]] = []
    for cap in _parse_caps(args.caps):
        config = _with_cap(base_config, cap)
        policy_specs = [("baseline", policy.name) for policy in default_baselines()]
        ppo_checkpoint = run_dir / "train" / "adaptive_timing_ppo_agent.pt"
        if ppo_checkpoint.exists():
            policy_specs.append(("ppo", str(ppo_checkpoint)))
        results = _evaluate_policy_specs(
            config,
            args.scale,
            manifest_rows,
            policy_specs,
            eval_workers=args.eval_workers,
            desc=f"eval cap {cap if cap > 0 else 'unlimited'}",
        )
        metrics = [result["metrics"] for result in results]
        summary_rows = summarize_metrics(metrics)
        paired_rows = _paired_policy_delta_summary(metrics)
        rows.extend(_cap_summary_rows(cap, summary_rows, paired_rows))
    write_csv(run_dir / "eval" / "cap_sensitivity_summary.csv", rows)


def _parse_caps(raw: str) -> list[int]:
    caps = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        caps.append(max(0, int(item)))
    return caps or [0]


def _with_cap(config: ProjectConfig, cap: int) -> ProjectConfig:
    return replace(
        config,
        environment=replace(config.environment, max_candidate_vehicles_per_order=int(cap)),
    )


def _cap_summary_rows(
    cap: int,
    summary_rows: list[dict[str, float | str]],
    paired_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary_by_policy = {str(row["policy_name"]): row for row in summary_rows}
    paired_by_policy = {str(row["policy_name"]): row for row in paired_rows}
    fixed1_score = float(summary_by_policy.get("fixed_1_tick_full_match", {}).get("future_v2v_score_mean", 0.0))
    fixed2_score = float(summary_by_policy.get("fixed_2_tick_full_match", {}).get("future_v2v_score_mean", 0.0))
    handcrafted_score = float(summary_by_policy.get("handcrafted_deadline_rule", {}).get("future_v2v_score_mean", 0.0))
    best = max(summary_rows, key=lambda row: float(row["future_v2v_score_mean"]), default={})
    rows = []
    for row in summary_rows:
        policy = str(row["policy_name"])
        score = float(row["future_v2v_score_mean"])
        paired = paired_by_policy.get(policy, {})
        rows.append(
            {
                "candidate_cap": cap if cap > 0 else "unlimited",
                "policy_name": policy,
                "future_v2v_score_mean": score,
                "future_v2v_score_sem": row.get("future_v2v_score_sem", 0.0),
                "service_rate_mean": row.get("service_rate_mean", 0.0),
                "expired_rate_mean": row.get("expired_rate_mean", 0.0),
                "cancelled_rate_mean": row.get("cancelled_rate_mean", 0.0),
                "platform_profit_mean": row.get("platform_profit_mean", 0.0),
                "mean_batch_interval_mean": row.get("mean_batch_interval_mean", 0.0),
                "mean_pickup_time_mean": row.get("mean_pickup_time_mean", 0.0),
                "total_pickup_distance_km_mean": row.get("total_pickup_distance_km_mean", 0.0),
                "score_delta_vs_fixed1_mean": score - fixed1_score,
                "score_delta_vs_fixed2_mean": score - fixed2_score,
                "score_delta_vs_handcrafted_mean": score - handcrafted_score,
                "paired_delta_vs_fixed1_mean": paired.get("score_delta_mean", ""),
                "paired_delta_vs_fixed1_sem": paired.get("score_delta_sem", ""),
                "paired_win_rate_vs_fixed1": paired.get("score_win_rate", ""),
                "best_policy_for_cap": best.get("policy_name", ""),
                "best_score_for_cap": best.get("future_v2v_score_mean", 0.0),
            }
        )
    return rows


if __name__ == "__main__":
    main()
