from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_v2v.algorithms.baselines import default_baselines
from future_v2v.config import load_project_config, resolve_run_dir
from future_v2v.metrics import summarize_metrics, write_csv
from scripts.run_experiment import (
    _evaluate_policy_specs,
    _load_or_build_eval_manifest,
    _paired_policy_delta_summary,
    _run_friction_sensitivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dispatch friction sensitivity for an existing Future V2V run.")
    parser.add_argument("--scale", choices=["smoke", "main"], default="main")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--observation-profile", choices=["legacy_full", "compact_v2v"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    if args.observation_profile:
        config = replace(
            config,
            environment=replace(config.environment, observation_profile=args.observation_profile),
            training=replace(config.training, observation_profile=args.observation_profile),
        )
    run_dir = resolve_run_dir(config, args.scale, args.run_name)
    scale = config.scale(args.scale)
    count = args.eval_episodes or scale.eval_episodes
    manifest_rows = _load_or_build_eval_manifest(config, args.scale, run_dir, seed=args.seed, count=count)
    policy_specs: list[tuple[str, str]] = [("baseline", policy.name) for policy in default_baselines()]
    checkpoint = run_dir / "train" / "adaptive_interval_dqn_agent.pt"
    if checkpoint.exists():
        policy_specs.append(("interval_dqn", str(checkpoint)))
    primary_results = _evaluate_policy_specs(
        config,
        args.scale,
        manifest_rows,
        policy_specs,
        eval_workers=args.eval_workers,
        desc="eval friction primary",
    )
    primary_metrics = [result["metrics"] for result in primary_results]
    primary_dispatch_rows = [row for result in primary_results for row in result["dispatch_trace"]]
    primary_summary_rows = summarize_metrics(primary_metrics)
    primary_paired_rows = _paired_policy_delta_summary(primary_metrics)
    sensitivity_rows = _run_friction_sensitivity(
        config,
        args.scale,
        manifest_rows,
        policy_specs,
        eval_workers=args.eval_workers,
        primary_summary_rows=primary_summary_rows,
        primary_dispatch_rows=primary_dispatch_rows,
        primary_paired_rows=primary_paired_rows,
    )
    write_csv(run_dir / "eval" / "friction_sensitivity_summary.csv", sensitivity_rows)


if __name__ == "__main__":
    main()
