from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_v2v.config import load_project_config, resolve_run_dir
from future_v2v.metrics import write_csv
from scripts.run_experiment import run_eval, run_generate, run_report, run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PBRS reward-shaping ablation for Future V2V adaptive timing.")
    parser.add_argument("--scale", choices=["smoke", "main"], default="smoke")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--run-name", default="pbrs_ablation")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--observation-profile", choices=["legacy_full", "compact_v2v"], default="compact_v2v")
    parser.add_argument("--agent", choices=["ppo", "dqn"], default="ppo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_project_config(args.config)
    root_dir = resolve_run_dir(base_config, args.scale, args.run_name)
    root_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    modes = ["none", "legacy_delta", "pbrs"]
    for mode in modes:
        config = replace(
            base_config,
            environment=replace(base_config.environment, observation_profile=args.observation_profile),
            training=replace(
                base_config.training,
                agent_type=args.agent,
                reward_shaping_mode=mode,
                observation_profile=args.observation_profile,
            ),
        )
        run_dir = root_dir / mode
        run_generate(config, args.scale, run_dir, args.seed, args.eval_episodes)
        run_train(config, args.scale, run_dir, args.seed, args.episodes, args.rollout_workers)
        run_eval(config, args.scale, run_dir, args.seed, args.eval_episodes, args.eval_workers)
        run_report(run_dir)
        summary_rows.extend(_tag_rows(_read_rows(run_dir / "train" / "pbrs_ablation_summary.csv"), mode))
        summary_rows.extend(_tag_eval_rows(_read_rows(run_dir / "eval" / "eval_summary.csv"), mode))
        curve_rows.extend(_tag_rows(_read_rows(run_dir / "train" / "training_curve_comparison.csv"), mode))
    _write_union_csv(root_dir / "train" / "pbrs_ablation_summary.csv", summary_rows)
    write_csv(root_dir / "train" / "training_curve_comparison.csv", curve_rows)


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _tag_rows(rows: list[dict[str, object]], mode: str) -> list[dict[str, object]]:
    tagged = []
    for row in rows:
        materialized = {"reward_shaping_mode": mode}
        materialized.update(row)
        tagged.append(materialized)
    return tagged


def _tag_eval_rows(rows: list[dict[str, object]], mode: str) -> list[dict[str, object]]:
    tagged = []
    for row in rows:
        if row.get("policy_name") not in {"adaptive_timing_ppo", "adaptive_interval_dqn"}:
            continue
        materialized = {"reward_shaping_mode": mode, "summary_source": "eval_summary"}
        materialized.update(row)
        tagged.append(materialized)
    return tagged


def _write_union_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
