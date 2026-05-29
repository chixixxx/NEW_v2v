from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_v2v.algorithms.baselines import TimingPolicy, default_baselines, policy_from_name, run_policy_episode
from future_v2v.algorithms.dqn import DQNTimingAgent
from future_v2v.config import ProjectConfig, load_project_config, resolve_run_dir
from future_v2v.envs.timing_env import FutureV2VTimingEnv
from future_v2v.metrics import summarize_metrics, write_csv
from future_v2v.progress import progress
from future_v2v.reporting import write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Future V2V adaptive timing experiment runner.")
    parser.add_argument("--stage", choices=["smoke", "generate", "train", "eval", "report", "all"], default="all")
    parser.add_argument("--scale", choices=["smoke", "main"], default="smoke")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Override training episodes.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Override evaluation episodes.")
    parser.add_argument("--rollout-workers", type=int, default=None, help="Parallel rollout workers for DQN training.")
    parser.add_argument("--eval-workers", type=int, default=1, help="Parallel workers for evaluation.")
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def make_env_factory(config: ProjectConfig, scale_name: str, seed: int) -> Callable[[], FutureV2VTimingEnv]:
    scale = config.scale(scale_name)

    def factory() -> FutureV2VTimingEnv:
        return FutureV2VTimingEnv(config.environment, scale, seed=seed)

    return factory


def run_generate(config: ProjectConfig, scale_name: str, run_dir: Path, seed: int, eval_episodes: int | None) -> None:
    env = make_env_factory(config, scale_name, seed)()
    scale = config.scale(scale_name)
    count = eval_episodes or min(8, scale.eval_episodes)
    manifest_rows = _build_eval_manifest(env, seed=seed, count=count)
    write_csv(run_dir / "env_health" / "eval_scenario_manifest.csv", manifest_rows)
    rows = []
    for scenario in progress(manifest_rows, desc="generate env health", total=len(manifest_rows), unit="seed"):
        if str(scenario.get("day", "")):
            rows.append(
                env.env_health_row_for_window(
                    seed=int(scenario["seed"]),
                    day=str(scenario["day"]),
                    start_tick_day=int(scenario["start_tick_day"]),
                    scenario_id=str(scenario["scenario_id"]),
                )
            )
        else:
            rows.append(env.env_health_row(int(scenario["seed"])))
    write_csv(run_dir / "env_health" / "env_health_summary.csv", rows)
    write_markdown_report(
        run_dir / "env_health" / "env_health_report.md",
        title=f"Future V2V Environment Health ({scale_name})",
        summary_lines=[
            "本报告用于检查动态匹配时机环境是否存在极端供需失衡或可行边过稀。",
            f"scale={scale_name}, seeds={count}",
        ],
        table_rows=rows[: min(10, len(rows))],
    )


def run_train(
    config: ProjectConfig,
    scale_name: str,
    run_dir: Path,
    seed: int,
    episodes: int | None,
    rollout_workers: int | None,
) -> None:
    env_factory = make_env_factory(config, scale_name, seed)
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    train_episodes = episodes or config.scale(scale_name).train_episodes
    worker_count = max(1, int(rollout_workers or config.training.rollout_workers))
    agent = DQNTimingAgent(obs_dim=len(obs), training_config=config.training)
    history = agent.train(
        env_factory,
        episodes=train_episodes,
        seed_start=seed,
        rollout_workers=worker_count,
        env_config=config.environment,
        scale_config=config.scale(scale_name),
    )
    write_csv(run_dir / "train" / "train_history.csv", history)
    write_csv(run_dir / "train" / "validation_history.csv", agent.validation_history)
    write_csv(run_dir / "train" / "action_distribution.csv", _action_distribution_rows(history))
    agent.save(run_dir / "train" / "dqn_timing_agent.pt")
    write_markdown_report(
        run_dir / "train" / "train_report.md",
        title=f"DQN Timing Training ({scale_name})",
        summary_lines=[
            "训练目标是学习 WAIT / MATCH_TOP_BATCH / MATCH_FULL 时机，匹配边仍由约束优化器决定。",
            f"episodes={train_episodes}, replay_prefill={config.training.teacher_prefill_episodes}, rollout_workers={worker_count}",
            f"validation_episodes={config.training.validation_episodes}, checkpoint_metric={config.training.checkpoint_selection_metric}",
        ],
        table_rows=history[-10:],
    )


def run_eval(
    config: ProjectConfig,
    scale_name: str,
    run_dir: Path,
    seed: int,
    eval_episodes: int | None,
    eval_workers: int,
) -> None:
    scale = config.scale(scale_name)
    count = eval_episodes or scale.eval_episodes
    manifest_rows = _load_or_build_eval_manifest(config, scale_name, run_dir, seed=seed, count=count)
    policy_specs: list[tuple[str, str]] = [("baseline", policy.name) for policy in default_baselines()]
    checkpoint = run_dir / "train" / "dqn_timing_agent.pt"
    if checkpoint.exists():
        policy_specs.append(("dqn", str(checkpoint)))
    tasks = [(policy_spec, scenario) for policy_spec in policy_specs for scenario in manifest_rows]
    if eval_workers > 1:
        with ProcessPoolExecutor(max_workers=eval_workers) as executor:
            futures = [
                executor.submit(_run_eval_task, config, scale_name, scenario, policy_spec)
                for policy_spec, scenario in tasks
            ]
            results = [
                future.result()
                for future in progress(futures, desc="eval policies", total=len(futures), unit="episode")
            ]
    else:
        results = []
        for policy_spec, scenario in progress(tasks, desc="eval policies", total=len(tasks), unit="episode"):
            results.append(_run_eval_task(config, scale_name, scenario, policy_spec))
    all_metrics = [result["metrics"] for result in results]
    action_rows = [row for result in results for row in result["action_trace"]]
    dispatch_rows = [row for result in results for row in result["dispatch_trace"]]
    wait_rows = [row for result in results for row in result["wait_tradeoff_trace"]]
    detail_rows = [metric.to_row() for metric in all_metrics]
    summary_rows = summarize_metrics(all_metrics)
    write_csv(run_dir / "eval" / "episode_metrics.csv", detail_rows)
    write_csv(run_dir / "eval" / "eval_summary.csv", summary_rows)
    write_csv(run_dir / "eval" / "action_trace_by_policy.csv", action_rows)
    write_csv(run_dir / "eval" / "dispatch_trace_by_policy.csv", dispatch_rows)
    write_csv(run_dir / "eval" / "wait_tradeoff_trace.csv", wait_rows)
    write_csv(run_dir / "eval" / "timing_policy_comparison.csv", _timing_policy_comparison(summary_rows, dispatch_rows))
    write_markdown_report(
        run_dir / "eval" / "eval_report.md",
        title=f"Future V2V Timing Evaluation ({scale_name})",
        summary_lines=[
            "主表按 future_v2v_score_mean 排序；profit、服务率、取消和过期是辅助解释指标。",
            "timing_degenerate_risk=True 表示策略可能退化为过于频繁的一步匹配。",
            f"episodes_per_policy={len(manifest_rows)}, eval_workers={eval_workers}",
        ],
        table_rows=summary_rows,
    )


def run_report(run_dir: Path) -> None:
    eval_report = run_dir / "eval" / "eval_report.md"
    train_report = run_dir / "train" / "train_report.md"
    health_report = run_dir / "env_health" / "env_health_report.md"
    lines = [
        "# Future V2V Adaptive Timing Run Report",
        "",
        "本总报告汇总环境健康、训练和评估三部分。详细 CSV 保留在各自目录中。",
        "",
    ]
    for label, path in [("环境健康", health_report), ("训练", train_report), ("评估", eval_report)]:
        lines.append(f"## {label}")
        lines.append("")
        if path.exists():
            lines.append(path.read_text(encoding="utf-8"))
        else:
            lines.append("尚未生成。")
        lines.append("")
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_eval_task(
    config: ProjectConfig,
    scale_name: str,
    scenario: dict[str, object],
    policy_spec: tuple[str, str],
) -> dict[str, object]:
    seed = int(scenario["seed"])
    env = FutureV2VTimingEnv(config.environment, config.scale(scale_name), seed=seed)
    kind, value = policy_spec
    if kind == "baseline":
        policy = policy_from_name(value)
    elif kind == "dqn":
        obs, _ = _reset_eval_env(env, scenario)
        policy = DQNTimingAgent.load(Path(value), config.training)
        _ = obs
    else:
        raise ValueError(f"unknown policy spec kind: {kind}")
    metrics = _run_policy_episode_for_scenario(env, policy, scenario=scenario)
    return {
        "metrics": metrics,
        "action_trace": _tag_trace_rows(
            env.action_trace,
            policy_name=metrics.policy_name,
            seed=seed,
            scenario_id=metrics.scenario_id,
        ),
        "dispatch_trace": _tag_trace_rows(
            env.dispatch_trace,
            policy_name=metrics.policy_name,
            seed=seed,
            scenario_id=metrics.scenario_id,
        ),
        "wait_tradeoff_trace": _tag_trace_rows(
            env.wait_tradeoff_trace,
            policy_name=metrics.policy_name,
            seed=seed,
            scenario_id=metrics.scenario_id,
        ),
    }


def _reset_eval_env(env: FutureV2VTimingEnv, scenario: dict[str, object]):
    day = str(scenario.get("day", ""))
    seed = int(scenario["seed"])
    if day:
        return env.reset_to_tlc_window(
            seed=seed,
            day=day,
            start_tick_day=int(scenario["start_tick_day"]),
            scenario_id=str(scenario.get("scenario_id", f"eval_{seed}")),
        )
    return env.reset(seed=seed)


def _run_policy_episode_for_scenario(env: FutureV2VTimingEnv, policy: TimingPolicy, *, scenario: dict[str, object]):
    seed = int(scenario["seed"])
    if not str(scenario.get("day", "")):
        return run_policy_episode(env, policy, seed=seed)
    obs, _ = _reset_eval_env(env, scenario)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy.act(env, obs)
        env.set_action_q_values(getattr(policy, "last_q_values", None))
        obs, _reward, terminated, truncated, _info = env.step(action)
    return env.episode_metrics(policy_name=policy.name, seed=seed)


def _tag_trace_rows(
    rows: list[dict[str, object]],
    *,
    policy_name: str,
    seed: int,
    scenario_id: str,
) -> list[dict[str, object]]:
    tagged = []
    for row in rows:
        materialized = {"policy_name": policy_name, "seed": seed, "scenario_id": scenario_id}
        materialized.update(row)
        tagged.append(materialized)
    return tagged


def _action_distribution_rows(history: list[dict[str, float | int]]) -> list[dict[str, object]]:
    return [
        {
            "episode": row["episode"],
            "wait_count": row.get("wait_count", 0),
            "top_batch_count": row.get("top_batch_count", 0),
            "full_match_count": row.get("full_match_count", 0),
        }
        for row in history
    ]


def _load_or_build_eval_manifest(
    config: ProjectConfig,
    scale_name: str,
    run_dir: Path,
    *,
    seed: int,
    count: int,
) -> list[dict[str, object]]:
    path = run_dir / "env_health" / "eval_scenario_manifest.csv"
    if path.exists():
        rows = _read_csv_rows(path)
        if len(rows) >= count:
            return rows[:count]
    env = make_env_factory(config, scale_name, seed)()
    rows = _build_eval_manifest(env, seed=seed, count=count)
    write_csv(path, rows)
    return rows


def _build_eval_manifest(env: FutureV2VTimingEnv, *, seed: int, count: int) -> list[dict[str, object]]:
    generator = getattr(env, "generator", None)
    rows: list[dict[str, object]] = []
    if hasattr(generator, "manifest_row"):
        for idx in range(count):
            rows.append(generator.manifest_row(seed=seed + idx, scenario_id=f"eval_{idx:03d}"))
        return rows
    for idx in range(count):
        rows.append(
            {
                "scenario_id": f"eval_{idx:03d}",
                "seed": seed + idx,
                "day": "",
                "start_tick_day": "",
                "time_of_day_bucket": "",
                "raw_tlc_rows": "",
                "mean_zone_pressure": "",
                "mean_wtp_proxy": "",
                "mean_duration_minutes": "",
            }
        )
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _timing_policy_comparison(
    summary_rows: list[dict[str, float | str]],
    dispatch_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    dispatch_by_policy: dict[str, list[dict[str, object]]] = {}
    for row in dispatch_rows:
        dispatch_by_policy.setdefault(str(row["policy_name"]), []).append(row)
    summary_by_policy = {str(row["policy_name"]): row for row in summary_rows}
    fixed_1 = summary_by_policy.get("fixed_1_tick_full_match")
    fixed_2 = summary_by_policy.get("fixed_2_tick_full_match")
    fixed_1_top = summary_by_policy.get("fixed_1_tick_top_batch")
    best_score = max((float(row["future_v2v_score_mean"]) for row in summary_rows), default=0.0)
    intervals = [float(row["mean_batch_interval_mean"]) for row in summary_rows]
    fixed_1_score = float(fixed_1["future_v2v_score_mean"]) if fixed_1 else 0.0
    fixed_1_profit = float(fixed_1["platform_profit_mean"]) if fixed_1 else 0.0
    fixed_1_service = float(fixed_1["service_rate_mean"]) if fixed_1 else 0.0
    fixed_2_service = float(fixed_2["service_rate_mean"]) if fixed_2 else 0.0
    fixed_1_top_score = float(fixed_1_top["future_v2v_score_mean"]) if fixed_1_top else 0.0
    policy_spread_score = best_score - fixed_1_score
    batch_interval_spread = max(intervals, default=0.0) - min(intervals, default=0.0)
    top_batch_viability = fixed_1_top_score / fixed_1_score if fixed_1_score > 0 else 0.0
    service_drop_fixed2_vs_fixed1 = fixed_1_service - fixed_2_service
    comparison = []
    for row in summary_rows:
        policy_name = str(row["policy_name"])
        dispatches = dispatch_by_policy.get(policy_name, [])
        profit_values = [float(item["platform_profit"]) for item in dispatches]
        accepted_values = [float(item["accepted_count"]) for item in dispatches]
        comparison.append(
            {
                "policy_name": policy_name,
                "future_v2v_score_mean": row["future_v2v_score_mean"],
                "platform_profit_mean": row["platform_profit_mean"],
                "service_rate_mean": row["service_rate_mean"],
                "expired_rate_mean": row["expired_rate_mean"],
                "cancelled_rate_mean": row["cancelled_rate_mean"],
                "mean_batch_interval_mean": row["mean_batch_interval_mean"],
                "dispatch_epoch_count_mean": row["dispatch_epoch_count_mean"],
                "mean_profit_per_dispatch": sum(profit_values) / max(1, len(profit_values)),
                "mean_accepted_per_dispatch": sum(accepted_values) / max(1, len(accepted_values)),
                "timing_degeneracy": bool(float(row["mean_batch_interval_mean"]) <= 1.15),
                "policy_spread_score": policy_spread_score,
                "batch_interval_spread": batch_interval_spread,
                "top_batch_viability": top_batch_viability,
                "service_drop_fixed2_vs_fixed1": service_drop_fixed2_vs_fixed1,
                "full_match_cost_gap": float(row["platform_profit_mean"]) - fixed_1_profit,
                "dynamic_timing_ready": bool(
                    policy_spread_score > 150.0
                    and batch_interval_spread >= 0.8
                    and service_drop_fixed2_vs_fixed1 <= 0.10
                    and top_batch_viability >= 0.95
                ),
            }
        )
    return comparison


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    scale_name = "smoke" if args.stage == "smoke" else args.scale
    run_dir = resolve_run_dir(config, scale_name, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        run_generate(config, scale_name, run_dir, args.seed, eval_episodes=3)
        run_train(
            config,
            scale_name,
            run_dir,
            args.seed,
            episodes=args.episodes or 5,
            rollout_workers=args.rollout_workers,
        )
        run_eval(config, scale_name, run_dir, args.seed, eval_episodes=args.eval_episodes or 3, eval_workers=args.eval_workers)
        run_report(run_dir)
        return
    stages = ["generate", "train", "eval", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "generate":
            run_generate(config, scale_name, run_dir, args.seed, args.eval_episodes)
        elif stage == "train":
            run_train(config, scale_name, run_dir, args.seed, args.episodes, args.rollout_workers)
        elif stage == "eval":
            run_eval(config, scale_name, run_dir, args.seed, args.eval_episodes, args.eval_workers)
        elif stage == "report":
            run_report(run_dir)


if __name__ == "__main__":
    main()
