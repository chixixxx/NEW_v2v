from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_v2v.algorithms.baselines import TimingPolicy, default_baselines, policy_from_name, run_policy_episode
from future_v2v.algorithms.interval_dqn import AdaptiveIntervalDQNAgent
from future_v2v.algorithms.ppo import AdaptiveTimingPPOAgent
from future_v2v.config import DispatchFrictionConfig, ProjectConfig, load_project_config, resolve_run_dir
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
    parser.add_argument("--agent", choices=["ppo", "dqn"], default=None, help="Training agent. Default comes from config.")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--reward-shaping", choices=["none", "legacy_delta", "pbrs"], default=None)
    parser.add_argument("--observation-profile", choices=["legacy_full", "compact_v2v"], default=None)
    parser.add_argument(
        "--include-friction-sensitivity",
        action="store_true",
        help="Run friction sensitivity inside eval. Default is off; prefer scripts/run_friction_sensitivity.py.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    training = config.training
    environment = config.environment
    if args.reward_shaping is not None:
        training = replace(training, reward_shaping_mode=args.reward_shaping)
    if args.agent is not None:
        training = replace(training, agent_type=args.agent)
    if args.observation_profile is not None:
        training = replace(training, observation_profile=args.observation_profile)
        environment = replace(environment, observation_profile=args.observation_profile)
    elif training.observation_profile != environment.observation_profile:
        environment = replace(environment, observation_profile=training.observation_profile)
    return replace(config, training=training, environment=environment)


def make_env_factory(
    config: ProjectConfig,
    scale_name: str,
    seed: int,
    scenario_phase: str = "train",
) -> Callable[[], FutureV2VTimingEnv]:
    scale = config.scale(scale_name)

    def factory() -> FutureV2VTimingEnv:
        return FutureV2VTimingEnv(config.environment, scale, seed=seed, scenario_phase=scenario_phase)

    return factory


def run_generate(config: ProjectConfig, scale_name: str, run_dir: Path, seed: int, eval_episodes: int | None) -> None:
    env = make_env_factory(config, scale_name, seed, scenario_phase="eval")()
    scale = config.scale(scale_name)
    count = eval_episodes or scale.eval_episodes
    manifest_rows = _load_or_build_eval_manifest(config, scale_name, run_dir, seed=seed, count=count)
    health_path = run_dir / "env_health" / "env_health_summary.csv"
    if health_path.exists():
        existing_rows = _read_csv_rows(health_path)
        rows = existing_rows[: len(manifest_rows)] if _health_rows_match_manifest(existing_rows, manifest_rows) else []
    else:
        rows = []
    if not rows:
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
        write_csv(health_path, rows)
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
    env_factory = make_env_factory(config, scale_name, seed, scenario_phase="train")
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    train_episodes = episodes or config.scale(scale_name).train_episodes
    worker_count = max(1, int(rollout_workers or config.training.rollout_workers))
    if config.training.agent_type == "ppo":
        agent = AdaptiveTimingPPOAgent(obs_dim=len(obs), training_config=config.training)
        checkpoint_path = run_dir / "train" / "adaptive_timing_ppo_agent.pt"
        train_title = f"PPO Timing Training ({scale_name})"
        method_line = "训练目标是学习 WAIT / MATCH_FULL 二元动作；动态匹配间隔由连续 WAIT 自然形成。"
    elif config.training.agent_type == "dqn":
        agent = AdaptiveIntervalDQNAgent(obs_dim=len(obs), training_config=config.training)
        checkpoint_path = run_dir / "train" / "adaptive_interval_dqn_agent.pt"
        train_title = f"DQN Timing Training ({scale_name})"
        method_line = "训练目标是学习固定步长选项；该路径保留为 DQN 对照。"
    else:
        raise ValueError(f"unknown training.agent_type={config.training.agent_type!r}; expected ppo or dqn")
    history = agent.train(
        env_factory,
        episodes=train_episodes,
        seed_start=seed,
        rollout_workers=worker_count,
        env_config=config.environment,
        scale_config=config.scale(scale_name),
    )
    write_csv(run_dir / "train" / "train_history.csv", history)
    write_csv(run_dir / "train" / "reward_shaping_history.csv", _reward_shaping_history_rows(history))
    write_csv(run_dir / "train" / "training_curve_comparison.csv", _training_curve_comparison_rows(history))
    write_csv(run_dir / "train" / "pbrs_ablation_summary.csv", _pbrs_ablation_summary_rows(history, config.training.reward_shaping_mode))
    write_csv(run_dir / "train" / "validation_history.csv", agent.validation_history)
    write_csv(run_dir / "train" / "interval_action_distribution.csv", _interval_action_distribution_rows(history))
    agent.save(checkpoint_path)
    write_markdown_report(
        run_dir / "train" / "train_report.md",
        title=train_title,
        summary_lines=[
            method_line,
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
    include_friction_sensitivity: bool = False,
) -> None:
    scale = config.scale(scale_name)
    count = eval_episodes or scale.eval_episodes
    manifest_rows = _load_or_build_eval_manifest(config, scale_name, run_dir, seed=seed, count=count)
    policy_specs: list[tuple[str, str]] = [("baseline", policy.name) for policy in default_baselines()]
    ppo_checkpoint = run_dir / "train" / "adaptive_timing_ppo_agent.pt"
    if ppo_checkpoint.exists():
        policy_specs.append(("ppo", str(ppo_checkpoint)))
    interval_checkpoint = run_dir / "train" / "adaptive_interval_dqn_agent.pt"
    if config.training.agent_type == "dqn" and interval_checkpoint.exists():
        policy_specs.append(("interval_dqn", str(interval_checkpoint)))
    results = _evaluate_policy_specs(
        config,
        scale_name,
        manifest_rows,
        policy_specs,
        eval_workers=eval_workers,
        desc="eval policies",
    )
    all_metrics = [result["metrics"] for result in results]
    action_rows = [row for result in results for row in result["action_trace"]]
    dispatch_rows = [row for result in results for row in result["dispatch_trace"]]
    wait_rows = [row for result in results for row in result["wait_tradeoff_trace"]]
    interval_rows = [row for result in results for row in result.get("interval_trace", [])]
    detail_rows = [metric.to_row() for metric in all_metrics]
    summary_rows = summarize_metrics(all_metrics)
    paired_rows = _paired_policy_delta_summary(all_metrics)
    comparison_rows = _timing_policy_comparison(summary_rows, dispatch_rows)
    sensitivity_rows = (
        _run_friction_sensitivity(
            config,
            scale_name,
            manifest_rows,
            policy_specs,
            eval_workers=eval_workers,
            primary_summary_rows=summary_rows,
            primary_dispatch_rows=dispatch_rows,
            primary_paired_rows=paired_rows,
        )
        if include_friction_sensitivity
        else []
    )
    acceptance_rows = _environment_acceptance_summary(
        summary_rows,
        dispatch_rows,
        action_rows,
        interval_rows,
        paired_rows,
        sensitivity_rows,
    )
    write_csv(run_dir / "eval" / "episode_metrics.csv", detail_rows)
    write_csv(run_dir / "eval" / "eval_summary.csv", summary_rows)
    write_csv(run_dir / "eval" / "distance_adjusted_eval_summary.csv", _distance_adjusted_summary_rows(summary_rows))
    write_csv(run_dir / "eval" / "paired_policy_delta_summary.csv", paired_rows)
    write_csv(run_dir / "eval" / "action_trace_by_policy.csv", action_rows)
    write_csv(run_dir / "eval" / "dispatch_trace_by_policy.csv", dispatch_rows)
    write_csv(run_dir / "eval" / "wait_tradeoff_trace.csv", wait_rows)
    write_csv(run_dir / "eval" / "interval_policy_trace.csv", interval_rows)
    write_csv(run_dir / "eval" / "interval_action_distribution_by_policy.csv", _interval_action_distribution_by_policy(interval_rows))
    write_csv(run_dir / "eval" / "interval_distribution_histogram.csv", _interval_distribution_histogram(interval_rows))
    write_csv(run_dir / "eval" / "state_action_policy_trace.csv", _state_action_policy_trace(interval_rows))
    write_csv(run_dir / "eval" / "state_action_bucket_summary.csv", _state_action_bucket_summary(interval_rows))
    write_csv(run_dir / "eval" / "timing_policy_comparison.csv", comparison_rows)
    if include_friction_sensitivity:
        write_csv(run_dir / "eval" / "friction_sensitivity_summary.csv", sensitivity_rows)
    else:
        stale_friction_path = run_dir / "eval" / "friction_sensitivity_summary.csv"
        if stale_friction_path.exists():
            stale_friction_path.unlink()
    write_csv(run_dir / "eval" / "environment_acceptance_summary.csv", acceptance_rows)
    write_csv(run_dir / "eval" / "eval_summary_zh.csv", _eval_summary_zh_rows(summary_rows), encoding="utf-8-sig")
    write_markdown_report(
        run_dir / "eval" / "eval_report.md",
        title=f"Future V2V Timing Evaluation ({scale_name})",
        summary_lines=[
            "主表按 future_v2v_score_mean 排序；该得分已包含平台利润、服务可靠性、接驾时间、服务占用和接驾距离软成本。",
            "正式判断优先看配对差值、标准误和胜率，不只看跨场景标准差。",
            "timing_degenerate_risk=True 表示策略可能退化为过于频繁的一步匹配。",
            "交易摩擦灵敏度默认关闭；需要稳健性检查时运行 scripts/run_friction_sensitivity.py。",
            f"episodes_per_policy={len(manifest_rows)}, eval_workers={eval_workers}",
        ],
        table_rows=summary_rows,
        extra_tables=[
            ("配对差值（相对固定1步完整匹配）", _paired_delta_report_rows(paired_rows)),
        ],
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


def _evaluate_policy_specs(
    config: ProjectConfig,
    scale_name: str,
    manifest_rows: list[dict[str, object]],
    policy_specs: list[tuple[str, str]],
    *,
    eval_workers: int,
    desc: str,
) -> list[dict[str, object]]:
    tasks = [(policy_spec, scenario) for policy_spec in policy_specs for scenario in manifest_rows]
    if eval_workers > 1:
        with ProcessPoolExecutor(max_workers=eval_workers) as executor:
            futures = [
                executor.submit(_run_eval_task, config, scale_name, scenario, policy_spec)
                for policy_spec, scenario in tasks
            ]
            return [
                future.result()
                for future in progress(futures, desc=desc, total=len(futures), unit="episode")
            ]
    results = []
    for policy_spec, scenario in progress(tasks, desc=desc, total=len(tasks), unit="episode"):
        results.append(_run_eval_task(config, scale_name, scenario, policy_spec))
    return results


def _run_eval_task(
    config: ProjectConfig,
    scale_name: str,
    scenario: dict[str, object],
    policy_spec: tuple[str, str],
) -> dict[str, object]:
    seed = int(scenario["seed"])
    env = FutureV2VTimingEnv(config.environment, config.scale(scale_name), seed=seed, scenario_phase="eval")
    kind, value = policy_spec
    if kind == "baseline":
        policy = policy_from_name(value)
    elif kind == "ppo":
        obs, _ = _reset_eval_env(env, scenario)
        policy = AdaptiveTimingPPOAgent.load(Path(value), config.training)
        interval_trace = policy.run_eval_episode(env, obs)
        metrics = env.episode_metrics(policy_name=policy.name, seed=seed)
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
            "interval_trace": _tag_trace_rows(
                interval_trace,
                policy_name=metrics.policy_name,
                seed=seed,
                scenario_id=metrics.scenario_id,
            ),
        }
    elif kind == "interval_dqn":
        obs, _ = _reset_eval_env(env, scenario)
        policy = AdaptiveIntervalDQNAgent.load(Path(value), config.training)
        interval_trace = policy.run_eval_episode(env, obs)
        metrics = env.episode_metrics(policy_name=policy.name, seed=seed)
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
            "interval_trace": _tag_trace_rows(
                interval_trace,
                policy_name=metrics.policy_name,
                seed=seed,
                scenario_id=metrics.scenario_id,
            ),
        }
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
        "interval_trace": [],
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


def _interval_action_distribution_rows(history: list[dict[str, float | int]]) -> list[dict[str, object]]:
    return [
        {
            "episode": row["episode"],
            "wait_action_count": row.get("wait_action_count", 0),
            "match_full_action_count": row.get("match_full_action_count", 0),
            "wait_action_share": row.get("wait_action_share", 0.0),
            "match_full_action_share": row.get("match_full_action_share", 0.0),
            "dispatch_now_count": row.get("interval_dispatch_now_count", 0),
            "delay_1_count": row.get("interval_delay_1_count", 0),
            "delay_2_count": row.get("interval_delay_2_count", 0),
            "delay_3_count": row.get("interval_delay_3_count", 0),
            "delay_4_plus_count": row.get("interval_delay_4_plus_count", 0),
            "delay_2_share": row.get("interval_delay_2_share", 0.0),
            "mean_action_interval": row.get("interval_mean_action_interval", 0.0),
            "interval_max_action_share": row.get("interval_max_action_share", 0.0),
        }
        for row in history
    ]


def _reward_shaping_history_rows(history: list[dict[str, float | int]]) -> list[dict[str, object]]:
    return [
        {
            "episode": row["episode"],
            "reward_shaping_mode": row.get("reward_shaping_mode", ""),
            "raw_return": row.get("raw_reward", row.get("reward", 0.0)),
            "legacy_return": row.get("legacy_reward", row.get("reward", 0.0)),
            "shaped_return": row.get("shaped_reward", row.get("reward", 0.0)),
            "potential_delta_sum": row.get("potential_delta_sum", 0.0),
            "episode_score": row.get("future_v2v_score", 0.0),
            "service_rate": row.get("service_rate", 0.0),
            "wait_action_count": row.get("wait_action_count", 0),
            "match_full_action_count": row.get("match_full_action_count", 0),
            "wait_action_share": row.get("wait_action_share", 0.0),
            "match_full_action_share": row.get("match_full_action_share", 0.0),
            "dispatch_now_count": row.get("interval_dispatch_now_count", 0),
            "delay_1_count": row.get("interval_delay_1_count", 0),
            "delay_2_count": row.get("interval_delay_2_count", 0),
            "delay_3_count": row.get("interval_delay_3_count", 0),
        }
        for row in history
    ]


def _training_curve_comparison_rows(history: list[dict[str, float | int]]) -> list[dict[str, object]]:
    rows = []
    scores: list[float] = []
    raw_returns: list[float] = []
    shaped_returns: list[float] = []
    for row in history:
        scores.append(float(row.get("future_v2v_score", 0.0)))
        raw_returns.append(float(row.get("raw_reward", row.get("reward", 0.0))))
        shaped_returns.append(float(row.get("shaped_reward", row.get("reward", 0.0))))
        window = min(10, len(scores))
        rows.append(
            {
                "episode": row["episode"],
                "reward_shaping_mode": row.get("reward_shaping_mode", ""),
                "score": scores[-1],
                "raw_return": raw_returns[-1],
                "shaped_return": shaped_returns[-1],
                "rolling_score_10": _mean(scores[-window:]),
                "rolling_raw_return_10": _mean(raw_returns[-window:]),
                "rolling_shaped_return_10": _mean(shaped_returns[-window:]),
            }
        )
    return rows


def _pbrs_ablation_summary_rows(
    history: list[dict[str, float | int]],
    reward_shaping_mode: str,
) -> list[dict[str, object]]:
    if not history:
        return []
    tail = history[-min(10, len(history)) :]
    return [
        {
            "reward_shaping_mode": reward_shaping_mode,
            "episodes": len(history),
            "tail_score_mean": _mean([float(row.get("future_v2v_score", 0.0)) for row in tail]),
            "tail_raw_return_mean": _mean([float(row.get("raw_reward", row.get("reward", 0.0))) for row in tail]),
            "tail_shaped_return_mean": _mean([float(row.get("shaped_reward", row.get("reward", 0.0))) for row in tail]),
            "tail_potential_delta_mean": _mean([float(row.get("potential_delta_sum", 0.0)) for row in tail]),
        }
    ]


def _distance_adjusted_summary_rows(summary_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    fields = [
        "policy_name",
        "future_v2v_score_mean",
        "distance_adjusted_score_mean",
        "platform_profit_mean",
        "service_rate_mean",
        "total_pickup_distance_km_mean",
        "pickup_distance_per_served_order_mean",
        "mean_pickup_distance_km_mean",
    ]
    rows = [{field: row.get(field, "") for field in fields} for row in summary_rows]
    rows.sort(key=lambda row: float(row.get("distance_adjusted_score_mean") or 0.0), reverse=True)
    return rows


def _eval_summary_zh_rows(summary_rows: list[dict[str, float | str]]) -> list[dict[str, object]]:
    return [
        {
            _translate_eval_summary_field(key): _translate_eval_summary_value(key, value)
            for key, value in row.items()
        }
        for row in summary_rows
    ]


def _translate_eval_summary_field(field: str) -> str:
    direct = {
        "policy_name": "策略名称",
        "episodes": "评估轮数",
        "energy_loss_rate": "能量损耗率",
        "seller_compensation_share": "卖方补偿占买方支付比例",
        "platform_margin_per_served_order": "单服务订单平台边际收益",
        "timing_degenerate_risk": "时机退化风险",
        "environment_target_band": "环境目标区间达标",
    }
    if field in direct:
        return direct[field]
    suffix_map = {
        "_mean": "_均值",
        "_std": "_标准差",
        "_sem": "_标准误",
    }
    for suffix, zh_suffix in suffix_map.items():
        if field.endswith(suffix):
            return f"{_translate_eval_metric_base(field[: -len(suffix)])}{zh_suffix}"
    return field


def _translate_eval_metric_base(field: str) -> str:
    mapping = {
        "future_v2v_score": "未来V2V得分",
        "platform_profit": "平台利润",
        "service_rate": "服务率",
        "urgent_service_rate": "急单服务率",
        "expired_rate": "过期率",
        "cancelled_rate": "取消率",
        "mean_wait_before_match": "匹配前平均等待步数",
        "mean_batch_interval": "平均匹配间隔",
        "mean_pickup_time": "平均接驾时间",
        "total_pickup_distance_km": "总接驾距离_公里",
        "mean_pickup_distance_km": "平均接驾距离_公里",
        "pickup_distance_per_served_order": "单服务订单接驾距离_公里",
        "distance_adjusted_score": "距离修正得分",
        "mean_commitment_ticks": "平均服务占用步数",
        "profit_per_served_order": "单服务订单利润",
        "fleet_utilization": "车队车辆利用率",
        "private_utilization": "私人车辆利用率",
        "energy_utilization": "可供电量利用率",
        "dispatch_epoch_count": "派单次数",
        "unmet_kwh": "未满足电量_kWh",
        "delivered_kwh": "实际交付电量_kWh",
        "donor_output_kwh": "供电车输出电量_kWh",
        "energy_loss_kwh": "传输损耗电量_kWh",
        "buyer_payment": "买方支付",
        "seller_reimbursement": "卖方补偿",
        "seller_energy_cost": "卖方电能成本",
        "seller_degradation_cost": "电池退化成本",
        "seller_service_premium": "卖方服务溢价",
        "platform_margin": "平台边际收益",
        "dispatch_friction_cost": "派单交易摩擦成本",
        "dispatch_setup_cost": "派单启动成本",
        "dispatch_pair_coordination_cost": "配对协调成本",
        "dispatch_refresh_cost": "报价刷新成本",
        "dispatch_full_mode_extra_cost": "完整匹配额外协调成本",
        "friction_share_of_gross_profit": "摩擦成本占毛收益比例",
        "mean_donor_soc_after": "供电后平均SOC",
        "min_donor_soc_after": "供电后最低SOC",
        "donor_soc_violation_count": "SOC违规数",
        "battery_health_rejection_count": "电池健康约束拒绝数",
    }
    return mapping.get(field, field)


def _translate_eval_summary_value(field: str, value: object) -> object:
    if field == "policy_name":
        names = {
            "fixed_1_tick_full_match": "固定1步完整匹配",
            "fixed_2_tick_full_match": "固定2步完整匹配",
            "fixed_3_tick_full_match": "固定3步完整匹配",
            "fixed_4_tick_full_match": "固定4步完整匹配",
            "handcrafted_observable_rule": "手写公平可观测规则",
            "handcrafted_lookahead_rule": "手写前视强规则",
            "handcrafted_deadline_rule": "手写前视强规则",
            "adaptive_timing_ppo": "自适应时机PPO",
            "adaptive_interval_dqn": "自适应间隔DQN",
        }
        return names.get(str(value), value)
    if isinstance(value, bool):
        return "是" if value else "否"
    return value


def _interval_action_distribution_by_policy(interval_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: dict[str, dict[str, int]] = {}
    binary_counts: dict[str, dict[str, int]] = {}
    for row in interval_rows:
        policy = str(row.get("policy_name", ""))
        action = str(row.get("interval_action_name", ""))
        binary_action = str(row.get("binary_action_name", ""))
        if policy and binary_action:
            bucket = binary_counts.setdefault(policy, {})
            bucket[binary_action] = bucket.get(binary_action, 0) + 1
        if not policy or not action or row.get("final_dispatch_executed") is not True:
            continue
        bucket = counts.setdefault(policy, {})
        bucket[action] = bucket.get(action, 0) + 1
    rows = []
    for policy in sorted(set(counts) | set(binary_counts)):
        action_counts = counts.get(policy, {})
        decision_counts = binary_counts.get(policy, {})
        total = sum(action_counts.values())
        decision_total = sum(decision_counts.values())
        rows.append(
            {
                "policy_name": policy,
                "total_decisions": decision_total,
                "total_interval_actions": total,
                "wait_action_share": decision_counts.get("wait", 0) / max(1, decision_total),
                "match_full_action_share": decision_counts.get("match_full", 0) / max(1, decision_total),
                "dispatch_now_share": action_counts.get("dispatch_now", 0) / max(1, total),
                "delay_1_share": action_counts.get("delay_1_then_dispatch", 0) / max(1, total),
                "delay_2_share": action_counts.get("delay_2_then_dispatch", 0) / max(1, total),
                "delay_3_share": action_counts.get("delay_3_then_dispatch", 0) / max(1, total),
                "delay_4_plus_share": action_counts.get("delay_4_plus_then_dispatch", 0) / max(1, total),
                "max_action_share": max(action_counts.values(), default=0) / max(1, total),
                "binary_max_action_share": max(decision_counts.values(), default=0) / max(1, decision_total),
            }
        )
    return rows


def _interval_distribution_histogram(interval_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: dict[tuple[str, int], int] = {}
    totals: dict[str, int] = {}
    for row in interval_rows:
        policy = str(row.get("policy_name", ""))
        if not policy or row.get("final_dispatch_executed") is not True:
            continue
        delay_value = row.get("delay_ticks", 0)
        if delay_value == "":
            continue
        interval = int(float(delay_value)) + 1
        counts[(policy, interval)] = counts.get((policy, interval), 0) + 1
        totals[policy] = totals.get(policy, 0) + 1
    return [
        {
            "policy_name": policy,
            "interval_ticks": interval,
            "count": count,
            "share": count / max(1, totals.get(policy, 0)),
        }
        for (policy, interval), count in sorted(counts.items())
    ]


def _state_action_policy_trace(interval_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = [
        "policy_name",
        "seed",
        "scenario_id",
        "start_tick",
        "binary_action_name",
        "match_probability",
        "wait_probability",
        "interval_action_name",
        "delay_ticks",
        "final_dispatch_executed",
        "active_orders",
        "available_vehicles",
        "supply_demand_ratio",
        "near_deadline_share",
        "candidate_density",
        "mean_pickup_distance_est",
        "mean_pickup_time_est",
        "projected_service_risk",
        "projected_urgent_service_risk",
        "raw_reward",
        "shaped_reward",
        "pbrs_delta",
        "value_estimate",
        "entropy",
    ]
    return [{field: row.get(field, "") for field in fields} for row in interval_rows]


def _state_action_bucket_summary(interval_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = {}
    for row in interval_rows:
        key = (
            str(row.get("policy_name", "")),
            _bucket(float(row.get("active_orders", 0.0)), [25, 50, 100], "orders"),
            _bucket(float(row.get("available_vehicles", 0.0)), [25, 50, 100], "vehicles"),
            _bucket(float(row.get("supply_demand_ratio", 0.0)), [0.6, 1.0, 1.6], "sd_ratio"),
            _bucket(float(row.get("near_deadline_share", 0.0)), [0.08, 0.18, 0.30], "deadline"),
            _bucket(float(row.get("candidate_density", 0.0)), [0.02, 0.06, 0.12], "density"),
        )
        grouped.setdefault(key, []).append(row)
    rows = []
    for key, values in sorted(grouped.items()):
        policy, order_bucket, vehicle_bucket, sd_bucket, deadline_bucket, density_bucket = key
        total = len(values)
        action_counts: dict[str, int] = {}
        binary_counts: dict[str, int] = {}
        for row in values:
            action = str(row.get("interval_action_name", ""))
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
            binary_action = str(row.get("binary_action_name", ""))
            if binary_action:
                binary_counts[binary_action] = binary_counts.get(binary_action, 0) + 1
        rows.append(
            {
                "policy_name": policy,
                "order_bucket": order_bucket,
                "vehicle_bucket": vehicle_bucket,
                "supply_demand_bucket": sd_bucket,
                "near_deadline_bucket": deadline_bucket,
                "candidate_density_bucket": density_bucket,
                "sample_count": total,
                "wait_action_share": binary_counts.get("wait", 0) / max(1, total),
                "match_full_action_share": binary_counts.get("match_full", 0) / max(1, total),
                "mean_match_probability": _mean(
                    [float(row.get("match_probability", 0.0)) for row in values if row.get("match_probability", "") != ""]
                ),
                "dispatch_now_share": action_counts.get("dispatch_now", 0) / max(1, total),
                "delay_1_share": action_counts.get("delay_1_then_dispatch", 0) / max(1, total),
                "delay_2_share": action_counts.get("delay_2_then_dispatch", 0) / max(1, total),
                "delay_3_share": action_counts.get("delay_3_then_dispatch", 0) / max(1, total),
                "delay_4_plus_share": action_counts.get("delay_4_plus_then_dispatch", 0) / max(1, total),
                "mean_raw_reward": _mean([float(row.get("raw_reward", 0.0)) for row in values]),
                "mean_shaped_reward": _mean([float(row.get("shaped_reward", 0.0)) for row in values]),
            }
        )
    return rows


def _bucket(value: float, cuts: list[float], prefix: str) -> str:
    if value < cuts[0]:
        return f"{prefix}_low"
    if value < cuts[1]:
        return f"{prefix}_mid"
    if value < cuts[2]:
        return f"{prefix}_high"
    return f"{prefix}_very_high"


def _paired_policy_delta_summary(
    metrics,
    reference_policy: str = "fixed_1_tick_full_match",
) -> list[dict[str, object]]:
    by_policy: dict[str, dict[str, object]] = {}
    for metric in metrics:
        by_policy.setdefault(metric.policy_name, {})[metric.scenario_id or str(metric.seed)] = metric
    reference = by_policy.get(reference_policy, {})
    rows: list[dict[str, object]] = []
    for policy_name, values in sorted(by_policy.items()):
        if policy_name == reference_policy:
            continue
        scenario_ids = sorted(set(values) & set(reference))
        if not scenario_ids:
            continue
        score_delta = [values[sid].future_v2v_score - reference[sid].future_v2v_score for sid in scenario_ids]
        profit_delta = [values[sid].platform_profit - reference[sid].platform_profit for sid in scenario_ids]
        service_delta = [values[sid].service_rate - reference[sid].service_rate for sid in scenario_ids]
        expired_delta = [values[sid].expired_rate - reference[sid].expired_rate for sid in scenario_ids]
        batch_delta = [values[sid].mean_batch_interval - reference[sid].mean_batch_interval for sid in scenario_ids]
        rows.append(
            {
                "policy_name": policy_name,
                "reference_policy": reference_policy,
                "paired_episodes": len(scenario_ids),
                "score_delta_mean": _mean(score_delta),
                "score_delta_std": _std(score_delta),
                "score_delta_sem": _sem(score_delta),
                "score_win_rate": _win_rate(score_delta),
                "profit_delta_mean": _mean(profit_delta),
                "profit_delta_std": _std(profit_delta),
                "profit_delta_sem": _sem(profit_delta),
                "profit_win_rate": _win_rate(profit_delta),
                "service_delta_mean": _mean(service_delta),
                "service_delta_sem": _sem(service_delta),
                "expired_delta_mean": _mean(expired_delta),
                "batch_interval_delta_mean": _mean(batch_delta),
            }
        )
    rows.sort(key=lambda row: float(row["score_delta_mean"]), reverse=True)
    return rows


def _paired_delta_report_rows(paired_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = [
        "policy_name",
        "reference_policy",
        "paired_episodes",
        "score_delta_mean",
        "score_delta_sem",
        "score_win_rate",
        "profit_delta_mean",
        "profit_delta_sem",
        "service_delta_mean",
        "expired_delta_mean",
        "batch_interval_delta_mean",
    ]
    return [{field: row.get(field, "") for field in fields} for row in paired_rows]


def _run_friction_sensitivity(
    config: ProjectConfig,
    scale_name: str,
    manifest_rows: list[dict[str, object]],
    policy_specs: list[tuple[str, str]],
    *,
    eval_workers: int,
    primary_summary_rows: list[dict[str, float | str]],
    primary_dispatch_rows: list[dict[str, object]],
    primary_paired_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = [
        _friction_sensitivity_row(
            "decomposed_transaction_cost",
            primary_summary_rows,
            primary_dispatch_rows,
            primary_paired_rows,
        )
    ]
    for variant_name, friction in _friction_sensitivity_variants(config.environment.dispatch_friction):
        variant_config = replace(
            config,
            environment=replace(config.environment, dispatch_friction=friction),
        )
        results = _evaluate_policy_specs(
            variant_config,
            scale_name,
            manifest_rows,
            policy_specs,
            eval_workers=eval_workers,
            desc=f"eval friction {variant_name}",
        )
        metrics = [result["metrics"] for result in results]
        dispatch_rows = [row for result in results for row in result["dispatch_trace"]]
        summary_rows = summarize_metrics(metrics)
        paired_rows = _paired_policy_delta_summary(metrics)
        rows.append(_friction_sensitivity_row(variant_name, summary_rows, dispatch_rows, paired_rows))
    no_refresh_delta = _variant_delta(rows, "no_refresh_friction")
    decomposed_delta = _variant_delta(rows, "decomposed_transaction_cost")
    friction_robust_ready = bool(decomposed_delta > 500.0 and no_refresh_delta >= 200.0)
    for row in rows:
        row["no_refresh_score_delta_mean"] = no_refresh_delta
        row["friction_robust_ready"] = friction_robust_ready
        row["friction_sensitive_risk"] = bool(no_refresh_delta < 200.0)
    return rows


def _friction_sensitivity_variants(
    base: DispatchFrictionConfig,
) -> list[tuple[str, DispatchFrictionConfig]]:
    return [
        (
            "no_refresh_friction",
            replace(base, refresh_cost=0.0),
        ),
        (
            "common_fixed_cost",
            replace(base, full_mode_extra_pair_cost=0.0),
        ),
        (
            "no_dispatch_friction",
            replace(
                base,
                enabled=False,
                setup_cost=0.0,
                pair_coordination_cost=0.0,
                full_mode_extra_pair_cost=0.0,
                refresh_cost=0.0,
            ),
        ),
    ]


def _friction_sensitivity_row(
    variant_name: str,
    summary_rows: list[dict[str, float | str]],
    dispatch_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
) -> dict[str, object]:
    summary_by_policy = {str(row["policy_name"]): row for row in summary_rows}
    fixed_1 = summary_by_policy.get("fixed_1_tick_full_match", {})
    fixed_2 = summary_by_policy.get("fixed_2_tick_full_match", {})
    best = max(summary_rows, key=lambda row: float(row["future_v2v_score_mean"]), default={})
    fixed_1_score = float(fixed_1.get("future_v2v_score_mean", 0.0))
    fixed_1_service = float(fixed_1.get("service_rate_mean", 0.0))
    fixed_2_service = float(fixed_2.get("service_rate_mean", 0.0))
    _ = dispatch_rows
    paired_best = paired_rows[0] if paired_rows else {}
    return {
        "variant": variant_name,
        "best_policy": best.get("policy_name", ""),
        "best_score_delta_vs_fixed1": float(best.get("future_v2v_score_mean", 0.0)) - fixed_1_score,
        "paired_best_policy": paired_best.get("policy_name", ""),
        "paired_best_score_delta_mean": paired_best.get("score_delta_mean", 0.0),
        "paired_best_score_win_rate": paired_best.get("score_win_rate", 0.0),
        "best_mean_batch_interval": best.get("mean_batch_interval_mean", 0.0),
        "fixed2_service_drop_vs_fixed1": fixed_1_service - fixed_2_service,
        "mean_friction_share_of_gross_profit": best.get("friction_share_of_gross_profit_mean", 0.0),
        "dispatch_friction_cost_mean": best.get("dispatch_friction_cost_mean", 0.0),
    }


def _variant_delta(rows: list[dict[str, object]], variant_name: str) -> float:
    for row in rows:
        if row.get("variant") == variant_name:
            return float(row.get("paired_best_score_delta_mean", 0.0))
    return 0.0


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
    env = make_env_factory(config, scale_name, seed, scenario_phase="eval")()
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


def _health_rows_match_manifest(
    health_rows: list[dict[str, object]],
    manifest_rows: list[dict[str, object]],
) -> bool:
    if len(health_rows) < len(manifest_rows):
        return False
    for health, manifest in zip(health_rows, manifest_rows):
        health_scenario = str(health.get("scenario_id", ""))
        manifest_scenario = str(manifest.get("scenario_id", ""))
        if health_scenario and manifest_scenario and health_scenario != manifest_scenario:
            return False
        if str(health.get("seed", "")) != str(manifest.get("seed", "")):
            return False
    return True


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)


def _sem(values: list[float]) -> float:
    return float(_std(values) / max(1.0, len(values) ** 0.5))


def _win_rate(values: list[float]) -> float:
    return float(sum(1 for value in values if value > 0.0) / max(1, len(values)))


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
    best_score = max((float(row["future_v2v_score_mean"]) for row in summary_rows), default=0.0)
    intervals = [float(row["mean_batch_interval_mean"]) for row in summary_rows]
    fixed_1_score = float(fixed_1["future_v2v_score_mean"]) if fixed_1 else 0.0
    fixed_1_profit = float(fixed_1["platform_profit_mean"]) if fixed_1 else 0.0
    fixed_1_service = float(fixed_1["service_rate_mean"]) if fixed_1 else 0.0
    fixed_2_service = float(fixed_2["service_rate_mean"]) if fixed_2 else 0.0
    energy_loss_rate = float(fixed_1["energy_loss_rate"]) if fixed_1 else 0.0
    seller_compensation_share = float(fixed_1["seller_compensation_share"]) if fixed_1 else 0.0
    policy_spread_score = best_score - fixed_1_score
    batch_interval_spread = max(intervals, default=0.0) - min(intervals, default=0.0)
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
                "service_drop_fixed2_vs_fixed1": service_drop_fixed2_vs_fixed1,
                "full_match_cost_gap": float(row["platform_profit_mean"]) - fixed_1_profit,
                "energy_loss_rate": row.get("energy_loss_rate", energy_loss_rate),
                "seller_compensation_share": row.get("seller_compensation_share", seller_compensation_share),
                "platform_margin_per_served_order": row.get("platform_margin_per_served_order", 0.0),
                "donor_soc_violation_count_mean": row.get("donor_soc_violation_count_mean", 0.0),
                "battery_health_rejection_count_mean": row.get("battery_health_rejection_count_mean", 0.0),
                "dynamic_timing_ready": bool(
                    policy_spread_score > 300.0
                    and batch_interval_spread >= 0.8
                    and 0.04 <= service_drop_fixed2_vs_fixed1 <= 0.10
                ),
            }
        )
    return comparison


def _environment_acceptance_summary(
    summary_rows: list[dict[str, float | str]],
    dispatch_rows: list[dict[str, object]],
    action_rows: list[dict[str, object]],
    interval_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary_by_policy = {str(row["policy_name"]): row for row in summary_rows}
    fixed_1 = summary_by_policy.get("fixed_1_tick_full_match", {})
    fixed_2 = summary_by_policy.get("fixed_2_tick_full_match", {})
    best = max(summary_rows, key=lambda row: float(row["future_v2v_score_mean"]), default={})
    _ = dispatch_rows
    _ = action_rows
    fixed_1_score = float(fixed_1.get("future_v2v_score_mean", 0.0))
    fixed_1_service = float(fixed_1.get("service_rate_mean", 0.0))
    fixed_2_service = float(fixed_2.get("service_rate_mean", 0.0))
    fixed_1_expired = float(fixed_1.get("expired_rate_mean", 0.0))
    best_score_delta = float(best.get("future_v2v_score_mean", 0.0)) - fixed_1_score
    learned_policy_name = "adaptive_timing_ppo"
    if not any(str(row.get("policy_name", "")) == learned_policy_name for row in interval_rows):
        learned_policy_name = "adaptive_interval_dqn"
    interval_actions: dict[str, int] = {}
    binary_actions: dict[str, int] = {}
    interval_delays: list[int] = []
    for row in interval_rows:
        if str(row.get("policy_name", "")) != learned_policy_name:
            continue
        binary_action = str(row.get("binary_action_name", ""))
        if binary_action:
            binary_actions[binary_action] = binary_actions.get(binary_action, 0) + 1
        action = str(row.get("interval_action_name", ""))
        if action:
            interval_actions[action] = interval_actions.get(action, 0) + 1
        if row.get("final_dispatch_executed") is True and row.get("delay_ticks", "") != "":
            interval_delays.append(int(float(row.get("delay_ticks", 0))))
    interval_total = sum(interval_actions.values())
    binary_total = sum(binary_actions.values())
    interval_mean_delay = float(sum(interval_delays) / max(1, len(interval_delays))) if interval_delays else 0.0
    interval_mean_action_interval = 1.0 + interval_mean_delay
    interval_max_action_share = max(interval_actions.values(), default=0) / max(1, interval_total) if interval_total else 0.0
    binary_max_action_share = max(binary_actions.values(), default=0) / max(1, binary_total) if binary_total else 0.0
    interval_delayed_action_rate = (
        interval_total - interval_actions.get("dispatch_now", 0)
    ) / max(1, interval_total) if interval_total else 0.0
    best_interval = float(best.get("mean_batch_interval_mean", 0.0))
    paired_best = paired_rows[0] if paired_rows else {}
    paired_best_delta = float(paired_best.get("score_delta_mean", 0.0))
    friction_sensitivity_run = bool(sensitivity_rows)
    no_refresh_delta = _variant_delta(sensitivity_rows, "no_refresh_friction") if friction_sensitivity_run else 0.0
    fixed_1_target_band = bool(fixed_1.get("environment_target_band", False))
    fixed_2_target_band = bool(fixed_2.get("environment_target_band", False))
    target_band_policy_count = sum(1 for row in summary_rows if bool(row.get("environment_target_band", False)))
    baseline_ready = bool(
        paired_best_delta > 250.0
        and 0.04 <= fixed_1_service - fixed_2_service <= 0.10
        and fixed_1_target_band
        and fixed_2_target_band
        and 1.00 <= best_interval <= 2.50
    )
    friction_robust_ready = bool(baseline_ready and (not friction_sensitivity_run or no_refresh_delta >= 200.0))
    interval_action_ready = bool(
        interval_total
        and interval_max_action_share <= 0.75
        and (not binary_total or binary_max_action_share <= 0.85)
        and 1.20 <= interval_mean_action_interval <= 2.20
    )
    learned_policy_total = interval_total
    dqn_ready = bool(interval_action_ready)
    return [
        {
            "best_policy": best.get("policy_name", ""),
            "best_score_delta_vs_fixed1": best_score_delta,
            "best_mean_batch_interval": best_interval,
            "fixed1_expired_rate": fixed_1_expired,
            "fixed1_environment_target_band": fixed_1_target_band,
            "fixed2_environment_target_band": fixed_2_target_band,
            "target_band_policy_count": target_band_policy_count,
            "fixed2_service_drop_vs_fixed1": fixed_1_service - fixed_2_service,
            "learned_policy_name": learned_policy_name,
            "learned_policy_binary_action_count": binary_total,
            "learned_policy_binary_max_action_share": binary_max_action_share,
            "adaptive_interval_action_count": interval_total,
            "adaptive_interval_mean_action_interval": interval_mean_action_interval,
            "adaptive_interval_delayed_action_rate": interval_delayed_action_rate,
            "adaptive_interval_max_action_share": interval_max_action_share,
            "paired_best_policy": paired_best.get("policy_name", ""),
            "paired_best_score_delta_mean": paired_best_delta,
            "paired_best_score_win_rate": paired_best.get("score_win_rate", 0.0),
            "no_refresh_score_delta_mean": no_refresh_delta,
            "friction_sensitivity_run": friction_sensitivity_run,
            "baseline_environment_ready": baseline_ready,
            "friction_robust_ready": friction_robust_ready,
            "friction_sensitive_risk": bool(friction_sensitivity_run and no_refresh_delta < 200.0),
            "dqn_action_ready": dqn_ready,
            "adaptive_interval_action_ready": interval_action_ready,
            "dynamic_timing_ready": bool(friction_robust_ready and (dqn_ready or not learned_policy_total)),
        }
    ]


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_project_config(args.config), args)
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
        run_eval(
            config,
            scale_name,
            run_dir,
            args.seed,
            eval_episodes=args.eval_episodes or 3,
            eval_workers=args.eval_workers,
            include_friction_sensitivity=args.include_friction_sensitivity,
        )
        run_report(run_dir)
        return
    stages = ["generate", "train", "eval", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "generate":
            run_generate(config, scale_name, run_dir, args.seed, args.eval_episodes)
        elif stage == "train":
            run_train(config, scale_name, run_dir, args.seed, args.episodes, args.rollout_workers)
        elif stage == "eval":
            run_eval(
                config,
                scale_name,
                run_dir,
                args.seed,
                args.eval_episodes,
                args.eval_workers,
                include_friction_sensitivity=args.include_friction_sensitivity,
            )
        elif stage == "report":
            run_report(run_dir)


if __name__ == "__main__":
    main()
