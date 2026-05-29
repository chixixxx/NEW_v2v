from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_v2v.algorithms.baselines import default_baselines, run_policy_episode
from future_v2v.algorithms.dqn import DQNTimingAgent
from future_v2v.config import ProjectConfig, load_project_config, resolve_run_dir
from future_v2v.envs.timing_env import FutureV2VTimingEnv
from future_v2v.metrics import EpisodeMetrics, summarize_metrics, write_csv
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
    rows = [
        env.env_health_row(seed + idx)
        for idx in progress(range(count), desc="generate env health", total=count, unit="seed")
    ]
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


def run_train(config: ProjectConfig, scale_name: str, run_dir: Path, seed: int, episodes: int | None) -> None:
    env_factory = make_env_factory(config, scale_name, seed)
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    train_episodes = episodes or config.scale(scale_name).train_episodes
    agent = DQNTimingAgent(obs_dim=len(obs), training_config=config.training)
    history = agent.train(env_factory, episodes=train_episodes, seed_start=seed)
    write_csv(run_dir / "train" / "train_history.csv", history)
    agent.save(run_dir / "train" / "dqn_timing_agent.pt")
    write_markdown_report(
        run_dir / "train" / "train_report.md",
        title=f"DQN Timing Training ({scale_name})",
        summary_lines=[
            "训练目标是学习 WAIT/MATCH 时机，匹配边仍由约束优化器决定。",
            f"episodes={train_episodes}, replay_prefill={config.training.teacher_prefill_episodes}",
        ],
        table_rows=history[-10:],
    )


def run_eval(config: ProjectConfig, scale_name: str, run_dir: Path, seed: int, eval_episodes: int | None) -> None:
    scale = config.scale(scale_name)
    count = eval_episodes or scale.eval_episodes
    policies = list(default_baselines())
    checkpoint = run_dir / "train" / "dqn_timing_agent.pt"
    if checkpoint.exists():
        env = make_env_factory(config, scale_name, seed)()
        obs, _ = env.reset(seed=seed)
        policies.append(DQNTimingAgent.load(checkpoint, config.training))
        _ = obs
    all_metrics: list[EpisodeMetrics] = []
    tasks = [(policy, idx) for policy in policies for idx in range(count)]
    for policy, idx in progress(tasks, desc="eval policies", total=len(tasks), unit="episode"):
        env = make_env_factory(config, scale_name, seed + idx)()
        all_metrics.append(run_policy_episode(env, policy, seed=seed + idx))
    detail_rows = [metric.to_row() for metric in all_metrics]
    summary_rows = summarize_metrics(all_metrics)
    write_csv(run_dir / "eval" / "episode_metrics.csv", detail_rows)
    write_csv(run_dir / "eval" / "eval_summary.csv", summary_rows)
    write_markdown_report(
        run_dir / "eval" / "eval_report.md",
        title=f"Future V2V Timing Evaluation ({scale_name})",
        summary_lines=[
            "主表按 future_v2v_score_mean 排序；profit、服务率、取消和过期是辅助解释指标。",
            f"episodes_per_policy={count}",
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
        "本总报告汇总环境健康、训练和评估三个部分。详细 CSV 保留在各自目录中。",
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


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    scale_name = "smoke" if args.stage == "smoke" else args.scale
    run_dir = resolve_run_dir(config, scale_name, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        run_generate(config, scale_name, run_dir, args.seed, eval_episodes=3)
        run_train(config, scale_name, run_dir, args.seed, episodes=args.episodes or 5)
        run_eval(config, scale_name, run_dir, args.seed, eval_episodes=args.eval_episodes or 3)
        run_report(run_dir)
        return
    stages = ["generate", "train", "eval", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "generate":
            run_generate(config, scale_name, run_dir, args.seed, args.eval_episodes)
        elif stage == "train":
            run_train(config, scale_name, run_dir, args.seed, args.episodes)
        elif stage == "eval":
            run_eval(config, scale_name, run_dir, args.seed, args.eval_episodes)
        elif stage == "report":
            run_report(run_dir)


if __name__ == "__main__":
    main()

