from __future__ import annotations

import argparse
import copy
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_v2v.algorithms.baselines import FixedIntervalPolicy, HandcraftedDeadlineRulePolicy, run_policy_episode
from future_v2v.config import ProjectConfig, load_project_config, resolve_run_dir
from future_v2v.envs.timing_env import MATCH_FULL, WAIT, FutureV2VTimingEnv
from future_v2v.metrics import EpisodeMetrics, summarize_metrics, write_csv
from future_v2v.progress import progress


@dataclass(frozen=True)
class OptionSpec:
    name: str
    actions: tuple[int, ...]


ORACLE_OPTIONS = [
    OptionSpec("match_full_now", (MATCH_FULL, WAIT, WAIT)),
    OptionSpec("wait1_then_full", (WAIT, MATCH_FULL, WAIT)),
    OptionSpec("wait2_then_full", (WAIT, WAIT, MATCH_FULL)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether the environment supports adaptive V2V timing.")
    parser.add_argument("--scale", choices=["smoke", "main"], default="main")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--run-name", default="timing_env_diagnostic")
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--max-probe-states", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    run_dir = resolve_run_dir(config, args.scale, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_or_build_manifest(config, args.scale, run_dir, seed=args.seed, count=args.eval_episodes)
    diagnostic_dir = run_dir / "env_diagnostics"

    oracle_rows = short_window_oracle_rows(
        config,
        args.scale,
        manifest,
        max_probe_states=args.max_probe_states,
    )
    oracle_summary = summarize_oracle_rows(oracle_rows)
    fixed_rows = fixed_interval_envelope_rows(config, args.scale, manifest)
    fixed_summary = summarize_fixed_interval_envelope(fixed_rows)
    research_value = research_value_summary(oracle_summary, fixed_summary)

    write_csv(diagnostic_dir / "short_window_oracle_states.csv", oracle_rows)
    write_csv(diagnostic_dir / "short_window_oracle_summary.csv", [oracle_summary])
    write_csv(diagnostic_dir / "fixed_interval_envelope.csv", fixed_rows)
    write_csv(diagnostic_dir / "fixed_interval_envelope_summary.csv", [fixed_summary])
    write_csv(diagnostic_dir / "environment_research_value_summary.csv", [research_value])
    print_research_value(research_value)


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
    rows = build_eval_manifest(env, seed=seed, count=count)
    write_csv(path, rows)
    return rows


def build_eval_manifest(env: FutureV2VTimingEnv, *, seed: int, count: int) -> list[dict[str, object]]:
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
            }
        )
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


def short_window_oracle_rows(
    config: ProjectConfig,
    scale_name: str,
    manifest: list[dict[str, object]],
    *,
    max_probe_states: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in progress(manifest, desc="short-window oracle", total=len(manifest), unit="scenario"):
        env = FutureV2VTimingEnv(config.environment, config.scale(scale_name), seed=int(scenario["seed"]))
        reset_eval_env(env, scenario)
        probe_ticks = selected_probe_ticks(env.scale_config.horizon_ticks, max_probe_states=max_probe_states)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            if env.current_tick in probe_ticks and env.snapshot().active_orders:
                rows.append(evaluate_oracle_state(env, scenario))
            action = MATCH_FULL if env.snapshot().active_orders else WAIT
            _obs, _reward, terminated, truncated, _info = env.step(action)
    return rows


def selected_probe_ticks(horizon_ticks: int, *, max_probe_states: int) -> set[int]:
    if max_probe_states <= 0:
        return set()
    upper = max(1, horizon_ticks - 2)
    count = min(max_probe_states, upper)
    return {int(value) for value in np.linspace(0, upper - 1, num=count)}


def evaluate_oracle_state(env: FutureV2VTimingEnv, scenario: dict[str, object]) -> dict[str, object]:
    snapshot = env.snapshot()
    option_rewards = {option.name: rollout_option_reward(env, option) for option in ORACLE_OPTIONS}
    now_best = option_rewards["match_full_now"]
    wait_best = max(option_rewards["wait1_then_full"], option_rewards["wait2_then_full"])
    best_option = max(option_rewards, key=option_rewards.get)
    waiting_ratios = [order.waiting_ratio(env.current_tick) for order in snapshot.active_orders]
    near_deadline_share = sum(
        1
        for order in snapshot.active_orders
        if order.max_wait_ticks - order.waiting_ticks(env.current_tick) <= 1
    ) / max(1, len(snapshot.active_orders))
    return {
        "scenario_id": scenario.get("scenario_id", ""),
        "seed": scenario.get("seed", ""),
        "time_of_day_bucket": scenario.get("time_of_day_bucket", ""),
        "tick": env.current_tick,
        "active_orders": len(snapshot.active_orders),
        "active_vehicles": len(snapshot.active_vehicles),
        "candidate_edges": len(snapshot.candidate_edges),
        "mean_waiting_ratio": float(np.mean(waiting_ratios)) if waiting_ratios else 0.0,
        "near_deadline_share": near_deadline_share,
        "match_full_now_reward": option_rewards["match_full_now"],
        "wait1_then_full_reward": option_rewards["wait1_then_full"],
        "wait2_then_full_reward": option_rewards["wait2_then_full"],
        "best_option": best_option,
        "wait_best_reward": wait_best,
        "now_best_reward": now_best,
        "wait_margin_vs_now": wait_best - now_best,
        "wait_option_best": best_option.startswith("wait"),
        "wait_positive_margin": wait_best > now_best,
    }


def rollout_option_reward(env: FutureV2VTimingEnv, option: OptionSpec) -> float:
    candidate = copy.deepcopy(env)
    total_reward = 0.0
    for action in option.actions:
        _obs, reward, terminated, truncated, _info = candidate.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    return total_reward


def summarize_oracle_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "probed_states": 0,
            "wait_option_best_share": 0.0,
            "wait_positive_margin_share": 0.0,
            "mean_wait_margin_vs_now": 0.0,
            "median_wait_margin_vs_now": 0.0,
            "dynamic_wait_supported": False,
        }
    margins = np.array([float(row["wait_margin_vs_now"]) for row in rows], dtype=float)
    wait_best = np.array([bool(row["wait_option_best"]) for row in rows], dtype=bool)
    wait_positive = np.array([bool(row["wait_positive_margin"]) for row in rows], dtype=bool)
    support = bool(wait_best.mean() >= 0.12 and wait_positive.mean() >= 0.20 and float(np.mean(margins)) > -12.0)
    return {
        "probed_states": len(rows),
        "wait_option_best_share": float(wait_best.mean()),
        "wait_positive_margin_share": float(wait_positive.mean()),
        "mean_wait_margin_vs_now": float(np.mean(margins)),
        "median_wait_margin_vs_now": float(np.median(margins)),
        "p75_wait_margin_vs_now": float(np.quantile(margins, 0.75)),
        "dynamic_wait_supported": support,
    }


def fixed_interval_envelope_rows(
    config: ProjectConfig,
    scale_name: str,
    manifest: list[dict[str, object]],
) -> list[dict[str, object]]:
    policies = [
        FixedIntervalPolicy(interval=1),
        FixedIntervalPolicy(interval=2),
        FixedIntervalPolicy(interval=3),
        FixedIntervalPolicy(interval=4),
        HandcraftedDeadlineRulePolicy(),
    ]
    metrics: list[EpisodeMetrics] = []
    for scenario in progress(manifest, desc="fixed interval envelope", total=len(manifest), unit="scenario"):
        for policy in policies:
            env = FutureV2VTimingEnv(config.environment, config.scale(scale_name), seed=int(scenario["seed"]))
            if str(scenario.get("day", "")):
                reset_eval_env(env, scenario)
                metrics.append(run_policy_on_reset_env(env, policy, seed=int(scenario["seed"])))
            else:
                metrics.append(run_policy_episode(env, policy, seed=int(scenario["seed"])))
    return summarize_metrics(metrics)


def run_policy_on_reset_env(env: FutureV2VTimingEnv, policy, *, seed: int) -> EpisodeMetrics:
    obs = env._observation()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy.act(env, obs)
        obs, _reward, terminated, truncated, _info = env.step(action)
    return env.episode_metrics(policy_name=policy.name, seed=seed)


def summarize_fixed_interval_envelope(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "best_policy": "",
            "best_score": 0.0,
            "fixed1_full_score": 0.0,
            "best_delta_vs_fixed1_full": 0.0,
            "best_interval": 0,
            "non_one_tick_best": False,
            "fixed_interval_dynamic_supported": False,
        }
    by_policy = {str(row["policy_name"]): row for row in rows}
    best = max(rows, key=lambda row: float(row["future_v2v_score_mean"]))
    fixed1 = by_policy.get("fixed_1_tick_full_match", {})
    best_policy = str(best["policy_name"])
    best_interval = parse_interval(best_policy)
    delta = float(best["future_v2v_score_mean"]) - float(fixed1.get("future_v2v_score_mean", 0.0))
    non_one_tick = bool(best_interval and best_interval > 1)
    interval_rows = [row for row in rows if str(row["policy_name"]).startswith("fixed_")]
    best_fixed = max(interval_rows, key=lambda row: float(row["future_v2v_score_mean"]), default=best)
    best_fixed_interval = parse_interval(str(best_fixed.get("policy_name", "")))
    supported = bool(delta > 200.0 and (non_one_tick or (best_fixed_interval and best_fixed_interval > 1)))
    return {
        "best_policy": best_policy,
        "best_score": float(best["future_v2v_score_mean"]),
        "fixed1_full_score": float(fixed1.get("future_v2v_score_mean", 0.0)),
        "best_delta_vs_fixed1_full": delta,
        "best_interval": best_interval,
        "best_fixed_policy": best_fixed.get("policy_name", ""),
        "best_fixed_interval": best_fixed_interval,
        "non_one_tick_best": non_one_tick,
        "fixed_interval_dynamic_supported": supported,
    }


def parse_interval(policy_name: str) -> int:
    parts = policy_name.split("_")
    if len(parts) >= 2 and parts[0] == "fixed":
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


def research_value_summary(oracle: dict[str, object], envelope: dict[str, object]) -> dict[str, object]:
    wait_supported = bool(oracle.get("dynamic_wait_supported", False))
    interval_supported = bool(envelope.get("fixed_interval_dynamic_supported", False))
    ready = bool(wait_supported and interval_supported)
    if ready:
        diagnosis = "READY: environment supports delayed V2V matching trade-offs."
    elif interval_supported:
        diagnosis = "PARTIAL: fixed intervals show value, but short-window WAIT opportunities are too weak."
    elif wait_supported:
        diagnosis = "PARTIAL: local WAIT opportunities exist, but fixed interval envelope is not strong enough."
    else:
        diagnosis = "NOT_READY: dynamic waiting is not sufficiently supported by the environment."
    row = {
        "research_value_ready": ready,
        "diagnosis": diagnosis,
    }
    row.update({f"oracle_{key}": value for key, value in oracle.items()})
    row.update({f"envelope_{key}": value for key, value in envelope.items()})
    return row


def print_research_value(row: dict[str, object]) -> None:
    keys = [
        "research_value_ready",
        "diagnosis",
        "oracle_wait_option_best_share",
        "oracle_wait_positive_margin_share",
        "oracle_mean_wait_margin_vs_now",
        "envelope_best_policy",
        "envelope_best_delta_vs_fixed1_full",
        "envelope_best_fixed_policy",
    ]
    width = max(len(key) for key in keys)
    for key in keys:
        print(f"{key:<{width}}  {row.get(key, '')}")


if __name__ == "__main__":
    main()
