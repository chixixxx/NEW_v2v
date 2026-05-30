from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_timing_run.py"
SPEC = importlib.util.spec_from_file_location("summarize_timing_run", SCRIPT_PATH)
assert SPEC is not None
summarize_timing_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summarize_timing_run)


def test_diagnose_interval_flags_high_frequency_dispatch() -> None:
    status = summarize_timing_run.diagnose_interval(
        paired_delta=100.0,
        best_gap=120.0,
        mean_batch_interval=1.02,
        service_rate=0.75,
        platform_profit=1000.0,
        no_refresh_delta=250.0,
        common_fixed_delta=250.0,
        interval_mean_action_interval=1.05,
        interval_max_action_share=0.60,
        off_peak_score=100.0,
        off_peak_penalty=0.0,
    )
    assert "INTERVAL_TOO_SHORT" in status


def test_diagnose_interval_flags_single_action_degeneracy() -> None:
    status = summarize_timing_run.diagnose_interval(
        paired_delta=100.0,
        best_gap=120.0,
        mean_batch_interval=1.50,
        service_rate=0.75,
        platform_profit=1000.0,
        no_refresh_delta=250.0,
        common_fixed_delta=250.0,
        interval_mean_action_interval=1.50,
        interval_max_action_share=0.90,
        off_peak_score=100.0,
        off_peak_penalty=0.0,
    )
    assert "SINGLE_INTERVAL_DEGENERACY" in status


def test_diagnose_interval_accepts_candidate_ready() -> None:
    status = summarize_timing_run.diagnose_interval(
        paired_delta=300.0,
        best_gap=100.0,
        mean_batch_interval=1.35,
        service_rate=0.74,
        platform_profit=1000.0,
        no_refresh_delta=250.0,
        common_fixed_delta=250.0,
        interval_mean_action_interval=1.45,
        interval_max_action_share=0.50,
        off_peak_score=100.0,
        off_peak_penalty=0.0,
    )
    assert status == "CANDIDATE_READY"


def test_interval_distribution_reports_action_concentration() -> None:
    rates = summarize_timing_run.eval_interval_distribution(
        [
            {"policy_name": "adaptive_interval_dqn", "interval_action_name": "dispatch_now"},
            {"policy_name": "adaptive_interval_dqn", "interval_action_name": "delay_1_then_dispatch"},
            {"policy_name": "adaptive_interval_dqn", "interval_action_name": "delay_2_then_dispatch"},
            {"policy_name": "adaptive_interval_dqn", "interval_action_name": "delay_3_then_dispatch"},
        ],
        "adaptive_interval_dqn",
    )
    assert rates["interval_max_action_share"] == 0.25
    assert rates["interval_mean_action_interval"] == 2.5


def test_select_primary_policy_uses_adaptive_interval_only() -> None:
    policy = summarize_timing_run.select_primary_policy(
        [
            {"policy_name": "fixed_1_tick_full_match"},
            {"policy_name": "adaptive_interval_dqn"},
        ]
    )
    assert policy == "adaptive_interval_dqn"
