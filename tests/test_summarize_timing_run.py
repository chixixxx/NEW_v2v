from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_timing_run.py"
SPEC = importlib.util.spec_from_file_location("summarize_timing_run", SCRIPT_PATH)
assert SPEC is not None
summarize_timing_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summarize_timing_run)


def test_diagnose_flags_high_frequency_dispatch() -> None:
    status = summarize_timing_run.diagnose(
        paired_delta=100.0,
        best_gap=120.0,
        wait_rate=0.05,
        full_rate=0.30,
        mean_batch_interval=1.02,
        service_rate=0.75,
        platform_profit=1000.0,
        no_refresh_delta=250.0,
        common_fixed_delta=250.0,
        off_peak_score=100.0,
        off_peak_penalty=0.0,
    )
    assert "HIGH_FREQUENCY_DISPATCH" in status


def test_diagnose_flags_over_waiting_when_service_drops() -> None:
    status = summarize_timing_run.diagnose(
        paired_delta=-50.0,
        best_gap=300.0,
        wait_rate=0.70,
        full_rate=0.20,
        mean_batch_interval=2.0,
        service_rate=0.60,
        platform_profit=1000.0,
        no_refresh_delta=250.0,
        common_fixed_delta=250.0,
        off_peak_score=100.0,
        off_peak_penalty=0.0,
    )
    assert "OVER_WAITING" in status


def test_diagnose_accepts_candidate_ready() -> None:
    status = summarize_timing_run.diagnose(
        paired_delta=300.0,
        best_gap=100.0,
        wait_rate=0.25,
        full_rate=0.25,
        mean_batch_interval=1.35,
        service_rate=0.74,
        platform_profit=1000.0,
        no_refresh_delta=250.0,
        common_fixed_delta=250.0,
        off_peak_score=100.0,
        off_peak_penalty=0.0,
    )
    assert status == "CANDIDATE_READY"


def test_eval_action_distribution_prefers_policy_trace() -> None:
    rates = summarize_timing_run.eval_action_distribution(
        [
            {"policy_name": "dqn_adaptive_timing", "action_name": "wait"},
            {"policy_name": "dqn_adaptive_timing", "action_name": "match_top_batch"},
            {"policy_name": "dqn_adaptive_timing", "action_name": "match_full"},
            {"policy_name": "fixed_1_tick_full_match", "action_name": "match_full"},
        ],
        "dqn_adaptive_timing",
    )
    assert rates["wait_rate"] == 1 / 3
    assert rates["top_batch_rate"] == 1 / 3
    assert rates["full_match_rate"] == 1 / 3
