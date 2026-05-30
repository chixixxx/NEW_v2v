from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_timing_environment.py"
SPEC = importlib.util.spec_from_file_location("diagnose_timing_environment", SCRIPT_PATH)
assert SPEC is not None
diagnose_timing_environment = importlib.util.module_from_spec(SPEC)
sys.modules["diagnose_timing_environment"] = diagnose_timing_environment
assert SPEC.loader is not None
SPEC.loader.exec_module(diagnose_timing_environment)


def test_oracle_summary_detects_dynamic_wait_support() -> None:
    rows = [
        {"wait_margin_vs_now": 10.0, "wait_option_best": True, "wait_positive_margin": True},
        {"wait_margin_vs_now": 4.0, "wait_option_best": True, "wait_positive_margin": True},
        {"wait_margin_vs_now": -2.0, "wait_option_best": False, "wait_positive_margin": False},
    ]
    summary = diagnose_timing_environment.summarize_oracle_rows(rows)
    assert summary["wait_option_best_share"] > 0.12
    assert summary["wait_positive_margin_share"] > 0.20
    assert summary["dynamic_wait_supported"] is True


def test_fixed_interval_summary_requires_non_one_tick_value() -> None:
    rows = [
        {
            "policy_name": "fixed_1_tick_full_match",
            "future_v2v_score_mean": 1000.0,
        },
        {
            "policy_name": "fixed_2_tick_full_match",
            "future_v2v_score_mean": 1300.0,
        },
    ]
    summary = diagnose_timing_environment.summarize_fixed_interval_envelope(rows)
    assert summary["best_fixed_policy"] == "fixed_2_tick_full_match"
    assert summary["best_delta_vs_fixed1_full"] == 300.0
    assert summary["fixed_interval_dynamic_supported"] is True
