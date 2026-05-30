from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_interval_envelope.py"
SPEC = importlib.util.spec_from_file_location("run_interval_envelope", SCRIPT_PATH)
assert SPEC is not None
run_interval_envelope = importlib.util.module_from_spec(SPEC)
sys.modules["run_interval_envelope"] = run_interval_envelope
assert SPEC.loader is not None
SPEC.loader.exec_module(run_interval_envelope)


def test_interval_diversity_summary_requires_more_than_fixed2() -> None:
    bucket_rows = [
        {"best_interval": 2, "best_delta_vs_fixed2": 0.0},
        {"best_interval": 1, "best_delta_vs_fixed2": 30.0},
        {"best_interval": 3, "best_delta_vs_fixed2": 20.0},
    ]
    summary = run_interval_envelope.interval_diversity_summary(bucket_rows, [], [])[0]
    assert summary["fixed2_dominance_rate"] < 0.70
    assert summary["unique_best_interval_count"] == 3
    assert summary["interval_diversity_ready"] is True


def test_interval_bucket_assignment_uses_same_reference_for_all_policies() -> None:
    rows = [
        {
            "scenario_id": "s1",
            "policy_name": "fixed_1_tick_full_match",
            "expired_rate": 0.10,
            "cancelled_rate": 0.10,
            "pickup_distance_per_served_order": 2.0,
        },
        {
            "scenario_id": "s1",
            "policy_name": "fixed_2_tick_full_match",
            "expired_rate": 0.40,
            "cancelled_rate": 0.20,
            "pickup_distance_per_served_order": 4.0,
        },
    ]
    out = run_interval_envelope.attach_consistent_bucket_fields(rows)
    assert out[0]["service_risk_bucket"] == out[1]["service_risk_bucket"]
    assert out[0]["pickup_distance_bucket"] == out[1]["pickup_distance_bucket"]
