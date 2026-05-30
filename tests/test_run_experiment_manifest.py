from __future__ import annotations

import importlib.util
from pathlib import Path

from future_v2v.config import load_project_config


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_experiment", SCRIPT_PATH)
assert SPEC is not None
run_experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_experiment)


def test_main_uses_eighty_fixed_eval_scenarios_by_default() -> None:
    config = load_project_config("configs/default.json")
    assert config.scale("main").eval_episodes == 80


def test_health_rows_match_manifest_requires_same_seed_and_scenario_order() -> None:
    manifest = [
        {"scenario_id": "eval_000", "seed": "10"},
        {"scenario_id": "eval_001", "seed": "11"},
    ]
    health = [
        {"scenario_id": "eval_000", "seed": "10"},
        {"scenario_id": "eval_001", "seed": "11"},
        {"scenario_id": "eval_002", "seed": "12"},
    ]
    assert run_experiment._health_rows_match_manifest(health, manifest)
    assert not run_experiment._health_rows_match_manifest(
        [{"scenario_id": "eval_001", "seed": "11"}],
        manifest,
    )
    assert not run_experiment._health_rows_match_manifest(
        [{"scenario_id": "eval_000", "seed": "99"}, {"scenario_id": "eval_001", "seed": "11"}],
        manifest,
    )
