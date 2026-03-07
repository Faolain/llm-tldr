import json
import runpy
import sys
from pathlib import Path

import pytest


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_phase_gate.py")


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_evaluate_phase_a_passes_with_expected_margins():
    mod = _load_mod()

    report = {
        "results": {
            "tasks_scored": 20,
            "win_rate_tldr_over_rg": 0.6,
            "f1_mean": {"tldr": 0.72, "rg": 0.6},
            "errors_total": 1,
            "bad_json": 1,
            "per_task": [
                {
                    "variants": [
                        {"trials": [{}, {}]},
                    ]
                }
                for _ in range(20)
            ],
        }
    }
    gates = {
        "win_rate_tldr_over_rg_min": 0.55,
        "f1_mean_tldr_min_delta": 0.05,
        "max_error_rate": 0.05,
        "min_tasks_completed": 20,
    }

    diagnostic = mod["evaluate_report"](report, gates)
    assert diagnostic["pass"] is True
    assert diagnostic["summary"]["f1_mean_delta"] == pytest.approx(0.12)
    assert diagnostic["summary"]["error_rate"] == pytest.approx(0.05)


def test_evaluate_preflight_gate_shape_from_metrics_report():
    mod = _load_mod()

    report = {
        "metrics": {
            "correct_first_tool_rate": 0.9,
            "workflow_compliance_rate": 0.85,
            "tldrf_usage_on_required_rate": 1.0,
            "rg_first_on_exact_rate": 1.0,
            "median_turns_before_first_appropriate_tool_use": 1.0,
            "systematic_failure_detected": False,
        }
    }
    gates = {
        "correct_first_tool_min": 0.8,
        "workflow_compliance_min": 0.8,
        "tldrf_usage_on_required_min": 0.9,
        "rg_first_on_exact_min": 0.9,
        "median_dead_end_turns_max": 1,
        "systematic_failure_detected_must_be": False,
    }

    diagnostic = mod["evaluate_report"](report, gates)
    assert diagnostic["pass"] is True
    assert diagnostic["phase"] == "preflight"


def test_evaluate_phase_d_allows_close_solve_rate_when_tokens_drop():
    mod = _load_mod()

    report = {
        "results": {
            "baseline": {
                "solve_rate": 0.5,
                "total_input_tokens": 800,
                "total_output_tokens": 200,
                "median_turn_count": 10,
                "median_wall_clock_s": 100,
                "tasks_completed": 10,
                "judge_config_hash": "same-hash",
            },
            "augmented": {
                "solve_rate": 0.53,
                "total_input_tokens": 500,
                "total_output_tokens": 100,
                "median_turn_count": 9,
                "median_wall_clock_s": 95,
                "tasks_completed": 10,
                "judge_config_hash": "same-hash",
            },
        }
    }
    gates = {
        "solve_rate_delta_min": 0.0,
        "token_reduction_min_pct": 15,
        "turn_reduction_min_pct": 20,
        "time_reduction_min_pct": 15,
        "min_tasks_completed": 10,
        "enforce_judge_config_hash_match": True,
    }

    diagnostic = mod["evaluate_report"](report, gates)
    assert diagnostic["pass"] is True
    assert diagnostic["summary"]["token_reduction_pct"] == pytest.approx(40.0)


def test_main_returns_exit_2_when_judge_hashes_differ(monkeypatch, tmp_path: Path):
    mod = _load_mod()
    globals_dict = mod["main"].__globals__

    report_path = tmp_path / "report.json"
    gates_path = tmp_path / "gates.json"
    out_path = tmp_path / "diagnostic.json"
    _write_json(
        report_path,
        {
            "results": {
                "baseline": {"solve_rate": 0.5, "tasks_completed": 10, "judge_config_hash": "hash-a"},
                "augmented": {"solve_rate": 0.55, "tasks_completed": 10, "judge_config_hash": "hash-b"},
            }
        },
    )
    _write_json(
        gates_path,
        {
            "solve_rate_delta_min": 0.0,
            "min_tasks_completed": 10,
            "enforce_judge_config_hash_match": True,
        },
    )

    monkeypatch.setitem(globals_dict, "get_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_phase_gate.py",
            "--report",
            str(report_path),
            "--gates",
            str(gates_path),
            "--out",
            str(out_path),
        ],
    )

    assert mod["main"]() == 2
    written = json.loads(out_path.read_text(encoding="utf-8"))
    judge_gate = next(row for row in written["gates"] if row["name"] == "judge_config_hash_match")
    assert judge_gate["pass"] is False
