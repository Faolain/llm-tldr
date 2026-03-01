import argparse
import json
import runpy
import sys
from pathlib import Path

import pytest


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_head_to_head.py")


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def test_cmd_score_result_shape_counters(tmp_path: Path):
    mod = _load_mod()
    suite_id = "h2h_llm_tldr_vs_contextplus_v1"

    suite = {
        "schema_version": 1,
        "suite_id": suite_id,
        "lanes": [
            {
                "id": "common",
                "required_for_all_tools": True,
                "categories": ["retrieval", "impact", "slice"],
            }
        ],
        "budgets": {
            "token_budgets": [2000],
            "retrieval_ks": [1, 5, 10],
        },
        "protocol": {
            "trials": 1,
        },
    }

    tasks = [
        {
            "task_id": "retrieval:R01",
            "category": "retrieval",
            "ground_truth": {
                "relevant_files": ["src/foo.py"],
                "is_negative": False,
            },
        },
        {
            "task_id": "impact:A01",
            "category": "impact",
            "ground_truth": {
                "callers": [{"file": "src/foo.py", "function": "entrypoint"}],
            },
        },
        {
            "task_id": "slice:B01",
            "category": "slice",
            "ground_truth": {
                "lines": [12, 13],
                "total_function_lines": 50,
            },
        },
    ]
    task_manifest_sha256 = mod["_sha256_json"](tasks)
    tasks_doc = {
        "schema_version": 1,
        "suite_id": suite_id,
        "task_manifest_sha256": task_manifest_sha256,
        "tasks": tasks,
    }

    repo_root = Path(__file__).resolve().parents[1]
    preds_fixture_path = repo_root / "tests" / "fixtures" / "head_to_head" / "predictions.malformed.json"
    preds_doc = json.loads(preds_fixture_path.read_text())
    preds_doc["task_manifest_sha256"] = task_manifest_sha256

    suite_path = tmp_path / "suite.json"
    tasks_path = tmp_path / "tasks.json"
    preds_path = tmp_path / "preds.json"
    out_path = tmp_path / "score.json"

    _write_json(suite_path, suite)
    _write_json(tasks_path, tasks_doc)
    _write_json(preds_path, preds_doc)

    rc = mod["cmd_score"](
        argparse.Namespace(
            suite=str(suite_path),
            tasks=str(tasks_path),
            predictions=str(preds_path),
            tool_profile=None,
            out=str(out_path),
        )
    )
    assert rc == 0

    report = json.loads(out_path.read_text())

    # Existing scoring/gating semantics stay unchanged.
    assert report["status_counts"]["expected_total"] == 3
    assert report["status_counts"]["ok"] == 3
    assert report["status_counts"]["error"] == 1
    assert report["status_counts"]["missing"] == 0
    assert report["rates"]["error_rate"] == pytest.approx(1 / 3)
    assert report["gates_passed"] is True

    # Additive typed telemetry for malformed/result-shape failures.
    counters = report["diagnostics"]["result_shape_counters"]
    assert counters["non_object_result"] == 1
    assert counters["empty_result_object"] == 1
    assert counters["category_shape_mismatch"] == 1
    assert counters["category_shape_mismatch_by_category"]["slice"] == 1
    assert counters["category_shape_mismatch_by_category"]["retrieval"] == 0
    assert counters["category_shape_mismatch_by_category"]["impact"] == 0
    assert counters["category_shape_mismatch_by_category"]["complexity"] == 0
    assert counters["category_shape_mismatch_by_category"]["data_flow"] == 0
    assert counters["total"] == 3


def test_score_emits_typed_parse_diagnostics_without_gate_math_drift(tmp_path: Path):
    test_cmd_score_result_shape_counters(tmp_path)
