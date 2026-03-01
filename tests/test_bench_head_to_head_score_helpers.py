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


def test_score_budget_violation_rate_counts_over_budget_ok_predictions(tmp_path: Path):
    mod = _load_mod()
    suite_id = "h2h_budget_violation_rate"

    suite = {
        "schema_version": 1,
        "suite_id": suite_id,
        "lanes": [
            {
                "id": "common",
                "required_for_all_tools": True,
                "categories": ["retrieval"],
            }
        ],
        "budgets": {
            "token_budgets": [100],
            "retrieval_ks": [1, 5, 10],
        },
        "protocol": {
            "trials": 1,
        },
    }

    tasks = [
        {
            "task_id": "retrieval:ok-over-budget",
            "category": "retrieval",
            "ground_truth": {
                "relevant_files": ["src/a.py"],
                "is_negative": False,
            },
        },
        {
            "task_id": "retrieval:error-over-budget",
            "category": "retrieval",
            "ground_truth": {
                "relevant_files": ["src/b.py"],
                "is_negative": False,
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

    preds_doc = {
        "schema_version": 1,
        "suite_id": suite_id,
        "task_manifest_sha256": task_manifest_sha256,
        "tool_id": "tool-under-test",
        "predictions": [
            {
                "task_id": "retrieval:ok-over-budget",
                "budget_tokens": 100,
                "trial": 1,
                "status": "ok",
                "payload_tokens": 140,
                "payload_bytes": 10,
                "latency_ms": 1.0,
                "result": {"ranked_files": ["src/a.py"]},
            },
            {
                "task_id": "retrieval:error-over-budget",
                "budget_tokens": 100,
                "trial": 1,
                "status": "error",
                "payload_tokens": 1000,
                "payload_bytes": 10,
                "latency_ms": 1.0,
                "result": {"ranked_files": ["src/b.py"]},
            },
        ],
    }

    suite_path = tmp_path / "suite.json"
    tasks_path = tmp_path / "tasks.json"
    preds_path = tmp_path / "predictions.json"
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
    assert report["status_counts"]["budget_violations"] == 1
    assert report["rates"]["budget_violation_rate"] == pytest.approx(0.5)
