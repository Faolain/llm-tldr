import runpy
import sys
from pathlib import Path

import pytest


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_h2h_baseline.py")


def _score_report(
    *,
    task_manifest_hash: str = "task-hash-1",
    valid_run: bool = True,
    mrr_2000: float = 0.6,
    recall5_2000: float = 0.7,
    mrr_500: float = 0.1,
    recall5_500: float = 0.2,
) -> dict[str, object]:
    return {
        "suite_id": "h2h_llm_tldr_vs_contextplus_v1",
        "tool_id": "llm-tldr",
        "inputs": {
            "task_manifest_sha256": task_manifest_hash,
        },
        "gate_checks": [
            {"name": "run_validity.max_timeout_rate", "pass": valid_run},
            {"name": "run_validity.max_error_rate", "pass": valid_run},
            {"name": "run_validity.max_budget_violation_rate", "pass": valid_run},
        ],
        "metrics": {
            "by_budget": {
                "500": {
                    "retrieval": {
                        "mrr_mean": mrr_500,
                        "recall@5_mean": recall5_500,
                    }
                },
                "2000": {
                    "retrieval": {
                        "mrr_mean": mrr_2000,
                        "recall@5_mean": recall5_2000,
                    }
                },
            }
        },
    }


def test_baseline_summary_rejects_mixed_task_manifest_hashes():
    mod = _load_mod()
    summarize = mod["_build_baseline_summary"]

    score_reports = [
        _score_report(task_manifest_hash="hash-a"),
        _score_report(task_manifest_hash="hash-b"),
    ]

    with pytest.raises(ValueError, match="task_manifest_hash"):
        summarize(score_reports)


def test_baseline_summary_requires_two_of_three_valid_runs():
    mod = _load_mod()
    summarize = mod["_build_baseline_summary"]

    one_valid = [
        _score_report(valid_run=True, mrr_2000=0.61, recall5_2000=0.71),
        _score_report(valid_run=False, mrr_2000=0.62, recall5_2000=0.72),
        _score_report(valid_run=False, mrr_2000=0.63, recall5_2000=0.73),
    ]
    with pytest.raises(ValueError, match="at least 2 valid runs"):
        summarize(one_valid, min_valid_runs=2, primary_budget=2000)

    two_valid = [
        _score_report(valid_run=True, mrr_2000=0.61, recall5_2000=0.71),
        _score_report(valid_run=True, mrr_2000=0.62, recall5_2000=0.72),
        _score_report(valid_run=False, mrr_2000=0.63, recall5_2000=0.73),
    ]
    summary = summarize(two_valid, min_valid_runs=2, primary_budget=2000)
    assert summary["tools"]["llm-tldr"]["valid_runs"] == 2


def test_baseline_variance_uses_budget_2000_mrr_and_recall5():
    mod = _load_mod()
    summarize = mod["_build_baseline_summary"]

    score_reports = [
        _score_report(mrr_500=0.10, recall5_500=0.20, mrr_2000=0.80, recall5_2000=0.90),
        _score_report(mrr_500=0.90, recall5_500=0.95, mrr_2000=0.80, recall5_2000=0.90),
        _score_report(mrr_500=0.50, recall5_500=0.60, mrr_2000=0.80, recall5_2000=0.90),
    ]

    summary = summarize(score_reports, min_valid_runs=2, primary_budget=2000)
    retrieval_var = summary["tools"]["llm-tldr"]["variance"]["retrieval"]

    assert retrieval_var["mrr_mean"]["values"] == [0.8, 0.8, 0.8]
    assert retrieval_var["recall@5_mean"]["values"] == [0.9, 0.9, 0.9]
    assert retrieval_var["mrr_mean"]["stdev"] == pytest.approx(0.0)
    assert retrieval_var["recall@5_mean"]["stdev"] == pytest.approx(0.0)
