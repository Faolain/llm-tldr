import runpy
import sys
from pathlib import Path

import pytest


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_h2h_predict.py")


def test_render_command_template_raises_on_missing_placeholder():
    mod = _load_mod()
    render = mod["_render_command_template"]

    with pytest.raises(ValueError, match="missing template placeholders"):
        render("contextplus search --repo {repo_root} --query {query}", {"repo_root": "/tmp/repo"})


def test_timeout_maps_to_timeout_status_not_error():
    mod = _load_mod()
    run_once = mod["_run_command_once"]

    result = run_once(
        argv=[sys.executable, "-c", "import time; time.sleep(0.2)"],
        cwd=Path.cwd(),
        timeout_s=0.05,
    )

    assert result.status == "timeout"
    assert result.status != "error"


def test_raw_log_path_is_tool_trial_task_layout(tmp_path: Path):
    mod = _load_mod()
    raw_log_path = mod["_raw_log_path"]

    path = raw_log_path(
        tmp_path,
        tool_id="llm-tldr",
        trial=2,
        task_id="retrieval:R01",
    )

    expected = tmp_path / "benchmark" / "runs" / "raw_logs" / "llm-tldr" / "2" / "retrieval:R01.log"
    assert path == expected


def test_segment_filters_limit_prediction_identity_matrix():
    mod = _load_mod()
    normalize_filters = mod["_normalize_segment_filters"]
    apply_filters = mod["_apply_segment_filters"]

    tasks_sorted = [
        {"task_id": "impact:I01", "category": "impact"},
        {"task_id": "retrieval:R01", "category": "retrieval"},
        {"task_id": "slice:S01", "category": "slice"},
    ]
    filters = normalize_filters(
        categories_raw=["retrieval"],
        task_ids_raw=["impact:I01,retrieval:R01"],
        trials_raw=["2,1"],
        budget_tokens_raw=["2000"],
    )

    selected_tasks, selected_budgets, selected_trials = apply_filters(
        tasks_sorted=tasks_sorted,
        budgets=[500, 2000],
        trials=3,
        segment_filters=filters,
    )

    identity_keys = {
        (str(task["task_id"]), int(budget), int(trial))
        for task in selected_tasks
        for budget in selected_budgets
        for trial in selected_trials
    }

    assert identity_keys == {
        ("retrieval:R01", 2000, 1),
        ("retrieval:R01", 2000, 2),
    }


def test_segment_filter_audit_doc_records_requested_filters():
    mod = _load_mod()
    normalize_filters = mod["_normalize_segment_filters"]
    apply_filters = mod["_apply_segment_filters"]
    build_audit_doc = mod["_segment_filter_audit_doc"]

    tasks_sorted = [
        {"task_id": "impact:I01", "category": "impact"},
        {"task_id": "retrieval:R01", "category": "retrieval"},
    ]
    filters = normalize_filters(
        categories_raw=["impact"],
        task_ids_raw=["impact:I01"],
        trials_raw=["3"],
        budget_tokens_raw=["2000"],
    )
    selected_tasks, selected_budgets, selected_trials = apply_filters(
        tasks_sorted=tasks_sorted,
        budgets=[500, 2000],
        trials=3,
        segment_filters=filters,
    )
    audit_doc = build_audit_doc(
        segment_filters=filters,
        selected_tasks=selected_tasks,
        selected_budgets=selected_budgets,
        selected_trials=selected_trials,
    )

    assert audit_doc["categories"] == ["impact"]
    assert audit_doc["task_ids"] == ["impact:I01"]
    assert audit_doc["trials"] == [3]
    assert audit_doc["budget_tokens"] == [2000]
    assert audit_doc["selected_identity_count"] == 1
