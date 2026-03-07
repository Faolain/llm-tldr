import json
import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_preflight_validate.py")


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_evaluate_report_passes_for_compliant_transcripts(tmp_path: Path):
    mod = _load_mod()

    transcript_a = tmp_path / "task-a.jsonl"
    transcript_b = tmp_path / "task-b.jsonl"
    _write_jsonl(
        transcript_a,
        [
            {"turn": 1, "tool_name": "rg"},
            {"turn": 2, "tool_name": "read_file"},
        ],
    )
    _write_jsonl(
        transcript_b,
        [
            {"turn": 1, "tool_calls": [{"name": "tldrf_semantic_search"}]},
            {"turn": 2, "tool_calls": [{"name": "tldrf_context"}]},
        ],
    )

    report = {
        "results": {
            "per_task": [
                {
                    "task_id": "PF01",
                    "workflow_class": "exact_lookup",
                    "expected_first_tool": "rg",
                    "expected_tool_set": ["rg", "read_file"],
                    "forbidden_first_tool": ["tldrf_semantic_search"],
                    "max_allowed_dead_end_turns": 1,
                    "transcript_path": str(transcript_a),
                },
                {
                    "task_id": "PF02",
                    "workflow_class": "concept_lookup",
                    "expected_first_tool": "tldrf_semantic_search",
                    "expected_tool_set": ["tldrf_semantic_search", "tldrf_context"],
                    "forbidden_first_tool": ["rg"],
                    "max_allowed_dead_end_turns": 1,
                    "transcript_path": str(transcript_b),
                },
            ]
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
    assert diagnostic["recommendation"] == "pass"
    assert diagnostic["metrics"]["correct_first_tool_rate"] == 1.0
    assert diagnostic["metrics"]["workflow_compliance_rate"] == 1.0
    assert diagnostic["metrics"]["tldrf_usage_on_required_rate"] == 1.0


def test_evaluate_report_detects_systematic_failure(tmp_path: Path):
    mod = _load_mod()

    tasks = []
    for idx in range(3):
        transcript = tmp_path / f"bad-{idx}.jsonl"
        _write_jsonl(
            transcript,
            [
                {"turn": 1, "tool_name": "rg"},
                {"turn": 2, "tool_name": "read_file"},
            ],
        )
        tasks.append(
            {
                "task_id": f"PF{idx}",
                "workflow_class": "concept_lookup",
                "expected_first_tool": "tldrf_semantic_search",
                "expected_tool_set": ["tldrf_semantic_search", "tldrf_context"],
                "forbidden_first_tool": ["rg"],
                "max_allowed_dead_end_turns": 0,
                "transcript_path": str(transcript),
            }
        )

    diagnostic = mod["evaluate_report"](
        {"results": {"per_task": tasks}},
        {
            "correct_first_tool_min": 0.8,
            "workflow_compliance_min": 0.8,
            "tldrf_usage_on_required_min": 0.9,
            "rg_first_on_exact_min": 0.9,
            "median_dead_end_turns_max": 1,
            "systematic_failure_detected_must_be": False,
        },
    )

    assert diagnostic["pass"] is False
    assert diagnostic["systematic_failure_detected"] is True
    assert diagnostic["failure_patterns"][0]["count"] == 3
    assert diagnostic["recommendation"] == "tune_and_rerun"


def test_main_returns_exit_2_on_gate_failure(monkeypatch, tmp_path: Path):
    mod = _load_mod()
    globals_dict = mod["main"].__globals__

    transcript = tmp_path / "bad.jsonl"
    _write_jsonl(transcript, [{"turn": 1, "tool_name": "rg"}])
    report_path = tmp_path / "report.json"
    gates_path = tmp_path / "gates.json"
    out_path = tmp_path / "diagnostic.json"
    _write_json(
        report_path,
        {
            "results": {
                "per_task": [
                    {
                        "task_id": "PF01",
                        "workflow_class": "concept_lookup",
                        "expected_first_tool": "tldrf_semantic_search",
                        "expected_tool_set": ["tldrf_semantic_search"],
                        "forbidden_first_tool": ["rg"],
                        "max_allowed_dead_end_turns": 0,
                        "transcript_path": str(transcript),
                    }
                ]
            }
        },
    )
    _write_json(
        gates_path,
        {
            "correct_first_tool_min": 1.0,
            "workflow_compliance_min": 1.0,
            "tldrf_usage_on_required_min": 1.0,
            "systematic_failure_detected_must_be": False,
        },
    )

    monkeypatch.setitem(globals_dict, "get_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_preflight_validate.py",
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
    assert written["pass"] is False
    assert written["preflight_status"] == "failed"
