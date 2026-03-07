import json
import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_tool_choice.py")


def test_resolve_workspace_path_rejects_escape(tmp_path: Path):
    mod = _load_mod()
    fn = mod["resolve_workspace_path"]
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert fn(workspace, "nested/file.py") == (workspace / "nested" / "file.py").resolve()

    try:
        fn(workspace, "../escape.py")
    except ValueError as exc:
        assert "escapes workspace" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected path escape rejection")


def test_evaluate_tool_choice_metrics():
    mod = _load_mod()
    fn = mod["evaluate_tool_choice_metrics"]
    metrics = fn(
        [
            {
                "first_tool": "rg",
                "expected_first_tool": "rg",
                "workflow_compliant": True,
                "tool_calls": ["rg"],
                "expected_tool_set": ["rg"],
                "workflow_class": "exact_lookup",
                "dead_end_turns": 0,
            },
            {
                "first_tool": "grep",
                "expected_first_tool": "tldrf_semantic_search",
                "workflow_compliant": False,
                "tool_calls": ["grep", "tldrf_semantic_search"],
                "expected_tool_set": ["tldrf_semantic_search"],
                "workflow_class": "concept_lookup",
                "dead_end_turns": 1,
            },
        ]
    )
    assert metrics["correct_first_tool_rate"] == 0.5
    assert metrics["workflow_compliance_rate"] == 0.5
    assert metrics["tldrf_usage_on_required_rate"] == 1.0
    assert metrics["rg_first_on_exact_rate"] == 1.0
    assert metrics["median_turns_before_first_appropriate_tool_use"] == 1.5


def test_main_writes_transcript_and_report(monkeypatch, tmp_path: Path):
    mod = _load_mod()
    globals_dict = mod["main"].__globals__

    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "PF01",
                        "workflow_class": "concept_lookup",
                        "question": "Where is the ORM query compiler entry point?",
                        "expected_first_tool": "tldrf_semantic_search",
                        "expected_tool_set": ["tldrf_semantic_search", "tldrf_context"],
                        "forbidden_first_tool": ["rg"],
                        "max_allowed_dead_end_turns": 1,
                        "repo_path": str(tmp_path / "workspace"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    instruction_path = tmp_path / "AGENTS.md"
    instruction_path.write_text("Use rg first for exact lookup.\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    out_path = tmp_path / "report.json"

    decisions = iter(
        [
            ('{"kind":"tool","tool":"tldrf_semantic_search","args":{"query":"orm compiler"}}', {"input_tokens": 3, "output_tokens": 2}),
            ('{"kind":"final","answer":"done"}', {"input_tokens": 2, "output_tokens": 1}),
        ]
    )

    def fake_call_provider(**kwargs):
        del kwargs
        return next(decisions)

    tool_invocation = mod["ToolInvocation"](
        name="tldrf_semantic_search",
        args={"query": "orm compiler"},
        observation="django/db/models/sql/query.py",
        ok=True,
    )

    monkeypatch.setitem(globals_dict, "call_provider", fake_call_provider)
    monkeypatch.setitem(globals_dict, "execute_tool", lambda **kwargs: tool_invocation)
    monkeypatch.setitem(globals_dict, "make_provider_runtime", lambda **kwargs: object())
    monkeypatch.setitem(globals_dict, "get_repo_root", lambda: tmp_path)
    monkeypatch.setitem(globals_dict, "gather_meta", lambda **kwargs: {"test_meta": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_tool_choice.py",
            "--tasks",
            str(tasks_path),
            "--provider",
            "kimi_cli",
            "--model",
            "kimi-code/kimi-for-coding",
            "--instruction-source",
            str(instruction_path),
            "--out",
            str(out_path),
        ],
    )

    assert mod["main"]() == 0

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["arm"] == "augmented"
    assert report["instruction_surface"]["path"] == str(instruction_path)
    assert "tldrf_semantic_search" in report["tools_available"]
    assert report["results"]["task_count"] == 1
    per_task = report["results"]["per_task"][0]
    transcript_path = Path(per_task["transcript_path"])
    transcript_rows = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines()]
    assert transcript_rows[0]["kind"] == "assistant_decision"
    assert transcript_rows[1]["kind"] == "tool_result"

