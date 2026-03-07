import json
import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_agent_tasks.py")


def test_load_completed_ids(tmp_path: Path):
    mod = _load_mod()
    fn = mod["_load_completed_ids"]
    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        json.dumps({"task_id": "T01"}) + "\n" + json.dumps({"task_id": "T02"}) + "\n",
        encoding="utf-8",
    )
    assert fn(answers) == {"T01", "T02"}


def test_changed_file_precision_recall():
    mod = _load_mod()
    fn = mod["_set_metrics"]
    precision, recall = fn(["a.py", "b.py"], ["a.py", "c.py"])
    assert precision == 0.5
    assert recall == 0.5


def test_main_resume_and_circuit_breaker(monkeypatch, tmp_path: Path):
    mod = _load_mod()
    globals_dict = mod["main"].__globals__

    source_workspace = tmp_path / "source"
    source_workspace.mkdir()
    tasks_path = tmp_path / "patch_tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "P01", "repo_path": str(source_workspace), "hidden_test_command": "true"},
                    {"id": "P02", "repo_path": str(source_workspace), "hidden_test_command": "true"},
                    {"id": "P03", "repo_path": str(source_workspace), "hidden_test_command": "true"},
                ]
            }
        ),
        encoding="utf-8",
    )
    instruction_path = tmp_path / "AGENTS.md"
    instruction_path.write_text("Patch carefully.\n", encoding="utf-8")
    out_path = tmp_path / "patch-report.json"
    resume_path = tmp_path / "resume.jsonl"
    resume_path.write_text(json.dumps({"task_id": "P01"}) + "\n", encoding="utf-8")

    calls: list[str] = []

    def fake_run_patch_task(**kwargs):
        task = kwargs["task"]
        calls.append(task["id"])
        if task["id"] == "P02":
            return {
                "task_id": "P02",
                "category": None,
                "transcript_path": str(tmp_path / "P02.jsonl"),
                "solve_rate": 0,
                "turn_count": 1,
                "tool_call_count": 0,
                "tool_calls": [],
                "changed_files": [],
                "changed_file_precision": 0.0,
                "changed_file_recall": 0.0,
                "wall_clock_s": 0.1,
                "first_pass_time_s": None,
                "workspace_root": str(tmp_path / "workspace-P02"),
                "error": "boom",
            }
        return {
            "task_id": task["id"],
            "category": None,
            "transcript_path": str(tmp_path / f"{task['id']}.jsonl"),
            "solve_rate": 1,
            "turn_count": 1,
            "tool_call_count": 1,
            "tool_calls": ["run_tests"],
            "changed_files": ["x.py"],
            "changed_file_precision": 1.0,
            "changed_file_recall": 1.0,
            "wall_clock_s": 0.1,
            "first_pass_time_s": 0.1,
            "workspace_root": str(tmp_path / f"workspace-{task['id']}"),
            "error": None,
        }

    monkeypatch.setitem(globals_dict, "run_patch_task", fake_run_patch_task)
    monkeypatch.setitem(globals_dict, "_copy_workspace", lambda src, dst: dst.mkdir(parents=True, exist_ok=True))
    monkeypatch.setitem(globals_dict, "make_provider_runtime", lambda **kwargs: object())
    monkeypatch.setitem(globals_dict, "get_repo_root", lambda: tmp_path)
    monkeypatch.setitem(globals_dict, "gather_meta", lambda **kwargs: {"test_meta": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_agent_tasks.py",
            "--tasks",
            str(tasks_path),
            "--provider",
            "kimi_cli",
            "--model",
            "kimi-code/kimi-for-coding",
            "--arm",
            "augmented",
            "--instruction-source",
            str(instruction_path),
            "--out",
            str(out_path),
            "--resume",
            str(resume_path),
            "--max-consecutive-errors",
            "1",
            "--max-error-rate-abort",
            "0.9",
        ],
    )

    assert mod["main"]() == 0
    assert calls == ["P02"]
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["results"]["task_count"] == 1
    assert report["arm"] == "augmented"
    assert "replace_text" in report["tools_available"]


def test_workspace_path_guard_shared_with_patch_runner(tmp_path: Path):
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import bench_tool_choice

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        bench_tool_choice.resolve_workspace_path(workspace, "../outside.txt")
    except ValueError as exc:
        assert "escapes workspace" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected workspace escape rejection")
