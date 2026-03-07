import json
import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_agentic_orchestrate.py")


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_pair_phase_reports_computes_reductions(tmp_path: Path):
    mod = _load_mod()
    pair_phase_reports = mod["pair_phase_reports"]

    baseline = tmp_path / "baseline.json"
    augmented = tmp_path / "augmented.json"
    out = tmp_path / "paired.json"
    _write_json(
        baseline,
        {
            "results": {
                "solve_rate": 0.4,
                "total_input_tokens": 1000,
                "per_task": [
                    {"task_id": "T1", "solve_rate": 0, "turn_count": 8, "wall_clock_s": 20.0, "category": "debug"},
                    {"task_id": "T2", "solve_rate": 1, "turn_count": 10, "wall_clock_s": 24.0, "category": "debug"},
                ],
            }
        },
    )
    _write_json(
        augmented,
        {
            "results": {
                "solve_rate": 0.6,
                "total_input_tokens": 700,
                "per_task": [
                    {"task_id": "T1", "solve_rate": 1, "turn_count": 5, "wall_clock_s": 15.0, "category": "debug"},
                    {"task_id": "T2", "solve_rate": 1, "turn_count": 6, "wall_clock_s": 18.0, "category": "debug"},
                ],
            }
        },
    )

    pair_phase_reports(phase="D", baseline_path=baseline, augmented_path=augmented, out_path=out)

    doc = json.loads(out.read_text(encoding="utf-8"))
    results = doc["results"]
    assert results["solve_rate_delta"] == 0.19999999999999996
    assert round(results["token_reduction_pct"], 2) == 30.0
    assert round(results["turn_reduction_pct"], 2) == 38.89
    assert len(results["per_task"]) == 2


def test_run_step_executes_command_without_outer_sandbox_wrapper(tmp_path: Path):
    mod = _load_mod()
    Step = mod["Step"]
    run_step = mod["_run_step"]

    step = Step(name="echo", command=["python", "-c", "print('ok')"])
    result = run_step(step, cwd=tmp_path)

    assert result["command"] == ["python", "-c", "print('ok')"]
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "ok"


def test_main_blocks_when_latest_preflight_failed(monkeypatch, tmp_path: Path):
    mod = _load_mod()
    globals_dict = mod["main"].__globals__
    runs_root = tmp_path / "benchmark" / "runs"
    _write_json(
        runs_root / "20260306-preflight.json",
        {
            "results": {
                "preflight_status": "failed",
            }
        },
    )
    monkeypatch.setitem(globals_dict, "get_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_agentic_orchestrate.py",
            "--start-from-phase",
            "C",
            "--end-at-phase",
            "C",
            "--kimi-model",
            "kimi-code/kimi-for-coding",
            "--out",
            str(tmp_path / "summary.json"),
        ],
    )

    assert mod["main"]() == 2
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["stopped_at_phase"] == "C"
    assert "preflight status" in summary["reason"]


def test_main_stops_on_failed_step(monkeypatch, tmp_path: Path):
    mod = _load_mod()
    globals_dict = mod["main"].__globals__
    Step = mod["Step"]
    monkeypatch.setitem(globals_dict, "get_repo_root", lambda: tmp_path)
    monkeypatch.setitem(
        globals_dict,
        "_phase_steps",
        lambda phase, args, repo_root, runs_root, ts: [
            Step(name="phase_a_run", command=["echo", "ok"], report_path=tmp_path / "phase-a.json"),
            Step(name="phase_a_gate", command=["echo", "fail"], gate_path=tmp_path / "phase-a-gate.json"),
        ],
    )

    results = [
        {
            "name": "phase_a_run",
            "command": ["echo", "ok"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "report_path": str(tmp_path / "phase-a.json"),
            "gate_path": None,
        },
        {
            "name": "phase_a_gate",
            "command": ["echo", "fail"],
            "returncode": 2,
            "stdout": "",
            "stderr": "gate failed",
            "report_path": None,
            "gate_path": str(tmp_path / "phase-a-gate.json"),
        },
    ]

    def fake_run_step(step, *, cwd):
        del step, cwd
        return results.pop(0)

    monkeypatch.setitem(globals_dict, "_run_step", fake_run_step)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_agentic_orchestrate.py",
            "--start-from-phase",
            "A",
            "--end-at-phase",
            "A",
            "--kimi-model",
            "kimi-code/kimi-for-coding",
            "--phase-a-prompts",
            str(tmp_path / "prompts.jsonl"),
            "--out",
            str(tmp_path / "summary.json"),
        ],
    )

    assert mod["main"]() == 2
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["stopped_at_phase"] == "A"
    assert summary["gate_diagnostic_path"] == str(tmp_path / "phase-a-gate.json")
