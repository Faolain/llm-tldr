#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_agentic_common import (
    DEFAULT_DOCKER_BINARY,
    DEFAULT_SANDBOX_IMAGE,
    load_json_obj,
)
from bench_util import bench_runs_root, get_repo_root, now_utc_compact, write_report

SCHEMA_VERSION = 1
PHASE_ORDER = ["A", "B", "C", "D", "E", "F"]


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    report_path: Path | None = None
    gate_path: Path | None = None


def _phase_index(phase: str) -> int:
    if phase not in PHASE_ORDER:
        raise ValueError(f"Unsupported phase: {phase}")
    return PHASE_ORDER.index(phase)


def _phase_range(start: str, end: str) -> list[str]:
    start_idx = _phase_index(start)
    end_idx = _phase_index(end)
    if end_idx < start_idx:
        raise ValueError(f"end phase {end!r} cannot come before start phase {start!r}")
    return PHASE_ORDER[start_idx : end_idx + 1]


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json_obj(path)
    except Exception:
        return None


def _find_latest_preflight_status(runs_root: Path) -> tuple[str | None, Path | None]:
    candidates = sorted(runs_root.glob("*preflight*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        doc = _load_json_if_exists(path)
        if not isinstance(doc, dict):
            continue
        results = doc.get("results")
        if isinstance(results, dict) and isinstance(results.get("preflight_status"), str):
            return str(results["preflight_status"]), path
        if isinstance(doc.get("preflight_status"), str):
            return str(doc["preflight_status"]), path
    return None, None


def _require_preflight_pass(runs_root: Path, target_phase: str) -> tuple[bool, str, Path | None]:
    status, path = _find_latest_preflight_status(runs_root)
    if status is None:
        return False, f"Phase {target_phase} requires a prior preflight report, but none was found.", path
    if status != "passed":
        return False, f"Phase {target_phase} blocked because latest preflight status is {status!r}.", path
    return True, "ok", path


def _report_results(doc: dict[str, Any]) -> dict[str, Any]:
    results = doc.get("results")
    return results if isinstance(results, dict) else {}


def _per_task_index(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = _report_results(doc)
    items = results.get("per_task")
    if not isinstance(items, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("task_id", "id", "instance_id"):
            value = item.get(key)
            if isinstance(value, str) and value:
                out[value] = item
                break
    return out


def _compute_rate(results: dict[str, Any], key: str) -> float | None:
    value = _safe_float(results.get(key))
    if value is not None:
        return value
    per_task = results.get("per_task")
    if not isinstance(per_task, list) or not per_task:
        return None
    values = []
    for row in per_task:
        if not isinstance(row, dict):
            continue
        row_value = _safe_float(row.get(key))
        if row_value is not None:
            values.append(row_value)
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _per_task_metric(results: dict[str, Any], key: str) -> float | None:
    value = _safe_float(results.get(key))
    if value is not None:
        return value
    per_task = results.get("per_task")
    if not isinstance(per_task, list) or not per_task:
        return None
    values = []
    for row in per_task:
        if not isinstance(row, dict):
            continue
        row_value = _safe_float(row.get(key))
        if row_value is not None:
            values.append(row_value)
    return _median(values)


def _pct_reduction(baseline: float | None, augmented: float | None) -> float | None:
    if baseline is None or augmented is None or baseline == 0:
        return None
    return ((baseline - augmented) / baseline) * 100.0


def pair_phase_reports(
    *,
    phase: str,
    baseline_path: Path,
    augmented_path: Path,
    out_path: Path,
) -> Path:
    baseline = load_json_obj(baseline_path)
    augmented = load_json_obj(augmented_path)
    baseline_results = _report_results(baseline)
    augmented_results = _report_results(augmented)

    baseline_tasks = _per_task_index(baseline)
    augmented_tasks = _per_task_index(augmented)
    paired_ids = sorted(set(baseline_tasks) | set(augmented_tasks))
    paired_tasks: list[dict[str, Any]] = []
    for task_id in paired_ids:
        paired_tasks.append(
            {
                "task_id": task_id,
                "baseline": baseline_tasks.get(task_id),
                "augmented": augmented_tasks.get(task_id),
                "category": (
                    (baseline_tasks.get(task_id) or {}).get("category")
                    or (augmented_tasks.get(task_id) or {}).get("category")
                ),
                "workflow_class": (
                    (baseline_tasks.get(task_id) or {}).get("workflow_class")
                    or (augmented_tasks.get(task_id) or {}).get("workflow_class")
                ),
            }
        )

    baseline_solve_rate = _compute_rate(baseline_results, "solve_rate")
    augmented_solve_rate = _compute_rate(augmented_results, "solve_rate")
    baseline_turn_median = _per_task_metric(baseline_results, "turn_count")
    augmented_turn_median = _per_task_metric(augmented_results, "turn_count")
    baseline_time_median = _per_task_metric(baseline_results, "wall_clock_s")
    augmented_time_median = _per_task_metric(augmented_results, "wall_clock_s")
    baseline_tokens = _safe_float(baseline_results.get("total_input_tokens"))
    augmented_tokens = _safe_float(augmented_results.get("total_input_tokens"))

    doc = {
        "schema_version": SCHEMA_VERSION,
        "phase": f"phase_{phase.lower()}_paired",
        "generated_at_utc": _timestamp_utc(),
        "paired_reports": {
            "baseline": str(baseline_path),
            "augmented": str(augmented_path),
        },
        "judge_config_hash": baseline_results.get("judge_config_hash") or augmented_results.get("judge_config_hash"),
        "results": {
            "baseline": baseline_results,
            "augmented": augmented_results,
            "paired_task_count": len(paired_tasks),
            "solve_rate_baseline": baseline_solve_rate,
            "solve_rate_augmented": augmented_solve_rate,
            "solve_rate_delta": (
                augmented_solve_rate - baseline_solve_rate
                if baseline_solve_rate is not None and augmented_solve_rate is not None
                else None
            ),
            "token_reduction_pct": _pct_reduction(baseline_tokens, augmented_tokens),
            "turn_reduction_pct": _pct_reduction(baseline_turn_median, augmented_turn_median),
            "time_reduction_pct": _pct_reduction(baseline_time_median, augmented_time_median),
            "per_task": paired_tasks,
        },
    }
    write_report(out_path, doc)
    return out_path


def _post_webhook(url: str, payload: dict[str, Any]) -> None:
    req = urllib.request.Request(
        str(url),
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
        resp.read()


def _run_step(step: Step, *, cwd: Path) -> dict[str, Any]:
    command = [str(part) for part in step.command]
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "name": step.name,
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "report_path": str(step.report_path) if step.report_path is not None else None,
        "gate_path": str(step.gate_path) if step.gate_path is not None else None,
    }


def _sandbox_args(args: argparse.Namespace) -> list[str]:
    return [
        "--sandbox-image",
        str(args.sandbox_image),
        "--docker-binary",
        str(args.docker_binary),
    ]


def _phase_a_steps(args: argparse.Namespace, repo_root: Path, runs_root: Path, ts: str) -> list[Step]:
    if not args.phase_a_prompts:
        raise ValueError("--phase-a-prompts is required when Phase A is selected.")
    report_path = runs_root / f"{ts}-phase-a.json"
    gate_path = runs_root / f"{ts}-phase-a-gate.json"
    return [
        Step(
            name="phase_a_run",
            command=[
                "uv",
                "run",
                "python",
                "scripts/bench_llm_ab_run.py",
                "--prompts",
                str(Path(args.phase_a_prompts).resolve()),
                "--provider",
                "kimi_cli",
                "--model",
                str(args.kimi_model),
                "--mode",
                "structured",
                "--trials",
                str(args.trials),
                "--timeout-s",
                str(args.timeout_s),
                "--out",
                str(report_path),
                *_sandbox_args(args),
            ],
            report_path=report_path,
        ),
        Step(
            name="phase_a_gate",
            command=[
                "uv",
                "run",
                "python",
                "scripts/bench_phase_gate.py",
                "--report",
                str(report_path),
                "--gates",
                str((repo_root / "benchmarks" / "agentic" / "phase_a_gates.json").resolve()),
                "--out",
                str(gate_path),
            ],
            report_path=report_path,
            gate_path=gate_path,
        ),
    ]


def _phase_b_steps(args: argparse.Namespace, repo_root: Path, runs_root: Path, ts: str) -> list[Step]:
    report_path = runs_root / f"{ts}-preflight.json"
    gate_path = runs_root / f"{ts}-preflight-gate.json"
    return [
        Step(
            name="phase_b_preflight_run",
            command=[
                "uv",
                "run",
                "python",
                "scripts/bench_tool_choice.py",
                "--tasks",
                str((repo_root / "benchmarks" / "agentic" / "preflight_tasks.json").resolve()),
                "--provider",
                "kimi_cli",
                "--model",
                str(args.kimi_model),
                "--instruction-source",
                str(Path(args.instruction_source).resolve()),
                "--max-turns",
                str(args.max_turns),
                "--timeout-s",
                str(args.timeout_s),
                "--trials",
                str(args.trials),
                "--out",
                str(report_path),
                *_sandbox_args(args),
            ],
            report_path=report_path,
        ),
        Step(
            name="phase_b_preflight_gate",
            command=[
                "uv",
                "run",
                "python",
                "scripts/bench_preflight_validate.py",
                "--report",
                str(report_path),
                "--gates",
                str((repo_root / "benchmarks" / "agentic" / "preflight_gates.json").resolve()),
                "--out",
                str(gate_path),
            ],
            report_path=report_path,
            gate_path=gate_path,
        ),
    ]


def _paired_runner_steps(
    *,
    phase: str,
    script_name: str,
    tasks_path: Path,
    gate_path: Path,
    args: argparse.Namespace,
    runs_root: Path,
    ts: str,
    extra_args: list[str] | None = None,
) -> list[Step]:
    baseline_path = runs_root / f"{ts}-phase-{phase.lower()}-baseline.json"
    augmented_path = runs_root / f"{ts}-phase-{phase.lower()}-augmented.json"
    combined_path = runs_root / f"{ts}-phase-{phase.lower()}.json"
    extra = list(extra_args or [])
    steps = [
        Step(
            name=f"phase_{phase.lower()}_baseline_run",
            command=[
                "uv",
                "run",
                "python",
                f"scripts/{script_name}",
                "--tasks",
                str(tasks_path),
                "--provider",
                "kimi_cli",
                "--model",
                str(args.kimi_model),
                "--arm",
                "baseline",
                "--max-turns",
                str(args.max_turns),
                "--timeout-s",
                str(args.timeout_s),
                "--out",
                str(baseline_path),
                *extra,
                *_sandbox_args(args),
            ],
            report_path=baseline_path,
        ),
        Step(
            name=f"phase_{phase.lower()}_augmented_run",
            command=[
                "uv",
                "run",
                "python",
                f"scripts/{script_name}",
                "--tasks",
                str(tasks_path),
                "--provider",
                "kimi_cli",
                "--model",
                str(args.kimi_model),
                "--arm",
                "augmented",
                "--max-turns",
                str(args.max_turns),
                "--timeout-s",
                str(args.timeout_s),
                "--out",
                str(augmented_path),
                *extra,
                *_sandbox_args(args),
            ],
            report_path=augmented_path,
        ),
        Step(
            name=f"phase_{phase.lower()}_pair_reports",
            command=[
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "from bench_agentic_orchestrate import pair_phase_reports; "
                    f"pair_phase_reports(phase={phase!r}, baseline_path=Path({str(baseline_path)!r}), "
                    f"augmented_path=Path({str(augmented_path)!r}), out_path=Path({str(combined_path)!r}))"
                ),
            ],
            report_path=combined_path,
        ),
        Step(
            name=f"phase_{phase.lower()}_gate",
            command=[
                "uv",
                "run",
                "python",
                "scripts/bench_phase_gate.py",
                "--report",
                str(combined_path),
                "--gates",
                str(gate_path),
                "--out",
                str(runs_root / f"{ts}-phase-{phase.lower()}-gate.json"),
            ],
            report_path=combined_path,
            gate_path=runs_root / f"{ts}-phase-{phase.lower()}-gate.json",
        ),
    ]
    return steps


def _phase_steps(phase: str, args: argparse.Namespace, repo_root: Path, runs_root: Path, ts: str) -> list[Step]:
    if phase == "A":
        return _phase_a_steps(args, repo_root, runs_root, ts)
    if phase == "B":
        return _phase_b_steps(args, repo_root, runs_root, ts)
    if phase == "C":
        return _paired_runner_steps(
            phase="C",
            script_name="bench_tool_choice.py",
            tasks_path=(repo_root / "benchmarks" / "agentic" / "tool_choice_tasks.json").resolve(),
            gate_path=(repo_root / "benchmarks" / "agentic" / "phase_c_gates.json").resolve(),
            args=args,
            runs_root=runs_root,
            ts=ts,
            extra_args=["--instruction-source", str(Path(args.instruction_source).resolve()), "--trials", str(args.trials)],
        )
    if phase == "D":
        return _paired_runner_steps(
            phase="D",
            script_name="bench_agent_tasks.py",
            tasks_path=(repo_root / "benchmarks" / "agentic" / "patch_tasks.json").resolve(),
            gate_path=(repo_root / "benchmarks" / "agentic" / "phase_d_gates.json").resolve(),
            args=args,
            runs_root=runs_root,
            ts=ts,
        )
    if phase == "E":
        return _paired_runner_steps(
            phase="E",
            script_name="bench_agent_tasks.py",
            tasks_path=Path(args.swebench_subset).resolve(),
            gate_path=(repo_root / "benchmarks" / "agentic" / "phase_e_gates.json").resolve(),
            args=args,
            runs_root=runs_root,
            ts=ts,
        )
    if phase == "F":
        return _paired_runner_steps(
            phase="F",
            script_name="bench_agent_tasks.py",
            tasks_path=Path(args.swebench_subset).resolve(),
            gate_path=(repo_root / "benchmarks" / "agentic" / "phase_f_gates.json").resolve(),
            args=args,
            runs_root=runs_root,
            ts=ts,
        )
    raise ValueError(f"Unsupported phase: {phase}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Chain the agentic Kimi-vs-native benchmark phases with gate checks.")
    ap.add_argument("--start-from-phase", choices=PHASE_ORDER, default="A")
    ap.add_argument("--end-at-phase", choices=PHASE_ORDER, default="F")
    ap.add_argument("--kimi-model", required=True, help="Kimi model identifier to use for answer-model runs.")
    ap.add_argument("--phase-a-prompts", default=None, help="Prompt packet JSONL used for Phase A llm_ab runs.")
    ap.add_argument(
        "--instruction-source",
        default="AGENTS.md",
        help="Canonical instruction document consumed by tool-choice and agent-task phases.",
    )
    ap.add_argument(
        "--swebench-subset",
        default="benchmarks/agentic/swebench_subset.json",
        help="Curated SWE-bench subset JSON used for Phase E/F.",
    )
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--sandbox-image", default=DEFAULT_SANDBOX_IMAGE)
    ap.add_argument("--docker-binary", default=DEFAULT_DOCKER_BINARY)
    ap.add_argument("--notify-webhook", default=None, help="Optional webhook URL for completion/failure payloads.")
    ap.add_argument("--out", default=None, help="Write orchestrator summary JSON here.")
    args = ap.parse_args()

    repo_root = get_repo_root()
    runs_root = bench_runs_root(repo_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = now_utc_compact()
    out_path = Path(args.out).resolve() if args.out else (runs_root / f"{ts}-agentic-orchestrate.json").resolve()

    phases = _phase_range(str(args.start_from_phase), str(args.end_at_phase))
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _timestamp_utc(),
        "start_from_phase": str(args.start_from_phase),
        "end_at_phase": str(args.end_at_phase),
        "kimi_model": str(args.kimi_model),
        "sandbox_image": str(args.sandbox_image),
        "docker_binary": str(args.docker_binary),
        "status": "running",
        "stopped_at_phase": None,
        "reason": None,
        "gate_diagnostic_path": None,
        "completed_phases": [],
        "phase_steps": [],
        "latest_preflight_path": None,
    }

    rc = 0
    try:
        for phase in phases:
            if phase in {"C", "D", "E", "F"}:
                ok, reason, latest_preflight = _require_preflight_pass(runs_root, phase)
                summary["latest_preflight_path"] = str(latest_preflight) if latest_preflight is not None else None
                if not ok:
                    summary["status"] = "blocked"
                    summary["stopped_at_phase"] = phase
                    summary["reason"] = reason
                    rc = 2
                    break
            if phase in {"E", "F"} and not Path(args.swebench_subset).resolve().exists():
                summary["status"] = "blocked"
                summary["stopped_at_phase"] = phase
                summary["reason"] = f"Missing SWE-bench subset file: {Path(args.swebench_subset).resolve()}"
                rc = 2
                break

            phase_ok = True
            for step in _phase_steps(phase, args, repo_root, runs_root, ts):
                step_result = _run_step(step, cwd=repo_root)
                summary["phase_steps"].append(step_result)
                if step_result["returncode"] != 0:
                    summary["status"] = "failed"
                    summary["stopped_at_phase"] = phase
                    summary["reason"] = f"{step.name} exited with code {step_result['returncode']}"
                    summary["gate_diagnostic_path"] = step_result.get("gate_path")
                    phase_ok = False
                    rc = 2
                    break
                if step.name.endswith("_gate") and step.gate_path is not None:
                    summary["gate_diagnostic_path"] = str(step.gate_path)
            if not phase_ok:
                break
            summary["completed_phases"].append(phase)

        if rc == 0:
            summary["status"] = "passed"
            summary["reason"] = "Completed requested phase range successfully."
    except Exception as exc:
        summary["status"] = "failed"
        summary["reason"] = f"{type(exc).__name__}: {exc}"
        summary["stopped_at_phase"] = summary.get("stopped_at_phase") or (phases[0] if phases else None)
        rc = 2

    write_report(out_path, summary)
    print(out_path)

    if args.notify_webhook:
        try:
            _post_webhook(str(args.notify_webhook), summary)
        except Exception:
            if rc == 0:
                rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
