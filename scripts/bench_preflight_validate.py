#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from bench_agentic_common import load_json_obj, load_jsonl, normalize_tool_name, sha256_file
from bench_util import bench_runs_root, get_repo_root, now_utc_compact, write_report

SCHEMA_VERSION = 1


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _coerce_float(value: Any) -> float | None:
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


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _add_gate(gates: list[dict[str, Any]], *, name: str, actual: Any, expected: Any, passed: bool) -> None:
    gates.append({"name": name, "actual": actual, "expected": expected, "pass": bool(passed)})


def _task_entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    results = report.get("results")
    if not isinstance(results, dict):
        return []
    for key in ("per_task", "tasks"):
        value = results.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _extract_tool_name(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    for key in ("normalized_tool", "tool_name", "tool", "name", "action", "command"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            normalized = normalize_tool_name(value)
            if normalized:
                return normalized
    return None


def _extract_tool_calls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        turn = _coerce_int(row.get("turn"))
        tool_calls = row.get("tool_calls")
        if isinstance(tool_calls, list):
            for call_index, call in enumerate(tool_calls):
                name = _extract_tool_name(call)
                if name is None:
                    continue
                calls.append(
                    {
                        "tool_name": name,
                        "turn": turn if turn is not None else row_index + 1,
                        "row_index": row_index,
                        "call_index": call_index,
                    }
                )
        else:
            name = _extract_tool_name(row)
            if name is None:
                continue
            calls.append(
                {
                    "tool_name": name,
                    "turn": turn if turn is not None else row_index + 1,
                    "row_index": row_index,
                    "call_index": 0,
                }
            )
    return calls


def _load_transcript_calls(path: Path) -> list[dict[str, Any]]:
    return _extract_tool_calls(load_jsonl(path))


def _first_matching_turn(calls: list[dict[str, Any]], expected_tools: set[str]) -> int | None:
    for idx, call in enumerate(calls, start=1):
        if call["tool_name"] in expected_tools:
            return idx
    return None


def _failure_signature(
    *,
    first_tool: str | None,
    expected_first_tool: str | None,
    expected_tools: set[str],
    skip_expected: bool,
    forbidden_first: bool,
    dead_end_turns: int,
    max_allowed_dead_end_turns: int,
) -> str | None:
    if forbidden_first and first_tool is not None:
        return f"forbidden_first:{first_tool}"
    if first_tool != expected_first_tool:
        return f"wrong_first:{first_tool or 'none'}->{expected_first_tool or 'none'}"
    if skip_expected:
        ordered = ",".join(sorted(expected_tools))
        return f"skip_expected:{ordered}"
    if dead_end_turns > max_allowed_dead_end_turns:
        return f"dead_end:{dead_end_turns}>{max_allowed_dead_end_turns}"
    return None


def _evaluate_task(task: dict[str, Any], transcript_path: Path) -> dict[str, Any]:
    task_id = str(task.get("task_id") or task.get("id") or transcript_path.stem)
    workflow_class = str(task.get("workflow_class") or "")
    expected_first_tool = normalize_tool_name(str(task.get("expected_first_tool") or ""))
    expected_tool_set = {
        normalize_tool_name(str(item))
        for item in (task.get("expected_tool_set") or [])
        if isinstance(item, str) and str(item).strip()
    }
    forbidden_first_tool = {
        normalize_tool_name(str(item))
        for item in (task.get("forbidden_first_tool") or [])
        if isinstance(item, str) and str(item).strip()
    }
    max_allowed_dead_end_turns = _coerce_int(task.get("max_allowed_dead_end_turns")) or 0
    calls = _load_transcript_calls(transcript_path)
    tool_names = [call["tool_name"] for call in calls]
    first_tool = tool_names[0] if tool_names else None
    first_appropriate_turn = _first_matching_turn(calls, expected_tool_set)
    dead_end_turns = first_appropriate_turn - 1 if first_appropriate_turn is not None else len(calls)
    used_expected_tools = sorted({name for name in tool_names if name in expected_tool_set})
    skip_expected = bool(expected_tool_set) and not bool(used_expected_tools)
    correct_first_tool = bool(first_tool == expected_first_tool)
    forbidden_first = bool(first_tool in forbidden_first_tool) if first_tool is not None else False
    workflow_compliant = (
        bool(expected_tool_set)
        and not forbidden_first
        and not skip_expected
        and dead_end_turns <= max_allowed_dead_end_turns
    )
    recovered_after_wrong_first = bool(
        expected_tool_set and not correct_first_tool and first_appropriate_turn is not None
    )
    unnecessary_tool_calls = sum(1 for name in tool_names if name not in expected_tool_set)
    expected_tldrf = any(name.startswith("tldrf_") for name in expected_tool_set)
    used_tldrf = any(name.startswith("tldrf_") for name in tool_names)
    signature = _failure_signature(
        first_tool=first_tool,
        expected_first_tool=expected_first_tool or None,
        expected_tools=expected_tool_set,
        skip_expected=skip_expected,
        forbidden_first=forbidden_first,
        dead_end_turns=dead_end_turns,
        max_allowed_dead_end_turns=max_allowed_dead_end_turns,
    )
    return {
        "task_id": task_id,
        "workflow_class": workflow_class,
        "transcript_path": str(transcript_path),
        "tool_call_count": len(calls),
        "tool_calls": tool_names,
        "first_tool": first_tool,
        "expected_first_tool": expected_first_tool or None,
        "correct_first_tool": correct_first_tool,
        "expected_tool_set": sorted(expected_tool_set),
        "used_expected_tools": used_expected_tools,
        "skip_expected_tool_set": skip_expected,
        "forbidden_first_tool": sorted(forbidden_first_tool),
        "forbidden_first_tool_used": forbidden_first,
        "workflow_compliant": workflow_compliant,
        "dead_end_turns": dead_end_turns,
        "max_allowed_dead_end_turns": max_allowed_dead_end_turns,
        "first_appropriate_tool_turn": first_appropriate_turn,
        "recovered_after_wrong_first_tool": recovered_after_wrong_first,
        "unnecessary_tool_calls": unnecessary_tool_calls,
        "tldrf_expected": expected_tldrf,
        "tldrf_used": used_tldrf,
        "failure_signature": signature,
    }


def _failure_patterns(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for row in task_rows:
        workflow_class = str(row.get("workflow_class") or "")
        signature = row.get("failure_signature")
        if not workflow_class or not isinstance(signature, str) or not signature:
            continue
        grouped.setdefault((workflow_class, signature), []).append(str(row.get("task_id") or ""))
    patterns: list[dict[str, Any]] = []
    for (workflow_class, signature), task_ids in sorted(grouped.items()):
        if len(task_ids) <= 2:
            continue
        patterns.append(
            {
                "workflow_class": workflow_class,
                "signature": signature,
                "count": len(task_ids),
                "task_ids": task_ids,
            }
        )
    return patterns


def _mean_bool(rows: list[dict[str, Any]], key: str, *, filter_key: str | None = None) -> float | None:
    values: list[float] = []
    for row in rows:
        if filter_key is not None and not bool(row.get(filter_key)):
            continue
        value = _coerce_bool(row.get(key))
        if value is not None:
            values.append(1.0 if value else 0.0)
    if not values:
        return None
    return sum(values) / len(values)


def evaluate_report(report: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    per_task = _task_entries(report)
    if not per_task:
        raise ValueError("report must include results.per_task or results.tasks")

    evaluated_tasks: list[dict[str, Any]] = []
    for task in per_task:
        transcript_value = task.get("transcript_path")
        if not isinstance(transcript_value, str) or not transcript_value.strip():
            raise ValueError(f"task is missing transcript_path: {task}")
        transcript_path = Path(transcript_value).resolve()
        if not transcript_path.exists():
            raise ValueError(f"transcript does not exist: {transcript_path}")
        evaluated_tasks.append(_evaluate_task(task, transcript_path))

    failure_patterns = _failure_patterns(evaluated_tasks)
    systematic_failure_detected = bool(failure_patterns)
    exact_rows = [row for row in evaluated_tasks if row.get("expected_first_tool") in {"rg", "grep"}]
    required_tldrf_rows = [row for row in evaluated_tasks if bool(row.get("tldrf_expected"))]
    wrong_first_rows = [row for row in evaluated_tasks if not bool(row.get("correct_first_tool"))]
    dead_end_rows = [row for row in evaluated_tasks if row.get("dead_end_turns") is not None]
    first_appropriate_turns = [
        int(turn)
        for row in evaluated_tasks
        for turn in [row.get("first_appropriate_tool_turn")]
        if isinstance(turn, int)
    ]
    metrics = {
        "tasks_evaluated": len(evaluated_tasks),
        "correct_first_tool_rate": _mean_bool(evaluated_tasks, "correct_first_tool"),
        "workflow_compliance_rate": _mean_bool(evaluated_tasks, "workflow_compliant"),
        "tldrf_usage_on_required_rate": _mean_bool(required_tldrf_rows, "tldrf_used"),
        "rg_first_on_exact_rate": _mean_bool(exact_rows, "correct_first_tool"),
        "tool_choice_accuracy": _mean_bool(evaluated_tasks, "correct_first_tool"),
        "unnecessary_tool_call_rate": (
            sum(int(row.get("unnecessary_tool_calls") or 0) for row in evaluated_tasks)
            / max(1, sum(int(row.get("tool_call_count") or 0) for row in evaluated_tasks))
        ),
        "dead_end_turn_rate": (
            sum(
                1
                for row in dead_end_rows
                if int(row.get("dead_end_turns") or 0) > int(row.get("max_allowed_dead_end_turns") or 0)
            )
            / len(dead_end_rows)
            if dead_end_rows
            else None
        ),
        "recovery_after_wrong_first_tool_rate": _mean_bool(
            wrong_first_rows, "recovered_after_wrong_first_tool"
        ),
        "median_turns_before_first_appropriate_tool_use": (
            statistics.median(first_appropriate_turns) if first_appropriate_turns else None
        ),
        "systematic_failure_detected": systematic_failure_detected,
    }

    gate_rows: list[dict[str, Any]] = []
    if "correct_first_tool_min" in gates:
        threshold = _coerce_float(gates.get("correct_first_tool_min"))
        actual = metrics["correct_first_tool_rate"]
        _add_gate(
            gate_rows,
            name="correct_first_tool_min",
            actual=actual,
            expected=threshold,
            passed=(actual is not None and threshold is not None and actual >= threshold),
        )
    if "workflow_compliance_min" in gates:
        threshold = _coerce_float(gates.get("workflow_compliance_min"))
        actual = metrics["workflow_compliance_rate"]
        _add_gate(
            gate_rows,
            name="workflow_compliance_min",
            actual=actual,
            expected=threshold,
            passed=(actual is not None and threshold is not None and actual >= threshold),
        )
    if "tldrf_usage_on_required_min" in gates:
        threshold = _coerce_float(gates.get("tldrf_usage_on_required_min"))
        actual = metrics["tldrf_usage_on_required_rate"]
        _add_gate(
            gate_rows,
            name="tldrf_usage_on_required_min",
            actual=actual,
            expected=threshold,
            passed=(actual is not None and threshold is not None and actual >= threshold),
        )
    if "rg_first_on_exact_min" in gates:
        threshold = _coerce_float(gates.get("rg_first_on_exact_min"))
        actual = metrics["rg_first_on_exact_rate"]
        _add_gate(
            gate_rows,
            name="rg_first_on_exact_min",
            actual=actual,
            expected=threshold,
            passed=(actual is not None and threshold is not None and actual >= threshold),
        )
    if "median_dead_end_turns_max" in gates:
        threshold = _coerce_float(gates.get("median_dead_end_turns_max"))
        dead_end_values = [int(row.get("dead_end_turns") or 0) for row in evaluated_tasks]
        actual = statistics.median(dead_end_values) if dead_end_values else None
        _add_gate(
            gate_rows,
            name="median_dead_end_turns_max",
            actual=actual,
            expected=threshold,
            passed=(actual is not None and threshold is not None and actual <= threshold),
        )
    if "systematic_failure_detected_must_be" in gates:
        expected = _coerce_bool(gates.get("systematic_failure_detected_must_be"))
        actual = metrics["systematic_failure_detected"]
        _add_gate(
            gate_rows,
            name="systematic_failure_detected_must_be",
            actual=actual,
            expected=expected,
            passed=(expected is not None and actual == expected),
        )

    passed = all(bool(row["pass"]) for row in gate_rows)
    recommendation = "pass" if passed else "tune_and_rerun"
    return {
        "schema_version": SCHEMA_VERSION,
        "report_path": str(Path(report.get("_source_path") or "").resolve()) if report.get("_source_path") else None,
        "report_hash": sha256_file(Path(report["_source_path"])) if report.get("_source_path") else None,
        "metrics": metrics,
        "failure_patterns": failure_patterns,
        "tasks": evaluated_tasks,
        "gates": gate_rows,
        "systematic_failure_detected": systematic_failure_detected,
        "recommendation": recommendation,
        "preflight_status": "passed" if passed else "failed",
        "pass": passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate agentic preflight transcript compliance.")
    ap.add_argument("--report", required=True, help="Path to the preflight report JSON.")
    ap.add_argument("--gates", required=True, help="Path to the preflight gate JSON.")
    ap.add_argument("--out", default=None, help="Write diagnostic JSON here (default under benchmark/runs/).")
    args = ap.parse_args()

    repo_root = get_repo_root()
    report_path = Path(args.report).resolve()
    gates_path = Path(args.gates).resolve()
    report = load_json_obj(report_path)
    report["_source_path"] = str(report_path)
    gates = load_json_obj(gates_path)
    diagnostic = evaluate_report(report, gates)
    diagnostic["gate_config_path"] = str(gates_path)
    diagnostic["gate_config_hash"] = sha256_file(gates_path)

    out_path = (
        Path(args.out).resolve()
        if args.out
        else bench_runs_root(repo_root) / f"{now_utc_compact()}-preflight-validate.json"
    )
    write_report(out_path, diagnostic)
    print(out_path)
    return 0 if diagnostic["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
