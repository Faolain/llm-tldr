#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from bench_util import bench_runs_root, get_repo_root, write_report

SCHEMA_VERSION = 1
RUN_VALIDITY_GATES = (
    "run_validity.max_timeout_rate",
    "run_validity.max_error_rate",
    "run_validity.max_budget_violation_rate",
)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"score report must be a JSON object: {path}")
    return data


def _coerce_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _task_manifest_hash(score_report: dict[str, Any]) -> str:
    inputs = score_report.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("score report missing inputs object")
    task_manifest_hash = inputs.get("task_manifest_sha256")
    if not isinstance(task_manifest_hash, str) or not task_manifest_hash:
        raise ValueError("score report missing inputs.task_manifest_sha256")
    return task_manifest_hash


def _tool_id(score_report: dict[str, Any]) -> str:
    tool_id = score_report.get("tool_id")
    if not isinstance(tool_id, str) or not tool_id:
        raise ValueError("score report missing tool_id")
    return tool_id


def _gate_pass_map(score_report: dict[str, Any]) -> dict[str, bool]:
    gate_checks = score_report.get("gate_checks")
    if not isinstance(gate_checks, list):
        raise ValueError("score report missing gate_checks list")

    out: dict[str, bool] = {}
    for gate in gate_checks:
        if not isinstance(gate, dict):
            continue
        name = gate.get("name")
        passed = gate.get("pass")
        if isinstance(name, str) and isinstance(passed, bool):
            out[name] = passed
    return out


def _is_valid_run(score_report: dict[str, Any]) -> bool:
    gate_map = _gate_pass_map(score_report)
    return all(gate_map.get(name) is True for name in RUN_VALIDITY_GATES)


def _retrieval_metric_at_budget(
    score_report: dict[str, Any],
    *,
    budget_tokens: int,
    metric_name: str,
) -> float:
    metrics = score_report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("score report missing metrics object")

    by_budget = metrics.get("by_budget")
    if not isinstance(by_budget, dict):
        raise ValueError("score report missing metrics.by_budget object")

    budget_obj = by_budget.get(str(budget_tokens))
    if not isinstance(budget_obj, dict):
        raise ValueError(f"score report missing metrics.by_budget.{budget_tokens}")

    retrieval = budget_obj.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError(f"score report missing metrics.by_budget.{budget_tokens}.retrieval")

    value = _coerce_float(retrieval.get(metric_name))
    if value is None:
        raise ValueError(
            f"score report missing numeric metrics.by_budget.{budget_tokens}.retrieval.{metric_name}"
        )
    return value


def _safe_stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.stdev(xs))


def _summarize_tool_runs(
    *,
    tool_id: str,
    score_reports: list[dict[str, Any]],
    min_valid_runs: int,
    primary_budget: int,
) -> dict[str, Any]:
    valid_reports = [report for report in score_reports if _is_valid_run(report)]
    if len(valid_reports) < min_valid_runs:
        raise ValueError(f"tool '{tool_id}' requires at least {min_valid_runs} valid runs")

    mrr_values = [
        _retrieval_metric_at_budget(report, budget_tokens=primary_budget, metric_name="mrr_mean")
        for report in valid_reports
    ]
    recall5_values = [
        _retrieval_metric_at_budget(report, budget_tokens=primary_budget, metric_name="recall@5_mean")
        for report in valid_reports
    ]

    return {
        "runs_total": len(score_reports),
        "valid_runs": len(valid_reports),
        "invalid_runs": len(score_reports) - len(valid_reports),
        "variance": {
            "budget_tokens": int(primary_budget),
            "retrieval": {
                "mrr_mean": {
                    "values": mrr_values,
                    "mean": float(statistics.mean(mrr_values)),
                    "stdev": _safe_stdev(mrr_values),
                },
                "recall@5_mean": {
                    "values": recall5_values,
                    "mean": float(statistics.mean(recall5_values)),
                    "stdev": _safe_stdev(recall5_values),
                },
            },
        },
    }


def _build_baseline_summary(
    score_reports: list[dict[str, Any]],
    *,
    min_valid_runs: int = 2,
    primary_budget: int = 2000,
) -> dict[str, Any]:
    if not score_reports:
        raise ValueError("baseline summary requires at least one score report")
    if min_valid_runs < 1:
        raise ValueError("min_valid_runs must be >= 1")
    if primary_budget <= 0:
        raise ValueError("primary_budget must be > 0")

    manifest_hashes = {_task_manifest_hash(report) for report in score_reports}
    if len(manifest_hashes) != 1:
        raise ValueError("baseline summary requires identical task_manifest_hash across runs")
    task_manifest_hash = next(iter(manifest_hashes))

    reports_by_tool: dict[str, list[dict[str, Any]]] = {}
    for report in score_reports:
        tool_id = _tool_id(report)
        reports_by_tool.setdefault(tool_id, []).append(report)

    tools_summary: dict[str, Any] = {}
    for tool_id in sorted(reports_by_tool):
        tools_summary[tool_id] = _summarize_tool_runs(
            tool_id=tool_id,
            score_reports=reports_by_tool[tool_id],
            min_valid_runs=min_valid_runs,
            primary_budget=primary_budget,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "phase2_h2h_baseline_summary",
        "task_manifest_sha256": task_manifest_hash,
        "primary_budget": int(primary_budget),
        "min_valid_runs_required": int(min_valid_runs),
        "tools": tools_summary,
    }


def cmd_summarize(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    score_paths = sorted(Path(p).resolve() for p in args.score)
    score_reports = [_read_json(path) for path in score_paths]

    summary = _build_baseline_summary(
        score_reports,
        min_valid_runs=int(args.min_valid_runs),
        primary_budget=int(args.primary_budget),
    )
    summary["inputs"] = {"score_files": [str(path) for path in score_paths]}

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = bench_runs_root(repo_root) / "h2h-baseline-summary.json"
    write_report(out_path, summary)
    print(out_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate head-to-head score reports into a baseline summary.")
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        help="Path to a bench_head_to_head.py score report. Repeat for each run/tool.",
    )
    parser.add_argument(
        "--min-valid-runs",
        type=int,
        default=2,
        help="Minimum number of valid runs required per tool (default: 2).",
    )
    parser.add_argument(
        "--primary-budget",
        type=int,
        default=2000,
        help="Budget used for variance metrics (default: 2000).",
    )
    parser.add_argument("--out", default=None, help="Write JSON report to this path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(cmd_summarize(args))


if __name__ == "__main__":
    raise SystemExit(main())
