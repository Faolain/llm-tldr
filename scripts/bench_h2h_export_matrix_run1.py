#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_util import get_repo_root, write_json

SCHEMA_VERSION = 1
PRIMARY_BUDGET = 2000

DEFAULT_LABEL_A = "llm-tldr"
DEFAULT_LABEL_B = "contextplus"
DEFAULT_BUDGETS = "500,1000,2000,5000"

DEFAULT_SCORE_A = (
    "benchmark/runs/h2h-llm-tldr-score-run1-fixed-stitched-allowlist-20260302T062602Z.json"
)
DEFAULT_SCORE_B = "benchmark/runs/h2h-contextplus-score-run1.json"
DEFAULT_COMPARE = "benchmark/runs/h2h-compare-run1-fixed-stitched-allowlist-20260302T062602Z.json"
DEFAULT_ASSERT = "benchmark/runs/h2h-assert-run1-fixed-stitched-allowlist-20260302T062602Z.json"
DEFAULT_META_A = "benchmark/runs/h2h-run-metadata-run1-llm-tldr-fixed.json"
DEFAULT_META_B = "benchmark/runs/h2h-run-metadata-run1-contextplus.json"
DEFAULT_PROFILE_A = "benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json"
DEFAULT_PROFILE_B = "benchmarks/head_to_head/tool_profiles/contextplus.v1.json"

DEFAULT_RUN_ID_A = "run1-fixed-stitched-allowlist-20260302T062602Z"
DEFAULT_RUN_ID_B = "run1"
DEFAULT_FEATURE_SET_A = "baseline.run1.fixed.stitched.allowlist"
DEFAULT_FEATURE_SET_B = "baseline.run1"
DEFAULT_EMBEDDING_BACKEND_A = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL_A = "profile_unpinned"
DEFAULT_EMBEDDING_BACKEND_B = "unknown"
DEFAULT_EMBEDDING_MODEL_B = "unknown"

DEFAULT_MATRIX_RUN_ID = DEFAULT_RUN_ID_A
DEFAULT_OUT_JSON = f"benchmark/runs/matrix/h2h-matrix-long-{DEFAULT_MATRIX_RUN_ID}.json"
DEFAULT_OUT_CSV = f"benchmark/runs/matrix/h2h-matrix-long-{DEFAULT_MATRIX_RUN_ID}.csv"
DEFAULT_SNAPSHOT_MD = "implementations/008-canonical-matrix-run1-snapshot.md"
DEFAULT_PIVOT_MD = "implementations/008-canonical-matrix-run1-pivot-by-budget.md"

ROW_COLUMNS = [
    "row_id",
    "row_scope",
    "is_optional_budget_row",
    "suite_id",
    "tool",
    "tool_version",
    "feature_set_id",
    "embedding_backend",
    "embedding_model",
    "feature_lane",
    "budget_tokens",
    "run_id",
    "task_manifest_sha256",
    "suite_sha256",
    "tool_profile_sha256",
    "tokenizer",
    "corpus_id",
    "corpus_git_sha",
    "tldr_git_sha",
    "tldr_git_describe",
    "prediction_count",
    "trials",
    "retrieval_mrr_mean",
    "retrieval_recall_at_5_mean",
    "retrieval_recall_at_10_mean",
    "retrieval_precision_at_5_mean",
    "retrieval_precision_at_10_mean",
    "retrieval_fpr_at_5_mean",
    "retrieval_fpr_at_10_mean",
    "impact_f1_mean",
    "impact_precision_mean",
    "impact_recall_mean",
    "slice_recall_mean",
    "slice_precision_mean",
    "slice_f1_mean",
    "slice_noise_reduction_mean",
    "data_flow_origin_accuracy_mean",
    "data_flow_flow_completeness_mean",
    "complexity_mae",
    "complexity_kendall_tau_b",
    "timeout_rate",
    "error_rate",
    "unsupported_rate",
    "budget_violation_rate",
    "common_lane_coverage",
    "capability_coverage",
    "parse_errors_count",
    "result_shape_counters_total",
    "non_object_result_count",
    "empty_result_object_count",
    "category_shape_mismatch_count",
    "bad_json",
    "judge_bad_json",
    "answer_errors_total",
    "judge_errors_total",
    "unclassified_failures_total",
    "retrieval_payload_tokens_median",
    "retrieval_payload_bytes_median",
    "retrieval_latency_ms_p50",
    "latency_ms_p95",
    "tok",
    "tok_per_tp",
    "noise_ratio_mean",
    "noise_reduction_mean",
    "index_size_bytes",
    "cache_size_bytes",
    "compare_winner",
    "compare_wins_tool",
    "compare_wins_other_tool",
    "compare_delta_mrr_mean",
    "compare_delta_recall_at_5_mean",
    "compare_delta_precision_at_5_mean",
    "assert_gates_passed",
    "assert_failed_gate_names",
    "assert_stability_two_of_three_pass",
    "assert_stability_reason",
    "score_gates_passed",
    "source_score_path",
    "source_compare_path",
    "source_assert_path",
    "source_run_metadata_path",
    "source_tool_profile_path",
]


@dataclass(frozen=True)
class ToolRowConfig:
    label: str
    feature_set_id: str
    run_id: str
    embedding_backend: str
    embedding_model: str
    source_score_path: str
    source_run_metadata_path: str
    source_tool_profile_path: str


def _read_json_obj(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _resolve_path(repo_root: Path, path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


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


def _coerce_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return None
    return None


def _suite_sha(score_report: dict[str, Any], run_metadata: dict[str, Any]) -> str | None:
    inputs = score_report.get("inputs")
    if isinstance(inputs, dict):
        suite_sha = inputs.get("suite_sha256")
        if isinstance(suite_sha, str) and suite_sha:
            return suite_sha
    suite_sha = run_metadata.get("suite_sha256")
    if isinstance(suite_sha, str) and suite_sha:
        return suite_sha
    return None


def _task_manifest_sha(score_report: dict[str, Any], run_metadata: dict[str, Any]) -> str | None:
    inputs = score_report.get("inputs")
    if isinstance(inputs, dict):
        manifest_sha = inputs.get("task_manifest_sha256")
        if isinstance(manifest_sha, str) and manifest_sha:
            return manifest_sha
    manifest_sha = run_metadata.get("task_manifest_sha256")
    if isinstance(manifest_sha, str) and manifest_sha:
        return manifest_sha
    return None


def _tool_profile_sha(score_report: dict[str, Any], run_metadata: dict[str, Any]) -> str | None:
    value = run_metadata.get("tool_profile_sha256")
    if isinstance(value, str) and value:
        return value
    inputs = score_report.get("inputs")
    if isinstance(inputs, dict):
        v = inputs.get("tool_profile_sha256")
        if isinstance(v, str) and v:
            return v
    return None


def _tokenizer(score_report: dict[str, Any], run_metadata: dict[str, Any]) -> str | None:
    inputs = score_report.get("inputs")
    if isinstance(inputs, dict):
        value = inputs.get("tokenizer")
        if isinstance(value, str) and value:
            return value
    value = run_metadata.get("tokenizer")
    if isinstance(value, str) and value:
        return value
    return None


def _feature_set_id(
    *,
    configured_value: Any,
    run_metadata: dict[str, Any],
    default_value: str,
) -> str:
    if isinstance(configured_value, str) and configured_value.strip():
        return configured_value.strip()
    value = run_metadata.get("feature_set_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default_value


def _budget_obj(score_report: dict[str, Any], budget_tokens: int) -> dict[str, Any]:
    metrics = score_report.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    by_budget = metrics.get("by_budget")
    if not isinstance(by_budget, dict):
        return {}
    budget = by_budget.get(str(budget_tokens))
    if not isinstance(budget, dict):
        return {}
    return budget


def _category_metric(
    score_report: dict[str, Any], budget_tokens: int, category: str, metric: str
) -> float | None:
    budget_obj = _budget_obj(score_report, budget_tokens)
    category_obj = budget_obj.get(category)
    if not isinstance(category_obj, dict):
        return None
    return _coerce_float(category_obj.get(metric))


def _rate(score_report: dict[str, Any], metric: str) -> float | None:
    rates = score_report.get("rates")
    if not isinstance(rates, dict):
        return None
    return _coerce_float(rates.get(metric))


def _result_shape_count(score_report: dict[str, Any], metric: str) -> int | None:
    diagnostics = score_report.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    counters = diagnostics.get("result_shape_counters")
    if not isinstance(counters, dict):
        return None
    return _coerce_int(counters.get(metric))


def _parse_errors_count(score_report: dict[str, Any]) -> int | None:
    parse_errors = score_report.get("parse_errors")
    if not isinstance(parse_errors, list):
        return None
    return int(len(parse_errors))


def _compare_metric_delta(compare_report: dict[str, Any], metric: str) -> float | None:
    comps = compare_report.get("metric_comparisons")
    if not isinstance(comps, list):
        return None
    for comp in comps:
        if not isinstance(comp, dict):
            continue
        if comp.get("metric") != metric:
            continue
        a = _coerce_float(comp.get("a"))
        b = _coerce_float(comp.get("b"))
        if a is None or b is None:
            return None
        return a - b
    return None


def _assert_failed_gate_names(assert_report: dict[str, Any]) -> str:
    failed: list[str] = []
    runs = assert_report.get("runs")
    if isinstance(runs, list) and runs:
        run = runs[0]
        if isinstance(run, dict):
            checks = run.get("gate_checks")
            if isinstance(checks, list):
                for check in checks:
                    if not isinstance(check, dict):
                        continue
                    if check.get("pass") is False:
                        name = check.get("name")
                        if isinstance(name, str) and name:
                            failed.append(name)
    stability = assert_report.get("stability_gate")
    if isinstance(stability, dict) and stability.get("pass") is False:
        name = stability.get("name")
        if isinstance(name, str) and name:
            failed.append(name)
    return ";".join(failed)


def _row_id(
    *,
    tool: str,
    tool_version: str | None,
    feature_set_id: str,
    embedding_backend: str,
    embedding_model: str,
    budget_tokens: int,
    run_id: str,
) -> str:
    version = tool_version or "unknown"
    return (
        f"{tool}|{version}|{feature_set_id}|{embedding_backend}|"
        f"{embedding_model}|{budget_tokens}|{run_id}"
    )


def _tool_order(compare_report: dict[str, Any], fallback_a: str, fallback_b: str) -> list[str]:
    labels = compare_report.get("labels")
    if isinstance(labels, dict):
        a = labels.get("a")
        b = labels.get("b")
        if isinstance(a, str) and a and isinstance(b, str) and b:
            return [a, b]
    return [fallback_a, fallback_b]


def _ensure_row_columns(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ROW_COLUMNS:
        out[key] = row.get(key)
    return out


def _build_row(
    *,
    tool_config: ToolRowConfig,
    score_report: dict[str, Any],
    run_metadata: dict[str, Any],
    compare_report: dict[str, Any],
    assert_report: dict[str, Any],
    budget_tokens: int,
    other_label: str,
    primary_budget: int,
) -> dict[str, Any]:
    meta = score_report.get("meta")
    meta_obj = meta if isinstance(meta, dict) else {}
    tool_version = meta_obj.get("tldr_git_sha")
    tool_version_str = tool_version if isinstance(tool_version, str) and tool_version else None
    row_scope = "required_primary_budget" if budget_tokens == primary_budget else "optional_budget_sensitivity"
    is_optional_budget_row = budget_tokens != primary_budget

    compare_winner: str | None = None
    compare_wins_tool: int | None = None
    compare_wins_other: int | None = None
    compare_delta_mrr: float | None = None
    compare_delta_recall5: float | None = None
    compare_delta_precision5: float | None = None
    assert_gates_passed: bool | None = None
    assert_failed_gate_names: str | None = None
    assert_stability_pass: bool | None = None
    assert_stability_reason: str | None = None
    if budget_tokens == primary_budget:
        compare_winner_val = compare_report.get("winner")
        if isinstance(compare_winner_val, str):
            compare_winner = compare_winner_val
        wins = compare_report.get("wins")
        if isinstance(wins, dict):
            compare_wins_tool = _coerce_int(wins.get(tool_config.label))
            compare_wins_other = _coerce_int(wins.get(other_label))
        compare_delta_mrr = _compare_metric_delta(compare_report, "mrr_mean")
        compare_delta_recall5 = _compare_metric_delta(compare_report, "recall@5_mean")
        compare_delta_precision5 = _compare_metric_delta(compare_report, "precision@5_mean")
        gates_passed = assert_report.get("gates_passed")
        if isinstance(gates_passed, bool):
            assert_gates_passed = gates_passed
        assert_failed_gate_names = _assert_failed_gate_names(assert_report)
        stability = assert_report.get("stability_gate")
        if isinstance(stability, dict):
            pass_val = stability.get("pass")
            if isinstance(pass_val, bool):
                assert_stability_pass = pass_val
            reason = stability.get("reason")
            if isinstance(reason, str) and reason:
                assert_stability_reason = reason

    row = {
        "row_id": _row_id(
            tool=tool_config.label,
            tool_version=tool_version_str,
            feature_set_id=tool_config.feature_set_id,
            embedding_backend=tool_config.embedding_backend,
            embedding_model=tool_config.embedding_model,
            budget_tokens=budget_tokens,
            run_id=tool_config.run_id,
        ),
        "row_scope": row_scope,
        "is_optional_budget_row": is_optional_budget_row,
        "suite_id": score_report.get("suite_id"),
        "tool": tool_config.label,
        "tool_version": tool_version_str,
        "feature_set_id": tool_config.feature_set_id,
        "embedding_backend": tool_config.embedding_backend,
        "embedding_model": tool_config.embedding_model,
        "feature_lane": "retrieval/common",
        "budget_tokens": int(budget_tokens),
        "run_id": tool_config.run_id,
        "task_manifest_sha256": _task_manifest_sha(score_report, run_metadata),
        "suite_sha256": _suite_sha(score_report, run_metadata),
        "tool_profile_sha256": _tool_profile_sha(score_report, run_metadata),
        "tokenizer": _tokenizer(score_report, run_metadata),
        "corpus_id": meta_obj.get("corpus_id"),
        "corpus_git_sha": meta_obj.get("corpus_git_sha"),
        "tldr_git_sha": meta_obj.get("tldr_git_sha"),
        "tldr_git_describe": meta_obj.get("tldr_git_describe"),
        "prediction_count": _coerce_int(run_metadata.get("prediction_count")),
        "trials": _coerce_int(run_metadata.get("trials")),
        "retrieval_mrr_mean": _category_metric(score_report, budget_tokens, "retrieval", "mrr_mean"),
        "retrieval_recall_at_5_mean": _category_metric(
            score_report, budget_tokens, "retrieval", "recall@5_mean"
        ),
        "retrieval_recall_at_10_mean": _category_metric(
            score_report, budget_tokens, "retrieval", "recall@10_mean"
        ),
        "retrieval_precision_at_5_mean": _category_metric(
            score_report, budget_tokens, "retrieval", "precision@5_mean"
        ),
        "retrieval_precision_at_10_mean": _category_metric(
            score_report, budget_tokens, "retrieval", "precision@10_mean"
        ),
        "retrieval_fpr_at_5_mean": _category_metric(score_report, budget_tokens, "retrieval", "fpr@5_mean"),
        "retrieval_fpr_at_10_mean": _category_metric(
            score_report, budget_tokens, "retrieval", "fpr@10_mean"
        ),
        "impact_f1_mean": _category_metric(score_report, budget_tokens, "impact", "f1_mean"),
        "impact_precision_mean": _category_metric(score_report, budget_tokens, "impact", "precision_mean"),
        "impact_recall_mean": _category_metric(score_report, budget_tokens, "impact", "recall_mean"),
        "slice_recall_mean": _category_metric(score_report, budget_tokens, "slice", "recall_mean"),
        "slice_precision_mean": _category_metric(score_report, budget_tokens, "slice", "precision_mean"),
        "slice_f1_mean": _category_metric(score_report, budget_tokens, "slice", "f1_mean"),
        "slice_noise_reduction_mean": _category_metric(
            score_report, budget_tokens, "slice", "noise_reduction_mean"
        ),
        "data_flow_origin_accuracy_mean": _category_metric(
            score_report, budget_tokens, "data_flow", "origin_accuracy_mean"
        ),
        "data_flow_flow_completeness_mean": _category_metric(
            score_report, budget_tokens, "data_flow", "flow_completeness_mean"
        ),
        "complexity_mae": _category_metric(score_report, budget_tokens, "complexity", "mae"),
        "complexity_kendall_tau_b": _category_metric(score_report, budget_tokens, "complexity", "kendall_tau_b"),
        "timeout_rate": _rate(score_report, "timeout_rate"),
        "error_rate": _rate(score_report, "error_rate"),
        "unsupported_rate": _rate(score_report, "unsupported_rate"),
        "budget_violation_rate": _rate(score_report, "budget_violation_rate"),
        "common_lane_coverage": _rate(score_report, "common_lane_coverage"),
        "capability_coverage": _rate(score_report, "capability_coverage"),
        "parse_errors_count": _parse_errors_count(score_report),
        "result_shape_counters_total": _result_shape_count(score_report, "total"),
        "non_object_result_count": _result_shape_count(score_report, "non_object_result"),
        "empty_result_object_count": _result_shape_count(score_report, "empty_result_object"),
        "category_shape_mismatch_count": _result_shape_count(score_report, "category_shape_mismatch"),
        "bad_json": None,
        "judge_bad_json": None,
        "answer_errors_total": None,
        "judge_errors_total": None,
        "unclassified_failures_total": None,
        "retrieval_payload_tokens_median": _category_metric(
            score_report, budget_tokens, "retrieval", "payload_tokens_median"
        ),
        "retrieval_payload_bytes_median": _category_metric(
            score_report, budget_tokens, "retrieval", "payload_bytes_median"
        ),
        "retrieval_latency_ms_p50": _category_metric(
            score_report, budget_tokens, "retrieval", "latency_ms_p50"
        ),
        "latency_ms_p95": None,
        "tok": None,
        "tok_per_tp": None,
        "noise_ratio_mean": None,
        "noise_reduction_mean": None,
        "index_size_bytes": None,
        "cache_size_bytes": None,
        "compare_winner": compare_winner,
        "compare_wins_tool": compare_wins_tool,
        "compare_wins_other_tool": compare_wins_other,
        "compare_delta_mrr_mean": compare_delta_mrr,
        "compare_delta_recall_at_5_mean": compare_delta_recall5,
        "compare_delta_precision_at_5_mean": compare_delta_precision5,
        "assert_gates_passed": assert_gates_passed,
        "assert_failed_gate_names": assert_failed_gate_names,
        "assert_stability_two_of_three_pass": assert_stability_pass,
        "assert_stability_reason": assert_stability_reason,
        "score_gates_passed": score_report.get("gates_passed"),
        "source_score_path": tool_config.source_score_path,
        "source_compare_path": compare_report.get("__source_path"),
        "source_assert_path": assert_report.get("__source_path"),
        "source_run_metadata_path": tool_config.source_run_metadata_path,
        "source_tool_profile_path": tool_config.source_tool_profile_path,
    }
    return _ensure_row_columns(row)


def _build_matrix_rows(
    *,
    tool_configs: list[ToolRowConfig],
    score_reports: dict[str, dict[str, Any]],
    run_metadata_reports: dict[str, dict[str, Any]],
    compare_report: dict[str, Any],
    assert_report: dict[str, Any],
    budgets: list[int],
    primary_budget: int = PRIMARY_BUDGET,
) -> list[dict[str, Any]]:
    if len(tool_configs) != 2:
        raise ValueError("exactly two tool configs are required for head-to-head matrix export")

    ordered_labels = _tool_order(compare_report, tool_configs[0].label, tool_configs[1].label)
    label_to_cfg = {cfg.label: cfg for cfg in tool_configs}
    ordered_cfgs = [label_to_cfg[label] for label in ordered_labels if label in label_to_cfg]
    if len(ordered_cfgs) != 2:
        raise ValueError("compare labels do not align with provided tool configs")

    rows: list[dict[str, Any]] = []
    for i, tool_cfg in enumerate(ordered_cfgs):
        other_label = ordered_cfgs[1 - i].label
        score_report = score_reports[tool_cfg.label]
        run_metadata = run_metadata_reports[tool_cfg.label]
        for budget in sorted(budgets):
            rows.append(
                _build_row(
                    tool_config=tool_cfg,
                    score_report=score_report,
                    run_metadata=run_metadata,
                    compare_report=compare_report,
                    assert_report=assert_report,
                    budget_tokens=budget,
                    other_label=other_label,
                    primary_budget=primary_budget,
                )
            )
    return rows


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return json.dumps(v, separators=(",", ":"))
    return str(v)


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k)) for k in ROW_COLUMNS})


def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return json.dumps(v)
    return str(v)


def _short_sha(v: Any) -> str:
    if isinstance(v, str) and len(v) >= 7:
        return v[:7]
    return "unknown"


def _tool_budget_row(rows: list[dict[str, Any]], tool: str, budget: int) -> dict[str, Any]:
    for row in rows:
        if row.get("tool") == tool and row.get("budget_tokens") == budget:
            return row
    raise ValueError(f"missing row for tool={tool} budget={budget}")


def _render_table(headers: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_snapshot_markdown(
    *,
    generated_date: str,
    rows: list[dict[str, Any]],
    source_paths: dict[str, str],
    out_json: str,
    out_csv: str,
    primary_budget: int,
) -> str:
    tools = sorted({str(row["tool"]) for row in rows}, key=lambda x: (0 if x == DEFAULT_LABEL_A else 1, x))
    budgets = sorted({int(row["budget_tokens"]) for row in rows})
    id_rows: list[list[str]] = []
    for tool in tools:
        row = _tool_budget_row(rows, tool, primary_budget)
        id_rows.append(
            [
                tool,
                f"`{row.get('tool_version')}`",
                f"`{row.get('feature_set_id')}`",
                f"`{row.get('embedding_backend')}`",
                f"`{row.get('embedding_model')}`",
                f"`{row.get('run_id')}`",
            ]
        )

    budget_policy_rows = []
    for budget in budgets:
        row = _tool_budget_row(rows, tools[0], budget)
        budget_policy_rows.append(
            [
                str(budget),
                "required" if row.get("is_optional_budget_row") is False else "optional",
                f"`{row.get('row_scope')}`",
            ]
        )

    retrieval_rows: list[list[str]] = []
    for tool in tools:
        for budget in budgets:
            row = _tool_budget_row(rows, tool, budget)
            retrieval_rows.append(
                [
                    tool,
                    str(budget),
                    _fmt(row.get("retrieval_mrr_mean")),
                    _fmt(row.get("retrieval_recall_at_5_mean")),
                    _fmt(row.get("retrieval_recall_at_10_mean")),
                    _fmt(row.get("retrieval_precision_at_5_mean")),
                    _fmt(row.get("retrieval_fpr_at_5_mean")),
                    _fmt(row.get("retrieval_payload_tokens_median")),
                    _fmt(row.get("retrieval_latency_ms_p50")),
                ]
            )

    holistic_rows: list[list[str]] = []
    for tool in tools:
        row = _tool_budget_row(rows, tool, primary_budget)
        holistic_rows.append(
            [
                tool,
                _fmt(row.get("impact_f1_mean")),
                _fmt(row.get("slice_recall_mean")),
                _fmt(row.get("slice_noise_reduction_mean")),
                _fmt(row.get("data_flow_origin_accuracy_mean")),
                _fmt(row.get("data_flow_flow_completeness_mean")),
                _fmt(row.get("complexity_mae")),
                _fmt(row.get("timeout_rate")),
                _fmt(row.get("error_rate")),
                _fmt(row.get("budget_violation_rate")),
                _fmt(row.get("parse_errors_count")),
                _fmt(row.get("result_shape_counters_total")),
            ]
        )

    primary_row = _tool_budget_row(rows, DEFAULT_LABEL_A, primary_budget)
    compare_winner = primary_row.get("compare_winner")
    delta_mrr = primary_row.get("compare_delta_mrr_mean")
    delta_recall5 = primary_row.get("compare_delta_recall_at_5_mean")
    delta_precision5 = primary_row.get("compare_delta_precision_at_5_mean")
    assert_failed = primary_row.get("assert_failed_gate_names")
    compare_wins_tool = primary_row.get("compare_wins_tool")
    compare_wins_other = primary_row.get("compare_wins_other_tool")
    stability_pass = primary_row.get("assert_stability_two_of_three_pass")
    stability_reason = primary_row.get("assert_stability_reason")
    strict_pass = primary_row.get("assert_gates_passed")

    lines = [
        "# 008 Canonical Matrix Snapshot (Run1 Baseline)",
        "",
        f"- Generated: {generated_date}",
        "- Scope: Existing run1 artifacts only (no new LLM execution)",
        f"- Canonical matrix artifacts: `{out_csv}` and `{out_json}`",
        "- Canonical identity axes: `tool`, `tool_version`, `feature_set_id`, `embedding_backend`, `embedding_model`, `budget_tokens`, `run_id`",
        "",
        "## Source Artifacts",
        "",
        f"- llm-tldr score: `{source_paths['score_a']}`",
        f"- contextplus score: `{source_paths['score_b']}`",
        f"- compare: `{source_paths['compare']}`",
        f"- assert: `{source_paths['assert']}`",
        f"- run metadata (llm-tldr): `{source_paths['meta_a']}`",
        f"- run metadata (contextplus): `{source_paths['meta_b']}`",
        f"- tool profile (llm-tldr): `{source_paths['profile_a']}`",
        f"- tool profile (contextplus): `{source_paths['profile_b']}`",
        "",
        "## Important Caveat",
        "",
        "This snapshot combines:",
        "- llm-tldr from `run1-fixed` stitched allowlist artifacts",
        "- contextplus from `run1` artifacts",
        "",
        "## Canonical Row Identity (Budget 2000 Rows)",
        "",
        _render_table(
            ["tool", "tool_version", "feature_set_id", "embedding_backend", "embedding_model", "run_id"],
            id_rows,
        ),
        "",
        "## Budget Row Policy",
        "",
        _render_table(["budget_tokens", "policy", "row_scope"], budget_policy_rows),
        "",
        "## Retrieval Metrics By Budget",
        "",
        _render_table(
            [
                "tool",
                "budget_tokens",
                "mrr_mean",
                "recall@5_mean",
                "recall@10_mean",
                "precision@5_mean",
                "fpr@5_mean",
                "payload_tokens_median",
                "latency_ms_p50",
            ],
            retrieval_rows,
        ),
        "",
        "## Holistic Metrics At Budget 2000 (Required Rows)",
        "",
        _render_table(
            [
                "tool",
                "impact_f1_mean",
                "slice_recall_mean",
                "slice_noise_reduction_mean",
                "data_flow_origin_accuracy_mean",
                "data_flow_flow_completeness_mean",
                "complexity_mae",
                "timeout_rate",
                "error_rate",
                "budget_violation_rate",
                "parse_errors_count",
                "result_shape_counters_total",
            ],
            holistic_rows,
        ),
        "",
        "## Compare / Assert Snapshot (Budget 2000)",
        "",
        f"- compare winner: `{compare_winner}`",
        f"- compare wins: `{DEFAULT_LABEL_A}={compare_wins_tool}`, `{DEFAULT_LABEL_B}={compare_wins_other}`",
        "- deltas at budget 2000:",
        f"  - `mrr_mean_delta = {_fmt(delta_mrr)}`",
        f"  - `recall@5_mean_delta = {_fmt(delta_recall5)}`",
        f"  - `precision@5_mean_delta = {_fmt(delta_precision5)}`",
        f"- strict assert overall: `{_fmt(strict_pass)}`",
        f"- failed strict gate(s): `{assert_failed}`",
        f"- stability gate: `stability.two_of_three = {_fmt(stability_pass)}` (`{_fmt(stability_reason)}`)",
        "",
    ]
    return "\n".join(lines)


def _render_pivot_markdown(
    *,
    generated_date: str,
    rows: list[dict[str, Any]],
    out_json: str,
    primary_budget: int,
) -> str:
    tools = _tool_order(
        {"labels": {"a": DEFAULT_LABEL_A, "b": DEFAULT_LABEL_B}}, DEFAULT_LABEL_A, DEFAULT_LABEL_B
    )
    tool_rows = [_tool_budget_row(rows, tool, primary_budget) for tool in tools]
    col_keys = [f"{row['tool']}@{_short_sha(row.get('tool_version'))} ({row.get('feature_set_id')})" for row in tool_rows]
    budgets = sorted({int(row["budget_tokens"]) for row in rows})

    lines = [
        "# 008 Canonical Matrix (Pivot View by Budget)",
        "",
        f"- Generated: {generated_date}",
        "- View style: one table per budget token, columns are tool/version variants",
        f"- Source matrix artifact: `{out_json}`",
        "",
        "## Column Keys",
        "",
    ]
    for col in col_keys:
        lines.append(f"- `{col}`")

    lines.extend(["", "## Caveat", "", "This pivot uses mixed run artifacts currently in plan baseline:"])
    lines.extend(
        [
            f"- `{DEFAULT_LABEL_A}`: {DEFAULT_RUN_ID_A} artifact family",
            f"- `{DEFAULT_LABEL_B}`: {DEFAULT_RUN_ID_B} artifact family",
            "",
        ]
    )

    for budget in budgets:
        lines.append(f"## Budget {budget}")
        lines.append("")
        lines.append(
            _render_table(
                ["Metric", col_keys[0], col_keys[1]],
                [
                    [
                        "row_policy",
                        "required" if budget == primary_budget else "optional",
                        "required" if budget == primary_budget else "optional",
                    ],
                    [
                        "mrr_mean",
                        _fmt(_tool_budget_row(rows, tools[0], budget).get("retrieval_mrr_mean")),
                        _fmt(_tool_budget_row(rows, tools[1], budget).get("retrieval_mrr_mean")),
                    ],
                    [
                        "recall@5_mean",
                        _fmt(_tool_budget_row(rows, tools[0], budget).get("retrieval_recall_at_5_mean")),
                        _fmt(_tool_budget_row(rows, tools[1], budget).get("retrieval_recall_at_5_mean")),
                    ],
                    [
                        "recall@10_mean",
                        _fmt(_tool_budget_row(rows, tools[0], budget).get("retrieval_recall_at_10_mean")),
                        _fmt(_tool_budget_row(rows, tools[1], budget).get("retrieval_recall_at_10_mean")),
                    ],
                    [
                        "precision@5_mean",
                        _fmt(_tool_budget_row(rows, tools[0], budget).get("retrieval_precision_at_5_mean")),
                        _fmt(_tool_budget_row(rows, tools[1], budget).get("retrieval_precision_at_5_mean")),
                    ],
                    [
                        "fpr@5_mean",
                        _fmt(_tool_budget_row(rows, tools[0], budget).get("retrieval_fpr_at_5_mean")),
                        _fmt(_tool_budget_row(rows, tools[1], budget).get("retrieval_fpr_at_5_mean")),
                    ],
                    [
                        "payload_tokens_median",
                        _fmt(_tool_budget_row(rows, tools[0], budget).get("retrieval_payload_tokens_median")),
                        _fmt(_tool_budget_row(rows, tools[1], budget).get("retrieval_payload_tokens_median")),
                    ],
                    [
                        "latency_ms_p50",
                        _fmt(_tool_budget_row(rows, tools[0], budget).get("retrieval_latency_ms_p50")),
                        _fmt(_tool_budget_row(rows, tools[1], budget).get("retrieval_latency_ms_p50")),
                    ],
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Global Reliability Snapshot (Run-Level)",
            "",
            _render_table(
                ["Metric", col_keys[0], col_keys[1]],
                [
                    [
                        "timeout_rate",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("timeout_rate")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("timeout_rate")),
                    ],
                    [
                        "error_rate",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("error_rate")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("error_rate")),
                    ],
                    [
                        "unsupported_rate",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("unsupported_rate")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("unsupported_rate")),
                    ],
                    [
                        "budget_violation_rate",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("budget_violation_rate")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("budget_violation_rate")),
                    ],
                    [
                        "common_lane_coverage",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("common_lane_coverage")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("common_lane_coverage")),
                    ],
                    [
                        "capability_coverage",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("capability_coverage")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("capability_coverage")),
                    ],
                ],
            ),
            "",
            "## Structural Quality Snapshot (Budget 2000)",
            "",
            _render_table(
                ["Metric", col_keys[0], col_keys[1]],
                [
                    [
                        "impact_f1_mean",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("impact_f1_mean")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("impact_f1_mean")),
                    ],
                    [
                        "slice_recall_mean",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("slice_recall_mean")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("slice_recall_mean")),
                    ],
                    [
                        "slice_noise_reduction_mean",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("slice_noise_reduction_mean")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("slice_noise_reduction_mean")),
                    ],
                    [
                        "data_flow_origin_accuracy_mean",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("data_flow_origin_accuracy_mean")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("data_flow_origin_accuracy_mean")),
                    ],
                    [
                        "data_flow_flow_completeness_mean",
                        _fmt(
                            _tool_budget_row(rows, tools[0], primary_budget).get(
                                "data_flow_flow_completeness_mean"
                            )
                        ),
                        _fmt(
                            _tool_budget_row(rows, tools[1], primary_budget).get(
                                "data_flow_flow_completeness_mean"
                            )
                        ),
                    ],
                    [
                        "complexity_mae",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("complexity_mae")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("complexity_mae")),
                    ],
                    [
                        "complexity_kendall_tau_b",
                        _fmt(_tool_budget_row(rows, tools[0], primary_budget).get("complexity_kendall_tau_b")),
                        _fmt(_tool_budget_row(rows, tools[1], primary_budget).get("complexity_kendall_tau_b")),
                    ],
                ],
            ),
            "",
            "## Primary-Gate Summary (Budget 2000)",
            "",
            _render_table(
                ["Check", "Result"],
                [
                    ["winner", f"`{_fmt(_tool_budget_row(rows, tools[0], primary_budget).get('compare_winner'))}`"],
                    [
                        "wins (`>=3/5` required)",
                        (
                            f"`{tools[0]}={_fmt(_tool_budget_row(rows, tools[0], primary_budget).get('compare_wins_tool'))}`, "
                            f"`{tools[1]}={_fmt(_tool_budget_row(rows, tools[0], primary_budget).get('compare_wins_other_tool'))}`"
                        ),
                    ],
                    [
                        "mrr_mean_delta",
                        f"`{_fmt(_tool_budget_row(rows, tools[0], primary_budget).get('compare_delta_mrr_mean'))}`",
                    ],
                    [
                        "recall@5_mean_delta",
                        f"`{_fmt(_tool_budget_row(rows, tools[0], primary_budget).get('compare_delta_recall_at_5_mean'))}`",
                    ],
                    [
                        "precision@5_mean_delta",
                        f"`{_fmt(_tool_budget_row(rows, tools[0], primary_budget).get('compare_delta_precision_at_5_mean'))}`",
                    ],
                    [
                        "strict assert overall",
                        f"`{_fmt(_tool_budget_row(rows, tools[0], primary_budget).get('assert_gates_passed'))}`",
                    ],
                    [
                        "failing strict gate(s)",
                        f"`{_fmt(_tool_budget_row(rows, tools[0], primary_budget).get('assert_failed_gate_names'))}`",
                    ],
                    [
                        "stability gate",
                        (
                            "`stability.two_of_three="
                            f"{_fmt(_tool_budget_row(rows, tools[0], primary_budget).get('assert_stability_two_of_three_pass'))}` "
                            f"(`{_fmt(_tool_budget_row(rows, tools[0], primary_budget).get('assert_stability_reason'))}`)"
                        ),
                    ],
                ],
            ),
            "",
        ]
    )

    return "\n".join(lines)


def _parse_budgets(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"invalid budget value: {value!r}") from exc
        if parsed <= 0:
            raise ValueError(f"budget must be positive: {parsed}")
        out.append(parsed)
    if not out:
        raise ValueError("at least one budget is required")
    return sorted(set(out))


def cmd_export(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    budgets = _parse_budgets(args.budgets)

    score_a_path = _resolve_path(repo_root, args.score_a)
    score_b_path = _resolve_path(repo_root, args.score_b)
    compare_path = _resolve_path(repo_root, args.compare)
    assert_path = _resolve_path(repo_root, args.assert_report)
    meta_a_path = _resolve_path(repo_root, args.meta_a)
    meta_b_path = _resolve_path(repo_root, args.meta_b)

    score_a = _read_json_obj(score_a_path)
    score_b = _read_json_obj(score_b_path)
    compare_report = _read_json_obj(compare_path)
    assert_report = _read_json_obj(assert_path)
    meta_a = _read_json_obj(meta_a_path)
    meta_b = _read_json_obj(meta_b_path)
    compare_report["__source_path"] = args.compare
    assert_report["__source_path"] = args.assert_report

    feature_set_a = _feature_set_id(
        configured_value=args.feature_set_a,
        run_metadata=meta_a,
        default_value=DEFAULT_FEATURE_SET_A,
    )
    feature_set_b = _feature_set_id(
        configured_value=args.feature_set_b,
        run_metadata=meta_b,
        default_value=DEFAULT_FEATURE_SET_B,
    )

    tool_a = ToolRowConfig(
        label=args.label_a,
        feature_set_id=feature_set_a,
        run_id=args.run_id_a,
        embedding_backend=args.embedding_backend_a,
        embedding_model=args.embedding_model_a,
        source_score_path=args.score_a,
        source_run_metadata_path=args.meta_a,
        source_tool_profile_path=args.profile_a,
    )
    tool_b = ToolRowConfig(
        label=args.label_b,
        feature_set_id=feature_set_b,
        run_id=args.run_id_b,
        embedding_backend=args.embedding_backend_b,
        embedding_model=args.embedding_model_b,
        source_score_path=args.score_b,
        source_run_metadata_path=args.meta_b,
        source_tool_profile_path=args.profile_b,
    )

    rows = _build_matrix_rows(
        tool_configs=[tool_a, tool_b],
        score_reports={tool_a.label: score_a, tool_b.label: score_b},
        run_metadata_reports={tool_a.label: meta_a, tool_b.label: meta_b},
        compare_report=compare_report,
        assert_report=assert_report,
        budgets=budgets,
        primary_budget=int(args.primary_budget),
    )

    now = datetime.now(timezone.utc)
    generated_date = args.generated_date or now.strftime("%Y-%m-%d")
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "matrix_kind": "h2h_canonical_long_format",
        "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_date": generated_date,
        "primary_budget_tokens": int(args.primary_budget),
        "identity_axes": [
            "tool",
            "tool_version",
            "feature_set_id",
            "embedding_backend",
            "embedding_model",
            "budget_tokens",
            "run_id",
        ],
        "row_columns": ROW_COLUMNS,
        "sources": {
            "score_a": args.score_a,
            "score_b": args.score_b,
            "compare": args.compare,
            "assert": args.assert_report,
            "meta_a": args.meta_a,
            "meta_b": args.meta_b,
            "profile_a": args.profile_a,
            "profile_b": args.profile_b,
        },
        "rows": rows,
    }

    out_json = _resolve_path(repo_root, args.out_json)
    out_csv = _resolve_path(repo_root, args.out_csv)
    write_json(out_json, bundle)
    _write_csv_rows(out_csv, rows)

    if not args.no_markdown:
        snapshot_path = _resolve_path(repo_root, args.snapshot_md)
        pivot_path = _resolve_path(repo_root, args.pivot_md)
        snapshot_text = _render_snapshot_markdown(
            generated_date=generated_date,
            rows=rows,
            source_paths=bundle["sources"],
            out_json=args.out_json,
            out_csv=args.out_csv,
            primary_budget=int(args.primary_budget),
        )
        pivot_text = _render_pivot_markdown(
            generated_date=generated_date,
            rows=rows,
            out_json=args.out_json,
            primary_budget=int(args.primary_budget),
        )
        snapshot_path.write_text(snapshot_text.rstrip() + "\n")
        pivot_path.write_text(pivot_text.rstrip() + "\n")

    print(out_json)
    print(out_csv)
    if not args.no_markdown:
        print(_resolve_path(repo_root, args.snapshot_md))
        print(_resolve_path(repo_root, args.pivot_md))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export canonical run1 matrix rows (long format CSV/JSON) from pinned h2h artifacts."
    )
    parser.add_argument("--label-a", default=DEFAULT_LABEL_A)
    parser.add_argument("--label-b", default=DEFAULT_LABEL_B)
    parser.add_argument("--score-a", default=DEFAULT_SCORE_A)
    parser.add_argument("--score-b", default=DEFAULT_SCORE_B)
    parser.add_argument("--compare", default=DEFAULT_COMPARE)
    parser.add_argument("--assert-report", default=DEFAULT_ASSERT)
    parser.add_argument("--meta-a", default=DEFAULT_META_A)
    parser.add_argument("--meta-b", default=DEFAULT_META_B)
    parser.add_argument("--profile-a", default=DEFAULT_PROFILE_A)
    parser.add_argument("--profile-b", default=DEFAULT_PROFILE_B)
    parser.add_argument("--run-id-a", default=DEFAULT_RUN_ID_A)
    parser.add_argument("--run-id-b", default=DEFAULT_RUN_ID_B)
    parser.add_argument(
        "--feature-set-a",
        default=None,
        help=(
            "Feature-set id for tool A. If omitted, falls back to run metadata "
            "(feature_set_id) and then script default."
        ),
    )
    parser.add_argument(
        "--feature-set-b",
        default=None,
        help=(
            "Feature-set id for tool B. If omitted, falls back to run metadata "
            "(feature_set_id) and then script default."
        ),
    )
    parser.add_argument("--embedding-backend-a", default=DEFAULT_EMBEDDING_BACKEND_A)
    parser.add_argument("--embedding-model-a", default=DEFAULT_EMBEDDING_MODEL_A)
    parser.add_argument("--embedding-backend-b", default=DEFAULT_EMBEDDING_BACKEND_B)
    parser.add_argument("--embedding-model-b", default=DEFAULT_EMBEDDING_MODEL_B)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--primary-budget", type=int, default=PRIMARY_BUDGET)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--snapshot-md", default=DEFAULT_SNAPSHOT_MD)
    parser.add_argument("--pivot-md", default=DEFAULT_PIVOT_MD)
    parser.add_argument("--generated-date", default=None)
    parser.add_argument("--no-markdown", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(cmd_export(args))


if __name__ == "__main__":
    raise SystemExit(main())
