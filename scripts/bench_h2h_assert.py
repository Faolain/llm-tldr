#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bench_util import bench_runs_root, get_repo_root, write_report

SCHEMA_VERSION = 1
DEFAULT_LABEL_A = "llm-tldr"
DEFAULT_LABEL_B = "contextplus"

DEFAULT_RUN_VALIDITY = {
    "max_timeout_rate": 0.02,
    "max_error_rate": 0.01,
    "max_budget_violation_rate": 0.0,
}
DEFAULT_MARGINS = {
    "mrr_mean_delta_min": 0.05,
    "recall@5_mean_delta_min": 0.08,
    "precision@5_mean_delta_min": 0.05,
}
DEFAULT_EFFICIENCY = {
    "max_payload_tokens_median_ratio": 0.90,
    "max_latency_ms_p50_ratio": 1.10,
}
DEFAULT_STABILITY = {
    "required_runs": 3,
    "min_pass_runs": 2,
}


def _read_json_obj(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
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


def _coerce_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return None
    return None


def _safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if b == 0.0:
        if a == 0.0:
            return 1.0
        return float("inf")
    return a / b


def _read_budget_retrieval(score_report: dict[str, Any], budget: int) -> dict[str, Any]:
    metrics = score_report.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    by_budget = metrics.get("by_budget")
    if not isinstance(by_budget, dict):
        return {}
    budget_obj = by_budget.get(str(budget))
    if not isinstance(budget_obj, dict):
        return {}
    retrieval = budget_obj.get("retrieval")
    if not isinstance(retrieval, dict):
        return {}
    return retrieval


def _retrieval_metric(score_report: dict[str, Any], budget: int, metric: str) -> float | None:
    retrieval = _read_budget_retrieval(score_report, budget)
    return _coerce_float(retrieval.get(metric))


def _rate(score_report: dict[str, Any], rate_name: str) -> float | None:
    rates = score_report.get("rates")
    if not isinstance(rates, dict):
        return None
    return _coerce_float(rates.get(rate_name))


def _suite_id_or_error(suite: dict[str, Any]) -> str:
    suite_id = suite.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("suite.suite_id must be a non-empty string")
    return suite_id


def _require_suite_id(doc: dict[str, Any], *, suite_id: str, source: str) -> None:
    actual = doc.get("suite_id")
    if actual != suite_id:
        raise ValueError(f"{source} suite_id mismatch: expected {suite_id!r}, got {actual!r}")


def _resolve_primary_budget(suite: dict[str, Any], strict_gates: dict[str, Any]) -> int:
    strict_budget = _coerce_int(strict_gates.get("primary_budget"))
    if strict_budget is not None and strict_budget > 0:
        return strict_budget

    gates = suite.get("gates")
    if isinstance(gates, dict):
        h2h = gates.get("head_to_head")
        if isinstance(h2h, dict):
            suite_budget = _coerce_int(h2h.get("primary_budget"))
            if suite_budget is not None and suite_budget > 0:
                return suite_budget
    return 2000


def _resolve_run_validity(suite: dict[str, Any], strict_gates: dict[str, Any]) -> dict[str, float]:
    resolved = dict(DEFAULT_RUN_VALIDITY)
    gates = suite.get("gates")
    if isinstance(gates, dict):
        suite_validity = gates.get("run_validity")
        if isinstance(suite_validity, dict):
            for k in resolved:
                v = _coerce_float(suite_validity.get(k))
                if v is not None:
                    resolved[k] = v
    strict_validity = strict_gates.get("run_validity")
    if isinstance(strict_validity, dict):
        for k in resolved:
            v = _coerce_float(strict_validity.get(k))
            if v is not None:
                resolved[k] = v
    return resolved


def _resolve_margin(strict_gates: dict[str, Any]) -> dict[str, float]:
    resolved = dict(DEFAULT_MARGINS)
    margin = strict_gates.get("margin")
    if not isinstance(margin, dict):
        return resolved

    v = _coerce_float(margin.get("mrr_mean_delta_min"))
    if v is not None:
        resolved["mrr_mean_delta_min"] = v

    v = _coerce_float(margin.get("recall@5_mean_delta_min"))
    if v is None:
        v = _coerce_float(margin.get("recall5_mean_delta_min"))
    if v is not None:
        resolved["recall@5_mean_delta_min"] = v

    v = _coerce_float(margin.get("precision@5_mean_delta_min"))
    if v is None:
        v = _coerce_float(margin.get("precision5_mean_delta_min"))
    if v is not None:
        resolved["precision@5_mean_delta_min"] = v

    return resolved


def _resolve_efficiency(strict_gates: dict[str, Any]) -> dict[str, float]:
    resolved = dict(DEFAULT_EFFICIENCY)
    efficiency = strict_gates.get("efficiency")
    if not isinstance(efficiency, dict):
        return resolved

    v = _coerce_float(efficiency.get("max_payload_tokens_median_ratio"))
    if v is not None:
        resolved["max_payload_tokens_median_ratio"] = v

    v = _coerce_float(efficiency.get("max_latency_ms_p50_ratio"))
    if v is not None:
        resolved["max_latency_ms_p50_ratio"] = v

    return resolved


def _resolve_stability(strict_gates: dict[str, Any]) -> dict[str, int]:
    resolved = dict(DEFAULT_STABILITY)
    stability = strict_gates.get("stability")
    if not isinstance(stability, dict):
        return resolved

    required_runs = _coerce_int(stability.get("required_runs"))
    min_pass_runs = _coerce_int(stability.get("min_pass_runs"))
    if required_runs is not None and required_runs > 0:
        resolved["required_runs"] = required_runs
    if min_pass_runs is not None and min_pass_runs > 0:
        resolved["min_pass_runs"] = min_pass_runs
    return resolved


def _resolve_winner_target(strict_gates: dict[str, Any], *, label_a: str, label_b: str) -> str:
    winner = strict_gates.get("winner")
    if not isinstance(winner, dict):
        return label_a

    must_equal = winner.get("must_equal")
    if must_equal == "label_a":
        return label_a
    if must_equal == "label_b":
        return label_b
    if isinstance(must_equal, str) and must_equal:
        return must_equal
    return label_a


def _add_gate(
    gates: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    actual: Any,
    expected: Any,
) -> None:
    gates.append(
        {
            "name": name,
            "pass": bool(passed),
            "actual": actual,
            "expected": expected,
        }
    )


def _evaluate_run(
    *,
    run_index: int,
    score_a: dict[str, Any],
    score_b: dict[str, Any],
    compare: dict[str, Any],
    label_a: str,
    label_b: str,
    winner_target: str,
    primary_budget: int,
    run_validity: dict[str, float],
    margin: dict[str, float],
    efficiency: dict[str, float],
) -> dict[str, Any]:
    mrr_a = _retrieval_metric(score_a, primary_budget, "mrr_mean")
    mrr_b = _retrieval_metric(score_b, primary_budget, "mrr_mean")
    recall5_a = _retrieval_metric(score_a, primary_budget, "recall@5_mean")
    recall5_b = _retrieval_metric(score_b, primary_budget, "recall@5_mean")
    precision5_a = _retrieval_metric(score_a, primary_budget, "precision@5_mean")
    precision5_b = _retrieval_metric(score_b, primary_budget, "precision@5_mean")
    payload_a = _retrieval_metric(score_a, primary_budget, "payload_tokens_median")
    payload_b = _retrieval_metric(score_b, primary_budget, "payload_tokens_median")
    latency_a = _retrieval_metric(score_a, primary_budget, "latency_ms_p50")
    latency_b = _retrieval_metric(score_b, primary_budget, "latency_ms_p50")

    mrr_delta = None if mrr_a is None or mrr_b is None else mrr_a - mrr_b
    recall5_delta = None if recall5_a is None or recall5_b is None else recall5_a - recall5_b
    precision5_delta = None if precision5_a is None or precision5_b is None else precision5_a - precision5_b

    payload_ratio = _safe_ratio(payload_a, payload_b)
    latency_ratio = _safe_ratio(latency_a, latency_b)

    rates_a = {
        "timeout_rate": _rate(score_a, "timeout_rate"),
        "error_rate": _rate(score_a, "error_rate"),
        "budget_violation_rate": _rate(score_a, "budget_violation_rate"),
    }
    rates_b = {
        "timeout_rate": _rate(score_b, "timeout_rate"),
        "error_rate": _rate(score_b, "error_rate"),
        "budget_violation_rate": _rate(score_b, "budget_violation_rate"),
    }

    gates: list[dict[str, Any]] = []

    winner = compare.get("winner")
    _add_gate(
        gates,
        name="compare.winner_is_label_a",
        passed=(winner == winner_target),
        actual=winner,
        expected=winner_target,
    )

    _add_gate(
        gates,
        name="margin.mrr_mean_delta_min",
        passed=(mrr_delta is not None and mrr_delta >= margin["mrr_mean_delta_min"]),
        actual=mrr_delta,
        expected=f">= {margin['mrr_mean_delta_min']}",
    )
    _add_gate(
        gates,
        name="margin.recall@5_mean_delta_min",
        passed=(recall5_delta is not None and recall5_delta >= margin["recall@5_mean_delta_min"]),
        actual=recall5_delta,
        expected=f">= {margin['recall@5_mean_delta_min']}",
    )
    _add_gate(
        gates,
        name="margin.precision@5_mean_delta_min",
        passed=(precision5_delta is not None and precision5_delta >= margin["precision@5_mean_delta_min"]),
        actual=precision5_delta,
        expected=f">= {margin['precision@5_mean_delta_min']}",
    )

    for label, rates in ((label_a, rates_a), (label_b, rates_b)):
        for rate_name, gate_key in (
            ("timeout_rate", "max_timeout_rate"),
            ("error_rate", "max_error_rate"),
            ("budget_violation_rate", "max_budget_violation_rate"),
        ):
            threshold = run_validity[gate_key]
            value = rates[rate_name]
            _add_gate(
                gates,
                name=f"validity.{label}.{rate_name}",
                passed=(value is not None and value <= threshold),
                actual=value,
                expected=f"<= {threshold}",
            )

    _add_gate(
        gates,
        name="efficiency.payload_tokens_median_ratio_max",
        passed=(
            payload_ratio is not None
            and payload_ratio <= efficiency["max_payload_tokens_median_ratio"]
        ),
        actual=payload_ratio,
        expected=f"<= {efficiency['max_payload_tokens_median_ratio']}",
    )
    _add_gate(
        gates,
        name="efficiency.latency_ms_p50_ratio_max",
        passed=(latency_ratio is not None and latency_ratio <= efficiency["max_latency_ms_p50_ratio"]),
        actual=latency_ratio,
        expected=f"<= {efficiency['max_latency_ms_p50_ratio']}",
    )

    strict_gates_passed = all(g["pass"] for g in gates)

    return {
        "run_index": int(run_index),
        "winner": winner,
        "metrics": {
            "a": {
                "mrr_mean": mrr_a,
                "recall@5_mean": recall5_a,
                "precision@5_mean": precision5_a,
                "payload_tokens_median": payload_a,
                "latency_ms_p50": latency_a,
            },
            "b": {
                "mrr_mean": mrr_b,
                "recall@5_mean": recall5_b,
                "precision@5_mean": precision5_b,
                "payload_tokens_median": payload_b,
                "latency_ms_p50": latency_b,
            },
        },
        "deltas": {
            "mrr_mean": mrr_delta,
            "recall@5_mean": recall5_delta,
            "precision@5_mean": precision5_delta,
        },
        "ratios": {
            "payload_tokens_median": payload_ratio,
            "latency_ms_p50": latency_ratio,
        },
        "rates": {
            label_a: rates_a,
            label_b: rates_b,
        },
        "gate_checks": gates,
        "strict_gates_passed": strict_gates_passed,
    }


def _build_assert_report(
    *,
    suite: dict[str, Any],
    strict_gates: dict[str, Any],
    score_a_reports: list[dict[str, Any]],
    score_b_reports: list[dict[str, Any]],
    compare_reports: list[dict[str, Any]],
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    if not score_a_reports or not score_b_reports or not compare_reports:
        raise ValueError("at least one score-a, score-b, and compare report is required")
    if len(score_a_reports) != len(score_b_reports) or len(score_a_reports) != len(compare_reports):
        raise ValueError("score-a, score-b, and compare inputs must have identical counts")

    suite_id = _suite_id_or_error(suite)
    strict_suite_id = strict_gates.get("suite_id")
    if isinstance(strict_suite_id, str) and strict_suite_id and strict_suite_id != suite_id:
        raise ValueError(
            f"strict gates suite_id mismatch: expected {suite_id!r}, got {strict_suite_id!r}"
        )

    primary_budget = _resolve_primary_budget(suite, strict_gates)
    run_validity = _resolve_run_validity(suite, strict_gates)
    margin = _resolve_margin(strict_gates)
    efficiency = _resolve_efficiency(strict_gates)
    stability = _resolve_stability(strict_gates)
    winner_target = _resolve_winner_target(strict_gates, label_a=label_a, label_b=label_b)

    runs: list[dict[str, Any]] = []
    for i, (score_a, score_b, compare) in enumerate(
        zip(score_a_reports, score_b_reports, compare_reports, strict=False),
        start=1,
    ):
        _require_suite_id(score_a, suite_id=suite_id, source=f"score_a[{i}]")
        _require_suite_id(score_b, suite_id=suite_id, source=f"score_b[{i}]")
        _require_suite_id(compare, suite_id=suite_id, source=f"compare[{i}]")

        run = _evaluate_run(
            run_index=i,
            score_a=score_a,
            score_b=score_b,
            compare=compare,
            label_a=label_a,
            label_b=label_b,
            winner_target=winner_target,
            primary_budget=primary_budget,
            run_validity=run_validity,
            margin=margin,
            efficiency=efficiency,
        )
        runs.append(run)

    runs_total = len(runs)
    runs_passed = sum(1 for r in runs if r["strict_gates_passed"])
    required_runs = stability["required_runs"]
    min_pass_runs = stability["min_pass_runs"]

    stability_passed = runs_total >= required_runs and runs_passed >= min_pass_runs
    stability_gate: dict[str, Any] = {
        "name": "stability.two_of_three",
        "pass": stability_passed,
        "actual": {
            "runs_total": runs_total,
            "runs_passed": runs_passed,
        },
        "expected": {
            "required_runs": required_runs,
            "min_pass_runs": min_pass_runs,
        },
    }
    if runs_total < required_runs:
        stability_gate["reason"] = "insufficient_runs_for_stability_check"

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "phase6_h2h_strict_assert",
        "suite_id": suite_id,
        "labels": {
            "a": label_a,
            "b": label_b,
        },
        "primary_budget": primary_budget,
        "strict_thresholds": {
            "run_validity": run_validity,
            "winner_target": winner_target,
            "margin": margin,
            "efficiency": efficiency,
            "stability": stability,
        },
        "runs": runs,
        "summary": {
            "runs_total": runs_total,
            "runs_passed": runs_passed,
            "runs_failed": runs_total - runs_passed,
        },
        "stability_gate": stability_gate,
        "gates_passed": bool(stability_gate["pass"]),
    }


def cmd_assert(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    suite_path = Path(args.suite).resolve()
    strict_gates_path = Path(args.strict_gates).resolve()
    score_a_paths = [Path(p).resolve() for p in args.score_a]
    score_b_paths = [Path(p).resolve() for p in args.score_b]
    compare_paths = [Path(p).resolve() for p in args.compare]

    suite = _read_json_obj(suite_path)
    strict_gates = _read_json_obj(strict_gates_path)
    score_a_reports = [_read_json_obj(path) for path in score_a_paths]
    score_b_reports = [_read_json_obj(path) for path in score_b_paths]
    compare_reports = [_read_json_obj(path) for path in compare_paths]

    report = _build_assert_report(
        suite=suite,
        strict_gates=strict_gates,
        score_a_reports=score_a_reports,
        score_b_reports=score_b_reports,
        compare_reports=compare_reports,
        label_a=str(args.label_a),
        label_b=str(args.label_b),
    )
    report["inputs"] = {
        "suite": str(suite_path),
        "strict_gates": str(strict_gates_path),
        "score_a": [str(p) for p in score_a_paths],
        "score_b": [str(p) for p in score_b_paths],
        "compare": [str(p) for p in compare_paths],
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = bench_runs_root(repo_root) / "h2h-assert-strict.json"
    write_report(out_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gates_passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assert strict head-to-head superiority gates.")
    parser.add_argument("--suite", default="benchmarks/head_to_head/suite.v1.json")
    parser.add_argument(
        "--score-a",
        action="append",
        required=True,
        help="Path to llm-tldr score report. Repeat for multi-run assertions.",
    )
    parser.add_argument(
        "--score-b",
        action="append",
        required=True,
        help="Path to contextplus score report. Repeat for multi-run assertions.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        required=True,
        help="Path to compare report. Repeat for multi-run assertions.",
    )
    parser.add_argument("--label-a", default=DEFAULT_LABEL_A)
    parser.add_argument("--label-b", default=DEFAULT_LABEL_B)
    parser.add_argument(
        "--strict-gates",
        required=True,
        help="Path to strict gate thresholds JSON.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path for deterministic JSON diagnostics output.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(cmd_assert(args))


if __name__ == "__main__":
    raise SystemExit(main())
