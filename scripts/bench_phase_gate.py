#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from bench_agentic_common import load_json_obj, sha256_file
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


def _first_number(obj: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _coerce_float(obj.get(key))
        if value is not None:
            return value
    return None


def _results(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results")
    if isinstance(results, dict):
        return results
    return report


def _task_count_from_phase_a(results: dict[str, Any]) -> int:
    direct = _coerce_int(results.get("tasks_scored"))
    if direct is not None:
        return direct
    direct = _coerce_int(results.get("tasks_completed"))
    if direct is not None:
        return direct
    per_task = results.get("per_task")
    if isinstance(per_task, list):
        return len([row for row in per_task if isinstance(row, dict)])
    return 0


def _phase_a_total_attempts(results: dict[str, Any]) -> int:
    per_task = results.get("per_task")
    if not isinstance(per_task, list):
        return 0
    attempts = 0
    for task in per_task:
        if not isinstance(task, dict):
            continue
        variants = task.get("variants")
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            trials = variant.get("trials")
            if isinstance(trials, list):
                attempts += len([row for row in trials if isinstance(row, dict)])
    return attempts


def _judge_bad_json(results: dict[str, Any]) -> int:
    bad_json = _coerce_int(results.get("judge_bad_json"))
    if bad_json is not None:
        return bad_json
    empty_total = _coerce_int(results.get("judge_empty_verdict_total")) or 0
    malformed_total = _coerce_int(results.get("judge_malformed_verdict_total")) or 0
    return empty_total + malformed_total


def _phase_a_error_rate(results: dict[str, Any]) -> float | None:
    attempts = _phase_a_total_attempts(results)
    if attempts <= 0:
        tasks = _task_count_from_phase_a(results)
        if tasks <= 0:
            return None
        attempts = tasks
    errors = (_coerce_int(results.get("errors_total")) or 0) + (_coerce_int(results.get("bad_json")) or 0)
    return errors / attempts


def _evaluate_phase_a(report: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    results = _results(report)
    gates: list[dict[str, Any]] = []
    win_rate = _coerce_float(results.get("win_rate_tldr_over_rg"))
    f1_mean = results.get("f1_mean")
    f1_tldr = _first_number(f1_mean, "tldr") if isinstance(f1_mean, dict) else None
    f1_rg = _first_number(f1_mean, "rg") if isinstance(f1_mean, dict) else None
    f1_delta = None if f1_tldr is None or f1_rg is None else f1_tldr - f1_rg
    error_rate = _phase_a_error_rate(results)
    tasks_completed = _task_count_from_phase_a(results)

    if "win_rate_tldr_over_rg_min" in gate_cfg:
        threshold = _coerce_float(gate_cfg.get("win_rate_tldr_over_rg_min"))
        _add_gate(
            gates,
            name="win_rate_tldr_over_rg_min",
            actual=win_rate,
            expected=threshold,
            passed=(win_rate is not None and threshold is not None and win_rate >= threshold),
        )
    if "f1_mean_tldr_min_delta" in gate_cfg:
        threshold = _coerce_float(gate_cfg.get("f1_mean_tldr_min_delta"))
        _add_gate(
            gates,
            name="f1_mean_tldr_min_delta",
            actual=f1_delta,
            expected=threshold,
            passed=(f1_delta is not None and threshold is not None and f1_delta >= threshold),
        )
    if "max_error_rate" in gate_cfg:
        threshold = _coerce_float(gate_cfg.get("max_error_rate"))
        _add_gate(
            gates,
            name="max_error_rate",
            actual=error_rate,
            expected=threshold,
            passed=(error_rate is not None and threshold is not None and error_rate <= threshold),
        )
    if "min_tasks_completed" in gate_cfg:
        threshold = _coerce_int(gate_cfg.get("min_tasks_completed"))
        _add_gate(
            gates,
            name="min_tasks_completed",
            actual=tasks_completed,
            expected=threshold,
            passed=(threshold is not None and tasks_completed >= threshold),
        )

    return {
        "phase": "phase_a",
        "summary": {
            "win_rate_tldr_over_rg": win_rate,
            "f1_mean_tldr": f1_tldr,
            "f1_mean_rg": f1_rg,
            "f1_mean_delta": f1_delta,
            "error_rate": error_rate,
            "tasks_completed": tasks_completed,
        },
        "gates": gates,
    }


def _evaluate_preflight(report: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    metrics_source = report.get("metrics") if isinstance(report.get("metrics"), dict) else _results(report).get("metrics")
    metrics = metrics_source if isinstance(metrics_source, dict) else {}
    gates: list[dict[str, Any]] = []

    for gate_name, metric_key, cmp in (
        ("correct_first_tool_min", "correct_first_tool_rate", "min"),
        ("workflow_compliance_min", "workflow_compliance_rate", "min"),
        ("tldrf_usage_on_required_min", "tldrf_usage_on_required_rate", "min"),
        ("rg_first_on_exact_min", "rg_first_on_exact_rate", "min"),
        ("median_dead_end_turns_max", "median_turns_before_first_appropriate_tool_use", "max"),
    ):
        if gate_name not in gate_cfg:
            continue
        actual = _coerce_float(metrics.get(metric_key))
        threshold = _coerce_float(gate_cfg.get(gate_name))
        passed = (
            actual is not None
            and threshold is not None
            and (actual >= threshold if cmp == "min" else actual <= threshold)
        )
        _add_gate(gates, name=gate_name, actual=actual, expected=threshold, passed=passed)

    if "systematic_failure_detected_must_be" in gate_cfg:
        actual = _coerce_bool(metrics.get("systematic_failure_detected"))
        threshold = _coerce_bool(gate_cfg.get("systematic_failure_detected_must_be"))
        _add_gate(
            gates,
            name="systematic_failure_detected_must_be",
            actual=actual,
            expected=threshold,
            passed=(actual is not None and threshold is not None and actual == threshold),
        )

    return {"phase": "preflight", "summary": metrics, "gates": gates}


def _category_metrics(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    for key in ("by_category", "per_category", "categories"):
        value = results.get(key)
        if isinstance(value, dict):
            return {str(name): data for name, data in value.items() if isinstance(data, dict)}
    return {}


def _category_augmented_win_rate(obj: dict[str, Any]) -> float | None:
    value = _first_number(
        obj,
        "win_rate_augmented_over_baseline",
        "win_rate_tldrf_over_rg",
        "augmented_win_rate",
        "win_rate_augmented",
    )
    if value is not None:
        return value
    baseline = _first_number(obj, "baseline_win_rate", "win_rate_baseline")
    if baseline is not None:
        return 1.0 - baseline
    return None


def _category_task_count(obj: dict[str, Any]) -> int | None:
    for key in ("tasks_completed", "task_count", "tasks"):
        value = _coerce_int(obj.get(key))
        if value is not None:
            return value
    return None


def _evaluate_phase_c(report: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    results = _results(report)
    categories = _category_metrics(results)
    expectations = gate_cfg.get("category_expectations") if isinstance(gate_cfg.get("category_expectations"), dict) else {}
    min_win_rate = _coerce_float(gate_cfg.get("category_win_rate_min")) or _coerce_float(gate_cfg.get("win_rate_min"))
    min_tasks = _coerce_int(gate_cfg.get("min_tasks_per_category"))
    max_error_rate = _coerce_float(gate_cfg.get("max_error_rate"))
    overall_error_rate = _first_number(results, "error_rate", "overall_error_rate")
    gates: list[dict[str, Any]] = []
    category_summary: dict[str, Any] = {}

    for category_name, expected_winner in expectations.items():
        cat_obj = categories.get(str(category_name), {})
        augmented_win = _category_augmented_win_rate(cat_obj)
        baseline_win = None if augmented_win is None else 1.0 - augmented_win
        if expected_winner == "augmented":
            actual_win = augmented_win
        elif expected_winner == "baseline":
            actual_win = baseline_win
        else:
            actual_win = max(v for v in (augmented_win, baseline_win) if v is not None) if (
                augmented_win is not None or baseline_win is not None
            ) else None
        task_count = _category_task_count(cat_obj)
        category_summary[str(category_name)] = {
            "expected_winner": expected_winner,
            "augmented_win_rate": augmented_win,
            "baseline_win_rate": baseline_win,
            "selected_win_rate": actual_win,
            "tasks_completed": task_count,
        }
        if min_win_rate is not None:
            _add_gate(
                gates,
                name=f"category.{category_name}.win_rate_min",
                actual=actual_win,
                expected=min_win_rate,
                passed=(actual_win is not None and actual_win >= min_win_rate),
            )
        if min_tasks is not None:
            _add_gate(
                gates,
                name=f"category.{category_name}.min_tasks_per_category",
                actual=task_count,
                expected=min_tasks,
                passed=(task_count is not None and task_count >= min_tasks),
            )

    if max_error_rate is not None:
        _add_gate(
            gates,
            name="max_error_rate",
            actual=overall_error_rate,
            expected=max_error_rate,
            passed=(overall_error_rate is not None and overall_error_rate <= max_error_rate),
        )

    return {"phase": "phase_c", "summary": {"categories": category_summary, "error_rate": overall_error_rate}, "gates": gates}


def _arm_obj(results: dict[str, Any], arm_name: str) -> dict[str, Any]:
    arms = results.get("arms")
    if isinstance(arms, dict):
        value = arms.get(arm_name)
        if isinstance(value, dict):
            return value
    value = results.get(arm_name)
    if isinstance(value, dict):
        return value
    return {}


def _arm_total_tokens(arm: dict[str, Any]) -> float | None:
    direct = _first_number(arm, "total_tokens", "tokens_total", "median_total_tokens")
    if direct is not None:
        return direct
    input_tokens = _first_number(arm, "total_input_tokens", "input_tokens_total")
    output_tokens = _first_number(arm, "total_output_tokens", "output_tokens_total")
    if input_tokens is None and output_tokens is None:
        return None
    return (input_tokens or 0.0) + (output_tokens or 0.0)


def _arm_turn_metric(arm: dict[str, Any]) -> float | None:
    return _first_number(arm, "median_turn_count", "turn_count_median", "turn_count_mean", "turn_count")


def _arm_time_metric(arm: dict[str, Any]) -> float | None:
    return _first_number(arm, "median_wall_clock_s", "wall_clock_s_median", "wall_clock_s_mean", "wall_clock_s")


def _arm_solve_rate(arm: dict[str, Any]) -> float | None:
    return _first_number(arm, "solve_rate", "solve_rate_mean")


def _arm_tasks_completed(arm: dict[str, Any]) -> int | None:
    for key in ("tasks_completed", "task_count"):
        value = _coerce_int(arm.get(key))
        if value is not None:
            return value
    return None


def _pct_reduction(baseline: float | None, augmented: float | None) -> float | None:
    if baseline is None or augmented is None:
        return None
    if baseline == 0.0:
        return 0.0 if augmented == 0.0 else None
    return ((baseline - augmented) / baseline) * 100.0


def _paired_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    per_task = results.get("per_task")
    if isinstance(per_task, list):
        return [row for row in per_task if isinstance(row, dict)]
    return []


def _paired_metric_values(results: dict[str, Any], key: str) -> tuple[list[float], list[float]]:
    baseline_values: list[float] = []
    augmented_values: list[float] = []
    for row in _paired_rows(results):
        baseline = row.get("baseline")
        augmented = row.get("augmented")
        if not isinstance(baseline, dict) or not isinstance(augmented, dict):
            continue
        if key == "tokens":
            base_value = _arm_total_tokens(baseline)
            aug_value = _arm_total_tokens(augmented)
        elif key == "turns":
            base_value = _arm_turn_metric(baseline)
            aug_value = _arm_turn_metric(augmented)
        elif key == "time":
            base_value = _arm_time_metric(baseline)
            aug_value = _arm_time_metric(augmented)
        else:
            base_value = _first_number(baseline, key)
            aug_value = _first_number(augmented, key)
        if base_value is None or aug_value is None:
            continue
        baseline_values.append(base_value)
        augmented_values.append(aug_value)
    return baseline_values, augmented_values


def _require_scipy():
    try:
        from scipy import stats
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for significance testing but is not available") from exc
    return stats


def _mcnemar_pvalue(results: dict[str, Any]) -> tuple[float | None, dict[str, int]]:
    stats = _require_scipy()
    better_aug = 0
    better_base = 0
    for row in _paired_rows(results):
        baseline = row.get("baseline")
        augmented = row.get("augmented")
        if not isinstance(baseline, dict) or not isinstance(augmented, dict):
            continue
        base_solve = _first_number(baseline, "solve_rate")
        aug_solve = _first_number(augmented, "solve_rate")
        if base_solve is None or aug_solve is None:
            continue
        if aug_solve > base_solve:
            better_aug += 1
        elif base_solve > aug_solve:
            better_base += 1
    discordant = better_aug + better_base
    if discordant == 0:
        return 1.0, {"augmented_better": better_aug, "baseline_better": better_base}
    test = stats.binomtest(min(better_aug, better_base), n=discordant, p=0.5, alternative="two-sided")
    return float(test.pvalue), {"augmented_better": better_aug, "baseline_better": better_base}


def _wilcoxon_pvalue(results: dict[str, Any], key: str) -> float | None:
    stats = _require_scipy()
    baseline, augmented = _paired_metric_values(results, key)
    if not baseline:
        return None
    if len(baseline) != len(augmented):
        return None
    if all(abs(a - b) < 1e-12 for a, b in zip(baseline, augmented)):
        return 1.0
    test = stats.wilcoxon(baseline, augmented, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(test.pvalue)


def _collect_key_values(obj: Any, target_key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key:
                values.append(value)
            values.extend(_collect_key_values(value, target_key))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(_collect_key_values(item, target_key))
    return values


def _comparison_summary(results: dict[str, Any]) -> dict[str, Any]:
    baseline = _arm_obj(results, "baseline")
    augmented = _arm_obj(results, "augmented")
    solve_rate_baseline = _arm_solve_rate(baseline)
    solve_rate_augmented = _arm_solve_rate(augmented)
    solve_rate_delta = (
        None if solve_rate_baseline is None or solve_rate_augmented is None else solve_rate_augmented - solve_rate_baseline
    )
    token_reduction_pct = _pct_reduction(_arm_total_tokens(baseline), _arm_total_tokens(augmented))
    turn_reduction_pct = _pct_reduction(_arm_turn_metric(baseline), _arm_turn_metric(augmented))
    time_reduction_pct = _pct_reduction(_arm_time_metric(baseline), _arm_time_metric(augmented))
    task_counts = [value for value in (_arm_tasks_completed(baseline), _arm_tasks_completed(augmented)) if value is not None]
    tasks_completed = min(task_counts) if task_counts else len(_paired_rows(results))
    return {
        "solve_rate_baseline": solve_rate_baseline,
        "solve_rate_augmented": solve_rate_augmented,
        "solve_rate_delta": solve_rate_delta,
        "token_reduction_pct": token_reduction_pct,
        "turn_reduction_pct": turn_reduction_pct,
        "time_reduction_pct": time_reduction_pct,
        "tasks_completed": tasks_completed,
    }


def _evaluate_phase_d(report: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    results = _results(report)
    summary = _comparison_summary(results)
    gates: list[dict[str, Any]] = []
    reduction_details: dict[str, dict[str, Any]] = {}
    solve_rate_delta_min = _coerce_float(gate_cfg.get("solve_rate_delta_min"))
    if solve_rate_delta_min is not None:
        _add_gate(
            gates,
            name="solve_rate_delta_min",
            actual=summary["solve_rate_delta"],
            expected=solve_rate_delta_min,
            passed=(
                summary["solve_rate_delta"] is not None and summary["solve_rate_delta"] >= solve_rate_delta_min
            ),
        )
    if summary["solve_rate_delta"] is not None and abs(summary["solve_rate_delta"]) <= 0.05:
        reductions: list[tuple[str, str]] = [
            ("token_reduction_min_pct", "token_reduction_pct"),
            ("turn_reduction_min_pct", "turn_reduction_pct"),
            ("time_reduction_min_pct", "time_reduction_pct"),
        ]
        reduction_passes: list[bool] = []
        for gate_name, metric_key in reductions:
            if gate_name not in gate_cfg:
                continue
            threshold = _coerce_float(gate_cfg.get(gate_name))
            actual = _coerce_float(summary.get(metric_key))
            passed = actual is not None and threshold is not None and actual >= threshold
            reduction_passes.append(passed)
            reduction_details[gate_name] = {"actual": actual, "expected": threshold, "pass": passed}
        if reduction_passes:
            _add_gate(
                gates,
                name="close_solve_rate_requires_one_reduction",
                actual=any(reduction_passes),
                expected=True,
                passed=any(reduction_passes),
            )
    if "min_tasks_completed" in gate_cfg:
        threshold = _coerce_int(gate_cfg.get("min_tasks_completed"))
        _add_gate(
            gates,
            name="min_tasks_completed",
            actual=summary["tasks_completed"],
            expected=threshold,
            passed=(threshold is not None and summary["tasks_completed"] >= threshold),
        )
    if reduction_details:
        summary["reduction_details"] = reduction_details
    return {"phase": "phase_d", "summary": summary, "gates": gates}


def _evaluate_phase_e(report: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    results = _results(report)
    summary = _comparison_summary(results)
    gates: list[dict[str, Any]] = []
    absolute_delta = _coerce_float(gate_cfg.get("solve_rate_augmented_min_delta")) or 0.05
    relaxed_delta = _coerce_float(gate_cfg.get("solve_rate_delta_min_relaxed")) or -0.02
    token_reduction_min = _coerce_float(gate_cfg.get("token_reduction_min_pct")) or 20.0

    option_a = (
        summary["solve_rate_baseline"] is not None
        and summary["solve_rate_augmented"] is not None
        and summary["solve_rate_augmented"] >= summary["solve_rate_baseline"] + absolute_delta
    )
    option_b = (
        summary["solve_rate_delta"] is not None
        and summary["token_reduction_pct"] is not None
        and summary["solve_rate_delta"] >= relaxed_delta
        and summary["token_reduction_pct"] >= token_reduction_min
    )
    _add_gate(gates, name="phase_e_primary_condition", actual={"option_a": option_a, "option_b": option_b}, expected="option_a OR option_b", passed=(option_a or option_b))
    if "min_tasks_completed" in gate_cfg:
        threshold = _coerce_int(gate_cfg.get("min_tasks_completed"))
        _add_gate(
            gates,
            name="min_tasks_completed",
            actual=summary["tasks_completed"],
            expected=threshold,
            passed=(threshold is not None and summary["tasks_completed"] >= threshold),
        )
    mcnemar_p, contingency = _mcnemar_pvalue(results)
    token_p = _wilcoxon_pvalue(results, "tokens")
    time_p = _wilcoxon_pvalue(results, "time")
    summary["statistics"] = {
        "solve_rate_mcnemar_pvalue": mcnemar_p,
        "solve_rate_contingency": contingency,
        "token_wilcoxon_pvalue": token_p,
        "time_wilcoxon_pvalue": time_p,
    }
    return {"phase": "phase_e", "summary": summary, "gates": gates}


def _evaluate_phase_f(report: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    results = _results(report)
    runs = results.get("runs")
    run_rows = [row for row in runs if isinstance(row, dict)] if isinstance(runs, list) else []
    gates: list[dict[str, Any]] = []
    required_runs = _coerce_int(gate_cfg.get("required_runs")) or 3
    min_pass_runs = _coerce_int(gate_cfg.get("min_pass_runs")) or 2
    pass_count = sum(1 for row in run_rows if bool(row.get("pass") or row.get("gate_pass") or row.get("phase_pass")))
    tasks_completed = _coerce_int(results.get("tasks_completed")) or len(_paired_rows(results))
    _add_gate(
        gates,
        name="required_runs",
        actual=len(run_rows),
        expected=required_runs,
        passed=len(run_rows) >= required_runs,
    )
    _add_gate(
        gates,
        name="min_pass_runs",
        actual=pass_count,
        expected=min_pass_runs,
        passed=pass_count >= min_pass_runs,
    )
    if "min_tasks_completed" in gate_cfg:
        threshold = _coerce_int(gate_cfg.get("min_tasks_completed"))
        _add_gate(
            gates,
            name="min_tasks_completed",
            actual=tasks_completed,
            expected=threshold,
            passed=(threshold is not None and tasks_completed >= threshold),
        )
    solve_p = _coerce_float(results.get("solve_rate_mcnemar_pvalue"))
    if solve_p is None:
        solve_p, _ = _mcnemar_pvalue(results)
    token_p = _coerce_float(results.get("token_wilcoxon_pvalue"))
    if token_p is None:
        token_p = _wilcoxon_pvalue(results, "tokens")
    time_p = _coerce_float(results.get("time_wilcoxon_pvalue"))
    if time_p is None:
        time_p = _wilcoxon_pvalue(results, "time")
    sig_pass = any(p is not None and p < 0.05 for p in (solve_p, token_p, time_p))
    _add_gate(
        gates,
        name="significance_required",
        actual={"solve_rate_p": solve_p, "token_p": token_p, "time_p": time_p},
        expected="any p < 0.05",
        passed=sig_pass,
    )
    return {
        "phase": "phase_f",
        "summary": {
            "runs_evaluated": len(run_rows),
            "pass_count": pass_count,
            "tasks_completed": tasks_completed,
            "solve_rate_mcnemar_pvalue": solve_p,
            "token_wilcoxon_pvalue": token_p,
            "time_wilcoxon_pvalue": time_p,
        },
        "gates": gates,
    }


def _enforce_judge_config_hash(report: dict[str, Any], diagnostics: dict[str, Any], gate_cfg: dict[str, Any]) -> None:
    unique_hashes = {
        value
        for value in _collect_key_values(_results(report), "judge_config_hash")
        if isinstance(value, str) and value.strip()
    }
    if len(unique_hashes) <= 1 and not gate_cfg.get("enforce_judge_config_hash_match"):
        return
    _add_gate(
        diagnostics["gates"],
        name="judge_config_hash_match",
        actual=sorted(unique_hashes),
        expected="single unique judge_config_hash",
        passed=len(unique_hashes) <= 1,
    )
    diagnostics.setdefault("summary", {})["judge_config_hashes"] = sorted(unique_hashes)


def evaluate_report(report: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    gate_keys = set(gate_cfg.keys())
    if {"correct_first_tool_min", "workflow_compliance_min"} & gate_keys:
        diagnostics = _evaluate_preflight(report, gate_cfg)
    elif {"win_rate_tldr_over_rg_min", "f1_mean_tldr_min_delta"} & gate_keys:
        diagnostics = _evaluate_phase_a(report, gate_cfg)
    elif "category_expectations" in gate_cfg:
        diagnostics = _evaluate_phase_c(report, gate_cfg)
    elif "min_pass_runs" in gate_cfg or "required_runs" in gate_cfg:
        diagnostics = _evaluate_phase_f(report, gate_cfg)
    elif "solve_rate_delta_min" in gate_cfg:
        diagnostics = _evaluate_phase_d(report, gate_cfg)
    elif {"solve_rate_delta_min_relaxed", "solve_rate_augmented_min_delta"} & gate_keys:
        diagnostics = _evaluate_phase_e(report, gate_cfg)
    else:
        raise ValueError(f"Unsupported gate config shape: {sorted(gate_keys)}")

    _enforce_judge_config_hash(report, diagnostics, gate_cfg)
    diagnostics["pass"] = all(bool(row["pass"]) for row in diagnostics["gates"])
    return diagnostics


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate benchmark phase gates from a report JSON.")
    ap.add_argument("--report", required=True, help="Path to benchmark report JSON.")
    ap.add_argument("--gates", required=True, help="Path to phase gate JSON.")
    ap.add_argument("--out", default=None, help="Write diagnostic JSON here (default under benchmark/runs/).")
    args = ap.parse_args()

    repo_root = get_repo_root()
    report_path = Path(args.report).resolve()
    gates_path = Path(args.gates).resolve()
    report = load_json_obj(report_path)
    gate_cfg = load_json_obj(gates_path)
    diagnostics = evaluate_report(report, gate_cfg)
    diagnostics.update(
        {
            "schema_version": SCHEMA_VERSION,
            "report_path": str(report_path),
            "gate_config_path": str(gates_path),
            "report_hash": sha256_file(report_path),
            "gate_config_hash": sha256_file(gates_path),
        }
    )

    out_path = (
        Path(args.out).resolve()
        if args.out
        else bench_runs_root(repo_root) / f"{now_utc_compact()}-phase-gate.json"
    )
    write_report(out_path, diagnostics)
    print(out_path)
    return 0 if diagnostics["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
