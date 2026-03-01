import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_h2h_assert.py")


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _suite() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": "h2h_llm_tldr_vs_contextplus_v1",
        "gates": {
            "head_to_head": {
                "primary_budget": 2000,
            },
            "run_validity": {
                "max_timeout_rate": 0.02,
                "max_error_rate": 0.01,
                "max_budget_violation_rate": 0.0,
            },
        },
    }


def _strict_gates(*, required_runs: int, min_pass_runs: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": "h2h_llm_tldr_vs_contextplus_v1",
        "primary_budget": 2000,
        "run_validity": {
            "max_timeout_rate": 0.02,
            "max_error_rate": 0.01,
            "max_budget_violation_rate": 0.0,
        },
        "margin": {
            "mrr_mean_delta_min": 0.05,
            "recall@5_mean_delta_min": 0.08,
            "precision@5_mean_delta_min": 0.05,
        },
        "efficiency": {
            "max_payload_tokens_median_ratio": 0.90,
            "max_latency_ms_p50_ratio": 1.10,
        },
        "stability": {
            "required_runs": required_runs,
            "min_pass_runs": min_pass_runs,
        },
    }


def _score_report(
    *,
    mrr: float = 0.8,
    recall5: float = 0.9,
    precision5: float = 0.8,
    payload_tokens_median: float = 90.0,
    latency_ms_p50: float = 100.0,
    timeout_rate: float = 0.0,
    error_rate: float = 0.0,
    budget_violation_rate: float = 0.0,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": "h2h_llm_tldr_vs_contextplus_v1",
        "rates": {
            "timeout_rate": timeout_rate,
            "error_rate": error_rate,
            "budget_violation_rate": budget_violation_rate,
        },
        "metrics": {
            "by_budget": {
                "2000": {
                    "retrieval": {
                        "mrr_mean": mrr,
                        "recall@5_mean": recall5,
                        "precision@5_mean": precision5,
                        "payload_tokens_median": payload_tokens_median,
                        "latency_ms_p50": latency_ms_p50,
                    }
                }
            }
        },
    }


def _compare_report(*, winner: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": "h2h_llm_tldr_vs_contextplus_v1",
        "winner": winner,
        "labels": {
            "a": "llm-tldr",
            "b": "contextplus",
        },
    }


def _run_assert(
    mod: dict[str, Any],
    tmp_path: Path,
    *,
    strict_gates: dict[str, Any],
    score_a_reports: list[dict[str, Any]],
    score_b_reports: list[dict[str, Any]],
    compare_reports: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    suite_path = tmp_path / "suite.json"
    strict_path = tmp_path / "strict.json"
    out_path = tmp_path / "assert.json"
    _write_json(suite_path, _suite())
    _write_json(strict_path, strict_gates)

    score_a_paths: list[str] = []
    score_b_paths: list[str] = []
    compare_paths: list[str] = []
    for i, (score_a, score_b, compare) in enumerate(
        zip(score_a_reports, score_b_reports, compare_reports, strict=False),
        start=1,
    ):
        score_a_path = tmp_path / f"score-a-{i}.json"
        score_b_path = tmp_path / f"score-b-{i}.json"
        compare_path = tmp_path / f"compare-{i}.json"
        _write_json(score_a_path, score_a)
        _write_json(score_b_path, score_b)
        _write_json(compare_path, compare)
        score_a_paths.append(str(score_a_path))
        score_b_paths.append(str(score_b_path))
        compare_paths.append(str(compare_path))

    rc = mod["cmd_assert"](
        argparse.Namespace(
            suite=str(suite_path),
            strict_gates=str(strict_path),
            score_a=score_a_paths,
            score_b=score_b_paths,
            compare=compare_paths,
            label_a="llm-tldr",
            label_b="contextplus",
            out=str(out_path),
        )
    )
    report = json.loads(out_path.read_text())
    return rc, report


def _gate_map(run_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gates = run_report.get("gate_checks")
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(gates, list):
        return out
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        name = gate.get("name")
        if isinstance(name, str):
            out[name] = gate
    return out


def test_assert_fails_when_compare_winner_is_not_llm_tldr(tmp_path: Path):
    mod = _load_mod()
    rc, report = _run_assert(
        mod,
        tmp_path,
        strict_gates=_strict_gates(required_runs=1, min_pass_runs=1),
        score_a_reports=[_score_report()],
        score_b_reports=[
            _score_report(
                mrr=0.70,
                recall5=0.80,
                precision5=0.70,
                payload_tokens_median=120.0,
                latency_ms_p50=110.0,
            )
        ],
        compare_reports=[_compare_report(winner="contextplus")],
    )

    assert rc == 2
    gate_map = _gate_map(report["runs"][0])
    assert gate_map["compare.winner_is_label_a"]["pass"] is False


def test_assert_fails_when_margin_gates_are_below_threshold(tmp_path: Path):
    mod = _load_mod()
    rc, report = _run_assert(
        mod,
        tmp_path,
        strict_gates=_strict_gates(required_runs=1, min_pass_runs=1),
        score_a_reports=[
            _score_report(
                mrr=0.74,
                recall5=0.86,
                precision5=0.74,
            )
        ],
        score_b_reports=[
            _score_report(
                mrr=0.70,
                recall5=0.80,
                precision5=0.70,
                payload_tokens_median=120.0,
                latency_ms_p50=110.0,
            )
        ],
        compare_reports=[_compare_report(winner="llm-tldr")],
    )

    assert rc == 2
    gate_map = _gate_map(report["runs"][0])
    assert gate_map["margin.mrr_mean_delta_min"]["pass"] is False
    assert gate_map["margin.recall@5_mean_delta_min"]["pass"] is False
    assert gate_map["margin.precision@5_mean_delta_min"]["pass"] is False


def test_assert_fails_on_validity_or_efficiency_gate_failure(tmp_path: Path):
    mod = _load_mod()
    rc, report = _run_assert(
        mod,
        tmp_path,
        strict_gates=_strict_gates(required_runs=1, min_pass_runs=1),
        score_a_reports=[
            _score_report(
                payload_tokens_median=114.0,
                timeout_rate=0.03,
            )
        ],
        score_b_reports=[
            _score_report(
                mrr=0.70,
                recall5=0.80,
                precision5=0.70,
                payload_tokens_median=120.0,
                latency_ms_p50=110.0,
            )
        ],
        compare_reports=[_compare_report(winner="llm-tldr")],
    )

    assert rc == 2
    gate_map = _gate_map(report["runs"][0])
    assert gate_map["validity.llm-tldr.timeout_rate"]["pass"] is False
    assert gate_map["efficiency.payload_tokens_median_ratio_max"]["pass"] is False


def test_assert_requires_stability_two_of_three_runs(tmp_path: Path):
    mod = _load_mod()
    rc, report = _run_assert(
        mod,
        tmp_path,
        strict_gates=_strict_gates(required_runs=3, min_pass_runs=2),
        score_a_reports=[
            _score_report(),
            _score_report(),
            _score_report(
                mrr=0.74,
                recall5=0.86,
                precision5=0.74,
            ),
        ],
        score_b_reports=[
            _score_report(
                mrr=0.70,
                recall5=0.80,
                precision5=0.70,
                payload_tokens_median=120.0,
                latency_ms_p50=110.0,
            ),
            _score_report(
                mrr=0.70,
                recall5=0.80,
                precision5=0.70,
                payload_tokens_median=120.0,
                latency_ms_p50=110.0,
            ),
            _score_report(
                mrr=0.70,
                recall5=0.80,
                precision5=0.70,
                payload_tokens_median=120.0,
                latency_ms_p50=110.0,
            ),
        ],
        compare_reports=[
            _compare_report(winner="llm-tldr"),
            _compare_report(winner="contextplus"),
            _compare_report(winner="llm-tldr"),
        ],
    )

    assert rc == 2
    assert report["summary"]["runs_passed"] == 1
    assert report["stability_gate"]["pass"] is False
    assert report["stability_gate"]["actual"]["runs_total"] == 3
    assert report["stability_gate"]["expected"]["min_pass_runs"] == 2


def test_assert_passes_only_when_all_strict_gates_pass(tmp_path: Path):
    mod = _load_mod()
    rc, report = _run_assert(
        mod,
        tmp_path,
        strict_gates=_strict_gates(required_runs=3, min_pass_runs=2),
        score_a_reports=[
            _score_report(),
            _score_report(),
            _score_report(),
        ],
        score_b_reports=[
            _score_report(
                mrr=0.70,
                recall5=0.80,
                precision5=0.70,
                payload_tokens_median=120.0,
                latency_ms_p50=110.0,
            ),
            _score_report(
                mrr=0.70,
                recall5=0.80,
                precision5=0.70,
                payload_tokens_median=120.0,
                latency_ms_p50=110.0,
            ),
            _score_report(
                mrr=0.70,
                recall5=0.80,
                precision5=0.70,
                payload_tokens_median=120.0,
                latency_ms_p50=110.0,
            ),
        ],
        compare_reports=[
            _compare_report(winner="llm-tldr"),
            _compare_report(winner="llm-tldr"),
            _compare_report(winner="llm-tldr"),
        ],
    )

    assert rc == 0
    assert report["gates_passed"] is True
    assert report["stability_gate"]["pass"] is True
    assert all(run["strict_gates_passed"] is True for run in report["runs"])
