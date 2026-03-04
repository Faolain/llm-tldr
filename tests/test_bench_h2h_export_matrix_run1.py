import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_h2h_export_matrix_run1.py")


def _score_report(*, tool_id: str, tool_sha: str, retrieval_mrr_2000: float) -> dict[str, object]:
    return {
        "suite_id": "h2h_llm_tldr_vs_contextplus_v1",
        "gates_passed": tool_id == "llm-tldr",
        "inputs": {
            "task_manifest_sha256": "task-hash",
            "suite_sha256": "suite-hash",
            "tokenizer": "cl100k_base",
        },
        "meta": {
            "tldr_git_sha": tool_sha,
            "tldr_git_describe": f"{tool_sha[:7]}-dirty",
            "corpus_id": None,
            "corpus_git_sha": None,
        },
        "rates": {
            "timeout_rate": 0.0,
            "error_rate": 0.0 if tool_id == "llm-tldr" else 0.43,
            "unsupported_rate": 0.0,
            "budget_violation_rate": 0.0,
            "common_lane_coverage": 1.0 if tool_id == "llm-tldr" else 0.9972222222222222,
            "capability_coverage": 1.0 if tool_id == "llm-tldr" else 0.5698412698412698,
        },
        "parse_errors": [],
        "diagnostics": {
            "result_shape_counters": {
                "total": 0,
                "non_object_result": 0,
                "empty_result_object": 0,
                "category_shape_mismatch": 0,
            }
        },
        "metrics": {
            "by_budget": {
                "500": {
                    "retrieval": {
                        "mrr_mean": retrieval_mrr_2000 - 0.01,
                        "recall@5_mean": 0.1,
                        "recall@10_mean": 0.2,
                        "precision@5_mean": 0.3,
                        "precision@10_mean": 0.4,
                        "fpr@5_mean": 0.0,
                        "fpr@10_mean": 0.0,
                        "payload_tokens_median": 50.0,
                        "payload_bytes_median": 200.0,
                        "latency_ms_p50": 4000.0,
                    },
                    "impact": {"f1_mean": 0.8, "precision_mean": 0.7, "recall_mean": 0.9},
                    "slice": {
                        "recall_mean": 0.88,
                        "precision_mean": 1.0,
                        "f1_mean": 0.92,
                        "noise_reduction_mean": 0.65,
                    },
                    "data_flow": {"origin_accuracy_mean": 1.0, "flow_completeness_mean": 1.0},
                    "complexity": {"mae": 1.8, "kendall_tau_b": None},
                },
                "2000": {
                    "retrieval": {
                        "mrr_mean": retrieval_mrr_2000,
                        "recall@5_mean": 0.789,
                        "recall@10_mean": 0.807,
                        "precision@5_mean": 0.158,
                        "precision@10_mean": 0.081,
                        "fpr@5_mean": 0.0 if tool_id == "llm-tldr" else 1.0,
                        "fpr@10_mean": 0.0 if tool_id == "llm-tldr" else 1.0,
                        "payload_tokens_median": 53.5 if tool_id == "llm-tldr" else 329.0,
                        "payload_bytes_median": 247.0 if tool_id == "llm-tldr" else 1134.0,
                        "latency_ms_p50": 5021.415 if tool_id == "llm-tldr" else 7717.107,
                    },
                    "impact": {
                        "f1_mean": 0.847 if tool_id == "llm-tldr" else None,
                        "precision_mean": 0.739 if tool_id == "llm-tldr" else None,
                        "recall_mean": 0.933 if tool_id == "llm-tldr" else None,
                    },
                    "slice": {
                        "recall_mean": 0.884 if tool_id == "llm-tldr" else None,
                        "precision_mean": 1.0 if tool_id == "llm-tldr" else None,
                        "f1_mean": 0.919 if tool_id == "llm-tldr" else None,
                        "noise_reduction_mean": 0.657 if tool_id == "llm-tldr" else None,
                    },
                    "data_flow": {
                        "origin_accuracy_mean": 1.0 if tool_id == "llm-tldr" else None,
                        "flow_completeness_mean": 1.0 if tool_id == "llm-tldr" else None,
                    },
                    "complexity": {
                        "mae": 1.8 if tool_id == "llm-tldr" else None,
                        "kendall_tau_b": None,
                    },
                },
            }
        },
    }


def _run_metadata(*, tool_id: str, profile_sha: str, feature_set_id: str | None = None) -> dict[str, object]:
    out: dict[str, object] = {
        "tool_id": tool_id,
        "tool_profile_sha256": profile_sha,
        "prediction_count": 1260 if tool_id == "llm-tldr" else 720,
        "trials": 3,
        "task_manifest_sha256": "task-hash",
        "suite_sha256": "suite-hash",
        "tokenizer": "cl100k_base",
    }
    if feature_set_id is not None:
        out["feature_set_id"] = feature_set_id
    return out


def _compare_report() -> dict[str, object]:
    return {
        "labels": {"a": "llm-tldr", "b": "contextplus"},
        "winner": "llm-tldr",
        "wins": {"llm-tldr": 5, "contextplus": 0},
        "metric_comparisons": [
            {"metric": "mrr_mean", "a": 0.612, "b": 0.216},
            {"metric": "recall@5_mean", "a": 0.789, "b": 0.298},
            {"metric": "precision@5_mean", "a": 0.158, "b": 0.06},
        ],
    }


def _assert_report() -> dict[str, object]:
    return {
        "gates_passed": False,
        "runs": [
            {
                "deltas": {
                    "mrr_mean": 0.396,
                    "recall@5_mean": 0.491,
                    "precision@5_mean": 0.098,
                },
                "gate_checks": [
                    {"name": "validity.contextplus.error_rate", "pass": False},
                ],
            }
        ],
        "stability_gate": {
            "name": "stability.two_of_three",
            "pass": False,
            "reason": "insufficient_runs_for_stability_check",
        },
    }


def test_build_matrix_rows_marks_optional_budgets_and_identity_axes():
    mod = _load_mod()
    ToolRowConfig = mod["ToolRowConfig"]
    build_rows = mod["_build_matrix_rows"]
    primary_budget = mod["PRIMARY_BUDGET"]

    tool_a = ToolRowConfig(
        label="llm-tldr",
        feature_set_id="baseline.run1.fixed.stitched.allowlist",
        run_id="run1-fixed-stitched-allowlist-20260302T062602Z",
        embedding_backend="sentence-transformers",
        embedding_model="profile_unpinned",
        source_score_path="score-a.json",
        source_run_metadata_path="meta-a.json",
        source_tool_profile_path="profile-a.json",
    )
    tool_b = ToolRowConfig(
        label="contextplus",
        feature_set_id="baseline.run1",
        run_id="run1",
        embedding_backend="unknown",
        embedding_model="unknown",
        source_score_path="score-b.json",
        source_run_metadata_path="meta-b.json",
        source_tool_profile_path="profile-b.json",
    )

    compare = _compare_report()
    compare["__source_path"] = "compare.json"
    assert_report = _assert_report()
    assert_report["__source_path"] = "assert.json"
    rows = build_rows(
        tool_configs=[tool_a, tool_b],
        score_reports={
            "llm-tldr": _score_report(
                tool_id="llm-tldr",
                tool_sha="bbfee65bc8cc5d5051edb447d689e7ebed987a7c",
                retrieval_mrr_2000=0.612,
            ),
            "contextplus": _score_report(
                tool_id="contextplus",
                tool_sha="b42853d7c2a2018f2d4376c664db30d65ea1af23",
                retrieval_mrr_2000=0.216,
            ),
        },
        run_metadata_reports={
            "llm-tldr": _run_metadata(tool_id="llm-tldr", profile_sha="profile-a-sha"),
            "contextplus": _run_metadata(tool_id="contextplus", profile_sha="profile-b-sha"),
        },
        compare_report=compare,
        assert_report=assert_report,
        budgets=[500, 2000],
        primary_budget=primary_budget,
    )

    assert len(rows) == 4
    row_required = next(row for row in rows if row["tool"] == "llm-tldr" and row["budget_tokens"] == 2000)
    row_optional = next(row for row in rows if row["tool"] == "llm-tldr" and row["budget_tokens"] == 500)

    assert row_required["is_optional_budget_row"] is False
    assert row_required["row_scope"] == "required_primary_budget"
    assert row_required["compare_winner"] == "llm-tldr"
    assert row_required["impact_f1_mean"] == 0.847
    assert (
        row_required["row_id"]
        == "llm-tldr|bbfee65bc8cc5d5051edb447d689e7ebed987a7c|baseline.run1.fixed.stitched.allowlist|"
        "sentence-transformers|profile_unpinned|2000|run1-fixed-stitched-allowlist-20260302T062602Z"
    )

    assert row_optional["is_optional_budget_row"] is True
    assert row_optional["row_scope"] == "optional_budget_sensitivity"
    assert row_optional["compare_winner"] is None
    assert row_optional["assert_gates_passed"] is None


def test_write_csv_rows_uses_fixed_schema_order(tmp_path: Path):
    mod = _load_mod()
    write_csv = mod["_write_csv_rows"]
    columns = mod["ROW_COLUMNS"]
    path = tmp_path / "matrix.csv"
    row = {key: None for key in columns}
    row["row_id"] = "row-1"
    row["tool"] = "llm-tldr"
    row["budget_tokens"] = 2000
    row["retrieval_mrr_mean"] = 0.612
    write_csv(path, [row])

    text = path.read_text().splitlines()
    assert text[0] == ",".join(columns)
    assert text[1].startswith("row-1,")
    assert ",2000," in text[1]
    assert ",0.612," in text[1]


def test_feature_set_id_prefers_cli_then_run_metadata_then_default():
    mod = _load_mod()
    feature_set = mod["_feature_set_id"]

    # CLI value wins.
    assert (
        feature_set(
            configured_value="feature.hybrid.v1",
            run_metadata={"feature_set_id": "baseline.run1"},
            default_value="baseline.default",
        )
        == "feature.hybrid.v1"
    )

    # If CLI is missing, run metadata value is used.
    assert (
        feature_set(
            configured_value=None,
            run_metadata={"feature_set_id": "feature.hybrid.v1"},
            default_value="baseline.default",
        )
        == "feature.hybrid.v1"
    )

    # If both are missing, default is used.
    assert (
        feature_set(
            configured_value=None,
            run_metadata={},
            default_value="baseline.default",
        )
        == "baseline.default"
    )
