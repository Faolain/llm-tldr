import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_h2h_stitch.py")


def _pred_row(task_id: str, *, trial: int = 1, budget: int = 2000, status: str = "ok", marker: str = "base"):
    return {
        "task_id": task_id,
        "trial": trial,
        "budget_tokens": budget,
        "status": status,
        "latency_ms": 1.0,
        "payload_tokens": 10,
        "payload_bytes": 10,
        "result": {"ranked_files": [f"{marker}:{task_id}"]},
    }


def _pred_doc(rows: list[dict], *, tool_id: str = "llm-tldr") -> dict:
    return {
        "schema_version": 1,
        "suite_id": "h2h_llm_tldr_vs_contextplus_v1",
        "tool_id": tool_id,
        "task_manifest_sha256": "task-hash",
        "tokenizer": "cl100k_base",
        "predictions": rows,
    }


def _class_doc(rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "suite_id": "h2h_llm_tldr_vs_contextplus_v1",
        "tool_id": "llm-tldr",
        "rows": rows,
    }


def test_stitch_replaces_only_provider_transport_rows(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc([
        _pred_row("retrieval:R1", status="timeout", marker="base-timeout"),
        _pred_row("retrieval:R2", status="error", marker="base-error"),
    ])
    rerun1 = _pred_doc([
        _pred_row("retrieval:R1", status="ok", marker="rerun1-ok"),
        _pred_row("retrieval:R2", status="ok", marker="rerun1-r2"),
    ])

    class_doc = _class_doc(
        [
            {
                "task_id": "retrieval:R1",
                "trial": 1,
                "budget_tokens": 2000,
                "failure_class": "provider_transport_runtime",
            },
            {
                "task_id": "retrieval:R2",
                "trial": 1,
                "budget_tokens": 2000,
                "failure_class": "product_failure",
            },
        ]
    )

    stitched, audit = stitch(
        base_doc=base,
        rerun_docs=[(tmp_path / "rerun1.json", rerun1)],
        classification_doc=class_doc,
        run_metadata_doc=None,
        base_path=tmp_path / "base.json",
    )

    rows = stitched["predictions"]
    assert rows[0]["result"]["ranked_files"] == ["rerun1-ok:retrieval:R1"]
    assert rows[1]["result"]["ranked_files"] == ["base-error:retrieval:R2"]
    assert len(audit["replacements"]) == 1
    assert len(audit["unresolved"]) == 0


def test_stitch_uses_first_non_provider_candidate_by_rerun_order(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc([_pred_row("retrieval:R1", status="timeout", marker="base")])
    rerun1 = _pred_doc([_pred_row("retrieval:R1", status="timeout", marker="rerun1-timeout")])
    rerun2 = _pred_doc([_pred_row("retrieval:R1", status="ok", marker="rerun2-ok")])
    rerun3 = _pred_doc([_pred_row("retrieval:R1", status="ok", marker="rerun3-ok")])

    class_doc = _class_doc(
        [
            {
                "task_id": "retrieval:R1",
                "trial": 1,
                "budget_tokens": 2000,
                "failure_class": "provider_transport_runtime",
            }
        ]
    )

    stitched, audit = stitch(
        base_doc=base,
        rerun_docs=[
            (tmp_path / "rerun1.json", rerun1),
            (tmp_path / "rerun2.json", rerun2),
            (tmp_path / "rerun3.json", rerun3),
        ],
        classification_doc=class_doc,
        run_metadata_doc=None,
        base_path=tmp_path / "base.json",
    )

    row = stitched["predictions"][0]
    assert row["result"]["ranked_files"] == ["rerun2-ok:retrieval:R1"]
    assert len(audit["replacements"]) == 1
    assert audit["replacements"][0]["replacement_artifact"].endswith("rerun2.json")


def test_stitch_keeps_base_when_all_candidates_provider_or_missing(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc([_pred_row("retrieval:R1", status="timeout", marker="base")])
    rerun1 = _pred_doc([_pred_row("retrieval:R1", status="timeout", marker="rerun1-timeout")])

    class_doc = _class_doc(
        [
            {
                "task_id": "retrieval:R1",
                "trial": 1,
                "budget_tokens": 2000,
                "failure_class": "provider_transport_runtime",
            }
        ]
    )

    stitched, audit = stitch(
        base_doc=base,
        rerun_docs=[(tmp_path / "rerun1.json", rerun1)],
        classification_doc=class_doc,
        run_metadata_doc=None,
        base_path=tmp_path / "base.json",
    )

    row = stitched["predictions"][0]
    assert row["result"]["ranked_files"] == ["base:retrieval:R1"]
    assert len(audit["replacements"]) == 0
    assert len(audit["unresolved"]) == 1


def test_stitch_replaces_explicit_preflight_semantic_index_missing_rows(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc([_pred_row("retrieval:R1", status="error", marker="base-preflight")])
    rerun1 = _pred_doc([_pred_row("retrieval:R1", status="ok", marker="rerun1-ok")])

    class_doc = _class_doc(
        [
            {
                "task_id": "retrieval:R1",
                "trial": 1,
                "budget_tokens": 2000,
                "status": "error",
                "failure_class": "preflight_semantic_index_missing",
            }
        ]
    )

    stitched, audit = stitch(
        base_doc=base,
        rerun_docs=[(tmp_path / "rerun1.json", rerun1)],
        classification_doc=class_doc,
        run_metadata_doc=None,
        base_path=tmp_path / "base.json",
    )

    row = stitched["predictions"][0]
    assert row["result"]["ranked_files"] == ["rerun1-ok:retrieval:R1"]
    assert len(audit["replacements"]) == 1


def test_stitch_allowlist_replaces_ok_row_without_class_remap(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc([_pred_row("retrieval:R1", status="ok", marker="base-ok")])
    rerun1 = _pred_doc([_pred_row("retrieval:R1", status="ok", marker="rerun1-ok")])
    class_doc = _class_doc([])

    stitched, audit = stitch(
        base_doc=base,
        rerun_docs=[(tmp_path / "rerun1.json", rerun1)],
        classification_doc=class_doc,
        run_metadata_doc=None,
        base_path=tmp_path / "base.json",
        explicit_allowlist_filters={
            "task_ids": ["retrieval:R1"],
            "trials": [],
            "budget_tokens": [],
        },
    )

    row = stitched["predictions"][0]
    assert row["result"]["ranked_files"] == ["rerun1-ok:retrieval:R1"]
    assert len(audit["replacements"]) == 1
    assert audit["replacements"][0]["base_failure_class"] == "none"
    assert audit["replacements"][0]["eligibility_source"] == "explicit_allowlist"
    assert stitched["stitch"]["explicit_allowlist_filters"] == {
        "task_ids": ["retrieval:R1"],
        "trials": [],
        "budget_tokens": [],
    }
    assert audit["explicit_allowlist_filters"] == {
        "task_ids": ["retrieval:R1"],
        "trials": [],
        "budget_tokens": [],
    }


def test_stitch_allowlist_non_matching_rows_do_not_replace(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc(
        [
            _pred_row("retrieval:R1", status="ok", marker="base-r1"),
            _pred_row("retrieval:R2", status="ok", marker="base-r2"),
        ]
    )
    rerun1 = _pred_doc(
        [
            _pred_row("retrieval:R1", status="ok", marker="rerun1-r1"),
            _pred_row("retrieval:R2", status="ok", marker="rerun1-r2"),
        ]
    )

    stitched, audit = stitch(
        base_doc=base,
        rerun_docs=[(tmp_path / "rerun1.json", rerun1)],
        classification_doc=_class_doc([]),
        run_metadata_doc=None,
        base_path=tmp_path / "base.json",
        explicit_allowlist_filters={
            "task_ids": ["retrieval:R1"],
            "trials": [],
            "budget_tokens": [],
        },
    )

    rows = stitched["predictions"]
    assert rows[0]["result"]["ranked_files"] == ["rerun1-r1:retrieval:R1"]
    assert rows[1]["result"]["ranked_files"] == ["base-r2:retrieval:R2"]
    assert len(audit["replacements"]) == 1
    assert audit["replacements"][0]["row_key"]["task_id"] == "retrieval:R1"


def test_stitch_allowlist_timeout_candidate_still_not_selected(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc([_pred_row("retrieval:R1", status="ok", marker="base-ok")])
    rerun1 = _pred_doc([_pred_row("retrieval:R1", status="timeout", marker="rerun1-timeout")])
    rerun2 = _pred_doc([_pred_row("retrieval:R1", status="ok", marker="rerun2-ok")])

    stitched, audit = stitch(
        base_doc=base,
        rerun_docs=[
            (tmp_path / "rerun1.json", rerun1),
            (tmp_path / "rerun2.json", rerun2),
        ],
        classification_doc=_class_doc([]),
        run_metadata_doc=None,
        base_path=tmp_path / "base.json",
        explicit_allowlist_filters={
            "task_ids": ["retrieval:R1"],
            "trials": [],
            "budget_tokens": [],
        },
    )

    row = stitched["predictions"][0]
    assert row["result"]["ranked_files"] == ["rerun2-ok:retrieval:R1"]
    assert len(audit["replacements"]) == 1
    assert audit["replacements"][0]["replacement_artifact"].endswith("rerun2.json")
    assert audit["replacements"][0]["eligibility_source"] == "explicit_allowlist"


def test_stitch_allowlist_union_with_failure_based_eligibility(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc(
        [
            _pred_row("retrieval:R1", status="timeout", marker="base-failure-only"),
            _pred_row("retrieval:R2", status="timeout", marker="base-both"),
            _pred_row("retrieval:R3", status="ok", marker="base-allow-only"),
            _pred_row("retrieval:R4", status="ok", marker="base-neither"),
        ]
    )
    rerun1 = _pred_doc(
        [
            _pred_row("retrieval:R1", status="ok", marker="rerun1-r1"),
            _pred_row("retrieval:R2", status="ok", marker="rerun1-r2"),
            _pred_row("retrieval:R3", status="ok", marker="rerun1-r3"),
            _pred_row("retrieval:R4", status="ok", marker="rerun1-r4"),
        ]
    )

    class_doc = _class_doc(
        [
            {
                "task_id": "retrieval:R1",
                "trial": 1,
                "budget_tokens": 2000,
                "failure_class": "provider_transport_runtime",
            },
            {
                "task_id": "retrieval:R2",
                "trial": 1,
                "budget_tokens": 2000,
                "failure_class": "provider_transport_runtime",
            },
        ]
    )

    stitched, audit = stitch(
        base_doc=base,
        rerun_docs=[(tmp_path / "rerun1.json", rerun1)],
        classification_doc=class_doc,
        run_metadata_doc=None,
        base_path=tmp_path / "base.json",
        explicit_allowlist_filters={
            "task_ids": ["retrieval:R2, retrieval:R3"],
            "trials": ["1"],
            "budget_tokens": ["2000"],
        },
    )

    rows = stitched["predictions"]
    assert rows[0]["result"]["ranked_files"] == ["rerun1-r1:retrieval:R1"]
    assert rows[1]["result"]["ranked_files"] == ["rerun1-r2:retrieval:R2"]
    assert rows[2]["result"]["ranked_files"] == ["rerun1-r3:retrieval:R3"]
    assert rows[3]["result"]["ranked_files"] == ["base-neither:retrieval:R4"]
    assert len(audit["replacements"]) == 3

    sources_by_task = {
        rep["row_key"]["task_id"]: rep["eligibility_source"]
        for rep in audit["replacements"]
    }
    assert sources_by_task == {
        "retrieval:R1": "failure_class",
        "retrieval:R2": "failure_class_and_explicit_allowlist",
        "retrieval:R3": "explicit_allowlist",
    }
    assert stitched["stitch"]["explicit_allowlist_filters"] == {
        "task_ids": ["retrieval:R2", "retrieval:R3"],
        "trials": [1],
        "budget_tokens": [2000],
    }


def test_stitch_allowlist_no_match_raises_value_error(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc([_pred_row("retrieval:R1", status="ok", marker="base")])
    rerun1 = _pred_doc([_pred_row("retrieval:R1", status="ok", marker="rerun1-ok")])

    try:
        stitch(
            base_doc=base,
            rerun_docs=[(tmp_path / "rerun1.json", rerun1)],
            classification_doc=_class_doc([]),
            run_metadata_doc=None,
            base_path=tmp_path / "base.json",
            explicit_allowlist_filters={
                "task_ids": ["retrieval:R999"],
                "trials": [],
                "budget_tokens": [],
            },
        )
        raised = False
    except ValueError as exc:
        raised = True
        assert "matched zero base rows" in str(exc)

    assert raised is True


def test_stitch_preflight_reason_fallback_does_not_override_explicit_product_failure(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc(
        [
            _pred_row("retrieval:R1", status="error", marker="base-heuristic-preflight"),
            _pred_row("retrieval:R2", status="error", marker="base-explicit-product"),
        ]
    )
    rerun1 = _pred_doc(
        [
            _pred_row("retrieval:R1", status="ok", marker="rerun1-r1"),
            _pred_row("retrieval:R2", status="ok", marker="rerun1-r2"),
        ]
    )

    class_doc = _class_doc(
        [
            {
                "task_id": "retrieval:R1",
                "trial": 1,
                "budget_tokens": 2000,
                "status": "error",
                "reason": "Semantic index not found for benchmark corpus",
            },
            {
                "task_id": "retrieval:R2",
                "trial": 1,
                "budget_tokens": 2000,
                "status": "error",
                "failure_class": "product_failure",
                "reason": "Semantic index not found for benchmark corpus",
            },
        ]
    )

    stitched, audit = stitch(
        base_doc=base,
        rerun_docs=[(tmp_path / "rerun1.json", rerun1)],
        classification_doc=class_doc,
        run_metadata_doc=None,
        base_path=tmp_path / "base.json",
    )

    rows = stitched["predictions"]
    assert rows[0]["result"]["ranked_files"] == ["rerun1-r1:retrieval:R1"]
    assert rows[1]["result"]["ranked_files"] == ["base-explicit-product:retrieval:R2"]
    assert len(audit["replacements"]) == 1


def test_stitch_rejects_identity_mismatch(tmp_path: Path):
    mod = _load_mod()
    stitch = mod["_stitch_predictions"]

    base = _pred_doc([_pred_row("retrieval:R1", status="timeout")], tool_id="llm-tldr")
    rerun_bad = _pred_doc([_pred_row("retrieval:R1", status="ok")], tool_id="other-tool")

    class_doc = _class_doc(
        [
            {
                "task_id": "retrieval:R1",
                "trial": 1,
                "budget_tokens": 2000,
                "failure_class": "provider_transport_runtime",
            }
        ]
    )

    try:
        stitch(
            base_doc=base,
            rerun_docs=[(tmp_path / "rerun-bad.json", rerun_bad)],
            classification_doc=class_doc,
            run_metadata_doc=None,
            base_path=tmp_path / "base.json",
        )
        raised = False
    except ValueError as exc:
        raised = True
        assert "identity mismatch" in str(exc)

    assert raised is True
