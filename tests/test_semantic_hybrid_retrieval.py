from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import tldr.semantic as semantic


def test_hybrid_rrf_boosts_files_supported_by_multiple_rankers() -> None:
    ranked = semantic._rrf_fuse_file_rankings(
        [
            ["a.py", "b.py", "d.py"],
            ["b.py", "c.py", "e.py"],
        ],
        rrf_k=60,
    )

    assert ranked[0] == "b.py"
    assert ranked.index("b.py") < ranked.index("a.py")


def test_hybrid_rrf_tie_break_is_deterministic_by_path() -> None:
    ranked = semantic._rrf_fuse_file_rankings(
        [
            ["b.py"],
            ["a.py"],
        ],
        rrf_k=60,
    )

    assert ranked == ["a.py", "b.py"]


def test_hybrid_search_guard_rg_empty_returns_empty_and_skips_semantic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(semantic, "_rg_rank_files", lambda *_, **__: [])

    def _semantic_should_not_run(*args, **kwargs):
        raise AssertionError("semantic stage should be skipped when guard triggers")

    monkeypatch.setattr(semantic, "_semantic_unit_search", _semantic_should_not_run)

    out = semantic.hybrid_file_search(
        str(tmp_path),
        query="needle that does not exist",
        k=5,
        no_result_guard="rg_empty",
    )

    assert out == []


def test_hybrid_search_without_guard_allows_semantic_results_when_lexical_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(semantic, "_rg_rank_files", lambda *_, **__: [])
    monkeypatch.setattr(
        semantic,
        "_semantic_unit_search",
        lambda *_, **__: [
            {"file": "pkg/b.py", "line": 10},
            {"file": "pkg/a.py", "line": 20},
        ],
    )

    out = semantic.hybrid_file_search(
        str(tmp_path),
        query="needle that does not exist",
        k=5,
        no_result_guard="none",
    )

    assert [row["file"] for row in out] == ["pkg/b.py", "pkg/a.py"]


def test_semantic_search_lane2_signature_exposes_abstain_rerank_and_bounds() -> None:
    params = set(inspect.signature(semantic.semantic_search).parameters)
    expected = {
        "abstain_threshold",
        "abstain_empty",
        "rerank",
        "rerank_top_n",
        "max_latency_ms_p50_ratio",
        "max_payload_tokens_median_ratio",
        "budget_tokens",
    }
    missing = expected - params
    assert not missing, f"missing lane2 semantic_search kwargs: {sorted(missing)}"


def test_lane3_effective_k_mapping_is_deterministic_and_clamped() -> None:
    map_k = semantic._effective_k_from_budget_tokens

    assert map_k(10, budget_tokens=2000) == 10
    assert map_k(10, budget_tokens=1000) == 5
    assert map_k(10, budget_tokens=500) == 3
    assert map_k(10, budget_tokens=5000) == 25
    assert map_k(10, budget_tokens=500000) == 50

    assert map_k(10, budget_tokens=0) == 10
    assert map_k(10, budget_tokens=-1) == 10
    assert map_k(10, budget_tokens="bad") == 10


def test_semantic_search_lane3_budget_tokens_scales_effective_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, int] = {}

    def _fake_semantic_unit_search(*args, **kwargs):
        observed["k"] = int(kwargs["k"])
        return []

    monkeypatch.setattr(semantic, "_semantic_unit_search", _fake_semantic_unit_search)

    out = semantic.semantic_search(
        str(tmp_path),
        query="query",
        k=10,
        budget_tokens=500,
    )

    assert out == []
    assert observed["k"] == 3


def test_hybrid_search_lane3_budget_tokens_scales_return_count_and_semantic_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lexical = [f"pkg/file_{i:02d}.py" for i in range(1, 21)]
    semantic_rows = [{"file": fp, "line": idx} for idx, fp in enumerate(reversed(lexical), start=1)]
    observed: dict[str, int] = {}

    monkeypatch.setattr(semantic, "_rg_rank_files", lambda *_, **__: lexical)

    def _fake_semantic_unit_search(*args, **kwargs):
        observed["k"] = int(kwargs["k"])
        return semantic_rows

    monkeypatch.setattr(semantic, "_semantic_unit_search", _fake_semantic_unit_search)

    out = semantic.hybrid_file_search(
        str(tmp_path),
        query="query",
        k=10,
        no_result_guard="none",
        budget_tokens=500,
    )

    assert len(out) == 3
    assert observed["k"] == 15


def test_hybrid_search_rows_include_lane2_confidence_and_rerank_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(semantic, "_rg_rank_files", lambda *_, **__: ["pkg/a.py", "pkg/b.py"])
    monkeypatch.setattr(
        semantic,
        "_semantic_unit_search",
        lambda *_, **__: [
            {"file": "pkg/b.py", "line": 10},
            {"file": "pkg/a.py", "line": 20},
        ],
    )

    out = semantic.hybrid_file_search(
        str(tmp_path),
        query="items for result",
        k=2,
        no_result_guard="none",
    )

    assert out, "expected at least one hybrid row"
    top = out[0]
    assert top.get("confidence") is not None
    assert top.get("rerank_applied") in {True, False}


def test_hybrid_search_rows_include_lane2_latency_and_payload_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(semantic, "_rg_rank_files", lambda *_, **__: ["pkg/a.py", "pkg/b.py"])
    monkeypatch.setattr(
        semantic,
        "_semantic_unit_search",
        lambda *_, **__: [
            {"file": "pkg/b.py", "line": 10},
            {"file": "pkg/a.py", "line": 20},
        ],
    )

    out = semantic.hybrid_file_search(
        str(tmp_path),
        query="items for result",
        k=2,
        no_result_guard="none",
    )

    assert out, "expected at least one hybrid row"
    top = out[0]
    assert top.get("latency_ms_p50") is not None
    assert top.get("payload_tokens_median") is not None


def test_hybrid_search_lane2_abstain_threshold_can_return_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(semantic, "_rg_rank_files", lambda *_, **__: ["pkg/a.py"])
    monkeypatch.setattr(
        semantic,
        "_semantic_unit_search",
        lambda *_, **__: [{"file": "pkg/a.py", "line": 10, "score": -0.8}],
    )

    out = semantic.hybrid_file_search(
        str(tmp_path),
        query="query with weak semantic signal",
        k=1,
        no_result_guard="none",
        abstain_threshold=0.7,
        abstain_empty=True,
    )

    assert out == []


def test_hybrid_search_lane2_rerank_reorders_by_confidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(semantic, "_rg_rank_files", lambda *_, **__: ["pkg/a.py", "pkg/b.py"])
    monkeypatch.setattr(
        semantic,
        "_semantic_unit_search",
        lambda *_, **__: [
            {"file": "pkg/b.py", "line": 20, "score": 0.95},
            {"file": "pkg/a.py", "line": 10, "score": -0.6},
        ],
    )

    no_rerank = semantic.hybrid_file_search(
        str(tmp_path),
        query="query",
        k=2,
        no_result_guard="none",
        rerank=False,
    )
    reranked = semantic.hybrid_file_search(
        str(tmp_path),
        query="query",
        k=2,
        no_result_guard="none",
        rerank=True,
        rerank_top_n=2,
    )

    assert [row["file"] for row in no_rerank] == ["pkg/a.py", "pkg/b.py"]
    assert [row["file"] for row in reranked] == ["pkg/b.py", "pkg/a.py"]
    assert reranked[0].get("rerank_applied") is True


def test_hybrid_search_lane2_bound_metadata_accepts_positive_ratios_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(semantic, "_rg_rank_files", lambda *_, **__: ["pkg/a.py"])
    monkeypatch.setattr(
        semantic,
        "_semantic_unit_search",
        lambda *_, **__: [{"file": "pkg/a.py", "line": 10, "score": 0.8}],
    )

    out = semantic.hybrid_file_search(
        str(tmp_path),
        query="query",
        k=1,
        no_result_guard="none",
        max_latency_ms_p50_ratio=1.2,
        max_payload_tokens_median_ratio=-1.0,
    )

    assert out
    row = out[0]
    assert row.get("max_latency_ms_p50_ratio") == 1.2
    assert "max_payload_tokens_median_ratio" not in row


def test_lane4_compound_signature_exposes_impact_controls() -> None:
    params = set(inspect.signature(semantic.compound_semantic_impact_search).parameters)
    expected = {"impact_depth", "impact_limit", "impact_language"}
    missing = expected - params
    assert not missing, f"missing lane4 compound kwargs: {sorted(missing)}"
