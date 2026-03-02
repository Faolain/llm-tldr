from __future__ import annotations

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
