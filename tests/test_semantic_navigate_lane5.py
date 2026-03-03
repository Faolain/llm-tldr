from __future__ import annotations

from pathlib import Path

import pytest

import tldr.semantic as semantic


def _lane5_navigate_callable():
    for name in (
        "semantic_navigate_search",
        "navigate_cluster_search",
        "semantic_navigate_cluster_search",
    ):
        candidate = getattr(semantic, name, None)
        if callable(candidate):
            return candidate
    raise AssertionError("missing lane5 semantic navigate callable")


def _install_deterministic_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = {"value": 0.0}

    def _fake_perf_counter() -> float:
        ticks["value"] += 0.001
        return ticks["value"]

    monkeypatch.setattr(semantic.time, "perf_counter", _fake_perf_counter)


def _cluster_sort_key(cluster: dict) -> tuple[str, str]:
    return (str(cluster.get("cluster_id")), str(cluster.get("label") or ""))


def _member_sort_key(member: dict) -> tuple[str, str, str]:
    symbol = member.get("symbol")
    if isinstance(symbol, dict):
        symbol_name = symbol.get("qualified_name") or symbol.get("name")
    else:
        symbol_name = member.get("qualified_name") or member.get("name")
    return (
        str(member.get("file") or ""),
        str(symbol_name or ""),
        str(member.get("line") or ""),
    )


def test_lane5_navigate_schema_contract_and_cluster_ordering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    navigate = _lane5_navigate_callable()
    _install_deterministic_clock(monkeypatch)
    monkeypatch.setattr(
        semantic,
        "semantic_search",
        lambda *_, **__: [
            {
                "file": "pkg/c.py",
                "qualified_name": "pkg.c.alpha",
                "name": "alpha",
                "line": 30,
                "unit_type": "function",
                "score": 0.90,
                "semantic_score": 0.80,
            },
            {
                "file": "pkg/a.py",
                "qualified_name": "pkg.a.beta",
                "name": "beta",
                "line": 10,
                "unit_type": "function",
                "score": 0.85,
                "semantic_score": 0.78,
            },
            {
                "file": "pkg/b.py",
                "qualified_name": "pkg.b.gamma",
                "name": "gamma",
                "line": 20,
                "unit_type": "function",
                "score": 0.84,
                "semantic_score": 0.77,
            },
        ],
    )

    out = navigate(
        str(tmp_path),
        query="navigate auth entrypoints",
        k=5,
        budget_tokens=2000,
    )

    assert out["schema_version"] == 1
    assert out["feature_set_id"] == "feature.navigate-cluster.v1"
    assert out["status"] == "ok"
    assert out["counts"]["retrieval_results"] == 3
    assert out["counts"]["cluster_count"] >= 1
    assert [row["rank"] for row in out["results"]] == [1, 2, 3]
    assert all(row.get("cluster_id") is not None for row in out["results"])

    clusters = out["clusters"]
    assert clusters == sorted(clusters, key=_cluster_sort_key)
    for cluster in clusters:
        members = cluster.get("members")
        assert isinstance(members, list)
        assert members == sorted(members, key=_member_sort_key)


def test_lane5_navigate_semantic_stage_error_handling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    navigate = _lane5_navigate_callable()
    _install_deterministic_clock(monkeypatch)
    monkeypatch.setattr(
        semantic,
        "semantic_search",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("semantic exploded")),
    )

    out = navigate(
        str(tmp_path),
        query="navigate auth entrypoints",
        k=5,
    )

    assert out["status"] == "error"
    assert out["counts"]["retrieval_results"] == 0
    assert out["counts"]["cluster_count"] == 0
    assert out["partial_failures"][0]["stage"] == "semantic"
    assert out["partial_failures"][0]["code"] == "semantic_runtime_error"


def test_lane5_navigate_is_deterministic_for_same_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    navigate = _lane5_navigate_callable()
    semantic_rows = [
        {
            "file": "pkg/z.py",
            "qualified_name": "pkg.z.one",
            "name": "one",
            "line": 50,
            "unit_type": "function",
            "score": 0.91,
        },
        {
            "file": "pkg/y.py",
            "qualified_name": "pkg.y.two",
            "name": "two",
            "line": 40,
            "unit_type": "function",
            "score": 0.90,
        },
    ]
    monkeypatch.setattr(semantic, "semantic_search", lambda *_, **__: semantic_rows)

    _install_deterministic_clock(monkeypatch)
    first = navigate(
        str(tmp_path),
        query="deterministic navigation",
        k=5,
        budget_tokens=2000,
    )
    _install_deterministic_clock(monkeypatch)
    second = navigate(
        str(tmp_path),
        query="deterministic navigation",
        k=5,
        budget_tokens=2000,
    )

    assert first == second
