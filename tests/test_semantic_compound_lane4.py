from __future__ import annotations

from pathlib import Path

import pytest

import tldr.semantic as semantic


def _graph_with_edges(edges: list[tuple[str, str, str, str]]):
    from tldr.cross_file_calls import ProjectCallGraph

    graph = ProjectCallGraph()
    for edge in edges:
        graph.add_edge(*edge)
    return graph


def _install_deterministic_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = {"value": 0.0}

    def _fake_perf_counter() -> float:
        ticks["value"] += 0.001
        return ticks["value"]

    monkeypatch.setattr(semantic.time, "perf_counter", _fake_perf_counter)


def test_lane4_compound_schema_contract_and_ordering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_deterministic_clock(monkeypatch)
    monkeypatch.setattr(
        semantic,
        "semantic_search",
        lambda *_, **__: [
            {
                "file": "pkg/a.py",
                "qualified_name": "pkg.a.foo",
                "name": "foo",
                "line": 10,
                "unit_type": "function",
                "score": 0.9,
                "semantic_score": 0.8,
            },
            {
                "file": "pkg/b.py",
                "qualified_name": "pkg.b.bar",
                "name": "bar",
                "line": 20,
                "unit_type": "function",
                "score": 0.7,
                "semantic_score": 0.6,
            },
        ],
    )
    monkeypatch.setattr(
        "tldr.cross_file_calls.build_project_call_graph",
        lambda *_, **__: _graph_with_edges(
            [
                ("pkg/c.py", "caller_z", "pkg/a.py", "foo"),
                ("pkg/b.py", "caller_a", "pkg/a.py", "foo"),
                ("pkg/a.py", "foo", "pkg/b.py", "bar"),
            ]
        ),
    )

    out = semantic.compound_semantic_impact_search(
        str(tmp_path),
        query="find impacted callsites",
        k=5,
        impact_limit=2,
        impact_language="python",
        budget_tokens=2000,
    )

    assert out["schema_version"] == 1
    assert out["feature_set_id"] == "feature.compound-semantic-impact.v1"
    assert out["status"] == "ok"
    assert out["counts"]["retrieval_results"] == 2
    assert out["counts"]["impact_attempted"] == 2
    assert out["counts"]["impact_ok"] == 2
    assert out["counts"]["impact_partial"] == 0
    assert out["counts"]["impact_error"] == 0

    assert [row["rank"] for row in out["results"]] == [1, 2]
    callers = out["results"][0]["impact"]["callers"]
    assert callers == sorted(
        callers,
        key=lambda item: (
            int(item.get("depth", 0)),
            str(item.get("file")),
            str(item.get("function")),
            str(item.get("line")),
        ),
    )


def test_lane4_compound_partial_failure_impact_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_deterministic_clock(monkeypatch)
    monkeypatch.setattr(
        semantic,
        "semantic_search",
        lambda *_, **__: [
            {
                "file": "pkg/a.py",
                "qualified_name": "pkg.a.foo",
                "name": "foo",
                "line": 10,
                "unit_type": "function",
            },
            {
                "file": "pkg/b.py",
                "qualified_name": "pkg.b.missing",
                "name": "missing",
                "line": 22,
                "unit_type": "function",
            },
        ],
    )
    monkeypatch.setattr(
        "tldr.cross_file_calls.build_project_call_graph",
        lambda *_, **__: _graph_with_edges(
            [("pkg/c.py", "caller", "pkg/a.py", "foo")]
        ),
    )

    out = semantic.compound_semantic_impact_search(
        str(tmp_path),
        query="find impacted callsites",
        k=5,
        impact_limit=2,
        impact_language="python",
    )

    assert out["status"] == "partial"
    assert out["counts"]["impact_ok"] == 1
    assert out["counts"]["impact_partial"] >= 1
    assert any(item.get("code") == "impact_not_found" for item in out["partial_failures"])


def test_lane4_compound_partial_failure_semantic_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_deterministic_clock(monkeypatch)
    monkeypatch.setattr(
        semantic,
        "semantic_search",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("semantic exploded")),
    )

    out = semantic.compound_semantic_impact_search(
        str(tmp_path),
        query="find impacted callsites",
        k=5,
    )

    assert out["status"] == "error"
    assert out["counts"]["retrieval_results"] == 0
    assert out["counts"]["impact_attempted"] == 0
    assert out["partial_failures"][0]["stage"] == "semantic"
    assert out["partial_failures"][0]["code"] == "semantic_runtime_error"


def test_lane4_compound_is_deterministic_for_same_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    semantic_rows = [
        {
            "file": "pkg/a.py",
            "qualified_name": "pkg.a.foo",
            "name": "foo",
            "line": 10,
            "unit_type": "function",
            "score": 0.9,
        },
        {
            "file": "pkg/b.py",
            "qualified_name": "pkg.b.bar",
            "name": "bar",
            "line": 12,
            "unit_type": "function",
            "score": 0.8,
        },
    ]
    monkeypatch.setattr(semantic, "semantic_search", lambda *_, **__: semantic_rows)
    monkeypatch.setattr(
        "tldr.cross_file_calls.build_project_call_graph",
        lambda *_, **__: _graph_with_edges(
            [
                ("pkg/c.py", "caller", "pkg/a.py", "foo"),
                ("pkg/a.py", "foo", "pkg/b.py", "bar"),
            ]
        ),
    )

    _install_deterministic_clock(monkeypatch)
    first = semantic.compound_semantic_impact_search(
        str(tmp_path),
        query="determinism",
        k=5,
        impact_limit=2,
        impact_language="python",
    )
    _install_deterministic_clock(monkeypatch)
    second = semantic.compound_semantic_impact_search(
        str(tmp_path),
        query="determinism",
        k=5,
        impact_limit=2,
        impact_language="python",
    )

    assert first == second


def test_lane4_compound_budget_latency_envelope_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_deterministic_clock(monkeypatch)
    monkeypatch.setattr(
        semantic,
        "semantic_search",
        lambda *_, **__: [
            {
                "file": "pkg/a.py",
                "qualified_name": "pkg.a.foo",
                "name": "foo",
                "line": 10,
                "unit_type": "function",
                "score": 0.9,
            }
        ],
    )
    monkeypatch.setattr(
        "tldr.cross_file_calls.build_project_call_graph",
        lambda *_, **__: _graph_with_edges(
            [("pkg/c.py", "caller", "pkg/a.py", "foo")]
        ),
    )

    out = semantic.compound_semantic_impact_search(
        str(tmp_path),
        query="budget envelope",
        k=5,
        budget_tokens=2000,
        max_latency_ms_p50_ratio=1.15,
        max_payload_tokens_median_ratio=1.10,
        impact_language="python",
    )

    envelope = out["regression_metadata"]
    assert envelope["budget_tokens"] == 2000
    assert envelope["max_latency_ms_p50_ratio"] == 1.15
    assert envelope["max_payload_tokens_median_ratio"] == 1.10
    assert isinstance(envelope["latency_ms_p50"], float)
    assert isinstance(envelope["payload_tokens_median"], float)
