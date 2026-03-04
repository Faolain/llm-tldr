from __future__ import annotations

from types import SimpleNamespace

from tldr.daemon.core import TLDRDaemon


def test_daemon_semantic_search_compound_flag_routes_to_lane4(
    monkeypatch, tmp_path
) -> None:
    import tldr.semantic as semantic

    dummy = SimpleNamespace(
        project=tmp_path,
        _semantic_config={},
        index_paths=None,
        index_config=None,
        _ignore_spec=None,
        _workspace_root=None,
    )

    monkeypatch.setattr(
        semantic,
        "compound_semantic_impact_search",
        lambda *_, **__: {"mode": "lane4"},
    )
    monkeypatch.setattr(
        semantic,
        "semantic_search",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("legacy semantic path should not run")
        ),
    )

    out = TLDRDaemon._handle_semantic(
        dummy,
        {"action": "search", "query": "find callers", "compound_impact": True},
    )

    assert out == {
        "status": "ok",
        "result": {"mode": "lane4"},
        "results": {"mode": "lane4"},
    }
