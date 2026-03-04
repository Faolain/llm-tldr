from __future__ import annotations

from types import SimpleNamespace

from tldr.daemon.core import TLDRDaemon


def test_daemon_semantic_search_navigate_cluster_flag_routes_to_lane5(
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

    for lane5_name in (
        "semantic_navigation_cluster_search",
        "semantic_navigate_search",
        "navigate_cluster_search",
        "semantic_navigate_cluster_search",
    ):
        monkeypatch.setattr(
            semantic,
            lane5_name,
            lambda *_, **__: {"mode": "lane5"},
            raising=False,
        )
    monkeypatch.setattr(
        semantic,
        "compound_semantic_impact_search",
        lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("lane4 compound path should not run")
        ),
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
        {"action": "search", "query": "navigate auth checks", "navigate_cluster": True},
    )

    assert out == {
        "status": "ok",
        "result": {"mode": "lane5"},
        "results": {"mode": "lane5"},
    }
