from pathlib import Path

import pytest


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "js-monorepo"


def test_javascript_tsconfig_fixture_uses_resolved_call_graph(monkeypatch):
    from tldr.cross_file_calls import build_project_call_graph, scan_project
    from tldr.ts.ts_callgraph import TsResolverError, build_ts_resolved_call_graph

    monkeypatch.delenv("TLDR_TS_RESOLVER", raising=False)

    js_files = scan_project(FIXTURE_ROOT, language="javascript")
    try:
        build_ts_resolved_call_graph(
            str(FIXTURE_ROOT),
            allow_files=js_files,
        )
    except TsResolverError as exc:
        pytest.skip(f"TS-resolved JS mode unavailable in this environment: {exc.code or 'resolver_error'}: {exc}")

    graph = build_project_call_graph(
        str(FIXTURE_ROOT),
        language="javascript",
    )

    assert graph.meta.get("graph_source") == "ts-resolved"
    assert (
        "packages/b/src/main.js",
        "main",
        "packages/a/src/foo.js",
        "foo",
    ) in graph.edges
