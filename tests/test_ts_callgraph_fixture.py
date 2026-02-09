import json
import subprocess
from pathlib import Path

import pytest
import shutil


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "ts-monorepo"


def _load_expected_edges() -> list[tuple[str, str, str, str]]:
    expected_path = FIXTURE_ROOT / "expected_edges.json"
    data = json.loads(expected_path.read_text())
    out = []
    for e in data:
        out.append(
            (
                e["caller"]["file"],
                e["caller"]["symbol"],
                e["callee"]["file"],
                e["callee"]["symbol"],
            )
        )
    return out


class TestTsCallGraphFixture:
    def test_fixture_compiles(self):
        if shutil.which("tsc") is None:
            pytest.skip("tsc not available")

        subprocess.run(
            ["tsc", "-p", str(FIXTURE_ROOT), "--noEmit"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_resolved_call_graph_matches_golden_edges(self, monkeypatch):
        from tldr.cross_file_calls import build_project_call_graph

        # Keep the resolver in its default mode unless the environment forces otherwise.
        monkeypatch.delenv("TLDR_TS_RESOLVER", raising=False)

        graph = build_project_call_graph(
            str(FIXTURE_ROOT),
            language="typescript",
        )

        if graph.meta.get("graph_source") != "ts-resolved":
            errs = graph.meta.get("ts_resolution_errors") or []
            pytest.skip(f"TS-resolved mode unavailable in this environment: {errs}")

        expected = _load_expected_edges()
        for edge in expected:
            assert edge in graph.edges

        # Explicit: exported const arrow/function expressions should resolve to the
        # variable name, not be dropped as callee_unnamed.
        assert (
            "packages/b/src/main.ts",
            "main",
            "packages/a/src/arrow.ts",
            "createCache",
        ) in graph.edges
        assert (
            "packages/b/src/main.ts",
            "main",
            "packages/a/src/arrow.ts",
            "createLibp2pExtended",
        ) in graph.edges

        # Explicit negative: dynamic element access should not produce an edge.
        assert (
            "packages/a/src/handlers.ts",
            "callUnsupported",
            "packages/a/src/handlers.ts",
            "onFoo",
        ) not in graph.edges

        # Determinism: repeated builds should match exactly.
        graph2 = build_project_call_graph(
            str(FIXTURE_ROOT),
            language="typescript",
        )
        assert graph2.meta.get("graph_source") == "ts-resolved"
        assert graph.sorted_edges() == graph2.sorted_edges()

    def test_impact_queries_work_on_fixture(self, monkeypatch):
        from tldr.analysis import impact_analysis
        from tldr.cross_file_calls import build_project_call_graph

        monkeypatch.delenv("TLDR_TS_RESOLVER", raising=False)

        graph = build_project_call_graph(
            str(FIXTURE_ROOT),
            language="typescript",
        )

        if graph.meta.get("graph_source") != "ts-resolved":
            errs = graph.meta.get("ts_resolution_errors") or []
            pytest.skip(f"TS-resolved mode unavailable in this environment: {errs}")

        # Function call across monorepo import.
        foo_impact = impact_analysis(graph, "foo", max_depth=2)
        foo_json = json.dumps(foo_impact)
        assert "packages/b/src/main.ts" in foo_json
        assert "main" in foo_json

        # Interface dispatch case should resolve to C.m in this fixture.
        m_impact = impact_analysis(graph, "C.m", max_depth=2)
        m_json = json.dumps(m_impact)
        assert "packages/a/src/dispatch.ts" in m_json
        assert "dispatch" in m_json

    def test_forced_syntax_only_mode(self, monkeypatch):
        from tldr.cross_file_calls import build_project_call_graph

        monkeypatch.setenv("TLDR_TS_RESOLVER", "syntax")
        graph = build_project_call_graph(str(FIXTURE_ROOT), language="typescript")
        assert graph.meta.get("graph_source") == "ts-syntax-only"
