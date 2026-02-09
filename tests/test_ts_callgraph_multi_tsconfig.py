from pathlib import Path

import pytest


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "ts-multi-tsconfig"


class TestTsCallGraphMultiTsconfigFixture:
    def test_multi_tsconfig_fallback_and_dist_dts_mapping(self, monkeypatch):
        from tldr.cross_file_calls import build_project_call_graph

        monkeypatch.delenv("TLDR_TS_RESOLVER", raising=False)

        graph = build_project_call_graph(
            str(FIXTURE_ROOT),
            language="typescript",
        )

        if graph.meta.get("graph_source") != "ts-resolved-multi":
            errs = graph.meta.get("ts_resolution_errors") or []
            pytest.skip(f"TS-resolved multi-tsconfig mode unavailable in this environment: {errs}")

        assert (
            "packages/b/src/main.ts",
            "main",
            "packages/a/src/index.ts",
            "foo",
        ) in graph.edges

        # Ensure we prefer mapping workspace dist declaration outputs back to workspace src paths.
        assert (
            "packages/b/src/main.ts",
            "main",
            "packages/a/dist/src/index.d.ts",
            "foo",
        ) not in graph.edges

        ts_projects = graph.meta.get("ts_projects") or []
        assert isinstance(ts_projects, list)
        assert any(
            isinstance(p, dict)
            and p.get("tsconfig") == "packages/b/tsconfig.json"
            and p.get("status") == "ok"
            for p in ts_projects
        )

