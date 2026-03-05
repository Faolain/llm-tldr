"""Regression tests for syntax-fallback TS/JS import resolution edge cases."""

from pathlib import Path

import pytest

from tldr.cross_file_calls import build_project_call_graph


@pytest.fixture
def force_syntax_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDR_TS_RESOLVER", "syntax")


def test_named_import_with_dotted_basename_and_explicit_extension_resolves(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "foo.bar.ts": "export function helper() { return 1; }\n",
            "main.ts": (
                'import { helper } from "./foo.bar.ts";\n'
                "export function run() {\n"
                "  return helper();\n"
                "}\n"
            ),
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="typescript")

    assert graph.meta.get("graph_source") == "ts-syntax-only"
    assert ("main.ts", "run", "foo.bar.ts", "helper") in graph.edges


def test_default_import_prefers_exact_relative_file_over_duplicate_basename(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "foo.js": "export default function rootFoo() { return 1; }\n",
            "nested/foo.js": "export default function nestedFoo() { return 2; }\n",
            "main.js": (
                'import foo from "./foo.js";\n'
                "export function run() {\n"
                "  return foo();\n"
                "}\n"
            ),
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="javascript")

    assert graph.meta.get("graph_source") == "js-syntax-only"
    assert ("main.js", "run", "foo.js", "default") in graph.edges
    assert ("main.js", "run", "nested/foo.js", "default") not in graph.edges


def test_default_import_with_explicit_extension_ignores_same_basename_variants(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "foo.cjs": "module.exports = function cjsFoo() { return 1; };\n",
            "foo.mjs": "export default function mjsFoo() { return 2; }\n",
            "foo.js": "export default function jsFoo() { return 3; }\n",
            "main.js": (
                'import foo from "./foo.js";\n'
                "export function run() {\n"
                "  return foo();\n"
                "}\n"
            ),
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="javascript")

    assert graph.meta.get("graph_source") == "js-syntax-only"
    assert ("main.js", "run", "foo.js", "default") in graph.edges
    assert ("main.js", "run", "foo.cjs", "default") not in graph.edges
    assert ("main.js", "run", "foo.mjs", "default") not in graph.edges


def test_require_alias_call_resolves_to_cjs_default_export(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "dep.cjs": "module.exports = function dep() { return 1; };\n",
            "main.js": (
                'const run = require("./dep.cjs");\n'
                "export function boot() {\n"
                "  return run();\n"
                "}\n"
            ),
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="javascript")

    assert graph.meta.get("graph_source") == "js-syntax-only"
    assert ("main.js", "boot", "dep.cjs", "default") in graph.edges
    assert ("main.js", "boot", "main.js", "run") not in graph.edges


def test_namespace_import_with_dotted_basename_resolves_attr_call(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "foo.bar.ts": "export function helper() { return 1; }\n",
            "main.ts": (
                'import * as mod from "./foo.bar.ts";\n'
                "export function run() {\n"
                "  return mod.helper();\n"
                "}\n"
            ),
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="typescript")

    assert graph.meta.get("graph_source") == "ts-syntax-only"
    assert ("main.ts", "run", "foo.bar.ts", "helper") in graph.edges


def test_javascript_syntax_path_requests_javascript_parser(
    tmp_path: Path,
    force_syntax_fallback: None,
    monkeypatch: pytest.MonkeyPatch,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "dep.js": "export function helper() { return 1; }\n",
            "main.js": (
                'import { helper } from "./dep.js";\n'
                "export function run() {\n"
                "  return helper();\n"
                "}\n"
            ),
        },
    )

    class _FakeNode:
        type = "program"
        children = []
        start_byte = 0
        end_byte = 0

        def child_by_field_name(self, _name: str):
            return None

    class _FakeTree:
        root_node = _FakeNode()

    class _FakeParser:
        def parse(self, _source: bytes) -> _FakeTree:
            return _FakeTree()

    parser_requests: list[str] = []

    def _spy_get_ts_parser(language: str = "typescript") -> _FakeParser:
        parser_requests.append(language)
        return _FakeParser()

    monkeypatch.setattr("tldr.cross_file_calls.TREE_SITTER_AVAILABLE", True)
    monkeypatch.setattr("tldr.cross_file_calls._get_ts_parser", _spy_get_ts_parser)

    build_project_call_graph(str(tmp_path), language="javascript")

    assert "javascript" in parser_requests
