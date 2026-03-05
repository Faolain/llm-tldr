"""Regression tests for syntax-fallback TS/JS import resolution edge cases."""

from pathlib import Path

import pytest

import tldr.cross_file_calls as cfc
from tldr.cross_file_calls import build_project_call_graph


@pytest.fixture
def force_syntax_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDR_TS_RESOLVER", "syntax")


def _build_graph_with_forced_scan_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    relative_scan_order: list[str],
):
    root = tmp_path.resolve()
    ordered_files = [str((root / rel_path).resolve()) for rel_path in relative_scan_order]

    with monkeypatch.context() as patch:
        original_scan_project = cfc.scan_project

        def _scan_project(
            scan_root: str | Path,
            scan_language: str = "python",
            workspace_config=None,
            respect_ignore: bool = True,
            ignore_spec=None,
            workspace_root: Path | None = None,
        ) -> list[str]:
            if Path(scan_root).resolve() == root and scan_language == language:
                return ordered_files
            return original_scan_project(
                scan_root,
                scan_language,
                workspace_config,
                respect_ignore,
                ignore_spec,
                workspace_root,
            )

        patch.setattr(cfc, "scan_project", _scan_project)
        return cfc.build_project_call_graph(str(root), language=language)


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


def test_default_import_extensionless_collision_is_scan_order_independent(
    tmp_path: Path,
    force_syntax_fallback: None,
    monkeypatch: pytest.MonkeyPatch,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "foo.js": "export default function fooJs() { return 1; }\n",
            "foo.cjs": "module.exports = function fooCjs() { return 2; };\n",
            "main.js": (
                'import foo from "./foo";\n'
                "export function run() {\n"
                "  return foo();\n"
                "}\n"
            ),
        },
    )

    graph_cjs_first = _build_graph_with_forced_scan_order(
        tmp_path,
        monkeypatch,
        language="javascript",
        relative_scan_order=["main.js", "foo.cjs", "foo.js"],
    )
    graph_js_first = _build_graph_with_forced_scan_order(
        tmp_path,
        monkeypatch,
        language="javascript",
        relative_scan_order=["main.js", "foo.js", "foo.cjs"],
    )

    assert graph_cjs_first.meta.get("graph_source") == "js-syntax-only"
    assert graph_js_first.meta.get("graph_source") == "js-syntax-only"
    assert graph_cjs_first.sorted_edges() == graph_js_first.sorted_edges()
    assert ("main.js", "run", "foo.js", "default") in graph_cjs_first.edges
    assert ("main.js", "run", "foo.js", "default") in graph_js_first.edges
    assert ("main.js", "run", "foo.cjs", "default") not in graph_cjs_first.edges
    assert ("main.js", "run", "foo.cjs", "default") not in graph_js_first.edges


def test_named_import_extensionless_collision_prefers_js_over_mjs_independent_of_scan_order(
    tmp_path: Path,
    force_syntax_fallback: None,
    monkeypatch: pytest.MonkeyPatch,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "foo.js": "export function helper() { return 1; }\n",
            "foo.mjs": "export function helper() { return 2; }\n",
            "main.js": (
                'import { helper } from "./foo";\n'
                "export function run() {\n"
                "  return helper();\n"
                "}\n"
            ),
        },
    )

    graph_mjs_first = _build_graph_with_forced_scan_order(
        tmp_path,
        monkeypatch,
        language="javascript",
        relative_scan_order=["main.js", "foo.mjs", "foo.js"],
    )
    graph_js_first = _build_graph_with_forced_scan_order(
        tmp_path,
        monkeypatch,
        language="javascript",
        relative_scan_order=["main.js", "foo.js", "foo.mjs"],
    )

    assert graph_mjs_first.meta.get("graph_source") == "js-syntax-only"
    assert graph_js_first.meta.get("graph_source") == "js-syntax-only"
    assert graph_mjs_first.sorted_edges() == graph_js_first.sorted_edges()
    assert ("main.js", "run", "foo.js", "helper") in graph_mjs_first.edges
    assert ("main.js", "run", "foo.js", "helper") in graph_js_first.edges
    assert ("main.js", "run", "foo.mjs", "helper") not in graph_mjs_first.edges
    assert ("main.js", "run", "foo.mjs", "helper") not in graph_js_first.edges


def test_namespace_import_extensionless_collision_prefers_js_over_mjs_independent_of_scan_order(
    tmp_path: Path,
    force_syntax_fallback: None,
    monkeypatch: pytest.MonkeyPatch,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "foo.js": "export function helper() { return 1; }\n",
            "foo.mjs": "export function helper() { return 2; }\n",
            "main.js": (
                'import * as mod from "./foo";\n'
                "export function run() {\n"
                "  return mod.helper();\n"
                "}\n"
            ),
        },
    )

    graph_mjs_first = _build_graph_with_forced_scan_order(
        tmp_path,
        monkeypatch,
        language="javascript",
        relative_scan_order=["main.js", "foo.mjs", "foo.js"],
    )
    graph_js_first = _build_graph_with_forced_scan_order(
        tmp_path,
        monkeypatch,
        language="javascript",
        relative_scan_order=["main.js", "foo.js", "foo.mjs"],
    )

    assert graph_mjs_first.meta.get("graph_source") == "js-syntax-only"
    assert graph_js_first.meta.get("graph_source") == "js-syntax-only"
    assert graph_mjs_first.sorted_edges() == graph_js_first.sorted_edges()
    assert ("main.js", "run", "foo.js", "helper") in graph_mjs_first.edges
    assert ("main.js", "run", "foo.js", "helper") in graph_js_first.edges
    assert ("main.js", "run", "foo.mjs", "helper") not in graph_mjs_first.edges
    assert ("main.js", "run", "foo.mjs", "helper") not in graph_js_first.edges


def test_default_reexport_extensionless_collision_is_scan_order_independent(
    tmp_path: Path,
    force_syntax_fallback: None,
    monkeypatch: pytest.MonkeyPatch,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "dep.js": "export default function depJs() { return 1; }\n",
            "dep.cjs": "module.exports = function depCjs() { return 2; };\n",
            "proxy.cjs": 'module.exports = require("./dep");\n',
            "main.js": (
                'import run from "./proxy.cjs";\n'
                "export function boot() {\n"
                "  return run();\n"
                "}\n"
            ),
        },
    )

    graph_cjs_first = _build_graph_with_forced_scan_order(
        tmp_path,
        monkeypatch,
        language="javascript",
        relative_scan_order=["main.js", "proxy.cjs", "dep.cjs", "dep.js"],
    )
    graph_js_first = _build_graph_with_forced_scan_order(
        tmp_path,
        monkeypatch,
        language="javascript",
        relative_scan_order=["main.js", "proxy.cjs", "dep.js", "dep.cjs"],
    )

    assert graph_cjs_first.meta.get("graph_source") == "js-syntax-only"
    assert graph_js_first.meta.get("graph_source") == "js-syntax-only"
    assert graph_cjs_first.sorted_edges() == graph_js_first.sorted_edges()
    assert ("main.js", "boot", "dep.js", "default") in graph_cjs_first.edges
    assert ("main.js", "boot", "dep.js", "default") in graph_js_first.edges
    assert ("main.js", "boot", "dep.cjs", "default") not in graph_cjs_first.edges
    assert ("main.js", "boot", "dep.cjs", "default") not in graph_js_first.edges


def test_ts_default_import_extensionless_collision_prefers_ts_over_tsx_independent_of_scan_order(
    tmp_path: Path,
    force_syntax_fallback: None,
    monkeypatch: pytest.MonkeyPatch,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "foo.ts": "export default function runTs() { return 1; }\n",
            "foo.tsx": "export default function runTsx() { return 2; }\n",
            "main.ts": (
                'import run from "./foo";\n'
                "export function boot() {\n"
                "  return run();\n"
                "}\n"
            ),
        },
    )

    graph_tsx_first = _build_graph_with_forced_scan_order(
        tmp_path,
        monkeypatch,
        language="typescript",
        relative_scan_order=["main.ts", "foo.tsx", "foo.ts"],
    )
    graph_ts_first = _build_graph_with_forced_scan_order(
        tmp_path,
        monkeypatch,
        language="typescript",
        relative_scan_order=["main.ts", "foo.ts", "foo.tsx"],
    )

    assert graph_tsx_first.meta.get("graph_source") == "ts-syntax-only"
    assert graph_ts_first.meta.get("graph_source") == "ts-syntax-only"
    assert graph_tsx_first.sorted_edges() == graph_ts_first.sorted_edges()
    assert ("main.ts", "boot", "foo.ts", "default") in graph_tsx_first.edges
    assert ("main.ts", "boot", "foo.ts", "default") in graph_ts_first.edges
    assert ("main.ts", "boot", "foo.tsx", "default") not in graph_tsx_first.edges
    assert ("main.ts", "boot", "foo.tsx", "default") not in graph_ts_first.edges


def test_named_import_resolves_exported_function_expression(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "dep.ts": (
                "export const foo = function () {\n"
                "  return 1;\n"
                "};\n"
            ),
            "main.ts": (
                'import { foo } from "./dep";\n'
                "export function run() {\n"
                "  return foo();\n"
                "}\n"
            ),
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="typescript")

    assert graph.meta.get("graph_source") == "ts-syntax-only"
    assert ("main.ts", "run", "dep.ts", "foo") in graph.edges


def test_index_typescript_file_indexes_variable_owned_function_expression(
    tmp_path: Path,
) -> None:
    src_path = tmp_path / "dep.ts"
    src_path.write_text(
        "export const foo = function () {\n"
        "  return 1;\n"
        "};\n"
    )

    index: dict[object, str] = {}

    cfc._index_typescript_file(
        src_path=src_path,
        rel_path=Path("dep.ts"),
        module_name="dep",
        simple_module="dep",
        index=index,
        language="typescript",
    )

    assert index[("dep.ts", "foo")] == "dep.ts"
