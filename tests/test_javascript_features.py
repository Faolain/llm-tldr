"""Focused regression tests for JavaScript/CommonJS call graph behavior."""

from pathlib import Path

import pytest

from tldr.cross_file_calls import (
    build_project_call_graph,
    parse_ts_imports,
    scan_project,
)


@pytest.fixture
def force_syntax_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDR_TS_RESOLVER", "syntax")


def test_javascript_scan_project_discovers_cjs_and_mjs(
    tmp_path: Path,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "a.cjs": "module.exports = function a() {};",
            "b.mjs": "export default function b() {}",
            "c.js": "export function c() {}",
        },
    )

    files = {Path(path).name for path in scan_project(tmp_path, language="javascript")}
    assert files == {"a.cjs", "b.mjs", "c.js"}


def test_default_import_from_module_exports_identifier_resolves_to_default(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "dep.cjs": """
function handler() { return 1; }
module.exports = handler;
""",
            "main.js": """
import run from "./dep.cjs";
export function boot() {
    return run();
}
""",
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="javascript")

    assert ("main.js", "boot", "dep.cjs", "default") in graph.edges
    assert ("main.js", "boot", "dep.cjs", "handler") not in graph.edges


def test_default_import_does_not_promote_singleton_named_export(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "dep.js": "export function only() { return 1; }",
            "main.js": """
import only from "./dep";
export function boot() {
    return only();
}
""",
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="javascript")

    assert ("main.js", "boot", "dep.js", "only") not in graph.edges
    assert ("main.js", "boot", "dep.js", "default") not in graph.edges


def test_module_exports_require_reexport_resolves_default_chain(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "dep.js": """
export default function original() {
    return 1;
}
""",
            "proxy.cjs": 'module.exports = require("./dep");',
            "main.js": """
import run from "./proxy.cjs";
export function boot() {
    return run();
}
""",
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="javascript")

    assert ("main.js", "boot", "dep.js", "default") in graph.edges


def test_multi_declarator_require_aliases_resolve_both_defaults(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "dep1.cjs": "module.exports = function dep1() { return 1; };",
            "dep2.cjs": "module.exports = function dep2() { return 2; };",
            "main.js": """
const one = require("./dep1.cjs"), two = require("./dep2.cjs");
export function boot() {
    one();
    return two();
}
""",
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="javascript")

    assert ("main.js", "boot", "dep1.cjs", "default") in graph.edges
    assert ("main.js", "boot", "dep2.cjs", "default") in graph.edges


def test_function_local_require_alias_scope_leak_does_not_cross_siblings(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "dep.cjs": "module.exports = function dep() { return 1; };",
            "main.js": """
export function owner() {
    const run = require("./dep.cjs");
    return run();
}

export function sibling() {
    return run();
}
""",
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="javascript")

    assert ("main.js", "owner", "dep.cjs", "default") in graph.edges
    assert ("main.js", "sibling", "dep.cjs", "default") not in graph.edges


def test_parse_ts_imports_scopes_require_alias_inside_function_expression(
    tmp_path: Path,
) -> None:
    source = tmp_path / "main.js"
    source.write_text("""
const owner = function () {
    const run = require("./dep.cjs");
    return run();
};
""")

    imports = parse_ts_imports(source, language="javascript")
    run_aliases = [
        imp
        for imp in imports
        if imp.get("module") == "./dep.cjs" and imp.get("default") == "run"
    ]

    assert len(run_aliases) == 1
    assert run_aliases[0].get("scope") == "owner"


def test_function_expression_local_require_alias_scope_does_not_leak_to_sibling(
    tmp_path: Path,
    force_syntax_fallback: None,
    write_project,
) -> None:
    write_project(
        tmp_path,
        {
            "dep.cjs": "module.exports = function dep() { return 1; };",
            "main.js": """
const owner = function () {
    const run = require("./dep.cjs");
    return run();
};

export function sibling() {
    return run();
}
""",
        },
    )

    graph = build_project_call_graph(str(tmp_path), language="javascript")

    assert ("main.js", "owner", "dep.cjs", "default") in graph.edges
    assert ("main.js", "sibling", "dep.cjs", "default") not in graph.edges
