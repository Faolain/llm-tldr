"""Regression tests for syntax-fallback TS/JS import resolution edge cases."""

from pathlib import Path

import pytest

from tldr.cross_file_calls import build_project_call_graph


def _write_project(root: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        file_path = root / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)


@pytest.fixture
def force_syntax_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDR_TS_RESOLVER", "syntax")


def test_named_import_with_dotted_basename_and_explicit_extension_resolves(
    tmp_path: Path,
    force_syntax_fallback: None,
) -> None:
    _write_project(
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
) -> None:
    _write_project(
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
