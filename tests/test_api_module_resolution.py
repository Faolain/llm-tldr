from pathlib import Path
from types import SimpleNamespace

import pytest

from tldr import api


def _install_fake_extractor(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    extracted_paths: list[Path] = []

    class FakeExtractor:
        def extract(self, file_path: str) -> SimpleNamespace:
            extracted_paths.append(Path(file_path))
            return SimpleNamespace(functions=[], classes=[])

    monkeypatch.setattr(api, "HybridExtractor", FakeExtractor)
    return extracted_paths


def test_typescript_module_resolves_tsx_when_ts_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    module_file = project / "src" / "components" / "Button.tsx"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("export const Button = () => null;\n")
    extracted_paths = _install_fake_extractor(monkeypatch)

    api._get_module_exports(project, "src/components/Button", language="typescript")

    assert extracted_paths == [module_file]


def test_javascript_module_resolves_jsx_when_js_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    module_file = project / "components" / "Widget.jsx"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("export function Widget() { return null; }\n")
    extracted_paths = _install_fake_extractor(monkeypatch)

    api._get_module_exports(project, "components/Widget", language="javascript")

    assert extracted_paths == [module_file]


def test_absolute_module_path_outside_project_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside_module = tmp_path / "outside.py"
    outside_module.write_text("def outside():\n    return 1\n")
    extracted_paths = _install_fake_extractor(monkeypatch)

    with pytest.raises(api.PathTraversalError):
        api._get_module_exports(project, str(outside_module.with_suffix("")), language="python")

    assert extracted_paths == []


def test_non_python_missing_module_error_does_not_mention_init_py(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(ValueError) as exc_info:
        api._get_module_exports(project, "components/Missing", language="typescript")

    assert "__init__.py" not in str(exc_info.value)


def test_python_package_fallback_via_init_py_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    init_file = project / "pkg" / "__init__.py"
    init_file.parent.mkdir(parents=True)
    init_file.write_text("def package_fn():\n    return 1\n")
    extracted_paths = _install_fake_extractor(monkeypatch)

    api._get_module_exports(project, "pkg", language="python")

    assert extracted_paths == [init_file]


def test_get_relevant_context_dispatches_to_module_mode_for_slash_without_dot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    fake_ctx = api.RelevantContext(entry_point="src/components/Button", depth=0, functions=[])
    module_call: dict[str, object] = {}

    def fake_module_exports(
        project_path: Path,
        module_path: str,
        language: str = "python",
        include_docstrings: bool = True,
    ) -> api.RelevantContext:
        module_call["project"] = project_path
        module_call["module_path"] = module_path
        module_call["language"] = language
        module_call["include_docstrings"] = include_docstrings
        return fake_ctx

    def fail_build_project_call_graph(*_args, **_kwargs):
        raise AssertionError("symbol mode call graph should not run for module entry")

    monkeypatch.setattr(api, "_get_module_exports", fake_module_exports)
    monkeypatch.setattr(api, "build_project_call_graph", fail_build_project_call_graph)

    result = api.get_relevant_context(
        project,
        "src/components/Button",
        depth=3,
        language="typescript",
        include_docstrings=False,
    )

    assert result is fake_ctx
    assert module_call == {
        "project": project,
        "module_path": "src/components/Button",
        "language": "typescript",
        "include_docstrings": False,
    }


def test_get_relevant_context_keeps_dotted_entry_in_symbol_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    call_graph_call: dict[str, object] = {}

    def fail_module_exports(*_args, **_kwargs):
        raise AssertionError("module mode should not run for dotted entry")

    def fake_build_project_call_graph(
        project_path: str,
        language: str = "python",
        ignore_spec=None,
        workspace_root=None,
    ) -> SimpleNamespace:
        call_graph_call["project_path"] = project_path
        call_graph_call["language"] = language
        call_graph_call["ignore_spec"] = ignore_spec
        call_graph_call["workspace_root"] = workspace_root
        return SimpleNamespace(edges=set())

    monkeypatch.setattr(api, "_get_module_exports", fail_module_exports)
    monkeypatch.setattr(api, "build_project_call_graph", fake_build_project_call_graph)
    monkeypatch.setattr("tldr.cross_file_calls.scan_project", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("tldr.workspace.load_workspace_config", lambda *_args, **_kwargs: None)

    result = api.get_relevant_context(project, "pkg/mod.py", depth=1, language="python")

    assert call_graph_call["project_path"] == str(project)
    assert call_graph_call["language"] == "python"
    assert result.entry_point == "pkg/mod.py"
    assert result.depth == 1
    assert [ctx.name for ctx in result.functions] == ["pkg/mod.py"]
