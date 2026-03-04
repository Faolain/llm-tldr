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
