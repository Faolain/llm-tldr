from __future__ import annotations

import pytest

import tldr.change_impact as change_impact


@pytest.mark.parametrize("extension", [".mjs", ".cjs"])
def test_analyze_change_impact_keeps_javascript_module_variants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    extension: str,
) -> None:
    changed_file = f"mod{extension}"
    (tmp_path / changed_file).write_text("export const value = 1;\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_find_affected_tests(
        project_path: str,
        changed_files: list[str],
        **_: object,
    ) -> dict[str, object]:
        captured["project_path"] = project_path
        captured["changed_files"] = list(changed_files)
        return {
            "changed_files": list(changed_files),
            "changed_functions": [],
            "affected_tests": ["tests/smoke.test.js"],
            "affected_count": 1,
            "skipped_count": 0,
            "total_tests": 1,
            "test_command": ["npm", "test", "--", "tests/smoke.test.js"],
        }

    monkeypatch.setattr(change_impact, "find_affected_tests", _fake_find_affected_tests)

    result = change_impact.analyze_change_impact(
        str(tmp_path),
        files=[changed_file],
        language="javascript",
    )

    assert captured["project_path"] == str(tmp_path.resolve())
    assert captured["changed_files"] == [changed_file]
    assert result["changed_files"] == [changed_file]
    assert result["affected_tests"] == ["tests/smoke.test.js"]
    assert result["source"] == "explicit"
