from pathlib import Path

import pytest

from tldr.tldrignore import load_ignore_patterns, should_ignore


def test_should_ignore_preserves_trailing_slash_for_directory_patterns(tmp_path: Path):
    pytest.importorskip("pathspec")
    (tmp_path / ".tldrignore").write_text(".venv/\n")
    spec = load_ignore_patterns(tmp_path)

    assert spec.match_file(".venv/")
    assert should_ignore(".venv/", tmp_path, spec=spec, use_gitignore=False)


def test_has_negation_for_file_public_helper():
    pathspec = pytest.importorskip("pathspec")
    import tldr.tldrignore as tldrignore

    assert hasattr(tldrignore, "has_negation_for_file")

    spec = pathspec.PathSpec.from_lines("gitwildmatch", ["!keep.py", "ignored.py"])
    assert tldrignore.has_negation_for_file(spec, "keep.py")
    assert not tldrignore.has_negation_for_file(spec, "ignored.py")
