import subprocess
import sys

import pytest

from tldr.indexing import get_index_context


def _git_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_cache_root_git_uses_repo_root_from_scan_root(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    scan_root = repo / "sub" / "dir"
    scan_root.mkdir(parents=True)

    ctx = get_index_context(
        scan_root=scan_root,
        cache_root_arg="git",
        index_id_arg=None,
        allow_create=True,
    )

    assert ctx.cache_root == repo.resolve()
    assert ctx.index_id == "path:sub/dir"


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_cache_root_git_falls_back_to_cwd(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.chdir(repo)

    ctx = get_index_context(
        scan_root=outside,
        cache_root_arg="git",
        index_id_arg=None,
        allow_create=True,
    )

    assert ctx.cache_root == repo.resolve()
    assert ctx.index_id.startswith("abs:")


def test_cache_root_git_errors_when_no_repo(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tldr.cli",
            "--cache-root",
            "git",
            "warm",
            str(project),
        ],
        capture_output=True,
        text=True,
        cwd=project,
    )

    assert result.returncode != 0
    assert "cache_root 'git' requested but no git repository found" in (
        result.stderr or ""
    )
