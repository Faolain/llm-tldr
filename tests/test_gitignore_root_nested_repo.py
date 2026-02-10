import subprocess
from pathlib import Path

import pytest

from tldr.cross_file_calls import scan_project
from tldr.indexing import get_index_context
from tldr.tldrignore import IgnoreSpec


def _git_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_index_mode_uses_scan_root_gitignore_for_nested_repo(tmp_path: Path):
    """Regression: when cache-root is inside a host repo that gitignores the corpus,
    index-mode should still respect the corpus repo's own .gitignore.
    """
    host = tmp_path / "host"
    host.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=host, check=True)
    (host / ".gitignore").write_text("benchmark/\n")

    cache_root = host / "benchmark" / "cache-root"
    cache_root.mkdir(parents=True)

    scan_root = host / "benchmark" / "corpora" / "dep"
    scan_root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=scan_root, check=True)

    (scan_root / "keep.py").write_text("def keep():\n    return 1\n")

    ctx = get_index_context(
        scan_root=scan_root,
        cache_root_arg=cache_root,
        index_id_arg="repo:dep",
        allow_create=True,
    )

    cfg = ctx.config
    ignore_spec = IgnoreSpec(
        project_dir=scan_root,
        use_gitignore=cfg.use_gitignore,
        cli_patterns=list(cfg.cli_patterns or ()),
        ignore_file=cfg.ignore_file,
        gitignore_root=cfg.gitignore_root,
    )

    files = scan_project(
        scan_root,
        language="python",
        respect_ignore=True,
        ignore_spec=ignore_spec,
    )
    assert any(Path(p).name == "keep.py" for p in files)

