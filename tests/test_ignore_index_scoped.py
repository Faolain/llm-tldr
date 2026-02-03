from pathlib import Path

import pytest

from tldr.cross_file_calls import scan_project
from tldr.indexing import get_index_context
from tldr.tldrignore import IgnoreSpec


def test_index_scoped_ignore_file_applies(tmp_path: Path):
    pytest.importorskip("pathspec")
    cache_root = tmp_path / "repo"
    scan_root = tmp_path / "dep"
    cache_root.mkdir()
    scan_root.mkdir()

    (scan_root / "keep.py").write_text("def keep():\n    return 1\n")
    (scan_root / "ignored.py").write_text("def ignored():\n    return 2\n")
    (scan_root / ".tldrignore").write_text("# empty\n")

    ctx = get_index_context(
        scan_root=scan_root,
        cache_root_arg=cache_root,
        index_id_arg="dep:ignore",
        allow_create=True,
    )

    ignore_path = ctx.paths.ignore_file
    ignore_path.parent.mkdir(parents=True, exist_ok=True)
    ignore_path.write_text("ignored.py\n")

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
    file_names = {Path(f).name for f in files}
    assert "keep.py" in file_names
    assert "ignored.py" not in file_names


def test_ignore_file_change_invalidates_call_graph(tmp_path: Path):
    cache_root = tmp_path / "repo"
    scan_root = tmp_path / "dep"
    cache_root.mkdir()
    scan_root.mkdir()

    (scan_root / "mod.py").write_text("def foo():\n    return 1\n")

    ctx = get_index_context(
        scan_root=scan_root,
        cache_root_arg=cache_root,
        index_id_arg="dep:invalidate",
        allow_create=True,
    )

    ignore_path = ctx.paths.ignore_file
    ignore_path.parent.mkdir(parents=True, exist_ok=True)
    ignore_path.write_text("ignored.py\n")

    ctx.paths.call_graph.parent.mkdir(parents=True, exist_ok=True)
    ctx.paths.call_graph.write_text('{"edges": []}')
    assert ctx.paths.call_graph.exists()

    # Change ignore file contents -> should invalidate caches without rebind
    ignore_path.write_text("ignored.py\nother.py\n")

    get_index_context(
        scan_root=scan_root,
        cache_root_arg=cache_root,
        index_id_arg="dep:invalidate",
        allow_create=False,
    )

    assert not ctx.paths.call_graph.exists()
