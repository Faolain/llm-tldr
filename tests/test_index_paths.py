from pathlib import Path

from tldr.indexing import IndexConfig, IndexPaths, compute_index_key


def test_index_key_deterministic():
    key1 = compute_index_key("node:zod@3.23.8")
    key2 = compute_index_key("node:zod@3.23.8")
    assert key1 == key2


def test_index_paths_distinct(tmp_path: Path):
    cache_root = tmp_path / "repo"
    scan_root = tmp_path / "scan"
    index_key_a = compute_index_key("dep:a")
    index_key_b = compute_index_key("dep:b")
    config_a = IndexConfig(
        cache_root=cache_root,
        scan_root=scan_root,
        index_id="dep:a",
        index_key=index_key_a,
        ignore_file=cache_root / ".tldr" / "indexes" / index_key_a / ".tldrignore",
        use_gitignore=True,
        gitignore_root=cache_root,
        cli_patterns=None,
        no_ignore=False,
    )
    config_b = IndexConfig(
        cache_root=cache_root,
        scan_root=scan_root,
        index_id="dep:b",
        index_key=index_key_b,
        ignore_file=cache_root / ".tldr" / "indexes" / index_key_b / ".tldrignore",
        use_gitignore=True,
        gitignore_root=cache_root,
        cli_patterns=None,
        no_ignore=False,
    )
    paths_a = IndexPaths.from_config(config_a)
    paths_b = IndexPaths.from_config(config_b)
    assert paths_a.index_dir != paths_b.index_dir
