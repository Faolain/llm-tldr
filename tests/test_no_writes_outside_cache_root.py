from pathlib import Path

from tldr.indexing import compute_index_key
from tldr import cli


def test_warm_does_not_write_under_scan_root(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "repo"
    scan_root = tmp_path / "dep"
    cache_root.mkdir()
    scan_root.mkdir()

    (scan_root / "mod.py").write_text("def foo():\n    return 1\n")

    index_id = "dep:test"
    argv = [
        "tldr",
        "warm",
        str(scan_root),
        "--cache-root",
        str(cache_root),
        "--index",
        index_id,
        "--lang",
        "python",
    ]
    monkeypatch.setattr("sys.argv", argv)
    cli.main()

    assert not (scan_root / ".tldr").exists()
    assert not (scan_root / ".tldrignore").exists()

    index_key = compute_index_key(index_id)
    call_graph = cache_root / ".tldr" / "indexes" / index_key / "cache" / "call_graph.json"
    assert call_graph.exists()
