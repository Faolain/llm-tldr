import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldr import cli
from tldr.indexing import compute_index_key, get_index_context
from tldr.indexing.management import gc_indexes, get_index_info, list_indexes, remove_index


def _create_index(cache_root: Path, scan_root: Path, index_id: str) -> None:
    get_index_context(
        scan_root=scan_root,
        cache_root_arg=cache_root,
        index_id_arg=index_id,
        allow_create=True,
        force_rebind=False,
        ignore_file_arg=None,
        use_gitignore_arg=False,
        cli_patterns_arg=None,
        no_ignore_arg=False,
    )


def test_index_cli_list_info_rm(tmp_path: Path, monkeypatch, capsys):
    cache_root = tmp_path / "repo"
    scan_root = tmp_path / "dep"
    cache_root.mkdir()
    scan_root.mkdir()
    (scan_root / "mod.py").write_text("def foo():\n    return 1\n")

    index_id = "dep:test"
    _create_index(cache_root, scan_root, index_id)

    monkeypatch.setattr(
        "sys.argv",
        ["tldr", "index", "list", "--cache-root", str(cache_root)],
    )
    cli.main()
    data = json.loads(capsys.readouterr().out)
    assert any(entry["index_id"] == index_id for entry in data["indexes"])

    monkeypatch.setattr(
        "sys.argv",
        [
            "tldr",
            "index",
            "info",
            index_id,
            "--cache-root",
            str(cache_root),
        ],
    )
    cli.main()
    info = json.loads(capsys.readouterr().out)
    assert info["index_id"] == index_id
    assert info["meta"]["index_id"] == index_id

    monkeypatch.setattr(
        "sys.argv",
        ["tldr", "index", "rm", index_id, "--cache-root", str(cache_root)],
    )
    cli.main()
    removed = json.loads(capsys.readouterr().out)
    assert removed["removed"] is True
    assert not (cache_root / ".tldr" / "indexes" / removed["index_key"]).exists()


def test_index_list_info_shared_scan_root(tmp_path: Path):
    cache_root = tmp_path / "repo"
    scan_root = tmp_path / "dep"
    cache_root.mkdir()
    scan_root.mkdir()
    (scan_root / "mod.py").write_text("def foo():\n    return 1\n")

    index_a = "dep:a"
    index_b = "dep:b"
    _create_index(cache_root, scan_root, index_a)
    _create_index(cache_root, scan_root, index_b)

    listing = list_indexes(cache_root)
    index_ids = {entry["index_id"] for entry in listing["indexes"]}
    assert {index_a, index_b}.issubset(index_ids)

    info_a = get_index_info(cache_root, index_a)
    info_b = get_index_info(cache_root, index_b)
    assert info_a["index_id"] == index_a
    assert info_b["index_id"] == index_b


def test_index_rm_requires_force_when_daemon_running(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "repo"
    scan_root = tmp_path / "dep"
    cache_root.mkdir()
    scan_root.mkdir()

    index_id = "dep:running"
    _create_index(cache_root, scan_root, index_id)

    from tldr.indexing import management

    monkeypatch.setattr(management, "_daemon_running", lambda *args, **kwargs: True)
    with pytest.raises(ValueError):
        remove_index(cache_root, index_id, force=False)

    result = remove_index(cache_root, index_id, force=True)
    assert result["removed"] is True


def test_index_gc_removes_old(tmp_path: Path):
    cache_root = tmp_path / "repo"
    scan_root_a = tmp_path / "dep_a"
    scan_root_b = tmp_path / "dep_b"
    cache_root.mkdir()
    scan_root_a.mkdir()
    scan_root_b.mkdir()

    index_a = "dep:old"
    index_b = "dep:new"
    _create_index(cache_root, scan_root_a, index_a)
    _create_index(cache_root, scan_root_b, index_b)

    meta_path = (
        cache_root
        / ".tldr"
        / "indexes"
        / compute_index_key(index_a)
        / "meta.json"
    )
    meta = json.loads(meta_path.read_text())
    meta["last_used_at"] = (
        datetime.now(timezone.utc) - timedelta(days=10)
    ).isoformat().replace("+00:00", "Z")
    meta_path.write_text(json.dumps(meta, indent=2))

    result = gc_indexes(cache_root, days=7, max_total_mb=None, force=False)
    removed_ids = {entry["index_id"] for entry in result["removed"]}
    assert index_a in removed_ids
    assert index_b not in removed_ids


def test_index_gc_skips_running_daemon(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "repo"
    scan_root = tmp_path / "dep"
    cache_root.mkdir()
    scan_root.mkdir()

    index_id = "dep:running"
    _create_index(cache_root, scan_root, index_id)

    meta_path = (
        cache_root
        / ".tldr"
        / "indexes"
        / compute_index_key(index_id)
        / "meta.json"
    )
    meta = json.loads(meta_path.read_text())
    meta["last_used_at"] = (
        datetime.now(timezone.utc) - timedelta(days=1)
    ).isoformat().replace("+00:00", "Z")
    meta_path.write_text(json.dumps(meta, indent=2))

    monkeypatch.setattr("tldr.indexing.management._daemon_running", lambda *_a, **_k: True)
    result = gc_indexes(cache_root, days=0, max_total_mb=None, force=False)
    removed_ids = {entry["index_id"] for entry in result["removed"]}
    skipped_reasons = {entry.get("reason") for entry in result["skipped"]}

    assert index_id not in removed_ids
    assert "daemon running" in skipped_reasons
