from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .index import IndexPaths


def _iter_index_dirs(cache_root: Path) -> list[Path]:
    indexes_dir = cache_root / ".tldr" / "indexes"
    if not indexes_dir.exists():
        return []
    try:
        entries = [p for p in indexes_dir.iterdir() if p.is_dir()]
    except OSError:
        return []
    return sorted(entries, key=lambda p: p.name)


def _read_meta(meta_path: Path) -> tuple[dict | None, str | None]:
    if not meta_path.exists():
        return None, "missing meta.json"
    try:
        return json.loads(meta_path.read_text()), None
    except (json.JSONDecodeError, OSError):
        return None, "invalid meta.json"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                if file_path.is_symlink():
                    continue
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _extract_last_used(meta: dict | None, meta_path: Path, index_dir: Path) -> str | None:
    if meta:
        for key in ("last_used_at", "created_at"):
            value = meta.get(key)
            if value:
                return value
    for path in (meta_path, index_dir):
        try:
            return _iso_from_timestamp(path.stat().st_mtime)
        except OSError:
            continue
    return None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _extract_scan_root(meta: dict | None) -> str | None:
    if not meta:
        return None
    return meta.get("scan_root_rel_to_cache_root") or meta.get("scan_root_abs")


def _daemon_running(cache_root: Path, index_key: str, meta: dict | None) -> bool:
    try:
        from tldr.daemon.identity import get_pid_path, resolve_daemon_identity
        from tldr.daemon.startup import is_daemon_alive
    except Exception:
        return False

    scan_root = meta.get("scan_root_abs") if meta else None
    if scan_root:
        scan_root_path = Path(scan_root)
    else:
        scan_root_path = cache_root
    index_id = meta.get("index_id") if meta else None
    identity = resolve_daemon_identity(
        scan_root_path,
        cache_root=cache_root,
        index_id=index_id,
        index_key=index_key,
    )
    if not get_pid_path(identity).exists():
        return False
    return is_daemon_alive(identity)


def _clear_port_file(cache_root: Path, index_key: str, meta: dict | None) -> None:
    try:
        from tldr.daemon.identity import clear_port_file, resolve_daemon_identity
    except Exception:
        return

    scan_root = meta.get("scan_root_abs") if meta else None
    if scan_root:
        scan_root_path = Path(scan_root)
    else:
        scan_root_path = cache_root
    index_id = meta.get("index_id") if meta else None
    identity = resolve_daemon_identity(
        scan_root_path,
        cache_root=cache_root,
        index_id=index_id,
        index_key=index_key,
    )
    clear_port_file(identity)


def list_indexes(cache_root: Path) -> dict:
    entries: list[dict] = []
    total_size = 0
    for index_dir in _iter_index_dirs(cache_root):
        meta_path = index_dir / "meta.json"
        meta, meta_error = _read_meta(meta_path)
        size_bytes = _dir_size_bytes(index_dir)
        total_size += size_bytes
        entry = {
            "index_id": meta.get("index_id") if meta else None,
            "index_key": meta.get("index_key") if meta else index_dir.name,
            "scan_root": _extract_scan_root(meta),
            "last_used_at": _extract_last_used(meta, meta_path, index_dir),
            "size_bytes": size_bytes,
            "index_dir": str(index_dir),
            "meta_path": str(meta_path),
            "meta_ok": meta is not None and meta_error is None,
            "meta_error": meta_error,
        }
        entries.append(entry)
    return {
        "cache_root": str(cache_root),
        "total_size_bytes": total_size,
        "indexes": entries,
    }


def _resolve_index_ref(
    cache_root: Path, index_ref: str
) -> tuple[Path, dict | None, str | None, str]:
    indexes_dir = cache_root / ".tldr" / "indexes"
    if not indexes_dir.exists():
        raise FileNotFoundError("No indexes found under cache root")

    matches: list[tuple[Path, dict | None, str | None, str]] = []
    fallback: tuple[Path, dict | None, str | None, str] | None = None

    for index_dir in _iter_index_dirs(cache_root):
        meta_path = index_dir / "meta.json"
        meta, meta_error = _read_meta(meta_path)
        if index_dir.name == index_ref:
            fallback = (index_dir, meta, meta_error, "index_key")
        if meta and meta.get("index_id") == index_ref:
            matches.append((index_dir, meta, meta_error, "index_id"))

    if matches:
        if len(matches) > 1:
            raise ValueError(f"Multiple indexes found for id {index_ref}")
        return matches[0]
    if fallback:
        return fallback
    raise FileNotFoundError(f"Index not found: {index_ref}")


def _artifact_info(path: Path) -> dict:
    info = {
        "path": str(path),
        "exists": False,
        "size_bytes": None,
        "modified_at": None,
    }
    try:
        if path.exists():
            stat = path.stat()
            info["exists"] = True
            info["size_bytes"] = stat.st_size
            info["modified_at"] = _iso_from_timestamp(stat.st_mtime)
    except OSError:
        pass
    return info


def get_index_info(cache_root: Path, index_ref: str) -> dict:
    index_dir, meta, meta_error, resolved_by = _resolve_index_ref(cache_root, index_ref)
    if meta is None or meta_error is not None:
        raise ValueError(
            f"Index metadata missing or invalid for {index_ref}. Use --force with index rm to delete."
        )

    index_key = meta.get("index_key") or index_dir.name
    ignore_file = None
    ignore_meta = meta.get("ignore") if meta else None
    if ignore_meta:
        ignore_file = ignore_meta.get("file_abs")
    ignore_path = Path(ignore_file) if ignore_file else None
    paths = IndexPaths.from_parts(
        cache_root, index_key, ignore_file=ignore_path
    )

    artifacts = {
        "meta": _artifact_info(paths.meta),
        "ignore_file": _artifact_info(paths.ignore_file),
        "call_graph": _artifact_info(paths.call_graph),
        "languages": _artifact_info(paths.languages),
        "dirty": _artifact_info(paths.dirty),
        "file_hashes": _artifact_info(paths.file_hashes),
        "content_index": _artifact_info(paths.content_index),
        "hook_activity": _artifact_info(paths.hook_activity),
        "semantic_faiss": _artifact_info(paths.semantic_faiss),
        "semantic_metadata": _artifact_info(paths.semantic_metadata),
        "daemon_status": _artifact_info(paths.daemon_status),
    }

    return {
        "index_id": meta.get("index_id"),
        "index_key": index_key,
        "scan_root": _extract_scan_root(meta),
        "cache_root": str(cache_root),
        "index_dir": str(index_dir),
        "resolved_by": resolved_by,
        "meta": meta,
        "artifacts": artifacts,
        "total_size_bytes": _dir_size_bytes(index_dir),
        "daemon_running": _daemon_running(cache_root, index_key, meta),
    }


def remove_index(cache_root: Path, index_ref: str, *, force: bool = False) -> dict:
    index_dir, meta, meta_error, resolved_by = _resolve_index_ref(cache_root, index_ref)
    index_key = index_dir.name
    if meta:
        index_key = meta.get("index_key") or index_key

    if (meta is None or meta_error is not None) and not force:
        raise ValueError(
            f"Index metadata missing or invalid for {index_ref}. Use --force to remove."
        )

    running = _daemon_running(cache_root, index_key, meta)
    if running and not force:
        raise ValueError(
            "Index appears to have a running daemon. Stop it first or pass --force."
        )

    size_bytes = _dir_size_bytes(index_dir)
    shutil.rmtree(index_dir)
    _clear_port_file(cache_root, index_key, meta)

    return {
        "index_id": meta.get("index_id") if meta else None,
        "index_key": index_key,
        "index_dir": str(index_dir),
        "resolved_by": resolved_by,
        "removed": True,
        "daemon_running": running,
        "freed_bytes": size_bytes,
    }


def gc_indexes(
    cache_root: Path,
    *,
    days: int | None = None,
    max_total_mb: float | None = None,
    force: bool = False,
) -> dict:
    if days is None and max_total_mb is None:
        raise ValueError("gc requires --days and/or --max-total-mb")

    entries = []
    for index_dir in _iter_index_dirs(cache_root):
        meta_path = index_dir / "meta.json"
        meta, meta_error = _read_meta(meta_path)
        size_bytes = _dir_size_bytes(index_dir)
        index_key = meta.get("index_key") if meta else index_dir.name
        last_used_at = _extract_last_used(meta, meta_path, index_dir)
        last_used_dt = _parse_iso(last_used_at)
        entry = {
            "index_dir": index_dir,
            "index_id": meta.get("index_id") if meta else None,
            "index_key": index_key,
            "meta": meta,
            "meta_error": meta_error,
            "size_bytes": size_bytes,
            "last_used_at": last_used_at,
            "last_used_dt": last_used_dt,
            "daemon_running": _daemon_running(cache_root, index_key, meta),
        }
        entries.append(entry)

    to_remove: dict[str, dict] = {}
    skipped: list[dict] = []
    now = datetime.now(timezone.utc)

    def mark_skip(entry: dict, reason: str) -> None:
        skipped.append(
            {
                "index_id": entry.get("index_id"),
                "index_key": entry.get("index_key"),
                "index_dir": str(entry.get("index_dir")),
                "reason": reason,
            }
        )

    def can_remove(entry: dict) -> bool:
        if entry.get("daemon_running") and not force:
            mark_skip(entry, "daemon running")
            return False
        if entry.get("meta_error") and not force:
            mark_skip(entry, entry.get("meta_error") or "invalid meta")
            return False
        return True

    if days is not None:
        cutoff = now - timedelta(days=days)
        for entry in entries:
            last_used = entry.get("last_used_dt")
            if last_used is None or last_used >= cutoff:
                continue
            if not can_remove(entry):
                continue
            to_remove[str(entry["index_dir"])] = {"entry": entry, "reason": "age"}

    if max_total_mb is not None:
        max_bytes = int(max_total_mb * 1024 * 1024)
        remaining = [
            entry
            for entry in entries
            if str(entry["index_dir"]) not in to_remove
        ]
        total_bytes = sum(entry.get("size_bytes", 0) for entry in remaining)
        if total_bytes > max_bytes:
            remaining_sorted = sorted(
                remaining,
                key=lambda e: (
                    e.get("last_used_dt") is None,
                    e.get("last_used_dt") or datetime.min.replace(tzinfo=timezone.utc),
                ),
            )
            for entry in remaining_sorted:
                if total_bytes <= max_bytes:
                    break
                if not can_remove(entry):
                    continue
                to_remove[str(entry["index_dir"])] = {
                    "entry": entry,
                    "reason": "size",
                }
                total_bytes -= entry.get("size_bytes", 0)

    removed: list[dict] = []
    freed_bytes = 0
    for item in to_remove.values():
        entry = item["entry"]
        index_dir = entry["index_dir"]
        size_bytes = entry.get("size_bytes", 0)
        try:
            shutil.rmtree(index_dir)
            _clear_port_file(cache_root, entry.get("index_key"), entry.get("meta"))
            freed_bytes += size_bytes
            removed.append(
                {
                    "index_id": entry.get("index_id"),
                    "index_key": entry.get("index_key"),
                    "index_dir": str(index_dir),
                    "reason": item["reason"],
                    "freed_bytes": size_bytes,
                }
            )
        except OSError as exc:
            mark_skip(entry, f"failed to remove: {exc}")

    remaining_entries = [
        entry
        for entry in entries
        if str(entry["index_dir"]) not in to_remove
    ]
    remaining_bytes = sum(entry.get("size_bytes", 0) for entry in remaining_entries)

    return {
        "cache_root": str(cache_root),
        "criteria": {
            "days": days,
            "max_total_mb": max_total_mb,
            "force": force,
        },
        "removed": removed,
        "skipped": skipped,
        "total_freed_bytes": freed_bytes,
        "remaining_bytes": remaining_bytes,
    }
