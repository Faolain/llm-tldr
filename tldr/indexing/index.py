from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexConfig:
    cache_root: Path
    scan_root: Path
    index_id: str
    index_key: str


@dataclass(frozen=True)
class IndexPaths:
    tldr_dir: Path
    indexes_dir: Path
    index_dir: Path
    meta: Path
    cache_dir: Path
    call_graph: Path
    languages: Path
    dirty: Path
    semantic_dir: Path
    semantic_faiss: Path
    semantic_metadata: Path

    @classmethod
    def from_config(cls, config: IndexConfig) -> "IndexPaths":
        tldr_dir = config.cache_root / ".tldr"
        indexes_dir = tldr_dir / "indexes"
        index_dir = indexes_dir / config.index_key
        meta = index_dir / "meta.json"
        cache_dir = index_dir / "cache"
        semantic_dir = cache_dir / "semantic"
        return cls(
            tldr_dir=tldr_dir,
            indexes_dir=indexes_dir,
            index_dir=index_dir,
            meta=meta,
            cache_dir=cache_dir,
            call_graph=cache_dir / "call_graph.json",
            languages=index_dir / "languages.json",
            dirty=cache_dir / "dirty.json",
            semantic_dir=semantic_dir,
            semantic_faiss=semantic_dir / "index.faiss",
            semantic_metadata=semantic_dir / "metadata.json",
        )


@dataclass(frozen=True)
class IndexContext:
    mode: Literal["legacy", "index"]
    scan_root: Path
    cache_root: Path | None
    index_id: str | None
    index_key: str | None
    paths: IndexPaths | None
    config: IndexConfig | None


def compute_index_key(index_id: str) -> str:
    digest = hashlib.sha256(index_id.encode("utf-8")).digest()
    encoded = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
    return encoded[:20]


def derive_index_id(scan_root: Path, cache_root: Path) -> str:
    scan_root = scan_root.resolve()
    cache_root = cache_root.resolve()
    try:
        rel = scan_root.relative_to(cache_root)
        return f"path:{rel.as_posix()}"
    except ValueError:
        return f"abs:{scan_root}"


def _normalize_abs(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _scan_root_rel(scan_root: Path, cache_root: Path) -> str | None:
    try:
        rel = scan_root.resolve().relative_to(cache_root.resolve())
        return rel.as_posix()
    except ValueError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _create_meta(config: IndexConfig) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "index_id": config.index_id,
        "index_key": config.index_key,
        "scan_root_abs": _normalize_abs(config.scan_root),
        "scan_root_rel_to_cache_root": _scan_root_rel(config.scan_root, config.cache_root),
        "cache_root_abs": _normalize_abs(config.cache_root),
        "created_at": _now_iso(),
        "last_used_at": _now_iso(),
    }


def load_meta(paths: IndexPaths) -> dict | None:
    if not paths.meta.exists():
        return None
    try:
        return json.loads(paths.meta.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_meta(paths: IndexPaths, meta: dict) -> None:
    paths.index_dir.mkdir(parents=True, exist_ok=True)
    paths.meta.write_text(json.dumps(meta, indent=2))


def _validate_meta(meta: dict, config: IndexConfig) -> None:
    if meta.get("index_id") != config.index_id:
        raise ValueError("Index ID mismatch for existing index directory.")

    # Compare scan root (prefer relative if available and applicable)
    rel = meta.get("scan_root_rel_to_cache_root")
    if rel:
        current_rel = _scan_root_rel(config.scan_root, config.cache_root)
        if current_rel and rel != current_rel:
            raise ValueError("Scan root mismatch for existing index directory.")
        if current_rel is None:
            # Fall back to absolute comparison if current scan_root isn't under cache_root
            if meta.get("scan_root_abs") != _normalize_abs(config.scan_root):
                raise ValueError("Scan root mismatch for existing index directory.")
        return

    if meta.get("scan_root_abs") != _normalize_abs(config.scan_root):
        raise ValueError("Scan root mismatch for existing index directory.")


def ensure_index(
    config: IndexConfig,
    *,
    allow_create: bool,
    force_rebind: bool = False,
) -> tuple[IndexPaths, dict]:
    paths = IndexPaths.from_config(config)
    meta = load_meta(paths)

    if meta is not None:
        try:
            _validate_meta(meta, config)
        except ValueError:
            if not force_rebind:
                raise
            if paths.index_dir.exists():
                shutil.rmtree(paths.index_dir)
            meta = None

    if meta is None:
        if not allow_create:
            raise FileNotFoundError(
                f"Index not initialized for {config.index_id}. Run warm or semantic index first."
            )
        meta = _create_meta(config)
        _write_meta(paths, meta)

    return paths, meta


def update_meta_semantic(
    paths: IndexPaths,
    config: IndexConfig,
    *,
    model: str,
    dim: int,
    lang: str | None = None,
) -> None:
    meta = load_meta(paths) or _create_meta(config)
    meta["semantic"] = {
        "model": model,
        "dim": dim,
        "lang": lang,
    }
    meta["last_used_at"] = _now_iso()
    _write_meta(paths, meta)


def get_index_context(
    *,
    scan_root: Path,
    cache_root_arg: str | Path | None,
    index_id_arg: str | None,
    allow_create: bool,
    force_rebind: bool = False,
) -> IndexContext:
    if cache_root_arg is None or str(cache_root_arg).strip() == "":
        return IndexContext(
            mode="legacy",
            scan_root=scan_root,
            cache_root=None,
            index_id=None,
            index_key=None,
            paths=None,
            config=None,
        )

    cache_root = Path(cache_root_arg).resolve()
    if index_id_arg is None or index_id_arg == "":
        index_id = derive_index_id(scan_root, cache_root)
    else:
        index_id = index_id_arg

    index_key = compute_index_key(index_id)
    config = IndexConfig(
        cache_root=cache_root,
        scan_root=scan_root.resolve(),
        index_id=index_id,
        index_key=index_key,
    )
    paths, _ = ensure_index(
        config, allow_create=allow_create, force_rebind=force_rebind
    )
    return IndexContext(
        mode="index",
        scan_root=config.scan_root,
        cache_root=config.cache_root,
        index_id=config.index_id,
        index_key=config.index_key,
        paths=paths,
        config=config,
    )
