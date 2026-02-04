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

_UNSET = object()


@dataclass(frozen=True)
class IndexConfig:
    cache_root: Path
    scan_root: Path
    index_id: str
    index_key: str
    ignore_file: Path
    use_gitignore: bool
    gitignore_root: Path | None
    cli_patterns: tuple[str, ...] | None
    no_ignore: bool


@dataclass(frozen=True)
class IndexPaths:
    tldr_dir: Path
    tldr_config: Path
    claude_settings: Path
    indexes_dir: Path
    index_dir: Path
    meta: Path
    ignore_file: Path
    cache_dir: Path
    call_graph: Path
    languages: Path
    dirty: Path
    file_hashes: Path
    content_index: Path
    stats_dir: Path
    hook_activity: Path
    semantic_dir: Path
    semantic_faiss: Path
    semantic_metadata: Path
    daemon_status: Path

    @classmethod
    def from_config(cls, config: IndexConfig) -> "IndexPaths":
        tldr_dir = config.cache_root / ".tldr"
        tldr_config = tldr_dir / "config.json"
        claude_settings = config.cache_root / ".claude" / "settings.json"
        indexes_dir = tldr_dir / "indexes"
        index_dir = indexes_dir / config.index_key
        meta = index_dir / "meta.json"
        ignore_file = config.ignore_file
        cache_dir = index_dir / "cache"
        stats_dir = index_dir / "stats"
        semantic_dir = cache_dir / "semantic"
        return cls(
            tldr_dir=tldr_dir,
            tldr_config=tldr_config,
            claude_settings=claude_settings,
            indexes_dir=indexes_dir,
            index_dir=index_dir,
            meta=meta,
            ignore_file=ignore_file,
            cache_dir=cache_dir,
            call_graph=cache_dir / "call_graph.json",
            languages=index_dir / "languages.json",
            dirty=cache_dir / "dirty.json",
            file_hashes=cache_dir / "file_hashes.json",
            content_index=cache_dir / "content_index.json",
            stats_dir=stats_dir,
            hook_activity=stats_dir / "hook_activity.jsonl",
            semantic_dir=semantic_dir,
            semantic_faiss=semantic_dir / "index.faiss",
            semantic_metadata=semantic_dir / "metadata.json",
            daemon_status=index_dir / "status",
        )

    @classmethod
    def from_parts(
        cls,
        cache_root: Path,
        index_key: str,
        *,
        ignore_file: Path | None = None,
    ) -> "IndexPaths":
        tldr_dir = cache_root / ".tldr"
        tldr_config = tldr_dir / "config.json"
        claude_settings = cache_root / ".claude" / "settings.json"
        indexes_dir = tldr_dir / "indexes"
        index_dir = indexes_dir / index_key
        meta = index_dir / "meta.json"
        if ignore_file is None:
            ignore_file = index_dir / ".tldrignore"
        cache_dir = index_dir / "cache"
        stats_dir = index_dir / "stats"
        semantic_dir = cache_dir / "semantic"
        return cls(
            tldr_dir=tldr_dir,
            tldr_config=tldr_config,
            claude_settings=claude_settings,
            indexes_dir=indexes_dir,
            index_dir=index_dir,
            meta=meta,
            ignore_file=ignore_file,
            cache_dir=cache_dir,
            call_graph=cache_dir / "call_graph.json",
            languages=index_dir / "languages.json",
            dirty=cache_dir / "dirty.json",
            file_hashes=cache_dir / "file_hashes.json",
            content_index=cache_dir / "content_index.json",
            stats_dir=stats_dir,
            hook_activity=stats_dir / "hook_activity.jsonl",
            semantic_dir=semantic_dir,
            semantic_faiss=semantic_dir / "index.faiss",
            semantic_metadata=semantic_dir / "metadata.json",
            daemon_status=index_dir / "status",
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


def _path_rel(path: Path | None, cache_root: Path) -> str | None:
    if path is None:
        return None
    try:
        rel = path.resolve().relative_to(cache_root.resolve())
        return rel.as_posix()
    except ValueError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _create_meta(config: IndexConfig) -> dict:
    from tldr.tldrignore import compute_ignore_hash

    content_hash = (
        compute_ignore_hash(config.ignore_file)
        if not config.no_ignore
        else "no-ignore"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "index_id": config.index_id,
        "index_key": config.index_key,
        "scan_root_abs": _normalize_abs(config.scan_root),
        "scan_root_rel_to_cache_root": _scan_root_rel(config.scan_root, config.cache_root),
        "cache_root_abs": _normalize_abs(config.cache_root),
        "ignore": {
            "file_abs": _normalize_abs(config.ignore_file),
            "file_rel_to_cache_root": _path_rel(config.ignore_file, config.cache_root),
            "use_gitignore": bool(config.use_gitignore),
            "gitignore_root_abs": _normalize_abs(config.gitignore_root) if config.gitignore_root else None,
            "gitignore_root_rel_to_cache_root": _path_rel(config.gitignore_root, config.cache_root),
            "cli_patterns": list(config.cli_patterns or ()),
            "no_ignore": bool(config.no_ignore),
            "content_hash": content_hash,
            "last_changed_at": _now_iso(),
        },
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

    ignore_meta = meta.get("ignore")
    if ignore_meta:
        if ignore_meta.get("file_abs") != _normalize_abs(config.ignore_file):
            raise ValueError("Ignore file mismatch for existing index directory.")
        if bool(ignore_meta.get("use_gitignore", True)) != bool(config.use_gitignore):
            raise ValueError("Gitignore usage mismatch for existing index directory.")
        if bool(ignore_meta.get("no_ignore", False)) != bool(config.no_ignore):
            raise ValueError("Ignore toggle mismatch for existing index directory.")
        meta_patterns = ignore_meta.get("cli_patterns") or []
        config_patterns = list(config.cli_patterns or ())
        if meta_patterns != config_patterns:
            raise ValueError("Ignore patterns mismatch for existing index directory.")
        if config.use_gitignore:
            meta_root = ignore_meta.get("gitignore_root_abs")
            config_root = _normalize_abs(config.gitignore_root) if config.gitignore_root else None
            if meta_root and config_root and meta_root != config_root:
                raise ValueError("Gitignore root mismatch for existing index directory.")


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
    else:
        meta_changed, ignore_changed = _sync_ignore_meta(meta, config)
        if meta_changed:
            _write_meta(paths, meta)
        if ignore_changed:
            _invalidate_index_caches(paths)

    return paths, meta


def _sync_ignore_meta(meta: dict, config: IndexConfig) -> tuple[bool, bool]:
    from tldr.tldrignore import compute_ignore_hash

    ignore_meta = meta.get("ignore")
    ignore_changed = False
    meta_changed = False

    expected_hash = (
        compute_ignore_hash(config.ignore_file)
        if not config.no_ignore
        else "no-ignore"
    )

    if not ignore_meta:
        meta["ignore"] = {
            "file_abs": _normalize_abs(config.ignore_file),
            "file_rel_to_cache_root": _path_rel(config.ignore_file, config.cache_root),
            "use_gitignore": bool(config.use_gitignore),
            "gitignore_root_abs": _normalize_abs(config.gitignore_root) if config.gitignore_root else None,
            "gitignore_root_rel_to_cache_root": _path_rel(config.gitignore_root, config.cache_root),
            "cli_patterns": list(config.cli_patterns or ()),
            "no_ignore": bool(config.no_ignore),
            "content_hash": expected_hash,
            "last_changed_at": _now_iso(),
        }
        meta_changed = True
        ignore_changed = True
        return meta_changed, ignore_changed

    if ignore_meta.get("content_hash") != expected_hash:
        ignore_meta["content_hash"] = expected_hash
        ignore_meta["last_changed_at"] = _now_iso()
        meta_changed = True
        ignore_changed = True

    meta["ignore"] = ignore_meta
    return meta_changed, ignore_changed


def _invalidate_index_caches(paths: IndexPaths) -> None:
    for path in [
        paths.call_graph,
        paths.languages,
        paths.dirty,
        paths.semantic_faiss,
        paths.semantic_metadata,
    ]:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


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
    ignore_file_arg: str | Path | None | object = _UNSET,
    use_gitignore_arg: bool | object | None = _UNSET,
    cli_patterns_arg: list[str] | None | object = _UNSET,
    no_ignore_arg: bool | object | None = _UNSET,
) -> IndexContext:
    if ignore_file_arg is None:
        ignore_file_arg = _UNSET
    if use_gitignore_arg is None:
        use_gitignore_arg = _UNSET
    if cli_patterns_arg is None:
        cli_patterns_arg = _UNSET
    if no_ignore_arg is None:
        no_ignore_arg = _UNSET
    cache_root_value = None if cache_root_arg is None else str(cache_root_arg).strip()
    if cache_root_value is None or cache_root_value == "":
        return IndexContext(
            mode="legacy",
            scan_root=scan_root,
            cache_root=None,
            index_id=None,
            index_key=None,
            paths=None,
            config=None,
        )

    if cache_root_value.lower() == "git":
        from tldr.tldrignore import resolve_git_root

        git_root = resolve_git_root(scan_root)
        if git_root is None:
            git_root = resolve_git_root(Path.cwd())
        if git_root is None:
            raise ValueError("cache_root 'git' requested but no git repository found")
        cache_root = git_root
    else:
        cache_root = Path(cache_root_arg).resolve()
    if index_id_arg is None or index_id_arg == "":
        index_id = derive_index_id(scan_root, cache_root)
    else:
        index_id = index_id_arg

    index_key = compute_index_key(index_id)

    meta_paths = IndexPaths.from_parts(cache_root, index_key)
    meta = load_meta(meta_paths)

    ignore_meta = meta.get("ignore") if meta else None

    if ignore_file_arg is _UNSET:
        if ignore_meta:
            rel = ignore_meta.get("file_rel_to_cache_root")
            abs_path = ignore_meta.get("file_abs")
            if rel:
                ignore_file = cache_root / rel
            elif abs_path:
                ignore_file = Path(abs_path)
            else:
                ignore_file = meta_paths.ignore_file
        else:
            ignore_file = meta_paths.ignore_file
    else:
        ignore_file = Path(ignore_file_arg)

    if use_gitignore_arg is _UNSET:
        use_gitignore = bool(ignore_meta.get("use_gitignore", True)) if ignore_meta else True
    else:
        use_gitignore = bool(use_gitignore_arg)

    if no_ignore_arg is _UNSET:
        no_ignore = bool(ignore_meta.get("no_ignore", False)) if ignore_meta else False
    else:
        no_ignore = bool(no_ignore_arg)

    if cli_patterns_arg is _UNSET:
        cli_patterns = (
            list(ignore_meta.get("cli_patterns") or []) if ignore_meta else []
        )
    else:
        cli_patterns = list(cli_patterns_arg or [])

    gitignore_root = None
    if use_gitignore:
        if ignore_meta and ignore_meta.get("gitignore_root_abs"):
            gitignore_root = Path(ignore_meta.get("gitignore_root_abs"))
        else:
            from tldr.tldrignore import resolve_git_root
            gitignore_root = resolve_git_root(cache_root)
            if gitignore_root is None:
                gitignore_root = cache_root

    config = IndexConfig(
        cache_root=cache_root,
        scan_root=scan_root.resolve(),
        index_id=index_id,
        index_key=index_key,
        ignore_file=ignore_file.resolve(),
        use_gitignore=use_gitignore,
        gitignore_root=gitignore_root.resolve() if gitignore_root else None,
        cli_patterns=tuple(cli_patterns) if cli_patterns else None,
        no_ignore=no_ignore,
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
