from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tldr.indexing import compute_index_key, derive_index_id, IndexConfig, IndexContext


@dataclass(frozen=True)
class DaemonIdentity:
    mode: Literal["legacy", "index"]
    scan_root: Path
    cache_root: Path | None
    index_id: str | None
    index_key: str | None
    seed: str
    hash: str


def resolve_daemon_identity(
    scan_root: str | Path,
    *,
    cache_root: str | Path | None = None,
    index_id: str | None = None,
    index_key: str | None = None,
) -> DaemonIdentity:
    scan_root_path = Path(scan_root).resolve()
    if cache_root is None:
        seed = str(scan_root_path)
        digest = hashlib.md5(seed.encode()).hexdigest()[:8]
        return DaemonIdentity(
            mode="legacy",
            scan_root=scan_root_path,
            cache_root=None,
            index_id=None,
            index_key=None,
            seed=seed,
            hash=digest,
        )

    cache_root_path = Path(cache_root).resolve()
    if index_key is None:
        if index_id is None:
            index_id = derive_index_id(scan_root_path, cache_root_path)
        index_key = compute_index_key(index_id)

    seed = f"{cache_root_path}\0{index_key}"
    digest = hashlib.md5(seed.encode()).hexdigest()[:8]
    return DaemonIdentity(
        mode="index",
        scan_root=scan_root_path,
        cache_root=cache_root_path,
        index_id=index_id,
        index_key=index_key,
        seed=seed,
        hash=digest,
    )


def resolve_daemon_identity_from_context(
    index_ctx: IndexContext | None,
    *,
    scan_root: str | Path | None = None,
) -> DaemonIdentity:
    if index_ctx is not None and index_ctx.cache_root is not None:
        return resolve_daemon_identity(
            index_ctx.scan_root,
            cache_root=index_ctx.cache_root,
            index_id=index_ctx.index_id,
            index_key=index_ctx.index_key,
        )
    if scan_root is None:
        raise ValueError("scan_root is required when no index context is provided")
    return resolve_daemon_identity(scan_root)


def resolve_daemon_identity_from_config(config: IndexConfig) -> DaemonIdentity:
    return resolve_daemon_identity(
        config.scan_root,
        cache_root=config.cache_root,
        index_id=config.index_id,
        index_key=config.index_key,
    )


def get_lock_path(identity: DaemonIdentity) -> Path:
    runtime_dir = _get_daemon_runtime_dir()
    return runtime_dir / f"tldr-{identity.hash}.lock"


def get_pid_path(identity: DaemonIdentity) -> Path:
    runtime_dir = _get_daemon_runtime_dir()
    return runtime_dir / f"tldr-{identity.hash}.pid"


def get_socket_path(identity: DaemonIdentity) -> Path:
    runtime_dir = _get_daemon_runtime_dir()
    return runtime_dir / f"tldr-{identity.hash}.sock"


def get_port_path(identity: DaemonIdentity) -> Path:
    runtime_dir = _get_daemon_runtime_dir()
    return runtime_dir / f"tldr-{identity.hash}.port"


def _unix_uid() -> int:
    # os.getuid is not available on Windows.
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        raise RuntimeError("os.getuid() is required on Unix platforms")
    return int(getuid())


def _prepare_runtime_dir(path: Path, *, source: str) -> Path:
    """Ensure the runtime dir exists and is safe for multi-user hosts (Unix)."""
    if sys.platform == "win32":
        return path

    # Create if missing with restricted permissions.
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    except PermissionError as e:
        raise RuntimeError(
            f"TLDR daemon runtime dir is not writable ({source}): {path}. "
            "Set TLDR_DAEMON_DIR to a short directory you own (e.g. /tmp/tldr-$UID)."
        ) from e
    except FileNotFoundError:
        # Parent missing (e.g. /tmp absent in weird environments) - caller should fallback.
        raise

    if not path.is_dir():
        raise RuntimeError(
            f"TLDR daemon runtime dir path exists but is not a directory ({source}): {path}. "
            "Set TLDR_DAEMON_DIR to a short directory you own."
        )

    try:
        st = path.stat()
    except OSError as e:
        raise RuntimeError(
            f"TLDR daemon runtime dir is not accessible ({source}): {path}. "
            "Set TLDR_DAEMON_DIR to a short directory you own."
        ) from e

    uid = _unix_uid()
    # If another user owns this directory, fail fast instead of misreporting
    # "daemon already running" due to PermissionError later.
    if getattr(st, "st_uid", uid) != uid:
        raise RuntimeError(
            f"TLDR daemon runtime dir is owned by another user ({source}): {path}. "
            "Set TLDR_DAEMON_DIR to a short directory you own."
        )

    if not os.access(path, os.W_OK | os.X_OK):
        raise RuntimeError(
            f"TLDR daemon runtime dir is not writable ({source}): {path}. "
            "Set TLDR_DAEMON_DIR to a short directory you own."
        )

    # Best-effort tighten permissions; ignore failures (e.g. filesystem restrictions).
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass

    return path


def _pick_daemon_runtime_dir_unix(*, uid: int) -> tuple[Path | None, Path | None, Path]:
    """Return (override, xdg_candidate, tmp_candidate) for Unix/macOS."""
    override = os.environ.get("TLDR_DAEMON_DIR")
    override_path = Path(override).expanduser().absolute() if override else None

    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    xdg_candidate = Path(xdg_runtime_dir) / "tldr" if xdg_runtime_dir else None

    tmp_candidate = Path("/tmp") / f"tldr-{uid}"
    return override_path, xdg_candidate, tmp_candidate


def _get_daemon_runtime_dir() -> Path:
    """
    Pick a stable directory for daemon runtime artifacts (pid/lock/sock/port).

    Why:
    - tempfile.gettempdir() can vary with TMPDIR across processes.
    - Unix domain socket paths have strict length limits; a short base dir helps.
    """
    if sys.platform == "win32":
        return Path(tempfile.gettempdir()).resolve()

    # Unix/macOS policy:
    # 1) TLDR_DAEMON_DIR
    # 2) $XDG_RUNTIME_DIR/tldr (if usable)
    # 3) /tmp/tldr-$UID
    # 4) fallback: tempfile.gettempdir()/tldr-$UID (only if /tmp is unavailable)
    uid = _unix_uid()
    override, xdg_candidate, tmp_candidate = _pick_daemon_runtime_dir_unix(uid=uid)

    if override is not None:
        return _prepare_runtime_dir(override, source="TLDR_DAEMON_DIR")

    if xdg_candidate is not None:
        try:
            return _prepare_runtime_dir(xdg_candidate, source="XDG_RUNTIME_DIR")
        except (PermissionError, FileNotFoundError, RuntimeError, OSError):
            # If XDG_RUNTIME_DIR is set but unusable, fall back to /tmp.
            pass

    try:
        return _prepare_runtime_dir(tmp_candidate, source="/tmp")
    except FileNotFoundError:
        fallback = Path(tempfile.gettempdir()) / f"tldr-{uid}"
        return _prepare_runtime_dir(fallback, source="tempfile.gettempdir()")


def read_port_file(identity: DaemonIdentity) -> int | None:
    port_path = get_port_path(identity)
    if not port_path.exists():
        return None
    try:
        value = port_path.read_text().strip()
        if not value:
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def write_port_file(identity: DaemonIdentity, port: int) -> None:
    port_path = get_port_path(identity)
    port_path.write_text(str(port))


def clear_port_file(identity: DaemonIdentity) -> None:
    port_path = get_port_path(identity)
    try:
        port_path.unlink(missing_ok=True)
    except OSError:
        pass


def get_connection_info(identity: DaemonIdentity) -> tuple[str, int | None]:
    """Return (address, port) - port is None for Unix sockets."""
    if sys.platform == "win32":
        port = read_port_file(identity)
        if port is None:
            port = 49152 + (int(identity.hash, 16) % 10000)
        return ("127.0.0.1", port)
    socket_path = get_socket_path(identity)
    return (str(socket_path), None)
