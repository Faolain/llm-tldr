from __future__ import annotations

import hashlib
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
    tmp_dir = tempfile.gettempdir()
    return Path(tmp_dir) / f"tldr-{identity.hash}.lock"


def get_pid_path(identity: DaemonIdentity) -> Path:
    tmp_dir = tempfile.gettempdir()
    return Path(tmp_dir) / f"tldr-{identity.hash}.pid"


def get_socket_path(identity: DaemonIdentity) -> Path:
    tmp_dir = tempfile.gettempdir()
    return Path(tmp_dir) / f"tldr-{identity.hash}.sock"


def get_port_path(identity: DaemonIdentity) -> Path:
    tmp_dir = tempfile.gettempdir()
    return Path(tmp_dir) / f"tldr-{identity.hash}.port"


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
