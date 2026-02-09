"""
Daemon lifecycle management: start, stop, query.

Uses file locking on PID file as the primary synchronization mechanism.
The lock is held for the daemon's entire lifetime, preventing duplicates.
Cross-platform: fcntl.flock() on Unix, msvcrt.locking() on Windows.
"""

import json
import logging
import os
import socket
import sys
import tempfile
import time

from pathlib import Path
from typing import Optional, IO

from tldr.indexing import IndexContext, get_index_context

from .identity import (
    DaemonIdentity,
    get_connection_info,
    get_lock_path,
    get_pid_path,
    get_socket_path,
    resolve_daemon_identity,
    resolve_daemon_identity_from_context,
)

# Platform-specific imports for file locking
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

logger = logging.getLogger(__name__)


def _legacy_unix_socket_candidates(addr: str) -> list[str]:
    """Return candidate socket paths for upgrade/back-compat on Unix."""
    addr_path = Path(addr)
    candidates: list[str] = [str(addr_path)]

    # Back-compat #1: TMPDIR-based tempdir location used by older clients/daemons.
    legacy_tmpdir = Path(tempfile.gettempdir()) / addr_path.name
    if str(legacy_tmpdir) not in candidates:
        candidates.append(str(legacy_tmpdir))

    # Back-compat #2: historical shared runtime dir (/tmp/tldr) used by older versions.
    legacy_shared = Path("/tmp/tldr") / addr_path.name
    if str(legacy_shared) not in candidates:
        candidates.append(str(legacy_shared))

    return candidates


def _resolve_identity(
    project: Path,
    *,
    index_ctx: IndexContext | None = None,
    cache_root: Path | None = None,
    index_id: str | None = None,
) -> DaemonIdentity:
    if index_ctx is not None:
        return resolve_daemon_identity_from_context(index_ctx, scan_root=project)
    if cache_root is not None or index_id is not None:
        return resolve_daemon_identity(project, cache_root=cache_root, index_id=index_id)
    return resolve_daemon_identity(project)


def _get_lock_path(identity: DaemonIdentity) -> Path:
    """Get lock file path for daemon startup synchronization."""
    return get_lock_path(identity)


def _get_pid_path(identity: DaemonIdentity) -> Path:
    """Get PID file path for daemon process tracking."""
    return get_pid_path(identity)


def _get_socket_path(identity: DaemonIdentity) -> Path:
    """Get socket path for daemon communication."""
    return get_socket_path(identity)


def _is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    if sys.platform == "win32":
        # Windows: use tasklist or ctypes
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)  # Signal 0 = check if process exists
            return True
        except (OSError, ProcessLookupError):
            return False


def _try_acquire_pidfile_lock(pid_path: Path) -> Optional[IO]:
    """Try to acquire exclusive lock on PID file.

    Returns:
        File handle if lock acquired (caller must keep it open!), None if locked by another process.
    """
    try:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode to create if not exists, don't truncate
        pidfile = open(pid_path, "a+")

        if sys.platform == "win32":
            # Windows: msvcrt.locking with LK_NBLCK (non-blocking)
            try:
                msvcrt.locking(pidfile.fileno(), msvcrt.LK_NBLCK, 1)
                return pidfile
            except (IOError, OSError):
                # Lock held by another process
                pidfile.close()
                return None
        else:
            # Unix: fcntl.flock with LOCK_NB (non-blocking)
            try:
                fcntl.flock(pidfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return pidfile
            except (IOError, BlockingIOError):
                pidfile.close()
                return None
    except PermissionError as e:
        # Distinguish permission problems from "daemon already running".
        raise RuntimeError(
            f"Cannot access TLDR daemon runtime artifacts under {pid_path.parent}: {e}. "
            "If this path is owned by another user or not writable, set TLDR_DAEMON_DIR "
            "to a short directory you own (e.g. /tmp/tldr-$UID)."
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            f"Cannot create TLDR daemon PID file under {pid_path.parent}: {pid_path}. "
            "Set TLDR_DAEMON_DIR to a short directory you own."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to open or lock TLDR daemon PID file at {pid_path}: {e}. "
            "Set TLDR_DAEMON_DIR to a short directory you own."
        ) from e


def _write_pid_to_locked_file(pidfile: IO, pid: int) -> None:
    """Write PID to an already-locked file."""
    pidfile.seek(0)
    pidfile.truncate()
    pidfile.write(str(pid))
    pidfile.flush()


def _is_socket_connectable(identity: DaemonIdentity, timeout: float = 1.0) -> bool:
    """Check if daemon socket exists and accepts connections.

    This is more robust than ping-based check because it doesn't
    depend on response format - just whether a daemon is listening.
    """
    addr, port = get_connection_info(identity)
    unix_candidates: list[str] | None = None
    if port is None:
        unix_candidates = _legacy_unix_socket_candidates(addr)
        if not any(Path(p).exists() for p in unix_candidates):
            return False

    try:
        if port is not None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.settimeout(timeout)
                sock.connect((addr, port))
            finally:
                sock.close()
        else:
            assert unix_candidates is not None
            last_err: OSError | None = None
            for candidate in unix_candidates:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    sock.settimeout(timeout)
                    sock.connect(candidate)
                    return True
                except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
                    last_err = e
                    continue
                finally:
                    sock.close()
            if last_err is not None:
                return False
        return True
    except (socket.error, OSError):
        return False



def _is_daemon_alive(identity: DaemonIdentity, retries: int = 3, delay: float = 0.1) -> bool:
    """Check if daemon is alive using file lock on PID file.

    This is the authoritative check - if we can't acquire the lock,
    another daemon is holding it and is therefore alive. No socket
    connectivity check needed (avoids race conditions with slow daemons).

    Args:
        identity: Daemon identity
        retries: Number of attempts (default 3) - used for brief retries
        delay: Seconds between attempts (default 0.1)

    Returns:
        True if daemon is alive (lock held by another process), False otherwise
    """
    pid_path = _get_pid_path(identity)

    for attempt in range(retries):
        # Try to acquire lock - if we can't, daemon is running
        pidfile = _try_acquire_pidfile_lock(pid_path)
        if pidfile is None:
            # Lock held by another process = daemon is alive
            return True

        # We got the lock - check if there's a stale PID
        pidfile.seek(0)
        content = pidfile.read().strip()
        if content:
            try:
                pid = int(content)
                if _is_process_running(pid):
                    # Process exists but we got the lock? Shouldn't happen normally.
                    # Could be a daemon that crashed after writing PID but before locking.
                    # Release lock and report alive (process still running).
                    if sys.platform == "win32":
                        msvcrt.locking(pidfile.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(pidfile.fileno(), fcntl.LOCK_UN)
                    pidfile.close()
                    return True
            except ValueError:
                pass  # Corrupt PID, ignore

        # Release lock - no daemon running
        if sys.platform == "win32":
            try:
                msvcrt.locking(pidfile.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        else:
            try:
                fcntl.flock(pidfile.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        pidfile.close()

        # If the PID lock indicates "not running", double-check socket connectivity.
        # This covers cases where the daemon is alive but PID files are in a different
        # runtime directory (e.g. TMPDIR drift or older versions' locations).
        if _is_socket_connectable(identity, timeout=0.2):
            return True

        if attempt < retries - 1:
            time.sleep(delay)

    return False


def is_daemon_alive(identity: DaemonIdentity) -> bool:
    """Public wrapper for daemon liveness checks."""
    return _is_daemon_alive(identity)


def _create_client_socket(identity: DaemonIdentity) -> socket.socket:
    """Create appropriate client socket for platform.

    Args:
        identity: Daemon identity to get connection info from

    Returns:
        Connected socket ready for communication
    """
    addr, port = get_connection_info(identity)

    if port is not None:
        # TCP socket for Windows
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((addr, port))
    else:
        # Unix socket for Linux/macOS
        last_err: OSError | None = None
        for candidate in _legacy_unix_socket_candidates(addr):
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                client.connect(candidate)
                return client
            except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
                last_err = e
                try:
                    client.close()
                except Exception:
                    pass
                continue
        if last_err is not None:
            raise last_err

    return client


def _ensure_index_ignore_file(index_ctx: IndexContext) -> None:
    cfg = index_ctx.config
    if cfg is None or cfg.no_ignore:
        return
    ignore_path = cfg.ignore_file
    if ignore_path.exists():
        return
    try:
        ignore_path.resolve().relative_to(cfg.cache_root.resolve())
    except ValueError:
        raise ValueError(
            f"Ignore file does not exist: {ignore_path}. Create it manually or choose a path under cache-root."
        )
    from ..tldrignore import ensure_tldrignore
    created, msg = ensure_tldrignore(index_ctx.paths.index_dir, ignore_file=ignore_path)
    if created:
        print(msg)


def start_daemon(
    project_path: str | Path,
    foreground: bool = False,
    *,
    index_ctx: IndexContext | None = None,
):
    """
    Start the TLDR daemon for a project.

    Uses file locking on the PID file as the primary synchronization mechanism.
    The lock is held for the daemon's entire lifetime, preventing duplicates.

    Args:
        project_path: Path to the project root
        foreground: If True, run in foreground; otherwise daemonize
    """
    from .core import TLDRDaemon
    project = Path(project_path).resolve()
    identity = _resolve_identity(project, index_ctx=index_ctx)
    pid_path = _get_pid_path(identity)

    # Try to acquire exclusive lock on PID file
    # If we can't, another daemon is running
    pidfile = _try_acquire_pidfile_lock(pid_path)
    if pidfile is None:
        print("Daemon already running")
        return

    # If a daemon is already accepting connections (e.g. from a different runtime dir),
    # don't start a second daemon. Release our PID lock and return.
    if _is_socket_connectable(identity, timeout=0.2):
        if sys.platform == "win32":
            try:
                msvcrt.locking(pidfile.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        else:
            try:
                fcntl.flock(pidfile.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        pidfile.close()
        print("Daemon already running")
        return

    # We have the lock - we're the only one starting a daemon
    if index_ctx is None:
        from ..tldrignore import ensure_tldrignore
        # Ensure .tldrignore exists (create with defaults if not)
        created, message = ensure_tldrignore(project)
        if created:
            print(f"\n\033[33m{message}\033[0m\n")  # Yellow warning
    else:
        _ensure_index_ignore_file(index_ctx)

    daemon = TLDRDaemon(project, index_ctx=index_ctx)

    if foreground:
        # Write PID and run - pidfile stays open (lock held)
        _write_pid_to_locked_file(pidfile, os.getpid())
        daemon._pidfile = pidfile  # Daemon keeps reference to hold lock
        daemon.run()
    else:
        if sys.platform == "win32":
            # Windows: Use subprocess to run in background
            # Release our lock - the subprocess will acquire its own
            import subprocess
            try:
                msvcrt.locking(pidfile.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            pidfile.close()

            # Acquire lock to prevent race conditions
            lock_path = _get_lock_path(identity)
            # Ensure lock file exists
            if not lock_path.exists():
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_path.touch()

            try:
                with open(lock_path, "w") as lock_file:
                    # Windows locking: try to acquire lock
                    # msvcrt.locking raises OSError if locked when using LK_NBLCK, 
                    # or blocks 10s with LK_RLCK. We want to wait until acquired.
                    start_lock = time.time()
                    while True:
                        try:
                            # Lock the first byte
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                            break
                        except OSError:
                            if time.time() - start_lock > 10.0:
                                print("Timeout waiting for daemon lock")
                                return
                            time.sleep(0.1)
                    
                    try:
                        # Re-check if daemon is alive (race condition handling)
                        if _is_daemon_alive(identity):
                            print("Daemon already running")
                            return

                        # Get the connection info for display
                        addr, port = get_connection_info(identity)

                        # Start detached process on Windows
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        startupinfo.wShowWindow = subprocess.SW_HIDE

                        cmd = [sys.executable, "-m", "tldr.daemon", str(project), "--foreground"]
                        if index_ctx is not None and index_ctx.config is not None:
                            cfg = index_ctx.config
                            cmd.extend(["--cache-root", str(cfg.cache_root)])
                            cmd.extend(["--index", str(cfg.index_id)])
                            if cfg.ignore_file is not None:
                                cmd.extend(["--ignore-file", str(cfg.ignore_file)])
                            if cfg.no_ignore:
                                cmd.append("--no-ignore")
                            elif cfg.use_gitignore is False:
                                cmd.append("--no-gitignore")
                            if cfg.cli_patterns:
                                for pattern in cfg.cli_patterns:
                                    cmd.extend(["--ignore", pattern])

                        proc = subprocess.Popen(
                            cmd,
                            startupinfo=startupinfo,
                            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        print(f"Daemon started with PID {proc.pid}")

                        # Verify daemon is listening
                        start_wait = time.time()
                        connected = False
                        while time.time() - start_wait < 5.0:
                            try:
                                with socket.create_connection((addr, port), timeout=0.5):
                                    connected = True
                                    break
                            except (OSError, ConnectionRefusedError):
                                time.sleep(0.1)

                        if connected:
                            print(f"Listening on {addr}:{port}")
                        else:
                            logger.error("Daemon started but failed to accept connections")
                            # Should we kill it? Maybe not strictly required but logging is good.

                    finally:
                        # Release lock
                        try:
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                        except OSError:
                            pass
            except Exception:
                logger.exception("Error starting daemon")

        else:
            # Unix: Fork and run in background
            # Child inherits the lock, parent releases it

            # Fork daemon process
            pid = os.fork()
            if pid == 0:
                # Child process - we inherit the lock
                os.setsid()
                # Detach stdio so the daemon doesn't inherit a dead pipe from the
                # launcher (common in non-interactive runners). Some libraries
                # used by semantic search emit progress to stdout/stderr and can
                # raise BrokenPipeError otherwise.
                try:
                    devnull = os.open(os.devnull, os.O_RDWR)
                    os.dup2(devnull, 0)
                    os.dup2(devnull, 1)
                    os.dup2(devnull, 2)
                    if devnull > 2:
                        os.close(devnull)
                except Exception:
                    # Best-effort; daemon can still run with inherited FDs.
                    pass
                # Write our PID to the locked file
                _write_pid_to_locked_file(pidfile, os.getpid())
                daemon._pidfile = pidfile  # Keep reference to hold lock
                daemon.run()
                sys.exit(0)  # Should not reach here
            else:
                # Parent process - release lock and wait for daemon
                try:
                    fcntl.flock(pidfile.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                pidfile.close()

                # Wait for daemon to be ready (socket exists)
                start_time = time.time()
                timeout = 10.0
                socket_path = _get_socket_path(identity)
                while time.time() - start_time < timeout:
                    if socket_path.exists() and _is_socket_connectable(identity, timeout=0.5):
                        print(f"Daemon started with PID {pid}")
                        print(f"Socket: {daemon.socket_path}")
                        return
                    time.sleep(0.1)

                # Daemon started but socket not ready - warn but don't fail
                print(f"Warning: Daemon (PID {pid}) socket not ready within {timeout}s")
                print(f"Socket: {daemon.socket_path}")


def stop_daemon(
    project_path: str | Path,
    *,
    index_ctx: IndexContext | None = None,
    cache_root: Path | None = None,
    index_id: str | None = None,
) -> bool:
    """
    Stop the TLDR daemon for a project.

    Args:
        project_path: Path to the project root

    Returns:
        True if daemon was stopped, False if not running
    """
    project = Path(project_path).resolve()
    identity = _resolve_identity(
        project,
        index_ctx=index_ctx,
        cache_root=cache_root,
        index_id=index_id,
    )

    try:
        client = _create_client_socket(identity)
        client.sendall(json.dumps({"cmd": "shutdown"}).encode() + b"\n")
        client.recv(4096)
        client.close()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False


def query_daemon(
    project_path: str | Path,
    command: dict,
    *,
    index_ctx: IndexContext | None = None,
    cache_root: Path | None = None,
    index_id: str | None = None,
) -> dict:
    """
    Send a command to the daemon and get the response.

    Args:
        project_path: Path to the project root
        command: Command dict to send

    Returns:
        Response dict from daemon
    """
    def _recv_json_line(sock: socket.socket) -> dict:
        buf = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
        line = buf.split(b"\n", 1)[0].strip()
        if not line:
            return {"status": "error", "message": "Empty response from daemon"}
        return json.loads(line.decode())

    project = Path(project_path).resolve()
    identity = _resolve_identity(
        project,
        index_ctx=index_ctx,
        cache_root=cache_root,
        index_id=index_id,
    )
    client = _create_client_socket(identity)
    try:
        client.sendall(json.dumps(command).encode() + b"\n")
        return _recv_json_line(client)
    finally:
        client.close()


def main():
    """CLI entry point for daemon management."""
    import argparse

    parser = argparse.ArgumentParser(description="TLDR Daemon")
    parser.add_argument("project", help="Project path")
    parser.add_argument("--scan-root", help="Scan root (overrides project)")
    parser.add_argument("--cache-root", help="Directory where .tldr caches live (enables index mode)")
    parser.add_argument("--index", dest="index_id", help="Logical index id (namespaces caches under cache-root)")
    parser.add_argument("--ignore-file", help="Path to a .tldrignore file (index-scoped in index mode)")
    gitignore_group = parser.add_mutually_exclusive_group()
    gitignore_group.add_argument(
        "--use-gitignore",
        dest="use_gitignore",
        action="store_true",
        default=None,
        help="Use .gitignore (default)",
    )
    gitignore_group.add_argument(
        "--no-gitignore",
        dest="use_gitignore",
        action="store_false",
        help="Ignore .gitignore",
    )
    parser.add_argument("--ignore", action="append", default=None, help="Additional ignore patterns")
    parser.add_argument("--no-ignore", action="store_true", help="Bypass all ignore rules")
    parser.add_argument("--force-rebind", action="store_true", help="Rebind index to a new scan root")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index artifacts")
    parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")
    parser.add_argument("--stop", action="store_true", help="Stop the daemon")
    parser.add_argument("--status", action="store_true", help="Get daemon status")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Env defaults (match CLI behavior)
    if args.cache_root is None:
        args.cache_root = os.environ.get("TLDR_CACHE_ROOT")
    if args.index_id is None:
        args.index_id = os.environ.get("TLDR_INDEX")
    if args.scan_root is None:
        args.scan_root = os.environ.get("TLDR_SCAN_ROOT")
    if args.ignore_file is None:
        args.ignore_file = os.environ.get("TLDR_IGNORE_FILE")
    if args.use_gitignore is None and "TLDR_USE_GITIGNORE" in os.environ:
        args.use_gitignore = os.environ.get("TLDR_USE_GITIGNORE") == "1"

    scan_root = args.scan_root or args.project
    if args.scan_root and Path(args.project).resolve() != Path(args.scan_root).resolve():
        print("Error: --scan-root conflicts with positional project path", file=sys.stderr)
        sys.exit(1)

    if args.index_id and not args.cache_root:
        print("Error: --index requires --cache-root", file=sys.stderr)
        sys.exit(1)

    index_ctx = None
    if args.cache_root:
        try:
            index_ctx = get_index_context(
                scan_root=Path(scan_root),
                cache_root_arg=args.cache_root,
                index_id_arg=args.index_id,
                allow_create=not (args.stop or args.status),
                force_rebind=args.force_rebind,
                ignore_file_arg=args.ignore_file,
                use_gitignore_arg=args.use_gitignore,
                cli_patterns_arg=args.ignore,
                no_ignore_arg=args.no_ignore,
            )
        except FileNotFoundError:
            index_ctx = None

    if args.stop:
        if stop_daemon(scan_root, index_ctx=index_ctx, cache_root=args.cache_root, index_id=args.index_id):
            print("Daemon stopped")
        else:
            print("Daemon not running")
    elif args.status:
        try:
            result = query_daemon(
                scan_root,
                {"cmd": "status"},
                index_ctx=index_ctx,
                cache_root=args.cache_root,
                index_id=args.index_id,
            )
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Daemon not running: {e}")
    else:
        start_daemon(scan_root, foreground=args.foreground, index_ctx=index_ctx)


if __name__ == "__main__":
    main()
