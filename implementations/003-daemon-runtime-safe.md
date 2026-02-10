Title: Make daemon runtime dir safe-by-default on multi-user machines and CI

  Context
  - Daemon runtime artifacts (sock/pid/lock/port) currently default to /tmp/tldr on Unix/macOS (overridable via TLDR_DAEMON_DIR).
  - This fixes TMPDIR drift but /tmp/tldr is shared on multi-user hosts and can cause permissions/collision issues.
  - Repo-local .tldr is for index artifacts; putting the Unix socket under .tldr can fail due to Unix socket path-length limits.

  Goal
  - Change the default daemon runtime directory selection to be per-user and permission-safe, while keeping paths short enough for Unix domain sockets.
  - Improve error reporting so PermissionError is not misreported as “Daemon already running”.
  - This is to solve in a more robust way the issues seen in docs/daemon.log

  Proposed Behavior (Unix/macOS)
  - Runtime dir resolution precedence:
    1) TLDR_DAEMON_DIR (explicit override)
    2) $XDG_RUNTIME_DIR/tldr if XDG_RUNTIME_DIR is set and usable (Linux)
    3) /tmp/tldr-$UID (or equivalent per-user stable short dir)
    4) fallback: tempfile.gettempdir()/tldr-$UID
  - Create the runtime dir if missing with mode 0700.
  - If the runtime dir is not writable by the current user, fail fast with an actionable message recommending TLDR_DAEMON_DIR.

  Implementation Notes
  - Update tldr/daemon/identity.py:_get_daemon_runtime_dir() to implement the new default policy.
  - Update tldr/daemon/startup.py PID-lock acquisition to distinguish PermissionError from “locked by daemon”.
  - Keep legacy socket connect fallbacks for upgrade compatibility.
  - Update daemon subprocess tests to set TLDR_DAEMON_DIR to a per-test tmp_path to avoid touching /tmp and to avoid parallel CI collisions.

  Acceptance Criteria
  - Two different users on the same host can run separate daemons concurrently without interfering.
  - Parallel CI jobs don’t collide on daemon runtime artifacts by default (or via tests setting TLDR_DAEMON_DIR).
  - If runtime dir permissions are wrong, the CLI errors clearly (no false “Daemon already running”).
  - uv run pytest passes (at least tests/test_daemon_*.py).

Plan:
  1. Define the default runtime dir policy (Unix/macOS)
      - Precedence:
          1. TLDR_DAEMON_DIR (explicit override)
          2. $XDG_RUNTIME_DIR/tldr if XDG_RUNTIME_DIR is set and usable (Linux)
          3. /tmp/tldr-$UID (per-user, stable, short)
          4. Last-resort fallback: tempfile.gettempdir()/tldr-$UID (only if /tmp is unavailable)
      - Keep Windows as-is (system temp).
  2. Enforce safe permissions on Unix
      - On first use, create the runtime dir with mode 0700.
      - If the dir exists but is not writable/readable by the current user, fail fast with a clear error telling the user to set TLDR_DAEMON_DIR (do not silently misreport “Daemon
        already running”).
  3. Implement in one place
      - Update tldr/daemon/identity.py:_get_daemon_runtime_dir() to compute the directory using the policy above.
      - Ensure all runtime artifacts (sock/pid/lock/port) continue to derive from this function.
  4. Fix the misleading “already running” path
      - In tldr/daemon/startup.py:_try_acquire_pidfile_lock(), distinguish PermissionError from “lock held”.
      - Propagate an actionable error message instead of returning None (which currently prints “Daemon already running”).
  5. Preserve backward compatibility
      - Keep the existing “legacy socket path” connect fallback in tldr/daemon/startup.py, but add one more fallback attempt to the old shared /tmp/tldr path to cover upgrades (optional,
        but helps smooth transitions).
  6. Make tests deterministic and CI-safe
      - In daemon subprocess tests, set TLDR_DAEMON_DIR to a tmp_path directory via the subprocess env so tests never touch /tmp and don’t collide across parallel runners.
  7. Docs update + validation
      - Update README.md and docs/TLDR.md to state the new default (per-user) and recommend setting TLDR_DAEMON_DIR in CI for strict isolation.
      - Run uv run ruff check tldr/daemon and uv run pytest tests/test_daemon_* (or full suite) to confirm behavior.

If you encounter any learnings, ahas/gotchas, or find next steps append them to this very same document.

Learnings / gotchas:
  - Unix domain socket path length limits are easy to hit on macOS if paths are "canonicalized":
    - Avoid calling Path.resolve() on Unix runtime dirs, because it can expand /tmp -> /private/tmp and make socket paths longer.
  - pytest tmp_path directories can be long enough to break Unix socket binding on macOS; tests can keep artifacts per-test while preserving short socket paths by using a short symlink
    under /tmp that points at a per-test tmp_path directory.

Possible next steps:
  - Consider a CI-specific override pattern (documented) where users set TLDR_DAEMON_DIR to a job-unique short path if multiple jobs share the same host + UID + /tmp.
