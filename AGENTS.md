# Repository Guidelines

## Project Structure & Module Organization
- `tldr/`: core library and CLI implementation (daemon, indexing, semantic search, analysis layers).
- `tests/`: pytest suite (`test_*.py`).
- `docs/`, `implementations/`, `specs/`: product docs, implementation plans, and feature specs.
- `scripts/`: helper utilities for local workflows.

## Build, Test, and Development Commands
- `uv venv && uv pip install -e ".[dev]"`: create a virtualenv and install dev dependencies.
- `uv run pytest`: run the full test suite (preferred; ensures all deps are available). DO NOT USE pip or "python -m pytest"
- `uv run pytest tests/test_some_area.py`: run a focused test file.
- `uv run ruff check tldr/`: lint the codebase.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and `snake_case` for functions/variables.
- Keep module names short and lowercase (e.g., `tldr/daemon/identity.py`).
- Favor explicit types and small, composable functions.
- Use `ruff` for linting; fix warnings before PRs.

## Testing Guidelines
- Framework: `pytest`.
- Test files live in `tests/` and follow `test_*.py` naming.
- Run tests with `uv run pytest` to avoid missing dependency issues.
- Add/adjust tests when changing behavior in CLI, daemon, or indexing paths.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits. Examples: `feat: add index-aware daemon identity`, `fix: handle ignore file mismatch`, `docs: update CLI examples`.
- Keep PRs focused on one logical change.
- Before opening a PR, rebase on `main`, run `uv run pytest` and `uv run ruff check tldr/`, and update docs when public behavior changes.

## Configuration & Runtime Notes
- The CLI honors `TLDR_*` env vars (e.g., `TLDR_CACHE_ROOT`, `TLDR_INDEX`, `TLDR_SCAN_ROOT`).
- Prefer `uv run ...` for any command that depends on optional native or ML deps.

## LLM Tool-Use Playbook (High-Signal, Token-Budgeted)

Use `tldrf` when the question is about **behavior/flow** (callers, slices, data flow). Use `rg/grep` when you need **exhaustive text search** (strings, docs, config).

- "Who calls X?": `uv run tldrf impact X .` (optionally `uv run tldrf context X --project .`)
- "Explain what a function does" (best default): `uv run tldrf context <entry> --project . --depth 2`
- "What affects this value/branch at file:function:line?":
  1. `uv run tldrf slice <file> <function> <line>`
  2. Always include a contiguous **anchor window** around `<line>` (largest that fits budget).
  3. Add a few small windows (+/-3 lines) around slice-selected lines *outside* the anchor (closest-to-target first; merge overlaps).
  4. If still unclear and budget allows, include defs/classes for 1-3 helpers referenced by those windows.
- "Trace a specific variable": `uv run tldrf dfg <file> <function>` (+ anchor window around the relevant line)
- "Find where something is implemented/configured":
  1. For definition-shaped lookups, use `rg` first (fast lexical check).
  2. For behavior/concept lookups, use `uv run tldrf semantic search "<intent>" --path . --k 8` (requires a semantic index).
  3. Treat "0 lexical hits" as "no result" and do not guess (semantic/hybrid will otherwise return false positives on negatives).

Budget heuristic:
- Tight: anchor only.
- Medium: anchor + 2-4 slice windows.
- Large: anchor + slice windows + 1-3 related defs/classes.

Always: answer using only gathered context; if insufficient, request the next **specific** snippet (file + line range or symbol).

## Long-Running / TTY-Sensitive Commands
  - If a command may run >60s: decide how you will keep it alive until completion.
  - Codex tool-runner note: interactive tool sessions (PTY/session ids) are not durable across assistant turns. If you start a long command and return before it finishes, the process/handle may be lost or terminated.
  - Claude Code note: `run_in_background: true` (shell) or background `Task` agents keep work running while the current session is alive, but do not survive if the runner/session exits.
  - If you need durability (jobs >5 min or anything you can't afford to lose): run it inside `tmux` (or `screen`) and tee output to a log file under the repo (e.g. `benchmark/logs/`).
  - Record a handle immediately (task id/session id/PID) and how to stop it (cancel/kill).
  - Ensure readable progress:
    - prefer plain/unbuffered output: `--no-progress` / `--progress=plain` / `--quiet`, and `PYTHONUNBUFFERED=1` (plus `NO_COLOR=1`/`CI=1` if helpful).
    - if output is still "silent" or clearly TTY-gated, rerun with pseudo-TTY/`tty` enabled.
  - Silence rule: if there's no new output within 10-15s, do a status check via the handle; never wait >60s without output or an explicit status check.
  - `tmux` pattern (example):
    - Start: `tmux new-session -d -s <name> 'cd <repo> && PYTHONUNBUFFERED=1 NO_PROGRESS=1 <cmd> 2>&1 | tee <log>'`
    - Status: `tmux has-session -t <name>` and `tmux capture-pane -pt <name>:0 | tail -n 50` (or `tail -n 50 <log>`)
    - Stop: `tmux kill-session -t <name>`
