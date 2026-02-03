# Repository Guidelines

## Project Structure & Module Organization
- `tldr/`: core library and CLI implementation (daemon, indexing, semantic search, analysis layers).
- `tests/`: pytest suite (`test_*.py`).
- `docs/`, `implementations/`, `specs/`: product docs, implementation plans, and feature specs.
- `scripts/`: helper utilities for local workflows.

## Build, Test, and Development Commands
- `uv venv && uv pip install -e ".[dev]"`: create a virtualenv and install dev dependencies.
- `uv run pytest`: run the full test suite (preferred; ensures all deps are available).
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
- Prefer `uv run …` for any command that depends on optional native or ML deps.
