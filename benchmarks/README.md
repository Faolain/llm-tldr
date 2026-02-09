# Benchmarks (Repeatable Corpora + Runs)

This directory contains **tracked** benchmark inputs (corpus manifests, curated edge sets, query sets).

All **untracked** run artifacts live under the gitignored `benchmark/` directory at repo root:

- `benchmark/corpora/<id>/`        Cloned corpora checkouts (pinned refs)
- `benchmark/cache-root/`          Index-mode caches (`--cache-root benchmark/cache-root`)
- `benchmark/runs/<timestamp>/`    JSON reports produced by scripts

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
```

## Fetch Corpora (Pinned)

Clones (or updates) corpora into `benchmark/corpora/<id>` at the pinned ref from `benchmarks/corpora.json`.

```bash
uv run python scripts/bench_fetch_corpora.py --corpus nextjs
uv run python scripts/bench_fetch_corpora.py --corpus peerbit
uv run python scripts/bench_fetch_corpora.py --all
```

Notes:
- Fetching does **not** run `pnpm install` (that is intentionally out-of-band and not timed).
- For TS callgraph correctness on monorepos, you will usually need to run `pnpm install` inside the corpus checkout at least once.

## Repeatability Rules

- Do not run benchmarks on a moving branch. Always use the pinned ref from `benchmarks/corpora.json`.
- Always run with index-mode cache isolation:
  - `--cache-root benchmark/cache-root`
  - `--index repo:<corpus-id>`
- Every report must record:
  - `tldr_git_sha`
  - corpus `git_sha`
  - platform + toolchain metadata (`python`, `node`, `pnpm`)

