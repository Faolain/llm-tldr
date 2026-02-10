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

## Phase 1: TS Curated Recall (Edge + Impact)

Curated edge sets (tracked):
- `benchmarks/ts/peerbit_curated_edges.json`
- `benchmarks/ts/nextjs_curated_edges.json`

Run curated scoring (writes a single JSON report under `benchmark/runs/` by default):

```bash
uv run python scripts/bench_ts_curated_recall.py \
  --repo-root benchmark/corpora/peerbit \
  --curated benchmarks/ts/peerbit_curated_edges.json \
  --cache-root benchmark/cache-root \
  --index repo:peerbit

uv run python scripts/bench_ts_curated_recall.py \
  --repo-root benchmark/corpora/nextjs \
  --curated benchmarks/ts/nextjs_curated_edges.json \
  --cache-root benchmark/cache-root \
  --index repo:nextjs
```

Optional:
- Add `--ts-trace` to capture a bounded resolver "skipped reasons" histogram for misses.
- Add `--rebuild` to ignore any cached call graph and rebuild from scratch.

## Phase 2: `rg` Baselines (Deterministic, Non-AST)

Runs a deterministic `rg` proxy for "what calls X?" against the same curated targets and emits a JSON report with token-budget curves:

```bash
uv run python scripts/bench_rg_impact_baseline.py \
  --repo-root benchmark/corpora/peerbit \
  --curated benchmarks/ts/peerbit_curated_edges.json \
  --strategy match_only \
  --budgets 500,1000,2000,5000,10000

uv run python scripts/bench_rg_impact_baseline.py \
  --repo-root benchmark/corpora/nextjs \
  --curated benchmarks/ts/nextjs_curated_edges.json \
  --strategy match_plus_enclosing_symbol \
  --budgets 500,1000,2000,5000,10000
```

## Phase 3: TS Perf (Build + Patch + Daemon Impact)

Measures:
- TS call graph build time
- TS-resolved incremental patch vs full rebuild after a deterministic one-file edit
- daemon warm + impact latency percentiles (optional)
- index/cache size under the benchmark cache root

```bash
uv run python scripts/bench_ts_perf.py --corpus peerbit --clear-callgraph-cache --daemon
uv run python scripts/bench_ts_perf.py --corpus nextjs --clear-callgraph-cache --daemon
```

Notes:
- `--corpus <id>` assumes the checkout exists at `benchmark/corpora/<id>` (use the fetcher first).
- Touch/edit files default to small, pinned-in-repo TS files; override with `--touch-file` if needed.

## Phase 3: Perf Microbench (Daemon vs CLI)

Measures warm daemon vs CLI spawn latency for key commands (`search`, `extract`, `impact`, `tree`, `structure`, `context`) and records index/cache sizing via the same JSON schema as `tldrf index list/info`:

```bash
uv run python scripts/bench_perf_daemon_vs_cli.py --corpus peerbit --lang typescript
uv run python scripts/bench_perf_daemon_vs_cli.py --corpus nextjs --lang typescript
```

Notes:
- The runner will start/stop the daemon automatically and will warm the call graph once if the index has no `call_graph.json` yet.
- Use `--curated` to pick a stable target from a specific curated edge set (defaults to `benchmarks/ts/<corpus>_curated_edges.json` when present).
- Optional:
  - `--include-calls` benchmarks `calls` (can be expensive; defaults to 1 iteration).
  - `--include-semantic` benchmarks `semantic search` only if semantic index artifacts already exist (no automatic indexing/downloading).

## Repeatability Rules

- Do not run benchmarks on a moving branch. Always use the pinned ref from `benchmarks/corpora.json`.
- Always run with index-mode cache isolation:
  - `--cache-root benchmark/cache-root`
  - `--index repo:<corpus-id>`
- Every report must record:
  - `tldr_git_sha`
  - corpus `git_sha`
  - platform + toolchain metadata (`python`, `node`, `pnpm`)

## Phase 4: Python Structural Quality (Django)

Fetch corpus:

```bash
uv run python scripts/bench_fetch_corpora.py --corpus django
```

Warm the Python call graph cache (recommended):

```bash
uv run tldrf warm --cache-root benchmark/cache-root --index repo:django --lang python benchmark/corpora/django
```

Run the structural benchmark suite:

```bash
uv run python scripts/bench_structural_analysis.py --corpus django
```

## Phase 5: Retrieval Quality (File-Level Search)

Query set (tracked):
- `benchmarks/retrieval/django_queries.json`

Run the retrieval-quality runner (semantic/hybrid metrics are skipped unless a semantic index already exists for the index id):

```bash
uv run python scripts/bench_retrieval_quality.py --corpus django
```

Optional (bench-only): allow semantic/hybrid to return "no results" on negative queries by suppressing semantic/hybrid when `rg_pattern` has zero matches:

```bash
uv run python scripts/bench_retrieval_quality.py --corpus django --no-result-guard rg_empty
```

Optional (setup, not timed): build a semantic index for the corpus so semantic/hybrid metrics run.
This may download embedding model weights on first run.

```bash
# Fast iteration (smaller embedding model):
uv run tldrf semantic index --cache-root benchmark/cache-root --index repo:django --lang python --model all-MiniLM-L6-v2 --rebuild benchmark/corpora/django

# Higher-quality (default) model (larger download):
uv run tldrf semantic index --cache-root benchmark/cache-root --index repo:django --lang python --rebuild benchmark/corpora/django
```

## Phase 6: Token Efficiency (Fixed Budgets)

Produces fixed-budget curves (`500/1000/2000/5000/10000` tokens by default) for:
- Python impact/slice/data-flow/complexity workflows (Django structural query set)
- optional retrieval workflows (Django retrieval query set)

Run structural-only:

```bash
uv run python scripts/bench_token_efficiency.py --corpus django --mode structural
```

Run both structural + retrieval:

```bash
uv run python scripts/bench_token_efficiency.py --corpus django --mode both
```

Notes:
- Retrieval metrics include semantic/hybrid only if semantic index artifacts already exist for the index id.
- Use `--budgets` to override token budgets (comma-separated).
- `--no-result-guard rg_empty` applies to retrieval mode only (suppresses semantic/hybrid when `rg_pattern` has zero matches).

## Phase 7: Downstream A/B Prompt Packets (LLM)

Generates randomized per-task A/B prompt packets with rg-derived context vs TLDR-derived context.
Outputs:
- a JSON report under `benchmark/runs/`
- a JSONL prompts file under `benchmark/llm/` (gitignored)

```bash
uv run python scripts/bench_llm_ab_prompts.py --corpus django --budget-tokens 2000
```

Optional: run the prompt packets against an answer model and score structured outputs (structured scoring vs ground truth; no judge model yet).

Task suite + scoring:
- Tasks live in `benchmarks/llm/tasks.json` (currently 30 tasks) and each task references a structural ground-truth query in `benchmarks/python/django_structural_queries.json` via `query_id`.
- Categories: `impact` (callers), `slice` (backward slice lines), `data_flow` (def/use events).
- `scripts/bench_llm_ab_run.py` uses deterministic scoring (no LLM judge yet): it parses the model JSON output, converts it to a set, and computes precision/recall/F1 against the `expected` set embedded in the JSONL prompt packet.
- `overall` metrics (e.g. `f1_mean`) aggregate across all tasks (impact + slice + data_flow).
- `win_rate_tldr_over_rg` is computed per-task by comparing TLDR vs rg F1 (win=1, loss=0, tie=0.5) and averaging.
- `--trials N` reruns each task variant N times and reports per-variant `f1_mean` as the mean across trials (plus timing percentiles).

Using Codex CLI (example model: `gpt-5.3-codex` with "medium" reasoning effort):

```bash
uv run python scripts/bench_llm_ab_run.py \
  --prompts benchmark/llm/<timestamp>-llm-ab-django.jsonl \
  --provider codex \
  --model gpt-5.3-codex \
  --codex-reasoning-effort medium \
  --enforce-json-schema
```

Using Claude Agent SDK (example model alias: `sonnet`):

```bash
uv run python scripts/bench_llm_ab_run.py \
  --prompts benchmark/llm/<timestamp>-llm-ab-django.jsonl \
  --provider claude_sdk \
  --model sonnet \
  --enforce-json-schema
```

Using Claude Code CLI (uses your local Claude login/subscription):

```bash
uv run python scripts/bench_llm_ab_run.py \
  --prompts benchmark/llm/<timestamp>-llm-ab-django.jsonl \
  --provider claude_cli \
  --model sonnet \
  --claude-home "$HOME" \
  --enforce-json-schema
```

Notes:
- `--provider claude_sdk` requires `claude-agent-sdk` and `ANTHROPIC_API_KEY` to be set.
- `--provider claude_cli` writes state under `~/.claude` / `~/.local/share/claude` (debug, todo/session metadata) even for `--print`. In workspace-restricted sandboxes, use `--claude-home "$HOME"` in a non-sandboxed environment, or expect to re-login if using the default isolated `benchmark/claude-home`.
- `--max-tokens` / `--temperature` only apply to the legacy `--provider anthropic` path.
- Use `--limit 3` for a cheap smoke-run before doing the full task set.
