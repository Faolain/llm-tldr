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

Generates randomized per-task prompt packets with multiple context variants (rg, TLDR structural context, and retrieval strategies).
Outputs:
- a JSON report under `benchmark/runs/`
- a JSONL prompts file under `benchmark/llm/` (gitignored)

```bash
uv run python scripts/bench_llm_ab_prompts.py --corpus django --budget-tokens 2000
```

Optional: run the prompt packets against an answer model.
There are two Phase 7 modes:
- `--mode structured` (default): deterministically scored against set-valued ground truth embedded in the prompt packet.
- `--mode judge`: open-ended tasks scored by a separate judge model (blinded A/B).

Task suite + scoring:
- Structural tasks live in `benchmarks/llm/tasks.json` (currently 30 tasks) and each task references a structural ground-truth query in `benchmarks/python/django_structural_queries.json` via `query_id`.
- Retrieval-type tasks live in `benchmarks/llm/retrieval_tasks.json` and reference retrieval ground truth in `benchmarks/retrieval/django_queries.json` via `query_id`.
Categories:
- `impact`: list direct callers (scored as a set of `(file, function)` tuples)
- `slice`: compute backward slice lines (scored as a set of line numbers)
- `data_flow`: trace def/use events (scored as a set of `(line, event)` tuples)
- `retrieval`: locate implementation/configuration files (scored as a set of repo-relative file paths from `{\"paths\": [...]}`)
- `scripts/bench_llm_ab_run.py` supports:
- `--mode structured`: deterministic scoring by parsing the model JSON output into a set and computing precision/recall/F1 against the `expected` set embedded in the JSONL prompt packet.
- `--mode judge`: open-ended scoring by running a separate judge model that compares A vs B (blinded) against a rubric and returns a structured verdict.
- `overall` metrics (e.g. `f1_mean`) aggregate across all tasks in the prompt packet.
- `win_rate_tldr_over_rg` is computed per-task by comparing TLDR vs rg F1 (win=1, loss=0, tie=0.5) and averaging (when both sources are present).
- When a prompt packet contains more than two sources, `win_rate_by_pair` and `win_rate_by_pair_by_category` are also reported (e.g. `semantic_over_rg`, `hybrid_rrf_over_rg`).
- `--trials N` reruns each task variant N times and reports per-variant `f1_mean` as the mean across trials (plus timing percentiles).

Retrieval-type tasks (structured mode):

```bash
uv run python scripts/bench_llm_ab_prompts.py \
  --corpus django \
  --tasks benchmarks/llm/retrieval_tasks.json \
  --budget-tokens 2000
```

Interpreting retrieval-type results:
- These tasks are testing the *retrieval + context materialization* path (snippets) and the model's ability to output the correct repo-relative file paths from that context. They are **not** testing TLDR's structural analysis layers (call graphs/slicing/DFG).
- Retrieval variants in the prompt packet:
- `rg`: deterministic ranking by `rg_pattern`, then render a small snippet around the first match in each ranked file.
- `semantic`: embedding-based ranking, then render the same snippet shape (still anchored by `rg_pattern` for determinism).
- `hybrid_rrf`: RRF fusion of `rg` and `semantic` rankings, then snippet rendering.
- If the retrieval queries have strong definition-ish `rg_pattern`s, `rg` is often already near-perfect and `hybrid_rrf` will mostly tie.
- Pure `semantic` can miss “where is X defined?” lookups by returning *usages/references* rather than the defining file; if the context does not surface the defining file, the answer model (correctly) returns an empty `paths` list when instructed not to guess.

Example retrieval structured run (Codex, Django) on 2026-02-10:
- Prompts report: `benchmark/runs/20260210-065101Z-llm-ab-prompts-django-retrieval.json`
- Run report: `benchmark/runs/20260210-065101Z-llm-ab-run-structured-retrieval.json` (`--trials 3`, 16 tasks x 3 variants = 144 calls)
- Key results from that run:
- `f1_mean`: `hybrid_rrf=0.9375`, `rg=0.8958`, `semantic=0.6875`
- Pairwise win-rate (ties=0.5): `hybrid_rrf_over_rg=0.5313`, `hybrid_rrf_over_semantic=0.6250`, `rg_over_semantic=0.5938`
- Note: that report included 1 negative query (`expected=[]`). As of 2026-02-10, Phase 7 structured scoring treats empty/empty as perfect (precision/recall/F1=`1.0`); older reports may undercount negative-query performance.

Recommendations based on retrieval-type results:
- Default retrieval comparisons to `hybrid_rrf` rather than pure semantic.
- If you want semantic to compete on definition lookups, add a “definition-intent” validation step (e.g., require the candidate snippet/file to contain `^def name` / `^class Name` or a symbol-index hit) so semantic doesn't keep surfacing references.
- Use the structural Phase 7 suite (`benchmarks/llm/tasks.json`) to evaluate TLDR's core advantage (impact/slice/data_flow context) rather than the retrieval suite where `rg_pattern` can already solve most tasks.

What TLDR is most useful for (and what would make it more useful):
- TLDR tends to shine on structural workflows: impact analysis (callers through indirection), slicing (“what actually influences this value”), and data-flow (def/use chains), especially under tight token budgets.
- For open-ended debugging tasks, the biggest win is improving TLDR context packing: keep the structured summary, but include a small contiguous code window around the target and around slice/DFG-selected lines (merged windows, budgeted) so the answer model can explain behavior without missing crucial nearby lines/comments.

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
- `--provider claude_sdk` requires `claude-agent-sdk` and a working local `claude` (Claude Code) install. It uses your local Claude Code login/subscription (no API key) and spawns `claude`, so it has the same state-write/sandbox caveats as `--provider claude_cli`.
- `--provider claude_cli` writes state under `~/.claude` / `~/.local/share/claude` (debug, todo/session metadata) even for `--print`. In workspace-restricted sandboxes, use `--claude-home "$HOME"` in a non-sandboxed environment, or expect to re-login if using the default isolated `benchmark/claude-home`.
- `--max-tokens` / `--temperature` only apply to the legacy `--provider anthropic` path.
- Use `--limit 3` for a cheap smoke-run before doing the full task set.

Open-ended tasks (judge mode):
- Tasks live in `benchmarks/llm/open_ended_tasks.json` and include `task_type=open_ended` plus a per-task rubric.
- Generate prompt packets by pointing `bench_llm_ab_prompts.py` at the open-ended suite:

```bash
uv run python scripts/bench_llm_ab_prompts.py \
  --corpus django \
  --tasks benchmarks/llm/open_ended_tasks.json \
  --budget-tokens 2000
```

- Run answer-model A/B and judge-model scoring (blinded) via:

```bash
uv run python scripts/bench_llm_ab_run.py \
  --mode judge \
  --prompts benchmark/llm/<timestamp>-llm-ab-django.jsonl \
  --provider codex \
  --model gpt-5.3-codex \
  --judge-provider claude_sdk \
  --judge-model sonnet \
  --enforce-json-schema
```
