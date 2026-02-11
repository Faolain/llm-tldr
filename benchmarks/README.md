# Benchmarks (Repeatable Corpora + Runs)

This directory contains **tracked** benchmark inputs (corpus manifests, curated edge sets, query sets).

All **untracked** run artifacts live under the gitignored `benchmark/` directory at repo root:

- `benchmark/corpora/<id>/`        Cloned corpora checkouts (pinned refs)
- `benchmark/cache-root/`          Index-mode caches (`--cache-root benchmark/cache-root`)
- `benchmark/runs/<timestamp>/`    JSON reports produced by scripts

## Results Snapshot (Pinned Runs)

Numbers below are copied from the JSON reports under `benchmark/runs/` (so they are reproducible and diffable).

### Phase 1-3: TS Fixture Smoke Runs (ts-monorepo)

| Phase | Report | Key result |
| --- | --- | --- |
| Phase 1 (TS curated recall) | `benchmark/runs/20260209-043537Z-ts-curated-recall-ts-monorepo.json` | edge recall 1.000 (9/9), build_s 1.225 |
| Phase 2 (rg impact baseline) | `benchmark/runs/20260209-043543Z-rg-impact-baseline-ts-monorepo-match_plus_enclosing_symbol.json` | F1 0.000 at budgets 200/500 (fixture-only smoke run) |
| Phase 3 (TS perf) | `benchmark/runs/20260209-044240Z-ts-perf-ts-monorepo.json` | build_s 1.231, patch_s 0.815, full_rebuild_after_touch_s 1.070 |

### Phase 4: Django Structural Quality (Deterministic)

Report: `benchmark/runs/20260210-005452Z-phase4-python-structural-django.json`

| Workload | TLDR | Baseline |
| --- | --- | --- |
| impact (caller set) | **F1 0.727 (P 0.588, R 0.952)** | rg F1 0.306 (P 0.216, R 0.524) |
| slice (line set) | **P_mean 1.000, R_mean 0.884, noise_reduction_mean 0.657** | (baseline is whole-function window; see Phase 6 for token curves) |
| data_flow | **origin_accuracy 0.900, flow_completeness_mean 1.000** | grep noise_ratio_mean 2.142 |
| complexity (cyclomatic) | **acc 0.600, MAE 1.80, tau-b 0.901** | grep heuristic MAE 6.20 |

### Phase 5: Django Retrieval Quality (Ranking Metrics)

Report (BGE + negative guard): `benchmark/runs/20260210-001934Z-retrieval-django-bge-guard-rg-empty.json`

| Strategy | MRR | Recall@5 | Recall@10 | FPR@5 | FPR@10 |
| --- | --- | --- | --- | --- | --- |
| rg | 0.820 | 0.877 | **0.947** | **0.000** | **0.000** |
| semantic | 0.602 | 0.772 | **0.789** | **0.000** | **0.000** |
| hybrid_rrf | **0.868** | **0.965** | **1.000** | **0.000** | **0.000** |

Notes:
- This run uses `--no-result-guard rg_empty` (bench-only): if `rg_pattern` yields 0 hits, semantic/hybrid are suppressed so negatives can return "no result".

### Phase 6: Django Token Efficiency (Fixed Budgets)

Structural report: `benchmark/runs/20260210-214935Z-token-efficiency-structural-django-multistep.json`

Bold: best per budget row and best per strategy column (ties: fewer tokens).

Impact (caller set):

| Budget | tldr_structured | rg_window_function | rg_match_plus_context | tldr_structured_plus_code |
| --- | --- | --- | --- | --- |
| 500 | **F1 0.727** (P 0.588, R 0.952), tok 70.5, tok/TP 52.9 | F1 0.629 (P 0.786, R 0.524), tok 184.9, tok/TP 252.1 | F1 0.516 (P 0.800, R 0.381), tok 300.9, tok/TP 564.1 | F1 0.533 (P 0.889, R 0.381), tok 141.8, tok/TP 265.9 |
| 1000 | **F1 0.727** (P 0.588, R 0.952), tok 70.5, tok/TP 52.9 | **F1 0.638** (P 0.577, R 0.714), tok 465.6, tok/TP 465.6 | **F1 0.619** (P 0.619, R 0.619), tok 621.4, tok/TP 717.0 | F1 0.682 (P 0.652, R 0.714), tok 464.9, tok/TP 464.9 |
| 2000 | **F1 0.727** (P 0.588, R 0.952), tok 70.5, tok/TP 52.9 | F1 0.576 (P 0.447, R 0.809), tok 812.1, tok/TP 716.6 | F1 0.520 (P 0.448, R 0.619), tok 805.9, tok/TP 929.9 | F1 0.704 (P 0.576, R 0.905), tok 828.2, tok/TP 653.8 |
| 5000 | **F1 0.727** (P 0.588, R 0.952), tok 70.5, tok/TP 52.9 | F1 0.567 (P 0.436, R 0.809), tok 882.5, tok/TP 778.7 | F1 0.481 (P 0.394, R 0.619), tok 883.1, tok/TP 1019.0 | **F1 0.727** (P 0.588, R 0.952), tok 900.1, tok/TP 675.1 |
| 10000 | **F1 0.727** (P 0.588, R 0.952), tok 70.5, tok/TP 52.9 | F1 0.567 (P 0.436, R 0.809), tok 882.5, tok/TP 778.7 | F1 0.481 (P 0.394, R 0.619), tok 883.1, tok/TP 1019.0 | F1 0.727 (P 0.588, R 0.952), tok 900.1, tok/TP 675.1 |

Slice (line set):

| Budget | tldr_structured | tldr_structured_plus_code | grep_window |
| --- | --- | --- | --- |
| 500 | **F1 0.938** / P 1.000 / R 0.884 / noise 0.88 / tok 40.4 | **F1 0.557** / P 0.386 / R 1.000 / noise 8.00 / tok 238.0 | **F1 0.557** / P 0.386 / R 1.000 / noise 8.40 / tok 200.0 |
| 1000 | **F1 0.938** / P 1.000 / R 0.884 / noise 0.88 / tok 40.4 | F1 0.556 / P 0.385 / R 1.000 / noise 12.40 / tok 273.6 | F1 0.556 / P 0.385 / R 1.000 / noise 12.40 / tok 232.2 |
| 2000 | **F1 0.938** / P 1.000 / R 0.884 / noise 0.88 / tok 40.4 | F1 0.556 / P 0.385 / R 1.000 / noise 12.40 / tok 273.6 | F1 0.556 / P 0.385 / R 1.000 / noise 12.40 / tok 232.2 |
| 5000 | **F1 0.938** / P 1.000 / R 0.884 / noise 0.88 / tok 40.4 | F1 0.556 / P 0.385 / R 1.000 / noise 12.40 / tok 273.6 | F1 0.556 / P 0.385 / R 1.000 / noise 12.40 / tok 232.2 |
| 10000 | **F1 0.938** / P 1.000 / R 0.884 / noise 0.88 / tok 40.4 | F1 0.556 / P 0.385 / R 1.000 / noise 12.40 / tok 273.6 | F1 0.556 / P 0.385 / R 1.000 / noise 12.40 / tok 232.2 |

Data flow:

| Budget | tldr_structured | tldr_structured_plus_code | grep_window_function |
| --- | --- | --- | --- |
| 500 | **flow 1.000** / noise 1.35 / tok 37.8 | **flow 1.000** / noise 5.27 / tok 199.4 | flow 0.900 / noise 8.92 / tok 223.7 |
| 1000 | **flow 1.000** / noise 1.35 / tok 37.8 | flow 1.000 / noise 5.27 / tok 199.4 | **flow 1.000** / noise 10.87 / tok 255.9 |
| 2000 | **flow 1.000** / noise 1.35 / tok 37.8 | flow 1.000 / noise 5.27 / tok 199.4 | flow 1.000 / noise 10.87 / tok 255.9 |
| 5000 | **flow 1.000** / noise 1.35 / tok 37.8 | flow 1.000 / noise 5.27 / tok 199.4 | flow 1.000 / noise 10.87 / tok 255.9 |
| 10000 | **flow 1.000** / noise 1.35 / tok 37.8 | flow 1.000 / noise 5.27 / tok 199.4 | flow 1.000 / noise 10.87 / tok 255.9 |

Complexity (cyclomatic, vs radon):

| Budget | tldr_structured | grep_heuristic |
| --- | --- | --- |
| 500 | acc 0.600 / **MAE 1.80** / tok 28.7 | acc 0.600 / **MAE 1.30** / tok 28.7 |
| 1000 | acc 0.600 / MAE 1.80 / tok 28.7 | acc 0.600 / **MAE 1.30** / tok 28.7 |
| 2000 | acc 0.600 / MAE 1.80 / tok 28.7 | acc 0.600 / **MAE 1.30** / tok 28.7 |
| 5000 | acc 0.600 / MAE 1.80 / tok 28.7 | acc 0.600 / **MAE 1.30** / tok 28.7 |
| 10000 | acc 0.600 / MAE 1.80 / tok 28.7 | acc 0.600 / **MAE 1.30** / tok 28.7 |

Retrieval report (BGE + negative guard): `benchmark/runs/20260210-001934Z-token-efficiency-retrieval-django-bge-guard-rg-empty.json`

| Budget | rg | semantic | hybrid_rrf |
| --- | --- | --- | --- |
| 500 | MRR 0.818 / FPR 0.000 / tok 111.1 | **MRR 0.612** / FPR 0.000 / tok 366.9 | **MRR 0.857** / FPR 0.000 / tok 375.9 |
| 1000 | **MRR 0.820** / FPR 0.000 / tok 159.0 | MRR 0.612 / FPR 0.000 / tok 404.5 | **MRR 0.860** / FPR 0.000 / tok 437.5 |
| 2000 | MRR 0.820 / FPR 0.000 / tok 220.4 | MRR 0.612 / FPR 0.000 / tok 419.4 | **MRR 0.860** / FPR 0.000 / tok 504.7 |
| 5000 | MRR 0.820 / FPR 0.000 / tok 252.8 | MRR 0.612 / FPR 0.000 / tok 419.4 | **MRR 0.860** / FPR 0.000 / tok 530.5 |
| 10000 | MRR 0.820 / FPR 0.000 / tok 252.8 | MRR 0.612 / FPR 0.000 / tok 419.4 | **MRR 0.860** / FPR 0.000 / tok 530.5 |

### Phase 7: Downstream A/B (LLM)

Structured tasks (fixed JSON outputs, deterministically scored; `budget_tokens=2000`, `--trials 3`):

Reports:
- Codex: `benchmark/runs/20260210-030111Z-llm-ab-run-codex.json`
- Claude: `benchmark/runs/20260210-040732Z-llm-ab-run-claude.json`

| Category | Codex TLDR F1 | Codex rg F1 | Codex win_rate | Claude TLDR F1 | Claude rg F1 | Claude win_rate |
| --- | --- | --- | --- | --- | --- | --- |
| overall | **0.865** | 0.619 | 0.683 | **0.865** | 0.655 | 0.700 |
| impact | **0.791** | 0.607 | 0.633 | **0.791** | 0.713 | 0.567 |
| slice | **0.919** | 0.471 | **0.800** | **0.919** | 0.406 | **1.000** |
| data_flow | **0.978** | **0.950** | 0.600 | **0.978** | **0.978** | 0.500 |

Retrieval-type structured tasks (file paths; deterministically scored; `budget_tokens=2000`, `--trials 3`):

Report: `benchmark/runs/20260210-065101Z-llm-ab-run-structured-retrieval.json`

| Strategy | f1_mean (report) | f1_mean (adjusted\*) |
| --- | --- | --- |
| hybrid_rrf | **0.9375** | **1.0000** |
| rg | 0.8958 | **0.9583** |
| semantic | 0.6875 | **0.7500** |

\* The run report was produced before the scorer fix that treats empty-expected + empty-predicted as F1=1.0 (one negative task in this suite).

Open-ended judge-mode tasks (free-form answers, judged A/B; `--trials 3`):

| Suite | Budget | Tasks | Overall win_rate | impact | slice | data_flow | Report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| open_ended_full | **2000** | **18** | **0.694** | **0.778** | **0.595** | **0.733** | `benchmark/runs/20260210-205924Z-llm-ab-run-judge-open-ended-t3.json` |
| slice_only | 1000 | 7 | **0.548** |  | **0.548** |  | `benchmark/runs/20260210-211226Z-llm-ab-run-judge-slice-1000-t3.json` |
| slice_only | 500 | 7 | **0.429** |  | **0.429** |  | `benchmark/runs/20260210-212621Z-llm-ab-run-judge-slice-500-t3.json` |

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
- Pure `semantic` can miss "where is X defined?" lookups by returning *usages/references* rather than the defining file; if the context does not surface the defining file, the answer model (correctly) returns an empty `paths` list when instructed not to guess.

Example retrieval structured run (Codex, Django) on 2026-02-10:
- Prompts report: `benchmark/runs/20260210-065101Z-llm-ab-prompts-django-retrieval.json`
- Run report: `benchmark/runs/20260210-065101Z-llm-ab-run-structured-retrieval.json` (`--trials 3`, 16 tasks x 3 variants = 144 calls)
- Key results from that run:
- `f1_mean`: `hybrid_rrf=0.9375`, `rg=0.8958`, `semantic=0.6875`
- Pairwise win-rate (ties=0.5): `hybrid_rrf_over_rg=0.5313`, `hybrid_rrf_over_semantic=0.6250`, `rg_over_semantic=0.5938`
- Note: that report included 1 negative query (`expected=[]`). As of 2026-02-10, Phase 7 structured scoring treats empty/empty as perfect (precision/recall/F1=`1.0`); older reports may undercount negative-query performance.

Recommendations based on retrieval-type results:
- Default retrieval comparisons to `hybrid_rrf` rather than pure semantic.
- If you want semantic to compete on definition lookups, add a "definition-intent" validation step (e.g., require the candidate snippet/file to contain `^def name` / `^class Name` or a symbol-index hit) so semantic doesn't keep surfacing references.
- Use the structural Phase 7 suite (`benchmarks/llm/tasks.json`) to evaluate TLDR's core advantage (impact/slice/data_flow context) rather than the retrieval suite where `rg_pattern` can already solve most tasks.

What TLDR is most useful for (and what would make it more useful):
- TLDR tends to shine on structural workflows: impact analysis (callers through indirection), slicing ("what actually influences this value"), and data-flow (def/use chains), especially under tight token budgets.
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
  --judge-retries 1 \
  --enforce-json-schema
```

Example open-ended judge run (Codex answers, Claude judge; Django) on 2026-02-10:
- Prompts report: `benchmark/runs/20260210-160641Z-llm-ab-prompts-django.json` (tasks_total=18; budget_tokens=2000)
- Run report: `benchmark/runs/20260210-161458Z-llm-ab-run-judge-open-ended.json` (`--trials 3`, `--judge-retries 1`)
- Key results (judge win_rate_tldr_over_rg; ties=0.5):
- Overall: `0.556`
- Impact: `0.833`
- Data flow: `0.600`
- Slice: `0.286` (slice remains the main weakness in open-ended judge mode)
- `judge_bad_json`: `0`
