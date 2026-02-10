# Benchmarking Structural Analysis Quality Implementation Plan

- Status: In progress (Phase 0 contract + manifests implemented; Phase 1 TS curated recall runner + Next.js curated edges implemented; Phase 2 `rg` baselines implemented; Phase 3 perf runners implemented; Phase 4 Python structural runner + Django query suite expanded; Phase 5 retrieval runner + Django query suite expanded; Phase 6 token efficiency runner + initial Django curves implemented; Phase 7 task suite + A/B prompt generator implemented; Phase 7 judge-mode harness implemented)
- Owner: TBD
- Last updated: 2026-02-10
- Source: `specs/006-benchmarking-retrieval-quality.md`

## Decisions & Assumptions (locked for this plan)

- Primary goal is to answer (with data): **what can TLDRF do that `rg/grep` fundamentally cannot**, and if/where does it win materially on quality-per-token for realistic workflows.
- We will prioritize **TypeScript monorepos first** (because TS-resolved call graphs, incremental patching, and daemon-warm workflows already exist), then extend to Python structural layers (CFG/DFG/slicing) once the TS harness is solid.
- The “core” benchmarks are **zero LLM cost**:
  - deterministic tool payloads
  - deterministic baseline approximations
  - deterministic scoring vs ground truth
- “Agent-in-the-loop” A/B comparisons are **optional** and live behind a separate phase because they are expensive and noisier.
- All runs must be **repeatable**:
  - pinned corpus git SHAs/tags
  - recorded TLDR git SHA
  - recorded environment/tooling metadata (python/node/pnpm, platform, CPU)
  - explicit cache/index identifiers via `--cache-root` + `--index`
- For TS corpora, the benchmark harness assumes:
  - `node` is available
  - `typescript` is available either from the repo’s `node_modules/typescript` or globally; otherwise the run must be marked `incomplete` and skipped (we do not silently compare “syntax-only” mode to `rg`).
- Token counts are standardized using `tldr.stats.count_tokens()` (cl100k_base), and we always record both `payload_tokens` and `payload_bytes`.

## Scope (v1)

We will benchmark, end-to-end, the workflows that TLDRF is designed for and that grep-style tools can only approximate:

- **Impact analysis (“what calls X?”)** using call graphs (TS monorepos first).
- **Warm-daemon interactive latency** for repeated structural queries (real agent workflow).
- **Incremental edits** (touch/edit one file) and the speed/behavior of incremental patching vs full rebuild.

Python-first structural layers in the spec (slice/cfg/dfg) remain in-scope, but after we have a credible TS harness and a second TS corpus (Next.js).

## Non-goals (v1)

- Proving SWE-bench patch success.
- Building a generic benchmarking framework. Each phase remains a script that emits a JSON report.
- Comparing TLDRF against IDE-grade LSP tooling (tsserver “find references”, Pyright, etc.). This plan focuses on the `rg/grep` baseline first.

## Target Corpora

### TypeScript (Phase 1-3)

- **Peerbit** (local checkout; curated edges already exist)
  - Existing artifacts:
    - `scripts/peerbit_curated_edges.json`
    - `scripts/validate_peerbit_callgraph.py`
    - `scripts/bench_ts_callgraph.py`
- **Next.js** (new; large TS/JS monorepo)
  - Repo: `vercel/next.js`
  - Pinned ref (latest stable as of 2026-02-09): `v16.1.6` (`adf8c612adddd103647c90ff0f511ea35c57076e`)
  - Requires a curated edge set similar to Peerbit.

### Python (future; Phase 4+)

- **Django** at a pinned tag (as described in the spec).

## High-level Architecture (Benchmark Harness)

To keep benchmarks repeatable and “diffable” over time:

- **Tracked inputs** (checked into git):
  - curated ground truth (edges/queries) JSON files
  - corpus manifest (git URLs + pinned refs + setup steps)
  - scoring/evaluation code (scripts)
- **Untracked outputs** (written under ignored dirs):
  - per-run JSON reports (timestamped)
  - caches/indexes (under a dedicated benchmark cache root)

Concretely:

- New tracked directory: `benchmarks/`
  - `benchmarks/corpora.json` (manifest)
  - `benchmarks/ts/peerbit_curated_edges.json` (optional: mirror/move from `scripts/…`)
  - `benchmarks/ts/nextjs_curated_edges.json` (new)
  - `benchmarks/README.md` (how to run)
- Untracked run artifacts:
  - `benchmark/` is already gitignored; use it for:
    - `benchmark/corpora/<id>/` (cloned corpora checkouts)
    - `benchmark/cache-root/` (index-mode caches)
    - `benchmark/runs/<timestamp>-<corpus>-<phase>.json` (reports)

## Phase 0: Lock The Benchmark Contract + Manifests

**Goals**
- Make “benchmarking structural advantage vs `rg`” a repeatable, single-command workflow.
- Standardize query formats, output formats, and metadata capture.

### Running Log (Phase 0)
- 2026-02-09: The repo already has ad-hoc TS benchmarks and validation:
  - `scripts/bench_ts_callgraph.py` (build/patch/daemon timings)
  - `scripts/validate_peerbit_callgraph.py` + `scripts/peerbit_curated_edges.json` (curated recall)
  - `benchmark/bench-tldrf-vs-rg.sh` (manual timing comparisons; not JSON)
  These need to be converted into a stable “contract” so results can be compared across runs and corpora.
- 2026-02-09: `benchmark/` is gitignored; that is the right place for caches and run outputs, but tracked manifests and curated edges must live elsewhere (e.g. `benchmarks/`).
- 2026-02-09: Next.js pinned to the latest stable tag visible via `git ls-remote` at plan authoring time: `v16.1.6` (`adf8c612adddd103647c90ff0f511ea35c57076e`).
- 2026-02-09: Implemented a shared JSON report “contract” helper in `scripts/bench_util.py` (`make_report(...)`, `schema_version`, `phase`, `meta`, `protocol`, `results`) and a compact UTC timestamp helper (`now_utc_compact()`). New benchmark scripts emit a single JSON report using this contract.
- 2026-02-09: Started moving curated benchmark inputs under tracked `benchmarks/`:
  - mirrored Peerbit curated edges to `benchmarks/ts/peerbit_curated_edges.json`
  - added Next.js curated edges to `benchmarks/ts/nextjs_curated_edges.json`
- 2026-02-09: Found two benchmark-blocking issues when running against corpora cloned under gitignored `benchmark/`:
  - **Index-mode gitignore root was derived from `cache_root`** (the host repo), so `git check-ignore` treated the entire corpus checkout under `benchmark/` as ignored, yielding **0 scanned files**.
  - `scan_project()` used `IgnoreSpec.match_file()` which triggers **per-path `git check-ignore` subprocess calls**, making large repo scans (Django) effectively unusable.
  Fixes implemented:
  - `tldr/indexing/index.py`: derive `gitignore_root` from `scan_root` (the scanned project) and auto-update stored index meta (`gitignore_root_abs`) without forcing a rebind.
  - `tldr/cross_file_calls.py`: batch gitignore checks during `scan_project()` to avoid per-file subprocess overhead.
  - Added regression test `tests/test_gitignore_root_nested_repo.py`.

**Deliverables**
- A pinned-corpus manifest (`benchmarks/corpora.json`).
- A stable, single-JSON output contract for each phase.
- A standard “meta” block captured in every run.

**Key tasks**
- [x] Add `benchmarks/README.md` describing:
  - prerequisites for TS corpora (node, pnpm, installed deps)
  - how to run each phase
  - where outputs and cache roots live
- [x] Add `benchmarks/corpora.json` with entries like:
  - `id` (e.g. `peerbit`, `nextjs`)
  - `git_url`
  - `pinned_ref` (tag or SHA)
  - `language` (typescript)
  - `setup` (commands; not timed as part of benchmarks)
  - `scan_root` (subdir, usually repo root)
  - `index_id` (recommended, e.g. `repo:peerbit`)
- [x] Define the shared JSON report schema used across scripts:
  - `meta` (date, tldr_git_sha, corpus id + git sha, platform, python/node, cpu)
  - `protocol` (warmups, iterations, budgets)
  - `results` (phase-specific)
- [x] Add a tiny helper module used by benchmark scripts (keep it minimal, no framework):
  - `scripts/bench_util.py`:
    - `gather_meta(...)`
    - `percentiles(...)`
    - `write_report(out_path, obj)`
- [x] Add a corpus fetcher that materializes pinned corpora under gitignored `benchmark/`:
  - `scripts/bench_fetch_corpora.py`

**Acceptance**
- Running any benchmark script produces a single JSON object that includes complete meta + protocol fields, and a deterministic result structure (stable key ordering is a nice-to-have, not required).

## Phase 1: TypeScript Structural Analysis Quality (Curated Edges + Impact)

**Goals**
- Demonstrate, on real TS monorepos, that TS-resolved call graphs and `impact` answer questions that `rg` cannot (or cannot answer within a sane token budget).
- Catch regressions in TS-resolved call graph quality using curated ground truth.

### Running Log (Phase 1)
- 2026-02-09: Peerbit curated validation already exists and (currently) achieves 100% recall on the curated set:
  - `scripts/peerbit_curated_edges.json`
  - `scripts/validate_peerbit_callgraph.py` (edge presence + `impact` caller recall)
- 2026-02-09: Next step is adding a second TS corpus (Next.js) with its own curated edge set so results are not “Peerbit-shaped”.
- 2026-02-09: Implemented a corpus-agnostic runner `scripts/bench_ts_curated_recall.py`:
  - flags: `--repo-root`, `--curated`, `--cache-root`, `--index`, `--ts-trace`, `--mode`, `--out`
  - uses index-mode cache paths when provided (writes/reads a call-graph cache under the index)
  - emits a single JSON report under `benchmark/runs/` by default
- 2026-02-09: Added `benchmarks/ts/nextjs_curated_edges.json` (37 edges) covering:
  - workspace package import (`@next/env`)
  - default export callers (`loadConfig`)
  - renamed import case (`parseUrl as parseUrlUtil`)
  - barrel/index-style import (`../build` -> `createStaticWorker`)
  - class methods called through concrete types (`Telemetry.flush`)
  - documented explicit exclusions for dynamic dispatch patterns and anonymous inline lambdas.
- 2026-02-09: Smoke-tested `scripts/bench_ts_curated_recall.py` on the TS fixture using `tests/fixtures/ts-monorepo/expected_edges.json`.

**Deliverables**
- A corpus-agnostic “curated edge recall” runner that outputs one JSON report.
- Curated TS edge sets for:
  - Peerbit (existing; potentially mirrored into `benchmarks/`)
  - Next.js (new)

**Key tasks**
- [x] Peerbit curated edges + validator exist:
  - `scripts/peerbit_curated_edges.json`
  - `scripts/validate_peerbit_callgraph.py`
- [x] Generalize `scripts/validate_peerbit_callgraph.py` into a benchmark runner (or add a new script) that:
  - takes `--repo-root`, `--curated`, `--cache-root`, `--index`, `--ts-trace`
  - builds the TS call graph once
  - scores:
    - direct edge recall (`present / total`)
    - impact recall for each callee target (`found / expected`)
  - emits trace reason histograms for misses (`ts_trace.top_reasons`) when enabled
  - writes a single JSON report to `--out` (default under `benchmark/runs/`)
- [x] Add `benchmarks/ts/nextjs_curated_edges.json` (or `scripts/nextjs_curated_edges.json`) with ~30-50 cross-package edges:
  - must include “grep-hard” patterns:
    - barrel re-exports
    - renamed imports
    - default exports
    - workspace package imports (monorepo)
    - class methods called through concrete types
  - explicitly exclude dynamic patterns (element access, `any` dispatch), and document exclusions in the file.

**Acceptance**
- Peerbit: curated edge recall >= 0.90 (target 1.0).
- Next.js: curated edge recall >= 0.80 (target 0.90), with misses explainable via trace output (not silent).

## Phase 2: `rg` Baselines For “What Calls X?” (Deterministic, Non-AST)

**Goals**
- Build a fair, reproducible “LLM with `rg`” proxy that does not cheat by using TLDR’s call graph.
- Quantify how noisy and incomplete the `rg` approximation is, especially under token budgets.

### Running Log (Phase 2)
- 2026-02-09: Implemented `scripts/bench_rg_impact_baseline.py` (deterministic `rg` runner) with:
  - strategies: `match_only`, `match_plus_context`, `match_plus_enclosing_symbol`
  - per-budget payload materialization with `payload_tokens`/`payload_bytes` (via `tldr.stats.count_tokens()`), plus micro-averaged precision/recall/F1 across targets
  - deterministic pattern derivation (including a constructor special-case for `Class.constructor` via `new Class(...)`)
  - optional per-edge `rg_pattern` override supported in curated JSON (edge-level or callee-level).
- 2026-02-09: Added unit tests for the baseline’s pattern derivation + enclosing-symbol heuristic (`tests/test_bench_rg_impact_baseline_helpers.py`) and a schema smoke test for the Next.js curated edge file (`tests/test_bench_curated_edges_nextjs_schema.py`).
- 2026-02-09: Smoke-tested `scripts/bench_rg_impact_baseline.py` on the TS fixture curated edge list (`tests/fixtures/ts-monorepo/expected_edges.json`).

### Baseline Strategies (v1)

- `rg:match_only`
  - Output: `file:line:match` hits for a callee identifier pattern.
  - Scoring: file-level hit coverage only (no caller symbol extraction).
- `rg:match_plus_context`
  - Output: `rg -n -B N -A M` windows around each hit (deterministic ordering).
  - Scoring: file-level hit coverage; token budgets applied to concatenated windows.
- `rg:match_plus_enclosing_symbol` (still deterministic, but more useful)
  - Output: same hits, but with a cheap heuristic to label the enclosing function/method:
    - scan backwards for `function`, `class`, or `methodName(` patterns
    - no parsing, no tsserver/tree-sitter
  - Scoring: compare (file, caller_symbol_guess) to curated callers to compute precision/recall/F1.

**Deliverables**
- A baseline runner that produces comparable metrics to TLDR impact (precision/recall/F1) and token-efficiency curves.

**Key tasks**
- [x] Implement `scripts/bench_rg_impact_baseline.py`:
  - input: `--repo-root`, `--curated`, `--strategy`, `--budgets 500,1000,2000,...`
  - output:
    - `payload_tokens` / `payload_bytes`
    - caller recall/precision/F1 where applicable
    - per-target breakdown (misses, false positives)
- [x] Define deterministic pattern selection for a curated target:
  - for `Class.method`, search for `\\.method\\s*\\(`
  - for `functionName`, search for `\\bfunctionName\\s*\\(`
  - allow curated edges to override the `rg_pattern` if needed (avoid accidental under/over-matching)

**Acceptance**
- Baseline runner can be executed on Peerbit + Next.js and produces stable results across repeated runs on the same checkout.
- We can plot/report a clear gap:
  - TLDR impact F1 materially higher than `rg` baseline F1 for curated targets
  - TLDR achieves higher recall at the same token budget (“quality-per-token” win)

## Phase 3: Performance Benchmarks (Warm Daemon + Incremental Edits)

**Goals**
- Measure realistic interactive latency (warm daemon) vs CLI spawn for structural queries.
- Measure indexing/warm costs (cold vs warm) and cache sizes (so we can reason about tradeoffs).
- Measure incremental patching speed after a small edit on large monorepos.

### Running Log (Phase 3)
- 2026-02-09: `scripts/bench_ts_callgraph.py` already measures:
  - fixture build time
  - incremental patch after touch (TS-resolved patcher)
  - full rebuild after touch
  - Peerbit daemon warm + 5 impact queries (optional)
  This should be evolved into the Phase 3 JSON contract and expanded to Next.js.
- 2026-02-09: Added `scripts/bench_ts_perf.py` (Phase 3 runner) that emits a single JSON report (`phase3_ts_perf`) with:
  - TS callgraph build time + graph metadata (`graph_source`, `incomplete`, edge count)
  - TS-resolved incremental patch vs full rebuild after a deterministic one-file edit (`--touch-file`, with defaults for `peerbit` + `nextjs`)
  - optional daemon warm + impact latency percentiles (`--daemon`, `p50/p95`, N iterations per target derived from curated edges)
  - index-scoped cache sizing (`index_bytes`, `call_graph_cache_bytes`) under `benchmark/cache-root`.
- 2026-02-09: Added `scripts/bench_perf_daemon_vs_cli.py` (Phase 3 runner) that emits `phase3_perf_daemon_vs_cli` reports with:
  - warm daemon vs CLI spawn latency for `search`, `extract`, `impact`, `tree`, `structure`, `context` (mean/stdev + p50/p95 + speedup)
  - optional `calls` microbench behind `--include-calls` (kept opt-in because daemon `calls` currently rebuilds the call graph and returns full edge payloads)
  - index/cache sizing via the same JSON shape as `tldrf index list/info` (`tldr.indexing.management.list_indexes/get_index_info`)
- 2026-02-09: Extended `scripts/bench_ts_perf.py` cache reporting to include `index_mgmt` (list/info) so per-index sizing is recorded in perf reports (not just filesystem walks).

**Deliverables**
- A stable perf microbench report for each TS corpus:
  - `warm_full_s` on cold cache
  - `warm_incremental_s` after a deterministic touch/edit
  - warm daemon `impact` latency distribution (p50/p95; optional mean/stdev)
  - CLI spawn vs daemon ratios for key commands
  - cache/index size under the benchmark cache root

**Protocol**
- Warm up each command once (not counted).
- Measure `N` iterations (default 10) and report mean/stdev + p50/p95.
- Explicitly measure two execution paths:
  - **CLI spawn path** (fresh `tldrf` process each iteration)
  - **Daemon path** (direct socket query; avoids CLI wrapper costs)

**Key tasks**
- [x] Existing manual runner: `scripts/bench_ts_callgraph.py`
- [x] Add `scripts/bench_ts_perf.py` (Phase 3 runner) that emits a Phase 3 JSON report and uses index-mode cache isolation (`--cache-root benchmark/cache-root --index repo:<id>`).
- [x] Add `scripts/bench_perf_daemon_vs_cli.py` (Phase 3 runner) that:
  - runs against a small TS fixture and a large TS corpus (Peerbit, Next.js)
  - measures: `search`, `extract`, `impact`, `tree`, `structure`, `context`, `calls`
  - optionally includes semantic search only if the semantic index exists and the embedding model is cached (no interactive downloads during benchmarks)
  - emits one JSON report under `benchmark/runs/`
- [x] Add a deterministic “touch plan” per corpus (which file to modify, and how) so patch benchmarks are comparable:
  - Implemented as defaults in `scripts/bench_ts_perf.py` (override via `--touch-file`).
- [x] Record cache sizes:
  - total bytes under `benchmark/cache-root/`
  - per-index bytes (from `tldr index list/info`)

**Acceptance**
- Warm daemon `impact` latency p50 is single-digit milliseconds on both Peerbit and Next.js.
- Incremental patch after a one-file touch is materially faster than full rebuild on multi-tsconfig monorepos.

## Phase 4: Python Structural Analysis Quality (Django) (After TS Harness)

**Goals**
- Execute the original spec’s structural benchmarks (impact/slice/cfg/dfg) on Django with deterministic ground truth.

### Running Log (Phase 4)
- 2026-02-09: Added Django to the pinned corpora manifest (`benchmarks/corpora.json`) and fetched it into `benchmark/corpora/django` via `scripts/bench_fetch_corpora.py`.
- 2026-02-09: Implemented an initial Phase 4 runner `scripts/bench_structural_analysis.py` that:
  - loads a query set (default `benchmarks/python/django_structural_queries.json`)
  - loads (or builds) a cached Python call graph under index-mode (`--cache-root benchmark/cache-root --index repo:django`)
  - evaluates per-query + aggregate metrics for:
    - impact (TLDR `impact_analysis` vs a deterministic `rg` + enclosing-symbol heuristic baseline)
    - slice (TLDR `get_slice` vs “read whole function” baseline)
    - complexity (TLDR `get_cfg_context` vs `radon` if installed, plus a naive keyword-count heuristic)
    - data flow (TLDR `get_dfg_context` vs simple in-function variable occurrence baseline)
  - records payload bytes/tokens and wall times for each query.
- 2026-02-09: Added an initial Django query set under tracked inputs (`benchmarks/python/django_structural_queries.json`) with a small starter set across categories. Next step is expanding this to the full spec-sized suite (~15 impact, 10 slice, 10 complexity, 10 data_flow) without cherry-picking after runs.
- 2026-02-09: Added `radon` as a benchmark-only dependency (dev extra) and updated `uv.lock` so complexity comparisons can use `radon cc` as the reference tool.
- 2026-02-09: Confirmed Django Python call graph warming works under index mode:
  - `tldrf warm --cache-root benchmark/cache-root --index repo:django --lang python benchmark/corpora/django` produced ~`2788` python files and ~`11443` edges (on this machine).
- 2026-02-09: Ran the initial suite: `uv run python scripts/bench_structural_analysis.py --corpus django` and produced a JSON report under `benchmark/runs/` (`phase4_python_structural`). Early signal on the starter set:
  - TLDR impact hit all curated expectations (F1=1.0 on this small set), while the `rg` baseline had lower precision/recall.
  - TLDR CFG complexity did not exactly match radon on all starter queries (accuracy 0.33 on 3 queries), but ranked similarly (tau-b ~0.82); heuristic keyword counting was substantially worse (MAE 14).
  Next step: expand the query set to a spec-sized suite so these metrics aren’t dominated by “toy” functions like `_boolean_icon`.
- 2026-02-09: Expanded `benchmarks/python/django_structural_queries.json` to a spec-sized suite (15 impact, 10 slice, 10 complexity, 10 data_flow) and tightened the schema test (`tests/test_bench_django_structural_queries_schema.py`) to enforce minimum counts.
- 2026-02-09: Ran the expanded suite (`uv run python scripts/bench_structural_analysis.py --corpus django`) and captured current signal + gaps:
  - Impact: TLDR beats the deterministic `rg` baseline on this suite (TLDR F1 ~0.73 vs `rg` F1 ~0.31), with high recall but only moderate precision (call graph still produces extra callers).
  - Slice: many targets return a slice equal to the entire function (noise reduction ~0), and slicing some non-control-flow target lines returns an empty set because current CFG blocks do not cover many sequential statements. Updated the runner to **not** treat empty slices as “perfect noise reduction”.
  - Complexity: TLDR cyclomatic complexity matches `radon` only ~50% of the time on the suite (tau-b ~0.84), suggesting decision-point counting and/or CFG semantics diverge from the reference tool.
  - Data flow: origin accuracy ~0.8 on this suite; grep baseline noise ratio mean ~2.1.
  Next step: fix Python CFG/PDG extraction (sequential statement coverage + better control-dependence modeling) and complexity decision-point counting, then re-run the suite to see if we can meet the spec’s gates.
- 2026-02-10: Implemented Python slicing/CFG/PDG fixes and added regression tests:
  - `tldr/cfg_extractor.py`:
    - ensure basic-block line coverage for sequential statements (`generic_visit` extends `end_line`)
    - treat `raise` as an exit (like `return`) so guard-branches that raise correctly gate downstream statements
    - model `assert` as a branch (false path exits) so slices include assertion guards
    - avoid spurious edges after terminators by ending the current flow path (`current_block=None`)
    - avoid overlapping after-block ranges by starting merge blocks at `end_lineno + 1`
  - `tldr/pdg_extractor.py`: Python PDG now uses **statement-level nodes** plus **control dependence** edges (post-dominator based), rather than CFG-flow edges; slicing defaults to a data slice for non-return lines and includes control dependencies for return lines (matches benchmark ground truth).
  - Added unit tests: `tests/test_python_slice_behavior.py`.
- 2026-02-10: Reran Phase 4 structural suite after the slicing fixes:
  - Report: `benchmark/runs/20260210-005452Z-phase4-python-structural-django.json`
  - Slice (10 queries): precision_mean=1.0, recall_mean=0.884, noise_reduction_mean=0.657 (slice is now materially better than “read whole function” and no longer returns empty sets on sequential target lines).
  - Data flow: origin_accuracy=0.9 (improved).

**Deliverables**
- `scripts/bench_structural_analysis.py` + `benchmarks/python/django_structural_queries.json`
- Metrics + decision gates as defined in `specs/006-benchmarking-retrieval-quality.md`

**Key tasks**
- [x] Implement the scripts and query sets as written in the spec (adapt paths/CLI where needed).
- [x] Add `radon` as a benchmark-only dependency (or run it via `uv run`) and record its version in output.

**Acceptance**
- Structural tool accuracy meets the spec’s gate thresholds (or we stop and fix the analysis pipeline).

## Phase 5: Retrieval Quality (File-Level Search) (Future)

**Goals**
- Quantify when TLDR semantic retrieval (and hybrids) find the right files faster/better than `rg`, and when `rg` still wins.

### Running Log (Phase 5)
- 2026-02-09: Implemented an initial retrieval-quality runner `scripts/bench_retrieval_quality.py`:
  - compares `rg` file ranking (hit-count + earliest-hit tie-break) vs TLDR semantic search (if semantic index artifacts exist) vs a hybrid Reciprocal Rank Fusion (RRF) ranker
  - reports Recall@K / Precision@K / MRR on positive queries and FPR@K on negative queries
  - writes a single JSON report (`phase5_retrieval_quality`) under `benchmark/runs/` by default.
- 2026-02-09: Added a starter Django retrieval query set `benchmarks/retrieval/django_queries.json` (12 queries, includes an explicit deterministic `rg_pattern` per query). Next step is expanding this to a spec-sized suite (~50) and adding query sets for TS corpora.
- 2026-02-09: Expanded `benchmarks/retrieval/django_queries.json` to 60 queries (mix of symbol-definition lookups, behavior/concept queries with deterministic regexes, and multiple negative queries) and tightened the schema test (`tests/test_bench_django_retrieval_queries_schema.py`) to enforce `>= 50` queries.
- 2026-02-09: Ran `uv run python scripts/bench_retrieval_quality.py --corpus django` (rg-only; semantic artifacts missing) and captured baseline metrics on this suite:
  - rg MRR ~0.82, Recall@5 ~0.88, Recall@10 ~0.95, FPR@5/10 = 0.0
  Next step: build the semantic index for `repo:django` under `benchmark/cache-root` and rerun to populate semantic + hybrid metrics, then evaluate whether hybrid is Pareto-improving on recall@k/MRR.
- 2026-02-09: Built a semantic index for `repo:django` under `benchmark/cache-root` using `sentence-transformers/all-MiniLM-L6-v2` (fast/local iteration model), then reran Phase 5. Observations on this query set:
  - Semantic-only underperformed `rg` on the mostly “symbol-definition lookup” mix (MRR ~0.25, Recall@5 ~0.46).
  - Hybrid (RRF) improved recall and precision over both pure `rg` and pure semantic (e.g. Recall@10 ~0.98), with MRR roughly tied to pure `rg` (slightly lower on this run).
  - Negative queries: semantic and hybrid have FPR@k = 1.0 because semantic search always returns some top-k results. If we want a “return none” behavior, we likely need a score threshold and/or a lexical guard for semantic hits.
  Next step: consider adding score-thresholding for semantic/hybrid (bench-only first), and run the same suite with the default `bge-large-en-v1.5` model for final numbers.
- 2026-02-09: Started building a separate semantic index for model comparison:
  - index id: `repo:django-bge`
  - embedding model: `bge-large-en-v1.5` (downloads ~1.3GB)
  - sandbox note: semantic unit extraction attempted parallelism but hit `Operation not permitted` and fell back to sequential; expect the first BGE build to be slow in restricted environments.
  - CPU note: semantic indexing defaults to `cpu` on macOS (it will not silently switch to `mps`). You can force it via `tldrf semantic index --device cpu ...` or `TLDR_DEVICE=cpu`.
  - Reliability note: running this long job inside an agent tool-session (PTY) is not durable. The BGE index build reached ~17% (embedding step) and then the tool session disappeared; no `index.faiss`/`metadata.json` were written because those are only emitted at the end. To avoid losing long runs, start them inside `tmux` (or a user terminal) and record the session name + log path immediately.
  Next steps (after index completes): rerun Phase 5 + Phase 6 retrieval with `--index repo:django-bge` and record the resulting `semantic_model`/`semantic_dimension` + MRR/Recall@K/FPR deltas vs MiniLM.
- 2026-02-09: Completed BGE semantic index build for Django:
  - `repo:django-bge` semantic metadata: `BAAI/bge-large-en-v1.5`, dim=1024, count=35712 (`benchmark/cache-root/.tldr/indexes/oiu3xcmk563zfef4bual/cache/semantic/metadata.json`)
  - index artifacts present: `index.faiss` (~146MB) + `metadata.json` (~22MB)
  - indexing printed a syntax error for a known fixture (`tests/test_runner_apps/tagged/tests_syntax_error.py`) but still completed and wrote the full index.
- 2026-02-09: Reran Phase 5 retrieval-quality with both embedding models (same query set, same corpus):
  - Reports:
    - MiniLM: `benchmark/runs/20260209-234341Z-retrieval-django-minilm.json` (`repo:django`, `sentence-transformers/all-MiniLM-L6-v2`, dim=384)
    - BGE: `benchmark/runs/20260209-234341Z-retrieval-django-bge.json` (`repo:django-bge`, `BAAI/bge-large-en-v1.5`, dim=1024)
  - Positive queries (MRR / Recall@5 / Recall@10):
    - rg (both): MRR=0.820, R@5=0.877, R@10=0.947 (unchanged)
    - semantic: MiniLM MRR=0.247, R@5=0.456, R@10=0.544; BGE MRR=0.602, R@5=0.772, R@10=0.789 (large improvement)
    - hybrid_rrf: MiniLM MRR=0.819, R@5=0.912, R@10=0.982; BGE MRR=0.868, R@5=0.965, R@10=1.000 (improvement over MiniLM and rg)
  - Negative queries:
    - rg FPR@5/10 = 0.0
    - semantic + hybrid_rrf FPR@5/10 = 1.0 (unchanged). This reinforces the need for a “return none” gate (score threshold and/or lexical guard) if we want semantic/hybrid to behave well on negative queries.
- 2026-02-10: Score-threshold gating is not viable on this suite: top semantic scores for negative queries overlap positive queries (for both MiniLM and BGE), so a simple “min score” cutoff cannot reliably produce “no result”.
- 2026-02-10: Implemented a **bench-only** “no result” gate and reran Phase 5:
  - `scripts/bench_retrieval_quality.py`: added `--no-result-guard rg_empty` which suppresses semantic/hybrid when `rg_pattern` yields zero matches.
  - This is intentionally benchmark-specific (it relies on the query suite’s deterministic `rg_pattern` being a high-precision “lexical existence check”).
  - Reports:
    - MiniLM + guard: `benchmark/runs/20260210-001934Z-retrieval-django-minilm-guard-rg-empty.json`
    - BGE + guard: `benchmark/runs/20260210-001934Z-retrieval-django-bge-guard-rg-empty.json`
  - Guard triggered on `3/60` queries (the negative queries), and semantic/hybrid FPR@5/10 dropped from `1.0` -> `0.0` without changing positive-query MRR/Recall@K.

Next step options:
1. Implement a bench-only “no result” gate (score threshold and/or lexical guard) and rerun Phase 5/6 to get semantic/hybrid FPR down on negative queries.
2. Move on to the next open quality gap (Python slicing/CFG coverage), then rerun Phase 4/6/7 to see slice flip.

**Deliverables**
- Deterministic retrieval-quality runner + pinned query set(s) per corpus.
- Metrics: Recall@5, Recall@10, MRR, Precision@5, FPR.

**Key tasks**
- [x] Add `benchmarks/retrieval/<corpus>_queries.json` (target: ~50 queries) with:
  - named symbol lookup
  - behavioral/semantic lookup (“where is X implemented?” without exact identifier)
  - cross-file queries (“where is request caching configured?”)
  - negative queries (should return none / no relevant files)
- [x] Implement `scripts/bench_retrieval_quality.py` to compare:
  - `rg` keyword ranking (and a small curated set of “expert regex” patterns)
  - TLDR semantic search
  - hybrid fusion (RRF: `rg` + semantic)
- [ ] Add a deterministic “scale dataset” (optional):
  - commit-subject -> touched-files mapping for a pinned commit range
  - evaluate retrieval against known touched files

**Acceptance**
- Bench output is stable (same checkout produces same top-k and metric totals).
- Hybrid strategy is Pareto-improving on at least one corpus (strictly better than both pure `rg` and pure semantic on recall@k or MRR).

## Phase 6: Token Efficiency (Deterministic Payload Curves)

**Goals**
- Produce fixed-budget curves (`500/1000/2000/5000/10000` tokens) for:
  - impact workflows
  - retrieval workflows
  - (python) slice/dfg/cfg workflows

### Running Log (Phase 6)
- 2026-02-09: Implemented `scripts/bench_token_efficiency.py` (Phase 6 runner) with `--mode structural|retrieval|both` and fixed-budget payload materialization. Structural strategies benchmarked (Django): `tldr_structured`, `tldr_structured_plus_code`, `rg_match_only`, `rg_match_plus_context`, `rg_window_function`, and a slice baseline `grep_window`. Retrieval strategies (Django): `rg`, `semantic`, `hybrid_rrf` using deterministic snippet payloads.
- 2026-02-09: Added helper tests for the new runner (`tests/test_bench_token_efficiency_helpers.py`) covering budget prefix selection and Python indentation-based span heuristics.
- 2026-02-09: Documented Phase 6 commands in `benchmarks/README.md`.
- 2026-02-09: Ran Phase 6 on Django (`uv run python scripts/bench_token_efficiency.py --corpus django --mode both`) and captured current signal:
  - Impact (caller F1, budget 500): `tldr_structured` F1 ~0.73 at ~70 tokens/query (~53 tokens per correct caller) vs `rg_window_function` F1 ~0.63 at ~185 tokens/query (~252 tokens per correct caller). `rg_match_plus_context` had higher precision but much lower recall at 500 tokens (F1 ~0.52).
  - Slice: `grep_window` around target achieves recall ~1.0 on this suite but with very high noise (noise_ratio_mean ~8-12x). `tldr_structured_plus_code` reduces noise (noise_ratio_mean ~2.3x) but recall_mean ~0.78, reflecting the known Python slicing gaps from Phase 4 (empty slices + overly-wide slices).
  - Data flow: on this query set, both TLDR and grep hit-list strategies cover expected flow lines easily (flow_completeness_mean ~1.0), so the main observable gap is noise: full-function windows are much noisier than TLDR’s line-focused payloads.
  - Complexity: TLDR cyclomatic complexity still diverges from `radon` on this suite (accuracy ~0.5, MAE ~1.9); the naive keyword-count heuristic is slightly better here (accuracy ~0.6, MAE ~1.3). This reinforces Phase 4’s “CFG complexity semantics diverge from radon” finding.
  - Retrieval (MiniLM semantic index): `rg` snippet payloads yield MRR ~0.82 with FPR ~0 on negative queries; semantic-only remains low MRR (~0.26) and has FPR ~1; hybrid improves MRR (~0.86) but inherits semantic’s FPR ~1. This matches Phase 5: hybrid can be Pareto-improving on positive queries but needs a “no-result” gate to avoid false positives.
  Next step: refine Phase 6 scoring to better represent multi-step workflows (cheap structured selector list first, then selective code materialization) and extend the query sets where current metrics are too “easy” (especially data-flow).
- 2026-02-09: Fixed a Phase 6 runner bug where a Python word-boundary regex was incorrectly escaped (`\\b` instead of `\b`), causing grep variable-hit baselines to report zero hits; reran Phase 6 after the fix.
- 2026-02-09: Reran Phase 6 retrieval-mode with both embedding models (fixed budgets; same query set):
  - Reports:
    - MiniLM: `benchmark/runs/20260209-234629Z-token-efficiency-retrieval-django-minilm.json` (`repo:django`, `sentence-transformers/all-MiniLM-L6-v2`, dim=384)
    - BGE: `benchmark/runs/20260209-234629Z-token-efficiency-retrieval-django-bge.json` (`repo:django-bge`, `BAAI/bge-large-en-v1.5`, dim=1024)
  - Summary (MRR_mean; budgets 500..10000):
    - rg: ~0.818-0.820 (unchanged)
    - semantic: MiniLM ~0.260-0.262; BGE ~0.612 (delta ~+0.35 across budgets)
    - hybrid_rrf: MiniLM ~0.856-0.857; BGE ~0.857-0.860 (small delta ~+0.003 at 10k)
  - Negative queries (FPR_mean):
    - rg: 0.0
    - semantic + hybrid_rrf: 1.0 (unchanged)
  - Token note: semantic/hybrid payloads are larger with BGE (higher MRR but higher `payload_tokens_mean`), because semantic-selected snippets differ.
- 2026-02-10: Reran Phase 6 retrieval-mode with the bench-only “no result” gate enabled (`--no-result-guard rg_empty`):
  - Reports:
    - MiniLM + guard: `benchmark/runs/20260210-001934Z-token-efficiency-retrieval-django-minilm-guard-rg-empty.json`
    - BGE + guard: `benchmark/runs/20260210-001934Z-token-efficiency-retrieval-django-bge-guard-rg-empty.json`
  - Negative-query FPR_mean for semantic + hybrid_rrf dropped from `1.0` -> `0.0` across budgets.
- 2026-02-10: Reran Phase 6 structural-mode after the Python slicing/CFG/PDG fixes:
  - Report: `benchmark/runs/20260210-005546Z-token-efficiency-structural-django.json`
  - Slice (TLDR structured): precision_mean=1.0, recall_mean=0.884, noise_ratio_mean=0.884 with payload_tokens_mean ~40 (vs `grep_window` noise_ratio_mean ~8-12 at payload_tokens_mean ~200+).

**Deliverables**
- A runner that materializes deterministic payloads for each strategy and scores quality under budgets.

**Key tasks**
- [x] Implement `scripts/bench_token_efficiency.py`:
  - strategy definitions:
    - `rg` match-only, match+context, match+window extraction
    - TLDR structured-only
    - TLDR structured + materialized code (selectors -> code lines)
  - per-budget quality metrics and “tokens per correct item”.

**Acceptance**
- Curves are reproducible and show a measurable “quality-per-token” gap on at least one category where grep is structurally disadvantaged (slice/dfg, or TS impact across monorepo indirection).

## Phase 7: Downstream Quality (LLM-as-Judge) (Future)

**Goals**
- Validate that TLDR-provided context yields better answers than `rg` context for debugging/refactor tasks.

**Deliverables**
- A small A/B task suite and an evaluation harness (judge model distinct from answer model).

**Key tasks**
- [x] Create `benchmarks/llm/tasks.json` (~30 tasks) with expected rubric.
- [x] Add a deterministic retrieval-type LLM suite (`benchmarks/llm/retrieval_tasks.json`) with expected file paths (ground truth from `benchmarks/retrieval/django_queries.json`).
- [x] Implement randomized A/B prompt generation:
  - Condition A: `rg`-derived context payload
  - Condition B: TLDR-derived context payload
  - Condition C (optional): hybrid
- [x] Implement an answer-model runner that supports multiple trials per task and reports p50/p95 timing + win-rate (structured scoring vs ground truth).
- [x] Add a judge-model path for open-ended tasks (`scripts/bench_llm_ab_run.py --mode judge`) and an open-ended suite (`benchmarks/llm/open_ended_tasks.json`).
- [x] Run a small blinded batch (answer model != judge model) and log judge win-rate + score distributions.

### Running Log (Phase 7)
- 2026-02-09: Added a first downstream task suite for Django under `benchmarks/llm/tasks.json` (30 tasks referencing Phase 4’s structural ground-truth query ids: all impact + all slice + first 5 data_flow).
- 2026-02-09: Added a schema/reference test `tests/test_bench_llm_tasks_schema.py` to ensure task ids are unique and all referenced query ids exist and match category.
- 2026-02-09: Implemented `scripts/bench_llm_ab_prompts.py` to generate per-task randomized A/B prompt packets with:
  - TLDR-derived context payloads (impact/slice/dfg) under a fixed context budget
  - rg-derived context payloads (ripgrep-driven windows) under the same budget
  - outputs:
    - JSON report under `benchmark/runs/` (`phase7_llm_ab_prompts`)
    - JSONL prompt packets under `benchmark/llm/` (gitignored) containing both variants per task + expected outputs.
- 2026-02-09: Generated prompt packets for Django (`uv run python scripts/bench_llm_ab_prompts.py --corpus django --budget-tokens 500`):
  - tasks_total=30
  - mean context tokens: rg ~181, tldr ~120
- 2026-02-09: Added `scripts/bench_llm_ab_run.py` to run the prompt packets against an answer model and deterministically score structured outputs vs ground truth (precision/recall/F1 + win-rate TLDR vs rg). Initially it supported only `anthropic` (API key).
- 2026-02-09: Updated `scripts/bench_llm_ab_run.py` to support programmatic, no-API-key execution via:
  - `--provider codex` (Codex CLI; example model `gpt-5.3-codex`)
  - `--provider claude_cli` (Claude Code CLI; example model alias `sonnet`)
  Added:
  - `--enforce-json-schema` (best-effort structured output enforcement)
  - `--timeout-s` per-call timeout
  - `--limit` + `--dry-run` for cheap smoke-runs
  Updated `benchmarks/README.md` accordingly.
- 2026-02-09: Codex output-schema enforcement requires a root JSON Schema of `type: object`, so Phase 7 impact tasks were updated to use `{\"callers\": [...]}` output format (instead of a root array). Updated:
  - `benchmarks/llm/tasks.json` questions
  - `scripts/bench_llm_ab_prompts.py` expected shape + schema hint
  - `scripts/bench_llm_ab_run.py` schema + parsing (accepts both old array and new object forms)
  Regenerated prompt packets (`benchmark/llm/20260209-081746Z-llm-ab-django.jsonl`).
- 2026-02-09: Ran a minimal end-to-end smoke batch (`--limit 1`) and successfully produced structured reports for both:
  - `--provider codex --model gpt-5.3-codex --enforce-json-schema` (`benchmark/runs/20260209-081808Z-llm-ab-run-structured.json`)
  - `--provider claude_cli --model sonnet --enforce-json-schema` (`benchmark/runs/20260209-081845Z-llm-ab-run-structured.json`)
  Next step: run `--limit 3` and then a full 30-task run with `--trials 3` to stabilize win-rate and percentile timing.
- 2026-02-09: Improved Phase 7 `rg` impact context generation in `scripts/bench_llm_ab_prompts.py`:
  - prior approach (full enclosing def) could exceed tight budgets and yield empty context on some targets
  - new approach emits compact per-caller snippets (def header + small windows around each call site) and uses a "skip oversize pieces" budget packer
  This makes the `rg` condition materially stronger and avoids pathological 0-token contexts at `--budget-tokens 500`.
- 2026-02-09: Ran a full 30-task, 3-trial structured A/B batch using Codex CLI:
  - command shape: `--provider codex --model gpt-5.3-codex --codex-reasoning-effort medium --enforce-json-schema --timeout-s 180 --trials 3`
  - prompts: `benchmark/llm/20260209-085046Z-llm-ab-django.jsonl`
  - report: `benchmark/runs/20260209-173450Z-llm-ab-run-structured.json`
  - key results (mean F1 across tasks):
    - overall: TLDR `f1_mean=0.709` vs rg `f1_mean=0.667`; win-rate (incl ties) `0.517`
    - impact (15 tasks): TLDR `f1_mean=0.791` vs rg `0.669`; win-rate (incl ties) `0.567`
    - slice (10 tasks): TLDR `f1_mean=0.452` vs rg `0.523`; win-rate (incl ties) `0.400` (slice remains the main weakness)
    - data_flow (5 tasks): TLDR `f1_mean=0.978` vs rg `0.950`; win-rate (incl ties) `0.600`
  - latency percentiles:
    - p50: TLDR ~2.11s vs rg ~4.06s
    - p95: TLDR ~4.86s vs rg ~14.11s
  Next step: fix Python slicing/CFG coverage (Phase 4/6) and rerun to see if slice flips.
- 2026-02-09: Claude Code CLI (`--provider claude_cli`) is not reliable in workspace-restricted sandboxes because it writes state under `~/.claude` / `~/.local/share/claude`. Implemented a Claude Agent SDK option instead:
  - Repro: even `claude --print "hello"` attempts to write debug/todo state under `~/.claude` and acquire version locks under `~/.local/share/claude`, which fails under a workspace-write sandbox with errors like:
    - `EPERM: operation not permitted, open '/Users/aristotle/.claude/debug/<session>.txt'`
    - `EPERM: operation not permitted, open '/Users/aristotle/.claude/todos/<session>...json'`
    - `NON-FATAL: Lock acquisition failed for /Users/aristotle/.local/share/claude/versions/...`
  - Verified local install: `which claude` -> `/Users/aristotle/.local/bin/claude`; `claude --version` -> `2.1.37 (Claude Code)`.
  - Sandbox workaround: `scripts/bench_llm_ab_run.py` redirects `HOME`/`XDG_*` to `benchmark/claude-home` by default for `--provider claude_cli|claude_sdk` (override via `--claude-home`). This avoids EPERM but **does not reuse the existing Claude subscription login**, so Claude CLI prints: `Not logged in · Please run /login`.
  - Added dependency: `claude-agent-sdk` and support for `--provider claude_sdk` (programmatic calls). Correction: `claude-agent-sdk` talks to the local `claude` (Claude Code) CLI, so it uses your Claude Code login/subscription (no API key) and inherits the same state-write/sandbox caveats.
- 2026-02-09: Fixed Claude Code CLI structured-output capture in `scripts/bench_llm_ab_run.py`:
  - `claude --json-schema ... --output-format text` yields empty stdout, even on successful calls.
  - For schema enforcement, we now use `--output-format json` and read `structured_output`, then feed that JSON into the existing scorer.
- 2026-02-09: Ran a full 30-task, 3-trial structured A/B batch using Claude Code CLI + Sonnet 4.5:
  - command shape: `--provider claude_cli --model claude-sonnet-4-5-20250929 --claude-home $HOME --enforce-json-schema --timeout-s 180 --trials 3`
  - prompts: `benchmark/llm/20260209-085046Z-llm-ab-django.jsonl`
  - report: `benchmark/runs/20260209-193314Z-llm-ab-run-structured.json`
  - answers: `benchmark/llm/20260209-193314Z-llm-ab-answers.jsonl`
  - key results (mean F1 across tasks):
    - overall: TLDR `f1_mean=0.709` vs rg `0.708`; win-rate (incl ties) `0.483`
    - impact (15 tasks): TLDR `f1_mean=0.791` vs rg `0.780`; win-rate (incl ties) `0.533`
    - slice (10 tasks): TLDR `f1_mean=0.452` vs rg `0.466`; win-rate (incl ties) `0.400`
    - data_flow (5 tasks): TLDR `f1_mean=0.978` vs rg `0.978`; win-rate (incl ties) `0.500`
  - latency percentiles:
    - p50: TLDR ~5.56s vs rg ~7.49s
    - p95: TLDR ~7.29s vs rg ~16.47s
  - cost (from per-call `total_cost_usd`): TLDR ~$5.72, rg ~$6.55 (total ~$12.27)

- 2026-02-10: After the Python slicing/CFG/PDG fixes (Phase 4/6), reran Phase 7 and observed the slice tasks flip:
  - prompts report: `benchmark/runs/20260210-005817Z-llm-ab-prompts-django.json` (budget_tokens=2000; tokens_context_mean: rg ~277.5, tldr ~37.6)
  - prompts: `benchmark/llm/20260210-005817Z-llm-ab-django.jsonl`
  - Codex run (trials=1, `model_reasoning_effort=medium`):
    - report: `benchmark/runs/20260210-005817Z-llm-ab-run-codex.json`
    - answers: `benchmark/llm/20260210-005817Z-llm-ab-answers-codex.jsonl`
  - key results:
    - overall: TLDR `f1_mean=0.865` vs rg `0.598`; win_rate_tldr_over_rg `0.683` (ties count as `0.5`)
    - slice (10 tasks): TLDR `f1_mean=0.919` vs rg `0.477`; slice win-rate (ties count as `0.5`) `0.800` (vs `0.400` in `benchmark/runs/20260209-173450Z-llm-ab-run-structured.json`)
  - Note: Phase 7 uses structural payloads (impact/slice/dfg); embedding model only affects retrieval benchmarks (Phase 5/6 retrieval).

- 2026-02-10: Stabilized Phase 7 numbers by rerunning the same prompt packet with `--trials 3` (Codex):
  - report: `benchmark/runs/20260210-030111Z-llm-ab-run-codex.json`
  - answers: `benchmark/llm/20260210-030111Z-llm-ab-answers-codex.jsonl`
  - key results:
    - overall: TLDR `f1_mean=0.865` vs rg `0.619`; win_rate_tldr_over_rg `0.683` (ties count as `0.5`)
    - slice (10 tasks): TLDR `f1_mean=0.919` vs rg `0.471`; slice win-rate (ties count as `0.5`) `0.800` (stable vs the `--trials 1` rerun above)
    - latency (p50/p95): TLDR `2.19s/4.39s` vs rg `4.26s/12.23s`

- 2026-02-10: Fixed `--provider claude_sdk` Phase 7 extraction and turn-budgeting:
  - `claude-agent-sdk` yields typed messages; the terminal output comes from `ResultMessage.structured_output` (it does not expose a `.type == "result"` attribute).
  - `max_turns=1` can terminate early with subtype `error_max_turns` and empty output; updated `scripts/bench_llm_ab_run.py` to use `max_turns=2` for the Claude SDK path.
  - Added regression tests for `_claude_sdk_result_to_text_and_usage` in `tests/test_bench_llm_ab_run_helpers.py`.

- 2026-02-10: Ran a full 30-task, 3-trial structured A/B batch using Claude Agent SDK + Sonnet 4.5 on the **new** prompt packet (same as the stabilized Codex run above):
  - command shape: `--provider claude_sdk --model claude-sonnet-4-5-20250929 --claude-home $HOME --enforce-json-schema --timeout-s 180 --trials 3`
  - prompts: `benchmark/llm/20260210-005817Z-llm-ab-django.jsonl`
  - report: `benchmark/runs/20260210-040732Z-llm-ab-run-claude.json`
  - answers: `benchmark/llm/20260210-040732Z-llm-ab-answers-claude.jsonl`
  - key results:
    - overall: TLDR `f1_mean=0.865` vs rg `0.655`; win_rate_tldr_over_rg `0.700` (ties count as `0.5`)
    - slice (10 tasks): TLDR `f1_mean=0.919` vs rg `0.406`; slice win-rate (ties count as `0.5`) `1.000`
    - latency (p50/p95): TLDR `5.26s/7.22s` vs rg `6.59s/12.41s`

  Comparison (same prompt packet `20260210-005817Z-llm-ab-django.jsonl`, `--trials 3`):

  | Category | Codex TLDR F1 | Codex rg F1 | Codex win_rate | Claude TLDR F1 | Claude rg F1 | Claude win_rate |
  | --- | --- | --- | --- | --- | --- | --- |
  | overall | 0.865 | 0.619 | 0.683 | 0.865 | 0.655 | 0.700 |
  | impact | 0.791 | 0.607 | 0.633 | 0.791 | 0.713 | 0.567 |
  | slice | 0.919 | 0.471 | 0.800 | 0.919 | 0.406 | 1.000 |
  | data_flow | 0.978 | 0.950 | 0.600 | 0.978 | 0.978 | 0.500 |

- 2026-02-10: Implemented the Phase 7 open-ended “judge model” path:
  - Added an open-ended task suite: `benchmarks/llm/open_ended_tasks.json` (tasks have `task_type=open_ended` + a per-task rubric).
  - Extended prompt generation (`scripts/bench_llm_ab_prompts.py`) to:
    - pass through `task_type` + `rubric` into the prompt packet
    - generate open-ended answer prompts (plain text) instead of forced JSON
    - for open-ended tasks, materialize more helpful TLDR context (structured + relevant code snippets) for `slice` and `data_flow` so the answer model can actually explain behavior
  - Extended the runner (`scripts/bench_llm_ab_run.py`) with `--mode judge`:
    - runs the answer model on A and B (free-form)
    - runs a separate judge model (blinded A/B) that returns a structured verdict JSON (winner + scores + notes)
    - aggregates judge win-rate TLDR vs rg and score means per dimension (correctness/groundedness/completeness/clarity/actionability)
  - Added schema tests for the open-ended suite: `tests/test_bench_llm_open_ended_tasks_schema.py`.
  Next step: generate an open-ended prompt packet and run a small blinded batch (e.g. Codex answers + Claude judge) and log results here.

- 2026-02-10: Ran a judge-mode open-ended smoke batch (Codex answers, Claude judge; limit=3):
  - prompts report: `benchmark/runs/20260210-052131Z-llm-ab-prompts-django.json` (open-ended suite; budget_tokens=2000)
  - prompts: `benchmark/llm/20260210-052131Z-llm-ab-django.jsonl`
  - command shape:
    - answer model: `--provider codex --model gpt-5.3-codex --codex-reasoning-effort medium`
    - judge model: `--judge-provider claude_sdk --judge-model claude-sonnet-4-5-20250929 --enforce-json-schema`
    - `--mode judge --trials 1 --limit 3`
  - report: `benchmark/runs/20260210-052355Z-llm-ab-run-judge.json`
  - answers+jury: `benchmark/llm/20260210-052355Z-llm-ab-answers-judge.jsonl`
  - key results (3 tasks, all `impact`):
    - judge win_rate_tldr_over_rg: `0.667` (2 wins, 1 loss; ties=0.5 but none here)
    - judge score means (rg vs TLDR):
      - correctness: `4.333` vs `4.667`
      - groundedness: `4.667` vs `5.000`
      - completeness: `3.333` vs `4.000`
      - clarity: `4.333` vs `4.667`
      - actionability: `3.333` vs `4.333`
    - answer latency p50/p95:
      - rg: `10.72s/11.83s`
      - tldr: `8.97s/16.01s` (p95 noisier at n=3)
    - judge latency p50/p95: `18.51s/21.83s`
  Next step: run the full 12 open-ended tasks (and ideally `--trials 3`) to stabilize win-rate and per-dimension score distributions (see next entry).

  Note: full judge-mode runs can take long enough that they are not reliable to run in a foreground terminal within an agent/tool session. Prefer running in `tmux` and tee logs to a file under `benchmark/logs/`, for example:

  ```bash
  tmux new-session -d -s bench-judge 'cd <repo-root> && PYTHONUNBUFFERED=1 NO_COLOR=1 CI=1 uv run python scripts/bench_llm_ab_run.py --mode judge --prompts benchmark/llm/<packet>.jsonl --provider codex --model gpt-5.3-codex --codex-reasoning-effort medium --judge-provider claude_sdk --judge-model claude-sonnet-4-5-20250929 --claude-home "$HOME" --enforce-json-schema --timeout-s 180 --judge-timeout-s 180 --trials 3 2>&1 | tee benchmark/logs/bench-judge.log'
  ```

- 2026-02-10: Ran the full open-ended judge-mode batch (12 tasks x 3 trials; Codex answers, Claude judge):
  - prompts: `benchmark/llm/20260210-052131Z-llm-ab-django.jsonl`
  - command shape:
    - answer model: `--provider codex --model gpt-5.3-codex --codex-reasoning-effort medium`
    - judge model: `--judge-provider claude_sdk --judge-model claude-sonnet-4-5-20250929 --enforce-json-schema`
    - `--mode judge --timeout-s 180 --judge-timeout-s 180 --trials 3`
  - report: `benchmark/runs/20260210-053918Z-llm-ab-run-judge.json`
  - answers+jury: `benchmark/llm/20260210-053918Z-llm-ab-answers-judge.jsonl` (108 rows; 2 answer rows + 1 judge row per task/trial)
  - key results:
    - overall judge win_rate_tldr_over_rg: `0.306`
    - by category win_rate_tldr_over_rg:
      - impact: `0.667`
      - slice: `0.208`
      - data_flow: `0.042`
    - judge score means (rg vs TLDR): correctness `4.900` vs `4.433`, groundedness `5.000` vs `4.767`, completeness `4.367` vs `3.533`, clarity `4.733` vs `4.267`, actionability `4.167` vs `3.533`
    - judge_bad_json: `6` (verdict parse failures; counted as ties)
    - latency p50/p95 (answer): TLDR `7.87s/11.81s` vs rg `7.68s/13.23s`; judge `15.49s/22.39s`
  Next step:
  1. Phase 7 (open-ended judge): improve TLDR open-ended slice/data_flow context packing and reduce judge parse failures.
    - This is the immediate follow-up after the 12-task x 3-trial judge run above.
    - Practically:
      - Adjust `scripts/bench_llm_ab_prompts.py` to include better budgeted code windows around slice/DFG-selected lines (not just the exact lines).
      - Make the judge path more robust in `scripts/bench_llm_ab_run.py` (retry / treat empty verdicts as errors, reduce `judge_bad_json`).

- 2026-02-10: Implemented the Phase 7 open-ended follow-ups (context packing + judge robustness):
  - `scripts/bench_llm_ab_prompts.py`: for open-ended `slice` and `data_flow`, TLDR context now materializes small **code windows** (radius=3) around slice/DFG-selected lines (plus a short function header), instead of emitting only the exact selected lines.
  - `scripts/bench_llm_ab_run.py`: judge-mode now retries on invalid/empty verdicts (`--judge-retries`, default=1) and records verdict parse failures as explicit judge errors (instead of silently treating them as ties). Judge rows now record `attempts` + `usage_attempts` for cost attribution.

- 2026-02-10: Fixed Phase 7 task/question mismatches:
  - `benchmarks/llm/tasks.json`: corrected 4 slice task questions (`L22`-`L25`) to match structural `query_id` targets (`B07`-`B10`) instead of outdated `parse_*` / `salted_hmac` text.
  - `benchmarks/llm/open_ended_tasks.json`: corrected `OE08` to match `query_id=B10` (`configure` in `django/conf/__init__.py` at `target_line=124`).
  - Note: the judge-mode prompt packet `benchmark/llm/20260210-052131Z-llm-ab-django.jsonl` (and the resulting report `benchmark/runs/20260210-053918Z-llm-ab-run-judge.json`) was generated before this fix, so interpret that run with caution and prefer regenerating prompts + rerunning for comparable numbers.

- 2026-02-10: Regenerated open-ended prompt packets and reran judge-mode after mismatch fixes + improved TLDR context packing/judge robustness:
  - prompts report: `benchmark/runs/20260210-160641Z-llm-ab-prompts-django.json` (open-ended suite; `tasks_total=18`, budget_tokens=2000; tokens_context_mean: rg ~250.2, tldr ~410.1)
  - prompts: `benchmark/llm/20260210-160641Z-llm-ab-django.jsonl`
  - command shape:
    - answer model: `--provider codex --model gpt-5.3-codex --codex-reasoning-effort medium`
    - judge model: `--judge-provider claude_sdk --judge-model claude-sonnet-4-5-20250929 --enforce-json-schema --judge-retries 1`
    - `--mode judge --timeout-s 180 --judge-timeout-s 180 --trials 3`
  - report: `benchmark/runs/20260210-161458Z-llm-ab-run-judge-open-ended.json`
  - answers+jury: `benchmark/llm/20260210-161458Z-llm-ab-answers-judge-open-ended.jsonl` (162 rows; 2 answer rows + 1 judge row per task/trial)
  - key results:
    - overall judge win_rate_tldr_over_rg: `0.556`
    - by category win_rate_tldr_over_rg:
      - impact: `0.833`
      - slice: `0.286` (slice remains the main weakness in open-ended judge mode)
      - data_flow: `0.600`
    - judge_bad_json: `0` (down from 6 in `benchmark/runs/20260210-053918Z-llm-ab-run-judge.json`)
    - judge score means (rg vs TLDR): correctness `4.704` vs `4.833`, groundedness `4.759` vs `4.981`, completeness `4.296` vs `4.481`, clarity `4.463` vs `4.611`, actionability `4.148` vs `4.278`
    - latency p50/p95 (answer): TLDR `9.76s/19.49s` vs rg `10.11s/19.11s`; judge `13.34s/26.92s`
  - notes: answer_errors_total=1 (rg only); judge_errors_total=0.
  Next step options:
  - Improve open-ended `slice` TLDR context further by including a larger contiguous window around `target_line` (rg-style) *plus* budgeted merged windows around slice-selected lines (to capture nearby guard conditions/comments that the slice may omit), then rerun judge-mode and compare slice win-rate.
  - If Phase 7 is “good enough” after slice improvements: proceed to Phase 8 SWE-bench Lite localization harness (file-level localization metrics vs `rg` baselines).

- 2026-02-10: Expanded Phase 7 task suites beyond structural-only:
  - Added a deterministic retrieval-type suite: `benchmarks/llm/retrieval_tasks.json` (expected file paths from `benchmarks/retrieval/django_queries.json`).
  - Extended `scripts/bench_llm_ab_prompts.py` to support `category=retrieval` and generate multiple retrieval variants (`rg`, `semantic`, `hybrid_rrf`) under the same token budget.
  - Extended `scripts/bench_llm_ab_run.py` to deterministically score retrieval outputs (`{\"paths\": [...]}`) and report pairwise win rates across all sources (not only `tldr_over_rg`).
  - Expanded the open-ended suite with more debugging-style questions (`OE13`-`OE18`).

- 2026-02-10: Ran Phase 7 deterministic retrieval-type structured batch (Codex):
  - prompts report: `benchmark/runs/20260210-065101Z-llm-ab-prompts-django-retrieval.json` (`tasks_total=16`, `budget_tokens=2000`, `tokens_context_mean`: rg ~403.7, semantic ~404.3, hybrid_rrf ~617.6)
  - prompts: `benchmark/llm/20260210-065101Z-llm-ab-django-retrieval.jsonl`
  - run report: `benchmark/runs/20260210-065101Z-llm-ab-run-structured-retrieval.json` (`--trials 3`, 16 tasks x 3 variants = 144 calls)
  - answers: `benchmark/llm/20260210-065101Z-llm-ab-answers-retrieval.jsonl`
  - results (`f1_mean` across tasks; old scorer treated empty/empty as `f1=0` so absolute means are underreported due to 1 negative task):
    - hybrid_rrf: `0.9375` (adjusted: `1.0000`)
    - rg: `0.8958` (adjusted: `0.9583`)
    - semantic: `0.6875` (adjusted: `0.7500`)
    - pairwise win rates (ties=0.5): `hybrid_rrf_over_rg=0.5313`, `rg_over_semantic=0.5938`, `hybrid_rrf_over_semantic=0.6250`
  Notes:
  - On 4 positive tasks (`LR04`, `LR11`, `LR13`, `LR14`), `semantic` consistently returned `{\"paths\": []}` (missed the definition file within the ranked+budgeted context); `hybrid_rrf` and `rg` remained perfect on this run.

- 2026-02-10: Fixed Phase 7 structured scorer empty-set handling:
  - `scripts/bench_llm_ab_run.py`: when both expected and predicted sets are empty, treat precision/recall/F1 as `1.0` (previously `0.0` due to 0/0 handling).
  - This mainly affects negative retrieval tasks (and any future tasks whose ground truth is legitimately empty).

**Acceptance**
- Clear win-rate signal on at least one task class (impact/slicing/debugging).

### Notes / Clarifications (Phase 7)
- Current Phase 7 (as implemented) is **deterministically scored**: tasks have a structured `expected` output embedded in the prompt packet (derived from `benchmarks/python/django_structural_queries.json` via `benchmarks/llm/tasks.json`), and `scripts/bench_llm_ab_run.py` computes precision/recall/F1 by parsing the model JSON output into a set:
  - `impact`: set of `(file, function)` tuples from `{"callers":[{"file":..., "function":...}, ...]}`
  - `slice`: set of line numbers from `{"lines":[...]}`
  - `data_flow`: set of `(line, event)` tuples from `{"flow":[{"line":..., "event":"defined"|"used"}, ...]}`
  The reported `f1_mean` is aggregated across all 30 tasks (impact + slice + data_flow). `win_rate_tldr_over_rg` is computed per-task (win=1, loss=0, tie=0.5) and averaged.

- Phase 7 retrieval-type structured tasks (`benchmarks/llm/retrieval_tasks.json`) are testing the **retrieval + context materialization** path (snippets) and the answer model’s ability to output correct repo-relative file paths from that context. They are **not** testing TLDR’s structural analysis layers (call graphs/slicing/DFG).
  - Variants in the retrieval prompt packet:
    - `rg`: deterministic ranking by `rg_pattern`, then render a small snippet around the first match per ranked file.
    - `semantic`: embedding ranking, then render the same snippet shape (still anchored by `rg_pattern`).
    - `hybrid_rrf`: RRF fusion of `rg` and `semantic` rankings, then snippet rendering.
  - Recommendation: default comparisons to `hybrid_rrf` (semantic-only is unreliable for “where is X defined?” because it often returns references/usages instead of the defining file).
  - To make semantic competitive on definition lookups: add a “definition-intent” validation/rerank step (e.g., require a `^def name` / `^class Name` match or a symbol-index hit in the candidate file/snippet).

- TLDR’s clearest “worth using” signal is still the structural workflows (Phase 6/7 structured suites): impact analysis through indirection, slicing to isolate influencing statements, and data-flow to separate definitions from uses under tight token budgets. Retrieval-only tasks can be `rg`-dominated when `rg_pattern` is already definition-shaped.

- “Judge model path for open-ended tasks” means adding a parallel evaluation mode for tasks that **do not have clean set-valued ground truth** (e.g., “diagnose why this fails”, “propose a fix”, “explain the root cause”, “recommend a refactor”):
  - Generate A/B prompt packets as usual (Condition A: rg-derived context, Condition B: TLDR-derived context), but allow free-form answers.
  - Run an answer model to produce responses for each condition.
  - Run a **separate judge model** that sees both answers (blinded as A/B) and a rubric, and emits a structured verdict like `{"winner":"A"|"B"|"tie","scores":{...},"notes":...}`.
  - Aggregate judge win-rate + score distributions (implemented via `scripts/bench_llm_ab_run.py --mode judge`; see Running Log for a full 12-task batch).

- Phase 8 (SWE-bench validation) would be a separate harness to validate TLDR on a standard benchmark dataset:
  - Select a fixed SWE-bench Lite subset relevant to a target corpus (e.g. Django-related instances).
  - Implement localization pipelines:
    - Condition A: `rg`-only localization
    - Condition B: TLDR-assisted localization (impact/context/structure and/or retrieval)
  - Score localization against known touched files (Recall@k/MRR/etc) and record tokens/time/cost (optionally extend to patch success later).

- Expanding/adjusting the Phase 7 task suite can mean:
  - Add **retrieval-type tasks** (“where is X implemented/configured?”) that are still deterministically scorable (expected file paths/symbols) and compare different context heuristics (`rg` vs semantic vs hybrid vs TLDR).
  - Add **open-ended debugging/refactor tasks** that require the judge path above.
  - Add additional experimental conditions (beyond rg vs TLDR) and tabulate results per category and per model (Codex vs Claude).

- Future: GEPA prompt optimization (for Phase 7 open-ended judge):
  - Scope: use GEPA (reflective evolutionary, Pareto-style prompt tuning) to optimize prompt + context-format knobs (e.g. slice/DFG window sizes/ordering + instructions), not just wording.
  - Objectives: improve judge win-rate / score means while reducing context tokens (Pareto frontier over quality vs cost).
  - Guardrails: train/holdout task split; keep judge prompt fixed (or use multiple judges); report only holdout deltas to avoid overfitting/judge-hacking.

## Phase 8: SWE-bench Validation (Future)

**Goals**
- Validate the localization advantage (and optionally patch success) on a subset of SWE-bench Lite relevant to the target corpora.

**Deliverables**
- A reproducible harness that runs a fixed subset and reports localization accuracy + cost.

**Key tasks**
- [ ] Select a fixed subset (e.g., 50 Django-related instances).
- [ ] Implement a minimal localization pipeline:
  - Condition A: `rg`-only localization
  - Condition B: TLDR-assisted localization (impact/context/structure)
- [ ] Record tokens/time and success metrics.

**Acceptance**
- TLDR-assisted localization improves localization accuracy and/or reduces tokens on the fixed subset.

## Files / Modules Likely to Change

- New: `benchmarks/README.md`
- New: `benchmarks/corpora.json`
- New: `benchmarks/ts/peerbit_curated_edges.json` (optional: mirror from `scripts/peerbit_curated_edges.json`)
- New: `benchmarks/ts/nextjs_curated_edges.json`
- New: `scripts/bench_util.py`
- New: `scripts/bench_fetch_corpora.py`
- Modify: `scripts/validate_peerbit_callgraph.py` (generalize) or add a new benchmark runner script
- Modify: `scripts/bench_ts_callgraph.py` (Phase 3 contract + Next.js support) or add `scripts/bench_ts_perf.py`
- New: `scripts/bench_rg_impact_baseline.py`
- New: `scripts/bench_perf_daemon_vs_cli.py`
- New: `scripts/bench_retrieval_quality.py`
- New: `scripts/bench_token_efficiency.py`
- New: `benchmarks/llm/tasks.json`
- New: `scripts/bench_llm_ab_prompts.py`

## Acceptance Criteria (Summary)

- Repeatability:
  - pinned corpora + recorded SHAs
  - deterministic query sets
  - index-mode cache isolation
- Structural advantage signal (TS):
  - curated-edge recall meets thresholds on Peerbit and Next.js
  - trace output explains misses (when enabled)
  - TLDR beats `rg` baselines on quality-per-token under fixed budgets
- Performance:
  - warm daemon `impact` queries are low-latency
  - incremental patch beats full rebuild on large monorepos

Important: Keep a running log within this document of all learnings, ahas, answers to questions, and next steps in the appropriate sections for each phase, and update it as implementation progresses.
