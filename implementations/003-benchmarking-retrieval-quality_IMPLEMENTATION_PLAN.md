# Benchmarking Structural Analysis Quality Implementation Plan

- Status: In progress (Phase 0 scaffolding + pinned corpora fetcher implemented; Next.js corpus pinned+fetchable; Next.js curated edges + `rg` baselines pending)
- Owner: TBD
- Last updated: 2026-02-09
- Source: `specs/003-benchmarking-retrieval-quality.md`

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
- [ ] Define the shared JSON report schema used across scripts:
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

**Deliverables**
- A corpus-agnostic “curated edge recall” runner that outputs one JSON report.
- Curated TS edge sets for:
  - Peerbit (existing; potentially mirrored into `benchmarks/`)
  - Next.js (new)

**Key tasks**
- [x] Peerbit curated edges + validator exist:
  - `scripts/peerbit_curated_edges.json`
  - `scripts/validate_peerbit_callgraph.py`
- [ ] Generalize `scripts/validate_peerbit_callgraph.py` into a benchmark runner (or add a new script) that:
  - takes `--repo-root`, `--curated`, `--cache-root`, `--index`, `--ts-trace`
  - builds the TS call graph once
  - scores:
    - direct edge recall (`present / total`)
    - impact recall for each callee target (`found / expected`)
  - emits trace reason histograms for misses (`ts_trace.top_reasons`) when enabled
  - writes a single JSON report to `--out` (default under `benchmark/runs/`)
- [ ] Add `benchmarks/ts/nextjs_curated_edges.json` (or `scripts/nextjs_curated_edges.json`) with ~30-50 cross-package edges:
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
- [ ] Implement `scripts/bench_rg_impact_baseline.py`:
  - input: `--repo-root`, `--curated`, `--strategy`, `--budgets 500,1000,2000,...`
  - output:
    - `payload_tokens` / `payload_bytes`
    - caller recall/precision/F1 where applicable
    - per-target breakdown (misses, false positives)
- [ ] Define deterministic pattern selection for a curated target:
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
- [ ] Add `scripts/bench_perf_daemon_vs_cli.py` (Phase 3 runner) that:
  - runs against a small TS fixture and a large TS corpus (Peerbit, Next.js)
  - measures: `search`, `extract`, `impact`, `tree`, `structure`, `context`, `calls`
  - optionally includes semantic search only if the semantic index exists and the embedding model is cached (no interactive downloads during benchmarks)
  - emits one JSON report under `benchmark/runs/`
- [ ] Extend/replace `scripts/bench_ts_callgraph.py` with a Phase 3-compatible runner that:
  - runs on `peerbit` and `nextjs` (by path or by manifest id)
  - emits one JSON report:
    - `build_s`, `edge_count`, `graph_source`, `incomplete`
    - `patch_after_touch_s` vs `full_rebuild_after_touch_s`
    - daemon `warm_s` and `impact_ms` p50/p95 (N iterations per target)
  - uses index mode (`--cache-root benchmark/cache-root --index repo:<id>`) so caches are isolated from dev workspaces
- [ ] Add a deterministic “touch plan” per corpus (which file to modify, and how) so patch benchmarks are comparable.
- [ ] Record cache sizes:
  - total bytes under `benchmark/cache-root/`
  - per-index bytes (from `tldr index list/info`)

**Acceptance**
- Warm daemon `impact` latency p50 is single-digit milliseconds on both Peerbit and Next.js.
- Incremental patch after a one-file touch is materially faster than full rebuild on multi-tsconfig monorepos.

## Phase 4: Python Structural Analysis Quality (Django) (After TS Harness)

**Goals**
- Execute the original spec’s structural benchmarks (impact/slice/cfg/dfg) on Django with deterministic ground truth.

**Deliverables**
- `scripts/bench_structural_analysis.py` + `benchmarks/python/django_structural_queries.json`
- Metrics + decision gates as defined in `specs/003-benchmarking-retrieval-quality.md`

**Key tasks**
- [ ] Implement the scripts and query sets as written in the spec (adapt paths/CLI where needed).
- [ ] Add `radon` as a benchmark-only dependency (or run it via `uv run`) and record its version in output.

**Acceptance**
- Structural tool accuracy meets the spec’s gate thresholds (or we stop and fix the analysis pipeline).

## Phase 5: Retrieval Quality (File-Level Search) (Future)

**Goals**
- Quantify when TLDR semantic retrieval (and hybrids) find the right files faster/better than `rg`, and when `rg` still wins.

**Deliverables**
- Deterministic retrieval-quality runner + pinned query set(s) per corpus.
- Metrics: Recall@5, Recall@10, MRR, Precision@5, FPR.

**Key tasks**
- [ ] Add `benchmarks/retrieval/<corpus>_queries.json` (target: ~50 queries) with:
  - named symbol lookup
  - behavioral/semantic lookup (“where is X implemented?” without exact identifier)
  - cross-file queries (“where is request caching configured?”)
  - negative queries (should return none / no relevant files)
- [ ] Implement `scripts/bench_retrieval_quality.py` to compare:
  - `rg` keyword ranking (and a small curated set of “expert regex” patterns)
  - TLDR semantic search
  - hybrid fusion (RRF: `rg` + semantic)
- [ ] Add a deterministic “scale dataset” (optional):
  - commit-subject -> touched-files mapping for a pinned commit range
  - evaluate retrieval against known touched files

**Acceptance**
- Bench output is stable (same checkout produces same top-k and metric totals).
- Hybrid strategy is Pareto-improving on at least one corpus (strictly better than both pure `rg` and pure semantic on recall@k or MRR).

## Phase 6: Token Efficiency (Deterministic Payload Curves) (Future)

**Goals**
- Produce fixed-budget curves (`500/1000/2000/5000/10000` tokens) for:
  - impact workflows
  - retrieval workflows
  - (python) slice/dfg/cfg workflows

**Deliverables**
- A runner that materializes deterministic payloads for each strategy and scores quality under budgets.

**Key tasks**
- [ ] Implement `scripts/bench_token_efficiency.py`:
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
- [ ] Create `benchmarks/llm/tasks.json` (~30 tasks) with expected rubric.
- [ ] Implement randomized A/B prompt generation:
  - Condition A: `rg`-derived context payload
  - Condition B: TLDR-derived context payload
  - Condition C (optional): hybrid
- [ ] Run multiple trials per task, report p50/p95, and compute win-rate under blinded judging.

**Acceptance**
- Clear win-rate signal on at least one task class (impact/slicing/debugging).

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
