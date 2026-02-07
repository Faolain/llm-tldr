# Spec: 003 - Benchmarking Structural Analysis Quality

## The Question

Does TLDR's AST-based structural analysis (impact analysis, program slicing, complexity assessment, data flow tracing) provide capabilities that an LLM agent equipped with grep/ripgrep fundamentally cannot replicate?

We need to answer this with evidence, not claims. TLDR's README says "95% token savings" but that's measured on our own repo. The community says ripgrep is sufficient. Neither side has data on external codebases with controlled comparisons.

The key insight: benchmarking TLDR on **file retrieval** (grep's home turf) misses the point. TLDR's value proposition is in structural analysis — understanding call graphs, tracing data flow, computing complexity, slicing programs. The benchmark should test **TLDR's capabilities** and ask whether grep can approximate them, not the other way around.

## What We're NOT Doing

- **Not starting with file retrieval benchmarks.** Retrieval (semantic search vs grep for finding files) is a secondary question. If TLDR's structural tools provide capabilities grep can't match, the retrieval story follows naturally. Retrieval benchmarks can be a future phase if structural analysis shows signal.
- **Not starting with SWE-bench.** It's the gold standard but costs hundreds of dollars per run and has too many confounding variables to iterate quickly. It's a future phase if the structural analysis benchmarks show clear signal.
- **Not building a framework.** Each phase is a single script that produces a JSON report. No abstractions until we need them.

## Target Repos

| Repo | LOC | Language | Why |
|------|-----|----------|-----|
| **Django** | ~500K | Python | Deep call chains, middleware layers, ORM -- exactly where grep struggles and `impact`/`slice` should shine |
| **FastAPI** | ~30K | Python | Smaller, heavily typed, good for validating results cross-repo |
| **Express** | ~15K | JavaScript | JS/TS coverage, different language family |

Start with **Django only**. It's large enough that grep genuinely struggles, well-documented enough that we can write ground-truth queries confidently, and Python so our full analysis stack (all 5 layers) applies.

Add FastAPI and Express only after Phase 1 produces results worth validating cross-repo.

---

## Benchmark Contract: What We Count

For every strategy we benchmark, we define a deterministic `payload(query) -> str` that represents the **tool output** provided to the LLM. We measure:
- `payload_tokens`: `tldr.stats.count_tokens(payload)`
- `payload_bytes`: `len(payload.encode("utf-8"))`
- `wall_time_s`: end-to-end time to produce the payload (including subprocess/tool time)

Tokenization is standardized via `tiktoken` using `cl100k_base` (through `tldr.stats.count_tokens()`).

---

## Benchmark Protocol: Warmup + Iterations (For Timing)

Timing is noisy. The daemon has cold-start effects. Semantic search has one-time model loads. We need a protocol that produces stable numbers.

For benchmarks that report time:
1. **Warm up:** run each command once (not counted). This loads caches and avoids first-run overhead skew.
2. **Measure:** run `N` iterations (default 10) and record:
   - mean and standard deviation (regression visibility)
   - p50 and p95 (tail latency)
3. **Report totals:** sum the measured means for a "workflow total time" number.

For large query sets, do not multiply runtime by doing `N` iterations per query. Instead:
- do a single warmup pass
- measure each query once
- report p50/p95 across queries (and optionally rerun the whole suite 2-3 times and compare distributions)

Two different execution paths must be measured explicitly:
- **CLI spawn path:** each iteration runs a fresh `tldrf <cmd>` process (includes process spawn, imports, cache reads).
- **Daemon path:** each iteration sends newline-delimited JSON to the daemon socket directly (avoids measuring the CLI wrapper).

Important implementation detail (borrowed from the Continuous Claude benchmark scripts):
- The daemon response must be read **until newline**, not with a single `recv()` call, so large JSON responses don't get truncated.
- Socket identity must be derived the same way TLDR derives it (legacy vs index mode). Do not assume `md5(project_path)[:8]`.

---

## Phase 0: Is The Daemon Actually Faster? (Latency Microbench)

**Timeline:** 0.5-1 day
**Cost:** Zero LLM tokens

This is not a retrieval-quality benchmark. It's a regression guardrail that answers:
- Do daemon queries outperform CLI spawns on the commands agents actually use?
- Which commands regress when we change caching/indexing internals?

### What We Measure

For a fixed project and fixed parameters, measure daemon vs CLI for:
- `search`
- `extract`
- `impact`
- `tree`
- `structure`
- `context`
- `semantic search` (optional if embeddings installed)

Run it on two corpora:
- **Small fixture** (fast regression test): ~20-100 files so it runs in seconds.
- **Large real repo** (realistic): Django (or another pinned large repo) to see scaling and tail latency.

For each command:
- daemon mean/stdev/p50/p95 (ms)
- CLI mean/stdev/p50/p95 (ms)
- speedup ratio (`cli_mean / daemon_mean`)

Additionally, measure **index build time** (important for evaluating whether indexing overhead is justified, and as a regression baseline for future vector DB migrations like USearch → Jina):
- `warm_full_s`: time for `tldrf warm <repo>` on a cold (no cache) project
- `warm_incremental_s`: time for `tldrf warm <repo>` after touching 2-3 files
- `semantic_index_build_s`: time to build the semantic/FAISS index specifically (if separable)
- `index_size_mb`: total size of `.tldr/cache/` after warm

Run in two modes:
1. **Warm daemon:** daemon already running with indexes loaded.
2. **Cold daemon (optional):** stop daemon, start daemon, and measure the first query latency separately (startup + load penalty).

### Protocol

```
1. Start daemon, let it reach ready state (or run `tldrf warm` first).
2. Warm up: run each command once via daemon and once via CLI (not counted).
3. Measure: N iterations (default 10) per command:
   - Daemon: direct socket query (newline-delimited JSON)
   - CLI: spawn `tldrf <cmd>` fresh each iteration
4. Aggregate and write JSON snapshot.
```

### Output Format

```json
{
  "meta": {
    "phase": 0,
    "date": "2026-02-XX",
    "tldr_git_sha": "…",
    "repo": "django",
    "repo_git_sha": "…",
    "python": "3.12.?.",
    "platform": "darwin",
    "cpu": "…"
  },
  "protocol": { "iterations": 10, "warmup": 1 },
  "index_build": {
    "warm_full_s": 8.2,
    "warm_incremental_s": 0.7,
    "semantic_index_build_s": 45.3,
    "index_size_mb": 12.4,
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "vector_backend": "faiss"
  },
  "results": [
    {
      "cmd": "impact",
      "daemon_ms": { "mean": 0.9, "stdev": 0.2, "p50": 0.8, "p95": 1.4 },
      "cli_ms":    { "mean": 55.0, "stdev": 4.1, "p50": 54.0, "p95": 62.0 },
      "speedup": 61.1
    }
  ]
}
```

---

## Phase 1: Can TLDR Analyze Code Better Than Raw Reading? (Structural Analysis Quality)

**Timeline:** 1-2 days
**Cost:** Zero LLM tokens (deterministic tool output vs ground truth)
**Kill signal:** If TLDR's structural tools are inaccurate (impact precision < 0.7, slice recall < 0.6), the analysis pipeline needs fixing before it can be marketed as a grep alternative.

### The Core Question

TLDR was built for structural analysis workflows (see TLDR.md): refactoring safely with impact analysis, understanding complexity via CFG, tracing data origins with DFG, debugging with program slices. This phase benchmarks **those capabilities directly** and asks whether an agent equipped with grep can approximate the same results.

If grep can do these things equally well, TLDR's value proposition collapses. If it can't, that's the headline.

### What We Measure

For a set of structural analysis tasks against Django, compare:
- **TLDR's structured tool output** (deterministic, AST-based) vs **ground truth**
- **Grep's best approximation** (pattern matching + heuristic extraction) vs **ground truth**

### Task Categories

**Category A: Impact Analysis — "What calls X?" (15 queries)**

The core refactoring question. Before renaming or changing a function's signature, you need to know every caller.

```json
{
  "id": "A01",
  "category": "impact",
  "function": "QuerySet._filter_or_exclude",
  "file": "django/db/models/query.py",
  "expected_callers": [
    {"function": "QuerySet.filter", "file": "django/db/models/query.py"},
    {"function": "QuerySet.exclude", "file": "django/db/models/query.py"},
    {"function": "Manager.filter", "file": "django/db/models/manager.py"}
  ],
  "difficulty": "medium",
  "why_grep_struggles": "grep finds string matches in comments, docstrings, and unrelated filter() calls on lists/dicts"
}
```

**TLDR:** `tldrf impact <function> <django_path>` → structured caller list from AST-based call graph.

**Grep baseline:** `rg "<function>(" --type py <django_path>` → extract file:line pairs, heuristically determine enclosing function name from context. This is what agents actually do today.

**Metrics:**
| Metric | Formula | What It Tells Us |
|--------|---------|-----------------|
| **Caller Precision** | (correctly identified callers) / (total reported callers) | How noisy is the output? |
| **Caller Recall** | (correctly identified callers) / (ground-truth callers) | Does it miss callers? |
| **F1** | harmonic mean of precision and recall | Overall accuracy |

**Category B: Program Slicing — "What affects line N?" (10 queries)**

The debugging question. Given a bug at a specific line, which lines in the function actually contribute to the value at that line?

```json
{
  "id": "B01",
  "category": "slice",
  "file": "django/middleware/csrf.py",
  "function": "CsrfViewMiddleware.process_view",
  "target_line": 185,
  "expected_slice_lines": [142, 147, 155, 162, 170, 178, 185],
  "total_function_lines": 68,
  "difficulty": "hard",
  "why_grep_cant": "slicing requires control+data dependency analysis — grep has no concept of execution flow"
}
```

**TLDR:** `tldrf slice <file> <function> <line>` → set of line numbers in the backward slice.

**Grep baseline:** Grep cannot do program slicing. The baseline is "return all lines of the function" (i.e., no filtering). This establishes the **capability gap** — how much noise reduction does TLDR's slicing provide?

**Metrics:**
| Metric | Formula | What It Tells Us |
|--------|---------|-----------------|
| **Slice Precision** | (correct lines in TLDR slice) / (total lines in TLDR slice) | Does TLDR include irrelevant lines? |
| **Slice Recall** | (correct lines in TLDR slice) / (ground-truth slice lines) | Does TLDR miss relevant lines? |
| **Noise Reduction** | 1 - (TLDR slice size / total function lines) | How much of the function can you skip? |
| **Grep Noise** | 1 - (ground-truth slice size / total function lines) | What fraction of grep's "just read it all" output is irrelevant? |

**Category C: Complexity Assessment — "How complex is X?" (10 queries)**

The refactoring prioritization question. Which functions need decomposition?

```json
{
  "id": "C01",
  "category": "complexity",
  "file": "django/db/models/sql/compiler.py",
  "function": "SQLCompiler.as_sql",
  "expected_complexity": 14,
  "expected_basic_blocks": 18,
  "reference_tool": "radon",
  "difficulty": "easy",
  "why_grep_cant": "cyclomatic complexity requires counting independent execution paths — grep sees text, not control flow"
}
```

**TLDR:** `tldrf cfg <file> <function>` → cyclomatic complexity number + basic block count + edge list.

**Grep baseline:** Grep cannot compute complexity. As a heuristic proxy, count lines matching `if |elif |else:|for |while |except |with ` in the function body. This is what a naive agent might try.

**Ground truth:** Verified against `radon cc` (the standard Python complexity tool).

**Metrics:**
| Metric | Formula | What It Tells Us |
|--------|---------|-----------------|
| **Complexity Accuracy** | (queries where TLDR complexity == radon complexity) / total | Does TLDR's CFG agree with the standard tool? |
| **Complexity MAE** | mean absolute error vs radon | How far off is TLDR when it disagrees? |
| **Grep Heuristic MAE** | mean absolute error of keyword-counting vs radon | How bad is the naive approach? |
| **Ranking Agreement** | Kendall's tau between TLDR ranking and radon ranking (over sets of 5 functions) | Even if absolute numbers differ, does TLDR rank functions correctly? |

**Category D: Data Flow Tracing — "Where does V come from?" (10 queries)**

The debugging and understanding question. Trace a variable back to its origin through assignments and transformations.

```json
{
  "id": "D01",
  "category": "data_flow",
  "file": "django/contrib/auth/hashers.py",
  "function": "make_password",
  "variable": "encoded",
  "expected_flow": [
    {"line": 82, "event": "defined", "via": "hasher.encode(password, salt)"},
    {"line": 75, "event": "dependency", "var": "salt", "via": "hasher.salt()"},
    {"line": 71, "event": "dependency", "var": "hasher", "via": "get_hasher(algorithm)"}
  ],
  "difficulty": "medium",
  "why_grep_struggles": "grep finds all occurrences of 'encoded' but can't distinguish definitions from uses or trace through intermediate variables"
}
```

**TLDR:** `tldrf dfg <file> <function>` → variable definitions, uses, and flow chains.

**Grep baseline:** `rg -n "<variable>" <file>` within the function body → list of lines containing the variable name. No def/use distinction, no flow chain, no transitive dependencies.

**Metrics:**
| Metric | Formula | What It Tells Us |
|--------|---------|-----------------|
| **Origin Accuracy** | (queries where TLDR correctly identifies the variable's origin) / total | Can TLDR trace to the source? |
| **Flow Chain Completeness** | (correct intermediate steps identified) / (ground-truth intermediate steps) | Does TLDR capture the full transformation chain? |
| **Grep Noise Ratio** | (lines matching variable name) / (lines in actual flow chain) | How much noise does raw grep produce for data flow questions? |

### Evaluation Procedure

```
1. Clone django/django at pinned tag 5.1
2. tldrf warm <django_path>
3. For each structural query:
   a. Run the TLDR tool (impact/slice/cfg/dfg) and capture structured output
   b. Run the grep approximation and capture output
   c. Compare both against ground truth
4. For complexity queries, also run `radon cc` as the reference standard
5. Compute metrics per category and overall
6. Output: bench_results/phase1_structural.json
```

### Output Format

```json
{
  "meta": {
    "phase": 1,
    "repo": "django",
    "repo_version": "5.1",
    "date": "2026-02-XX",
    "radon_version": "6.0.1"
  },
  "results": {
    "impact": {
      "tldr":  { "precision": 0.92, "recall": 0.85, "f1": 0.88, "avg_time_ms": 25 },
      "grep":  { "precision": 0.35, "recall": 0.70, "f1": 0.47, "avg_time_ms": 15 }
    },
    "slice": {
      "tldr":  { "precision": 0.88, "recall": 0.82, "noise_reduction": 0.85, "avg_time_ms": 30 },
      "grep":  { "precision": 0.15, "recall": 1.00, "noise_reduction": 0.00, "avg_time_ms": 5 }
    },
    "complexity": {
      "tldr":  { "accuracy": 0.90, "mae": 0.4, "ranking_tau": 0.92 },
      "grep_heuristic": { "accuracy": 0.30, "mae": 3.2, "ranking_tau": 0.65 }
    },
    "data_flow": {
      "tldr":  { "origin_accuracy": 0.85, "flow_completeness": 0.78, "avg_time_ms": 20 },
      "grep":  { "origin_accuracy": 0.40, "noise_ratio": 4.5, "avg_time_ms": 8 }
    },
    "per_query": [ "..." ]
  }
}
```

### Decision Gate

| Outcome | Action |
|---------|--------|
| TLDR impact F1 > grep F1 by >20% AND slice noise_reduction > 0.7 AND complexity ranking_tau > 0.85 | Strong signal. TLDR's structural tools provide capabilities grep cannot match. Proceed to future phases |
| TLDR wins on slice+dfg but impact F1 only marginally better than grep | Grep can approximate caller lookup but not deeper analysis. TLDR's value is in layers 3-5, not layer 2 alone |
| TLDR impact precision < 0.7 or slice recall < 0.6 | The analysis tools need accuracy fixes before benchmarking further. Fix the pipeline |
| Complexity accuracy < 0.8 vs radon | CFG analysis has bugs. Fix before claiming complexity metrics are reliable |

---

## Future Phases (Contingent on Phase 1 Signal)

These phases are designed but not spec'd in detail. Each becomes its own spec if Phase 1 results justify proceeding.

### Retrieval Quality (File-Level Search)

Test whether TLDR's semantic search finds files that grep can't (and vice versa). Compare ripgrep keyword ranking, expert regex patterns, semantic search, and a hybrid (RRF fusion of grep + semantic). 50 hand-written queries against Django across named symbol lookup, behavioral/semantic lookup, cross-file queries, and negative queries. Plus a deterministic dataset of commit-subject → touched-files for scale. Metrics: Recall@5, Recall@10, MRR, Precision@5, FPR. Zero LLM cost.

### Token Efficiency

For queries where both systems find the right code, measure how many tokens each needs. Compare grep excerpts (`rg -C 3`), grep full-file reads, and TLDR structured context. The key test: fixed-budget recall at 500/1000/2000/5000/10000 tokens — what fraction of ground-truth code is captured under each budget? Zero LLM cost.

### Downstream Quality (LLM-as-Judge)

Does an LLM produce better answers about Django when given TLDR context vs grep context? 30 tasks (localization, impact analysis, debugging) evaluated by a different model family (e.g., GPT-4o judging Claude Sonnet answers). Randomized A/B ordering to prevent position bias. Three pipelines: grep-only, TLDR-only, hybrid. ~$50 LLM cost.

### SWE-bench Validation

50 SWE-bench Lite instances from Django. Minimal Agentless-style pipeline with TLDR vs grep for the localization step. Same model/prompt/test harness for patch generation. Measures resolve rate, localization accuracy, token consumption. $100-300 LLM cost.

---

## Implementation Notes

### File Organization

```
scripts/
  bench_perf_daemon_vs_cli.py        # Phase 0 runner
  bench_structural_analysis.py       # Phase 1 runner (impact, slice, cfg, dfg accuracy)
  bench_django_structural_queries.json # Structural analysis query set
bench_results/
  phase0_perf.json                   # Daemon vs CLI perf microbench
  phase1_structural.json             # Structural analysis accuracy
  summary.md                         # Auto-generated comparison table
```

### Running

```bash
# Setup: clone Django at a pinned tag
git clone --depth 1 --branch 5.1 https://github.com/django/django.git /tmp/django

# Phase 0 (no LLM cost): perf microbench + index build time
uv run python scripts/bench_perf_daemon_vs_cli.py --repo /tmp/django --iterations 10

# Phase 1 (no LLM cost): structural analysis accuracy
uv run python scripts/bench_structural_analysis.py --repo /tmp/django --queries scripts/bench_django_structural_queries.json
```

### Snapshot & Reproducibility

Each run saves:
- Git hash of TLDR at time of run
- Git hash (or tag) of target repo
- Full query set used
- Raw results JSON with per-query breakdowns
- Hardware info (for timing comparisons)
- Timing protocol (warmup count, iterations, cold vs warm mode, command list)
- Index configuration (legacy vs index mode, scan_root, cache_root/index_id if used, ignore settings)
- Semantic config (embedding model id + dimension, device, vector DB backend)

**Pinned embedding model:** All benchmarks in this spec use `BAAI/bge-large-en-v1.5` (1024-dimensional embeddings) with FAISS as the vector backend. This is the current default in TLDR. When we migrate to Jina embeddings and/or USearch, the full benchmark suite must be re-run to produce comparable before/after data. The Phase 0 index build time metrics are specifically designed to capture this migration's performance impact.

Results are committed to `bench_results/` so future changes can compare against the baseline.

### Query Set Construction Principles

1. **No cherry-picking.** Write all queries BEFORE running any benchmarks. Don't add or remove queries after seeing results.
2. **Category balance.** Roughly equal queries per category so no single type dominates the aggregate.
3. **Multiple ground-truth answers.** Where possible, list 2-3 acceptable callers/lines/flows so partial matches are captured.
4. **Grep baselines must be reasonable.** Use patterns a competent developer would actually try, not strawman single-word searches.
5. **Difficulty labels.** Mark each query easy/medium/hard based on the indirection level and number of expected results.

---

## What This Answers

After Phase 0-1 (~2-3 days of work, zero LLM cost), you'll know:

- Whether TLDR's impact analysis finds callers that grep misses (and how many false positives grep produces)
- Whether program slicing meaningfully reduces noise vs reading entire functions
- Whether TLDR's complexity metrics agree with established tools (radon)
- Whether TLDR's data flow tracing provides information grep fundamentally cannot
- The concrete capability gap: what can TLDR do that an agent with grep cannot?
- Daemon vs CLI performance characteristics and index build costs

This is enough data to either (a) confidently invest in future phases (retrieval, tokens, downstream, SWE-bench), or (b) identify which structural analysis tools need accuracy fixes before making broader claims.
