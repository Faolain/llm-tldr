# Spec: 003 - Benchmarking Retrieval Quality & Value Proposition

## The Question

Is TLDR's structured, AST-based context retrieval meaningfully better than grep/ripgrep for LLM coding agents -- or is grep all you need?

We need to answer this with evidence, not claims. The README says "95% token savings" but that's measured on our own repo. The community says ripgrep is sufficient. Neither side has data on external codebases with controlled comparisons.

## What We're NOT Doing

- **Not using RepoBench as the primary harness.** RepoBench measures cross-file completion retrieval with pre-chunked candidate pools. It doesn't directly measure token efficiency, structured context quality, or multi-hop traces (call graphs, slices). It's the wrong primary tool for our question.
  - We may borrow ideas (e.g., "retrieval-only" evaluation) or run a small subset later as a sanity check, but our core regression suite must match real agent workflows (grep/keyword search vs TLDR outputs).
- **Not starting with SWE-bench.** It's the gold standard but costs hundreds of dollars per run and has too many confounding variables to iterate quickly. We'll get there in Phase 4 if earlier phases show signal.
- **Not building a framework.** Each phase is a single script that produces a JSON report. No abstractions until we need them.

## Target Repos

| Repo | LOC | Language | Why |
|------|-----|----------|-----|
| **Django** | ~500K | Python | Deep call chains, middleware layers, ORM -- exactly where grep struggles and `impact`/`slice` should shine |
| **FastAPI** | ~30K | Python | Smaller, heavily typed, good for semantic search on API patterns |
| **Express** | ~15K | JavaScript | JS/TS coverage, different language family |

Start with **Django only**. It's large enough that grep genuinely struggles, well-documented enough that we can write ground-truth queries confidently, and Python so our full analysis stack (all 5 layers) applies.

Add FastAPI and Express only after Phase 1 produces results worth validating cross-repo.

## Retrieval Strategies Under Test

Every phase compares these strategies/baselines:

| ID | Strategy | What It Does |
|----|----------|-------------|
| **G0** | `ripgrep` (keyword file ranking) | Extract keywords from the natural-language query, run `rg` per keyword, rank files by hits, return top-k files. Models what agents actually do without hand-writing regexes. |
| **G1** | `ripgrep` (expert regex) | `rg <grep_pattern>` using a hand-written regex pattern (when provided). This is a strong baseline: "what a competent dev might try after a minute." |
| **S** | `tldrf semantic` | Semantic search -- natural language query over 5-layer embeddings |
| **C** | `tldrf context` + `tldrf impact` | Structured context -- AST summaries + call graph + impact analysis |

Baselines **G0/G1** represent what agents do today (ripgrep + skim excerpts, sometimes escalating to reading whole files). Strategies **S** and **C** represent what TLDR adds. They are tested independently because they answer different questions: **S** asks "can we find the right code?", **C** asks "can we provide better context about it?"

---

## Benchmark Contract: What We Count

If we can't precisely define "the thing an agent would paste into the model," token metrics are easy to game and hard to compare.

For every strategy we benchmark, we define a deterministic `payload(query) -> str` that represents the **tool output** provided to the LLM. We measure:
- `payload_tokens`: `tldr.stats.count_tokens(payload)`
- `payload_bytes`: `len(payload.encode("utf-8"))`
- `wall_time_s`: end-to-end time to produce the payload (including subprocess/tool time)

Tokenization is standardized via `tiktoken` using `cl100k_base` (through `tldr.stats.count_tokens()`).

Two important rules:
1. **Token comparisons must be done on the payload**, not on "full file contents" unless we are explicitly modeling an agent that reads entire files.
2. When comparing under a fixed budget, **all payloads are truncated to N tokens using the same truncation policy** (e.g., stop after N tokens, do not reflow text).

---

## Benchmark Protocol: Warmup + Iterations (For Timing)

Timing is noisy. The daemon has cold-start effects. Semantic search has one-time model loads. We need a protocol that produces stable numbers.

For benchmarks that report time:
1. **Warm up:** run each command once (not counted). This loads caches and avoids first-run overhead skew.
2. **Measure:** run `N` iterations (default 10) and record:
   - mean and standard deviation (regression visibility)
   - p50 and p95 (tail latency)
3. **Report totals:** sum the measured means for a “workflow total time” number.

For large query sets (e.g., Phase 1 with 50-1000 queries), do not multiply runtime by doing `N` iterations per query. Instead:
- do a single warmup pass
- measure each query once
- report p50/p95 across queries (and optionally rerun the whole suite 2-3 times and compare distributions)

Two different execution paths must be measured explicitly:
- **CLI spawn path:** each iteration runs a fresh `tldrf <cmd>` process (includes process spawn, imports, cache reads).
- **Daemon path:** each iteration sends newline-delimited JSON to the daemon socket directly (avoids measuring the CLI wrapper).

Important implementation detail (borrowed from the Continuous Claude benchmark scripts):
- The daemon response must be read **until newline**, not with a single `recv()` call, so large JSON responses don’t get truncated.
- Socket identity must be derived the same way TLDR derives it (legacy vs index mode). Do not assume `md5(project_path)[:8]`.

---

## Phase 0: Is The Daemon Actually Faster? (Latency Microbench)

**Timeline:** 0.5-1 day
**Cost:** Zero LLM tokens

This is not a retrieval-quality benchmark. It’s a regression guardrail that answers:
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

Run in two modes:
1. **Warm daemon:** daemon already running with indexes loaded.
2. **Cold daemon (optional):** stop daemon, start daemon, and measure the first query latency separately (startup + load penalty).

### Protocol (Suggested)

```
1. Start daemon, let it reach ready state (or run `tldrf warm` first).
2. Warm up: run each command once via daemon and once via CLI (not counted).
3. Measure: N iterations (default 10) per command:
   - Daemon: direct socket query (newline-delimited JSON)
   - CLI: spawn `tldrf <cmd>` fresh each iteration
4. Aggregate and write JSON snapshot.
```

### Output Format (Sketch)

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

## Phase 1: Can TLDR Find the Right Code? (Retrieval Quality)

**Timeline:** 1-2 days to implement, minutes to run
**Cost:** Zero LLM tokens
**Kill signal:** If TLDR semantic search can't match the stronger grep baseline (max of G0/G1) on Recall@5, stop here -- the indexing overhead isn't justified.

### What We Measure

Pure retrieval accuracy: given a natural language query, does the system return results containing the ground-truth file/function?

### Query Set Design

Write 50 queries against Django, split into 4 categories that test different retrieval strengths:

**Category A: Named Symbol Lookup (12 queries)**
Grep's home turf. The query names the thing directly.
```json
{
  "id": "A01",
  "category": "named_symbol",
  "query": "ModelForm metaclass",
  "grep_pattern": "class ModelFormMetaclass|class ModelForm",
  "expected_files": ["django/forms/models.py"],
  "expected_functions": ["ModelFormMetaclass"],
  "difficulty": "easy"
}
```

**Category B: Behavioral / Semantic Lookup (15 queries)**
Where semantic search should beat grep. The query describes *what code does*, not what it's called.
```json
{
  "id": "B01",
  "category": "behavioral",
  "query": "where does Django validate CSRF tokens on incoming POST requests",
  "grep_pattern": "csrf|CsrfViewMiddleware",
  "expected_files": ["django/middleware/csrf.py"],
  "expected_functions": ["CsrfViewMiddleware.process_view"],
  "difficulty": "medium"
}
```

**Category C: Cross-File / Architectural (13 queries)**
Where `impact`/`calls` should shine. The query requires understanding relationships between files.
```json
{
  "id": "C01",
  "category": "cross_file",
  "query": "what functions call QuerySet.filter internally",
  "grep_pattern": "def filter|QuerySet",
  "expected_files": ["django/db/models/query.py", "django/db/models/manager.py"],
  "expected_functions": ["QuerySet.filter", "Manager.filter"],
  "difficulty": "hard"
}
```

**Category D: Negative Queries (10 queries)**
Sanity check -- queries about things that don't exist in Django.
```json
{
  "id": "D01",
  "category": "negative",
  "query": "React component lifecycle hooks",
  "grep_pattern": "componentDidMount|useEffect",
  "expected_files": [],
  "expect_none": true
}
```

### Dataset Expansion: Commit Subjects -> Touched Files (Deterministic, Scales Well)

Hand-written queries are high-signal but small and prone to accidental cherry-picking. Add a second dataset that is:
- large (hundreds to thousands of queries)
- deterministic (no LLM)
- grounded in "real developer intent"

Approach:
- Sample N non-merge commits from the pinned Django history.
- Query = commit subject line.
- Ground truth = files changed in that commit (optionally filtered to source files).

This benchmark answers a specific question: **localization quality at the file level** ("which files would I inspect/edit?").

Suggested JSON schema (no `grep_pattern` required):
```json
{
  "id": "H01234",
  "category": "commit_subject",
  "query": "Fix CSRF token rotation for streaming responses",
  "expected_files": ["django/middleware/csrf.py", "django/http/response.py"],
  "commit": { "sha": "abc1234", "date": "2024-01-02" }
}
```

### Metrics

| Metric | Formula | What It Tells Us |
|--------|---------|-----------------|
| **Recall@5** | (queries where ground-truth file appears in top 5) / total | Can the system find the right file? |
| **Recall@10** | Same, top 10 | Looser threshold for harder queries |
| **MRR** | Mean of 1/rank of first ground-truth hit | How high does the right answer rank? |
| **Precision@5** | (relevant results in top 5) / 5 | How noisy are the results? |
| **False Positive Rate** | (negative queries returning results) / negative total | Does it hallucinate matches? |
| **Latency p50/p95** | percentiles of per-query wall time | Does it feel interactive? Are there long tails? |

### Evaluation Procedure

```
1. Clone django/django at a pinned tag (e.g., 5.1)
2. tldrf warm <django_path>                    # Build all indexes
3. For each query:
   a. G0 (keyword ranking):
      - Extract keywords from <query> (drop stopwords, short tokens, and punctuation).
      - For each keyword, run `rg -l <kw> <django_path>` and accumulate per-file hit counts.
      - Rank files by score; take top 10.
   b. G1 (expert regex), if grep_pattern present:
      - Run: rg -l "<grep_pattern>" <django_path> | head -10
   c. Run: tldrf semantic search "<query>" --path <django_path> --k 10
      → collect file list, measure time
   d. Compare each result set against expected_files
4. Compute metrics per category and overall
5. Output: bench_results/phase1_retrieval.json
```

### Output Format

```json
{
  "meta": {
    "phase": 1,
    "repo": "django",
    "repo_version": "5.1",
    "repo_loc": 498000,
    "date": "2026-02-XX",
    "index_time_s": 120.5,
    "index_size_mb": 45.2
  },
  "results": {
    "overall": {
      "grep_keywords": { "recall_at_5": 0.78, "recall_at_10": 0.88, "mrr": 0.62, "precision_at_5": 0.28, "avg_query_time_s": 0.03 },
      "grep_regex":    { "recall_at_5": 0.85, "recall_at_10": 0.92, "mrr": 0.71, "precision_at_5": 0.32, "avg_query_time_s": 0.02 },
      "semantic":      { "recall_at_5": 0.82, "recall_at_10": 0.90, "mrr": 0.69, "precision_at_5": 0.45, "avg_query_time_s": 0.15 }
    },
    "by_category": {
      "named_symbol":  { "grep_keywords": { "recall_at_5": 0.88 }, "grep_regex": { "recall_at_5": 0.95 }, "semantic": { "recall_at_5": 0.85 } },
      "behavioral":    { "grep_keywords": { "recall_at_5": 0.55 }, "grep_regex": { "recall_at_5": 0.60 }, "semantic": { "recall_at_5": 0.87 } },
      "cross_file":    { "grep_keywords": { "recall_at_5": 0.40 }, "grep_regex": { "recall_at_5": 0.45 }, "semantic": { "recall_at_5": 0.70 } },
      "negative":      { "grep_keywords": { "fpr": 0.80 },         "grep_regex": { "fpr": 0.00 },         "semantic": { "fpr": 0.30 } }
    },
    "per_query": [ "..." ]
  }
}
```

### Decision Gate

| Outcome | Action |
|---------|--------|
| Semantic Recall@5 > max(G0,G1) on behavioral+cross_file queries by >10% | Proceed to Phase 2 -- TLDR finds code grep can't |
| Semantic Recall@5 within +-5% of max(G0,G1) across all categories | Proceed to Phase 2 anyway -- value may be in token efficiency, not retrieval |
| Semantic Recall@5 < max(G0,G1) by >10% across all categories | Investigate why. Fix indexing or query pipeline before proceeding |

---

## Phase 2: Does TLDR Use Fewer Tokens? (Token Efficiency)

**Timeline:** 1 day (reuses Phase 1 query set + infrastructure)
**Cost:** Zero LLM tokens (just tiktoken counting)
**Kill signal:** If TLDR returns more tokens than the grep excerpts baseline for the same information, the "95% savings" claim doesn't generalize to external repos.

### What We Measure

For queries where both systems correctly find the ground-truth code, how many tokens does each system need to deliver that information?

### Methodology

For each query in the Phase 1 set where both strategies found the right file:

**Grep strategies (model two realistic agent behaviors):**
1. **Grep excerpts (recommended baseline):**
   - Use the Phase 1 file list from G0 or G1 (top-k files).
   - Build a payload from `rg -n -C <context_lines>` matches in those files, capped by a token budget.
   - This matches how agents typically work: grep to narrow, then skim localized snippets before deciding what to open.
2. **Grep full-file escalation (upper bound):**
   - If modeling "agent opens entire files," read the full contents of the top-k files and count tokens.
   - This is useful as a worst-case comparison, but should not be the only baseline (it makes grep look artificially expensive).

**TLDR strategy:**
1. Run `tldrf context <function> --project <django_path> --depth 2`
2. Count total tokens of the structured output

**TLDR semantic strategy:**
1. Run `tldrf semantic search "<query>" --path <django_path> --k 5`
2. Count total tokens of the returned snippets

### Metrics

| Metric | Formula | What It Tells Us |
|--------|---------|-----------------|
| **Compression Ratio** | grep_tokens / tldr_tokens | How much more compact is TLDR? |
| **Tokens-to-Answer** | Tokens consumed before ground-truth code appears | How quickly do you get to the relevant part? |
| **Information Density** | relevant_tokens / total_tokens | What fraction of returned tokens is actually useful? |
| **Fixed-Budget Recall** | At 2000 tokens, what fraction of ground-truth is captured? | Token-constrained retrieval quality |

### The Key Test: Fixed-Budget Recall

This is the metric that settles the debate. Give both systems a budget of N tokens:

```
For N in [500, 1000, 2000, 5000, 10000]:
  grep_output = first N tokens of (rg excerpts payload)   # recommended
  grep_output_full = first N tokens of (top-k full files) # optional upper bound
  tldr_output = first N tokens of (tldrf context output)

  grep_covered = fraction of expected_functions found in grep_output
  tldr_covered = fraction of expected_functions found in tldr_output
```

If TLDR captures 80% of relevant code in 2000 tokens while grep captures 30%, that's the headline number.

### Output Format

```json
{
  "meta": { "phase": 2, "repo": "django", "..." },
  "results": {
    "compression_ratio": {
      "mean": 12.5,
      "median": 8.3,
      "p90": 45.2,
      "by_category": { "named_symbol": 5.1, "behavioral": 15.8, "cross_file": 22.4 }
    },
    "fixed_budget_recall": {
      "500":   { "grep": 0.10, "tldr_context": 0.55, "tldr_semantic": 0.35 },
      "1000":  { "grep": 0.20, "tldr_context": 0.75, "tldr_semantic": 0.50 },
      "2000":  { "grep": 0.35, "tldr_context": 0.90, "tldr_semantic": 0.70 },
      "5000":  { "grep": 0.55, "tldr_context": 0.95, "tldr_semantic": 0.85 },
      "10000": { "grep": 0.75, "tldr_context": 0.98, "tldr_semantic": 0.92 }
    },
    "per_query": [ "..." ]
  }
}
```

### Scenario Token Bench (Optional Sanity Check)

Query-level token benchmarks (above) are the primary evidence, because they compare against realistic grep payloads. Still, it’s useful to also run a small set of **scenario-style** measurements to:
- sanity-check the “headline” token numbers in docs
- build intuition about where TLDR saves tokens (and where it doesn’t)

This is adapted from the Continuous Claude benchmark approach, but with stricter baselines:
- “Raw files” is treated as an **upper bound** baseline.
- Always include a **grep excerpts** payload baseline when a scenario includes search/localization.

Suggested scenarios (run on a pinned repo + pinned file lists to avoid cherry-picking):
1. **Single file analysis**
   - Raw: `cat <file>`
   - TLDR: `tldrf extract <file>`
2. **Function context (depth=2)**
   - Raw (upper bound): concatenate the known file set containing the entry + its callees (predefined list)
   - TLDR: `tldrf context <entry> --project <repo> --depth 2`
3. **Codebase overview**
   - Raw (upper bound): concatenate all source files in the chosen sub-tree
   - TLDR: `tldrf structure <repo> --lang python`
4. **Deep call chain (depth=3)**
   - Raw (upper bound): concatenate the predefined file set for the chain
   - TLDR: `tldrf context <entry> --project <repo> --depth 3`

All scenarios should:
- count tokens via `tldr.stats.count_tokens()`
- emit per-scenario rows (`raw_tokens`, `grep_excerpt_tokens` if applicable, `tldr_tokens`, `savings_percent`)

### Decision Gate

| Outcome | Action |
|---------|--------|
| Compression ratio > 5x on cross_file queries AND fixed-budget recall advantage at 2000 tokens > 20% | Strong signal. Proceed to Phase 3 -- TLDR genuinely saves tokens |
| Compression ratio 2-5x, fixed-budget advantage 5-20% | Moderate signal. Proceed to Phase 3 to test if the savings translate to better agent performance |
| Compression ratio < 2x or fixed-budget advantage < 5% | The "95% savings" claim doesn't hold on external repos. Investigate whether the structured context format is the bottleneck |

---

## Phase 3: Does Better Context Make Agents Better? (Downstream Quality)

**Timeline:** 2-3 days
**Cost:** Moderate LLM tokens (~$20-50 for 50 queries x 2 strategies)
**Kill signal:** If grep-based context produces equal or better agent answers, TLDR's retrieval is not adding value.

### What We Measure

Given a coding question about Django, does an LLM produce a better answer when given TLDR context vs grep context?

### Task Design

Write 30 tasks against Django, split into three types:

**Type 1: Localization (10 tasks)**
"Find where X happens and explain how it works."
```json
{
  "id": "L01",
  "type": "localization",
  "task": "Find where Django validates that uploaded files don't exceed FILE_UPLOAD_MAX_MEMORY_SIZE and explain the validation flow",
  "ground_truth_files": ["django/core/files/uploadhandler.py"],
  "evaluation": "mentions MemoryFileUploadHandler, explains size check, identifies the handler chain"
}
```

**Type 2: Impact Analysis (10 tasks)**
"If I change X, what breaks?"
```json
{
  "id": "I01",
  "type": "impact",
  "task": "If I rename QuerySet._filter_or_exclude, what other code would need to change?",
  "ground_truth_callers": ["QuerySet.filter", "QuerySet.exclude", "..."],
  "evaluation": "identifies all direct callers and explains the downstream impact"
}
```

**Type 3: Debugging (10 tasks)**
"Why does X happen?"
```json
{
  "id": "D01",
  "type": "debugging",
  "task": "A user reports that prefetch_related doesn't work when chained after values(). Why?",
  "ground_truth_explanation": "values() returns a ValuesQuerySet which overrides _clone() and doesn't carry prefetch lookups",
  "evaluation": "identifies ValuesQuerySet behavior, explains the mechanism, points to the right code"
}
```

### Evaluation Protocol

For each task, run two pipelines:

**Pipeline A: Grep + Read (baseline agent workflow)**
1. Extract keywords from the task description
2. Run `rg <keywords> <django_path>` to find relevant files
3. Read matching files (up to 10K token budget)
4. Send to LLM: "Given this context, answer the task"
5. Record: answer, tokens consumed, wall time

**Pipeline B: TLDR-augmented**
1. Run `tldrf semantic search "<task>" --path <django_path>` to find entry points
2. For top results, run `tldrf context <function> --project <django_path> --depth 2`
3. For impact tasks, also run `tldrf impact <function> <django_path>`
4. Send to LLM: "Given this context, answer the task" (same prompt template)
5. Record: answer, tokens consumed, wall time

### Evaluation Metrics

**Automated metrics:**
| Metric | How |
|--------|-----|
| **Token cost** | Total input+output tokens per task |
| **Answer length** | Output token count |
| **File coverage** | Does the answer reference the ground-truth files? |
| **Function coverage** | Does the answer mention the ground-truth functions? |

**LLM-as-judge (for answer quality):**

Use a separate LLM call with the ground truth to score each answer on a 1-5 scale:

```
Given this coding task about Django:
  {task}

Ground truth:
  {ground_truth}

Answer A (grep-based context):
  {answer_a}

Answer B (TLDR-based context):
  {answer_b}

Score each answer 1-5 on:
1. Correctness: Does it identify the right code?
2. Completeness: Does it cover all relevant aspects?
3. Precision: Does it avoid irrelevant information?

Output JSON: { "a": { "correctness": N, "completeness": N, "precision": N }, "b": { ... } }
```

Run this 3 times per task and average to reduce noise.

### Output Format

```json
{
  "meta": { "phase": 3, "repo": "django", "model": "claude-sonnet-4-5-20250929", "..." },
  "results": {
    "overall": {
      "grep":  { "avg_score": 3.2, "avg_tokens": 8500, "file_coverage": 0.65 },
      "tldr":  { "avg_score": 4.1, "avg_tokens": 3200, "file_coverage": 0.85 }
    },
    "by_type": {
      "localization": { "grep": { "avg_score": 3.5 }, "tldr": { "avg_score": 3.8 } },
      "impact":       { "grep": { "avg_score": 2.5 }, "tldr": { "avg_score": 4.3 } },
      "debugging":    { "grep": { "avg_score": 3.1 }, "tldr": { "avg_score": 4.0 } }
    },
    "per_task": [ "..." ]
  }
}
```

### Decision Gate

| Outcome | Action |
|---------|--------|
| TLDR avg score > grep avg score by >0.5 AND tokens < 50% of grep | Clear win. Proceed to Phase 4 for the gold-standard validation |
| TLDR wins on impact+debugging but ties on localization | Expected. TLDR adds value for structural tasks. Publish results, Phase 4 optional |
| Scores within 0.3 of each other, tokens similar | TLDR isn't adding enough value over grep. Investigate which TLDR features matter and which don't |
| Grep outscores TLDR | Investigate why. Likely the structured context is losing information that raw code preserves. Consider hybrid strategies |

---

## Phase 4: SWE-bench Validation (Gold Standard)

**Timeline:** 1 week
**Cost:** $100-300 in LLM API tokens
**Prerequisite:** Phase 3 shows clear signal

### What We Measure

Can TLDR-augmented retrieval improve an agent's resolve rate on real GitHub issues?

### Methodology

1. **Select 50 SWE-bench Lite instances** from Django (since we already have it indexed). Filter for:
   - Issues where the patch touches <= 3 files (so localization is the bottleneck, not the edit itself)
   - Issues with clear reproduction steps
   - Issues across difficulty levels (easy/medium/hard based on historical resolve rates)

2. **Build a minimal Agentless-style pipeline:**
   ```
   Input: Issue description
   Step 1: Localization (find files/functions to edit)
   Step 2: Patch generation (given localized context, generate the fix)
   Step 3: Test (apply patch, run tests)
   Output: pass/fail
   ```

3. **Run Step 1 with two backends:**
   - **A:** grep + file listing + file reading (the standard approach)
   - **B:** `tldrf semantic` + `tldrf context` + `tldrf impact`

   Steps 2-3 are identical for both -- same model, same prompt, same test harness.

4. **Measure:**
   | Metric | What |
   |--------|------|
   | **Resolve rate** | % of issues producing a test-passing patch |
   | **Localization accuracy** | % of issues where Step 1 found the right file(s) |
   | **Tokens consumed** | Total tokens for the full pipeline |
   | **Time to solution** | Wall clock time |

### Decision Gate

| Outcome | Action |
|---------|--------|
| TLDR resolve rate > grep by 5%+ (e.g., 42% vs 36%) | Publishable result. TLDR demonstrably improves agent capability |
| TLDR localization accuracy higher but resolve rate similar | TLDR helps find code but patch generation is the bottleneck -- still valuable for different tasks |
| No difference | TLDR's value is in developer experience (speed, token savings) not in agent capability. Reframe the product positioning |

---

## Implementation Notes

### File Organization

```
scripts/
  bench_perf_daemon_vs_cli.py # Phase 0 runner (warmup + iterations, daemon socket vs CLI spawn)
  bench_gen_commit_queries.py # Generate commit-subject -> touched-files query set
  bench_retrieval.py          # Phase 1 runner
  bench_token_efficiency.py   # Phase 2 runner
  bench_token_scenarios.py    # Phase 2 optional scenario-style token bench (sanity/comms)
  bench_downstream.py         # Phase 3 runner
  bench_swebench.py           # Phase 4 runner
  bench_django_queries.json   # Shared query set (phases 1-2)
  bench_django_commit_queries.json # Auto-generated query set (Phase 1)
  bench_django_tasks.json     # Task set (phase 3)
bench_results/
  phase0_perf.json            # Daemon vs CLI perf microbench
  phase1_retrieval.json       # Raw results
  phase2_tokens.json
  phase3_downstream.json
  phase4_swebench.json
  summary.md                  # Auto-generated comparison table
```

### Running

```bash
# Phase 1 (no LLM cost)
git clone --depth 1 --branch 5.1 https://github.com/django/django.git /tmp/django

# Phase 0 (no LLM cost): perf microbench (daemon vs CLI spawn)
uv run python scripts/bench_perf_daemon_vs_cli.py --repo /tmp/django --iterations 10

# Optional: generate a large, deterministic commit-subject dataset
uv run python scripts/bench_gen_commit_queries.py --repo /tmp/django --out scripts/bench_django_commit_queries.json --n 1000

uv run python scripts/bench_retrieval.py --repo /tmp/django --queries scripts/bench_django_queries.json
uv run python scripts/bench_retrieval.py --repo /tmp/django --queries scripts/bench_django_commit_queries.json

# Phase 2 (no LLM cost, reuses Phase 1 data)
uv run python scripts/bench_token_efficiency.py --repo /tmp/django --phase1-results bench_results/phase1_retrieval.json

# Phase 2 optional: scenario-style token sanity check
uv run python scripts/bench_token_scenarios.py --repo /tmp/django

# Phase 3 (moderate LLM cost)
export ANTHROPIC_API_KEY=...
uv run python scripts/bench_downstream.py --repo /tmp/django --tasks scripts/bench_django_tasks.json --model claude-sonnet-4-5-20250929

# Phase 4 (high LLM cost)
uv run python scripts/bench_swebench.py --repo /tmp/django --instances 50 --model claude-sonnet-4-5-20250929
```

### Snapshot & Reproducibility

Each run saves:
- Git hash of TLDR at time of run
- Git hash (or tag) of target repo
- Full query/task set used
- Raw results JSON with per-query breakdowns
- Hardware info (for timing comparisons)
- Timing protocol (warmup count, iterations, cold vs warm mode, command list)
- Index configuration (legacy vs index mode, scan_root, cache_root/index_id if used, ignore settings)
- Semantic config (model id + dimension, device)
- Model version (for Phase 3-4)

Results are committed to `bench_results/` so future changes can compare against the baseline.

### Query Set Construction Principles

1. **No cherry-picking.** Write all queries BEFORE running any benchmarks. Don't add or remove queries after seeing results.
2. **Category balance.** Roughly equal queries per category so no single type dominates the aggregate.
3. **Multiple ground-truth answers.** Where possible, list 2-3 acceptable files/functions so partial matches are captured.
4. **Grep baselines must be reasonable.**
   - For **G0 (keywords)**: drop stopwords/short tokens and cap the number of keywords so it's not an unbounded advantage.
   - For **G1 (regex)**: use patterns a competent developer would actually try, not strawman single-word searches. If testing "CSRF validation", the grep pattern should be `csrf|CsrfViewMiddleware|_check_token`, not just `csrf`.
5. **Difficulty labels.** Mark each query easy/medium/hard based on how many files need to be found and how indirect the naming is.

---

## What This Answers

After Phase 2 (2-3 days of work, zero LLM cost), you'll know:

- Whether TLDR's semantic search finds code that grep can't (or vice versa)
- Whether the "95% token savings" claim holds on a repo TLDR has never seen
- The exact compression ratio on a well-known codebase
- Whether the indexing time is justified by the retrieval quality

After Phase 3 (~1 week, ~$50 LLM cost), you'll know:

- Whether TLDR context produces better LLM answers than grep context
- Which task types (localization, impact, debugging) benefit most
- Whether the token savings translate to cost savings without quality loss

This is enough data to either (a) confidently invest more time in TLDR, or (b) pivot to focusing on the features where TLDR has a clear edge and stop claiming general superiority over grep.
