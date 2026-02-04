# Implementation Plan: 002 - Benchmark Isolated Indexes

Spec: N/A (benchmarks and validation plan)

## Summary

Short answer: yes, but benchmark the right dimension. The new implementation doesn’t change the search model; it changes index isolation, cache location, and management. So you need both a parity benchmark (same corpus, same results) and a workflow benchmark (dependency discovery and isolation).

## What Changed (Scope of Comparison)

- It enables multiple isolated indexes under one cache root.
- It avoids writing `.tldr` inside dependency folders like `site-packages` or `node_modules`.
- It makes dependency indexing manageable and repeatable via `tldr index list/info/rm`.

Apples-to-apples means: compare the same corpus indexed both ways.

## Benchmarks

### Benchmark 1: Parity (same corpus, same answers)

Goal: prove index mode does not change search quality for a given corpus.

1. Pick a dependency repo (or sdist extraction) as the corpus.
2. Build an index using old behavior (inside repo, no `--cache-root`).
3. Build an index using new index mode (`--cache-root` + `--index`).
4. Run the same query list and compare results (Recall@k, MRR, top hit path match).

If results match, the new path is functionally equivalent, and differences are only in storage/management.

### Benchmark 2: Workflow Benefit (dependency debugging)

Goal: show new capability that the old workflow can’t do cleanly.

1. Create a query set that targets dependency behavior.
2. Baseline: search only the main repo index.
3. New workflow: search the dependency index directly.
4. Measure recall@k, time-to-first-relevant-hit, and number of commands.

This shows why per-dependency indexes actually improve debugging tasks.

### Benchmark 3: Query Scope Precision (pollution)

Goal: measure how often dependency queries return top hits **inside** the dependency corpus vs “polluted” hits from the main repo index.

Metric:
- `scope_hit_rate`: top hit is within the dependency root.
- `off_scope_rate`: top hit is outside the dependency root.

This is reported by `scripts/bench_dep_indexes.py` under `scope_precision`.

### Benchmark 4: Storage + Time

Goal: quantify overhead.

- Index time (per corpus).
- Disk usage of index artifacts.
- Total number of indexes and their sizes (`tldr index list --cache-root ...`).

## Practical Query Set (objective metrics)

Create a small JSON list like:

```json
[
  {
    "dep": "requests",
    "query": "Session.request implementation",
    "expect_path_contains": "requests/sessions.py"
  },
  {
    "dep": "pydantic",
    "query": "BaseModel.model_validate",
    "expect_path_contains": "pydantic/main.py"
  }
]
```

Run the same list across both setups and score “hit in top-k”.

## What This Answers

- The “standard way” (index the main repo only) will miss dependencies.
- The new way doesn’t improve the model, but it enables dependency-scoped indexes so you can find answers in the right codebase without pollution.

## Optional Automation

If needed, add a small `scripts/bench_dep_indexes.py` to automate parity + workflow benchmarks and output Recall@k, MRR, time-to-first-hit, and timings.

## Implementation

Added:
- `scripts/bench_dep_indexes.py` (runs parity + workflow + storage/time benchmarks)
- `scripts/bench_queries.json` (query set)

Run:
```bash
uv run python scripts/bench_dep_indexes.py --dep requests,urllib3
```

Optional flags:
- `--dep <import-name[,import-name]>` (default: `requests`)
- `--model all-MiniLM-L6-v2`
- `--device cpu`
- `--k 5`
- `--queries scripts/bench_queries.json`

## Results (2026-02-04)

Environment:
- Repo: `llm-tldr`
- Dependency corpora:
  - `requests` from `.venv` (7 positive queries, 2 negative queries)
  - `urllib3` from `.venv` (4 positive queries, 2 negative queries)
- Main repo scan root: `tldr/`
- Model: `all-MiniLM-L6-v2`
- Device: `cpu`
- `--no-ignore` for all runs

### Benchmark 1: Parity (same corpus)

Legacy index (writes `.tldr` in corpus):
- Indexed units: 276
- Index time: 8.12s
- Disk usage: ~608 KB
- Recall@5: 1.0
- MRR: 0.93
- Avg time-to-first-relevant-hit: 4.38s

Index mode (`--cache-root` + `--index`):
- Indexed units: 276
- Index time: 8.03s
- Disk usage: ~611 KB
- Recall@5: 1.0
- MRR: 0.93
- Avg time-to-first-relevant-hit: 4.23s

### Benchmark 2: Workflow (dependency debugging)

Baseline (search only main repo index):
- Recall@5: 0.0
- MRR: 0.0
- No relevant hits for dependency queries

Dependency index (search the dependency directly):
- Recall@5: 1.0
- MRR: 0.93
- Avg time-to-first-relevant-hit: 4.23s

### Benchmark 3: Query scope precision (pollution)

Metric: `scope_precision` in the benchmark output.

Results:
- Dependency index: `scope_hit_rate = 1.0`, `off_scope_rate = 0.0`, `topk_in_scope_rate = 1.0`, `any_in_scope_rate = 1.0`
- Main repo index: `scope_hit_rate = 0.0`, `off_scope_rate = 1.0`, `topk_in_scope_rate = 0.0`, `any_in_scope_rate = 0.0`
- Negative queries: `negative_any_in_scope_rate = 1.0` for the dependency index (semantic search returns in-scope files even for “absent” queries)

Cross-dependency check:
- Requests queries run against the `urllib3` index yield Recall@5 = 0.0 (expected).

### Benchmark 4: Storage + time

Index sizes (from `tldr index list --cache-root ...`):
- `main:llm-tldr`: ~2.53 MB
- `dep:requests`: ~0.61 MB
- `dep:urllib3`: ~1.26 MB
- Total cache root size: ~4.40 MB

## Gotchas / Learnings

- Semantic search stores file paths relative to the scan root. When indexing just the package directory, results look like `sessions.py` instead of `requests/sessions.py`. The benchmark matcher now accepts either a full path substring or matching basename.
