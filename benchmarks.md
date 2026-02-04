# Benchmarks

## Summary

These benchmarks validate that the isolated index feature preserves search quality while enabling dependency-scoped search and clean cache management.

Results date: 2026-02-03

## How To Run

```bash
uv run python scripts/bench_dep_indexes.py
```

Common flags:
```bash
uv run python scripts/bench_dep_indexes.py --dep requests --model all-MiniLM-L6-v2 --device cpu --k 5 --queries scripts/bench_queries.json
```

## Environment

- Repo: `llm-tldr`
- Dependency corpus: `requests` from `.venv`
  - Source path: `.venv/lib/python3.12/site-packages/requests`
- Main repo scan root: `tldr/`
- Model: `all-MiniLM-L6-v2`
- Device: `cpu`
- `--no-ignore` used for all runs

## Benchmark Results

### Benchmark 1: Parity (same corpus)

Goal: show that legacy indexing and index-mode indexing produce the same search quality for the same corpus.

Legacy index (writes `.tldr` inside the corpus):
- Indexed units: 276
- Index time: 8.27s
- Disk usage: ~608 KB
- Recall@5: 1.0
- MRR: 1.0
- Avg time-to-first-relevant-hit: 3.94s

Index mode (`--cache-root` + `--index`):
- Indexed units: 276
- Index time: 8.43s
- Disk usage: ~611 KB
- Recall@5: 1.0
- MRR: 1.0
- Avg time-to-first-relevant-hit: 3.93s

Conclusion: search quality is equivalent. The change is storage/management, not ranking.

### Benchmark 2: Workflow (dependency debugging)

Goal: show why per-dependency indexes matter.

Baseline (search only main repo index):
- Recall@5: 0.0
- MRR: 0.0
- No relevant hits for dependency queries

Dependency index (search the dependency directly):
- Recall@5: 1.0
- MRR: 1.0
- Avg time-to-first-relevant-hit: 3.93s

Conclusion: dependency queries are missed if you only index the main repo. Per-dependency indexes fix that.

### Benchmark 3: Query Scope Precision (pollution check)

Goal: quantify whether dependency queries return results **inside** the dependency corpus vs “polluted” hits from the main repo index.

Metric:
- `scope_hit_rate`: fraction of queries where the top hit is inside the dependency root.
- `off_scope_rate`: fraction of queries where the top hit is **outside** the dependency root.

This is reported in `scripts/bench_dep_indexes.py` under `scope_precision` for:
- `dependency_index` (expected high scope_hit_rate)
- `main_repo_index` (expected low scope_hit_rate)

Results:
- Dependency index: `scope_hit_rate = 1.0`, `off_scope_rate = 0.0`
- Main repo index: `scope_hit_rate = 0.0`, `off_scope_rate = 1.0`

### Benchmark 4: Storage + Time

Index sizes (from `tldr index list --cache-root ...`):
- `main:llm-tldr`: ~2.53 MB
- `dep:requests`: ~0.61 MB
- Total cache root size: ~3.14 MB

Conclusion: per-dependency indexes are small and cheap to keep around.

## Why The New Feature Enables This

Isolated indexes were added via two CLI flags:
- `--cache-root`: base directory where all index data is stored.
- `--index`: logical index id that namespaces caches under the cache root.

This allows multiple independent indexes under a single cache root (one per repo or dependency) without writing `.tldr` inside each dependency folder. As a result:
- You can index multiple dependencies separately.
- You can query each dependency directly without “pollution” from other corpora.
- You can list, inspect, and remove indexes cleanly using `tldr index list/info/rm`.

## Where The Cache Is Stored

Legacy mode (no `--cache-root`):
- Cache lives inside the corpus: `<scan_root>/.tldr/cache/...`

Index mode (with `--cache-root` + `--index`):
- Cache lives under the cache root, namespaced by a stable hash of the index id:
  - `<cache_root>/.tldr/indexes/<index_key>/cache/...`
  - `<cache_root>/.tldr/indexes/<index_key>/meta.json`
  - `<cache_root>/.tldr/indexes/<index_key>/.tldrignore`

`<index_key>` is derived from the `--index` value, and `meta.json` stores the index id and scan root so `tldr index list/info` can display it.

## Notes

Semantic search returns file paths relative to the scan root. When indexing a package directory directly, results may look like `sessions.py` rather than `requests/sessions.py`. The benchmark evaluator accepts either a basename match or a substring match.
