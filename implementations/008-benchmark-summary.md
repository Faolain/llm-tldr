# 008 Benchmark Summary and Repro Runbook

## Purpose
This document is the operator handoff for the 008 comparison-first benchmark program.

It explains:
- what each benchmark answers,
- how to run and compare tools reproducibly,
- where to find canonical outputs and decisions,
- how to add additional competitors beyond `contextplus` and `rg-native`.

## Benchmark Families and What They Mean

| Family | Primary scripts | Answers |
| --- | --- | --- |
| Head-to-head (`h2h`) | `bench_h2h_predict.py`, `bench_head_to_head.py`, `bench_h2h_assert.py`, `bench_h2h_stitch.py`, `bench_h2h_export_matrix_run1.py` | Is tool A better than tool B on identical tasks, budgets, and scoring? |
| Retrieval deterministic | `bench_retrieval_quality.py` | How good are retrieval strategies (`rg`, `semantic`, `hybrid`, lane controls) without LLM-judge variance? |
| Lane workflow deterministic | `bench_compound_semantic_impact.py`, `bench_navigate_cluster.py` | Do workflow features (lane4/lane5) improve utility while keeping deterministic behavior? |
| End-to-end judge (`llm_ab`) | `bench_llm_ab_prompts.py`, `bench_llm_ab_run.py` | Does improved context actually produce better final model answers? |
| Token efficiency | `bench_token_efficiency.py` | What quality/coverage gains are achieved per token budget increase? |

## Current Canonical Decision Sources
- Implementation authority: `implementations/008-beat-contextplus_IMPLEMENTATION_PLAN.md`
- Lane decision log: `implementations/008-canonical-matrix-lane-decisions.md`
- Canonical matrix snapshot/pivot:
  - `implementations/008-canonical-matrix-run1-snapshot.md`
  - `implementations/008-canonical-matrix-run1-pivot-by-budget.md`

## Current Program Snapshot (Run1 Comparison Track)

### Gate A (shared-capability retrieval @ budget 2000)
- lane1 (`llm-tldr`) vs `contextplus`: `llm-tldr` wins `5-0`
- lane2 (`llm-tldr`) vs `contextplus`: `llm-tldr` wins `5-0`
- lane3 (`llm-tldr`) vs `contextplus`: `llm-tldr` wins `5-0`
- lane4 (`llm-tldr`, bounded subset) vs `contextplus`: `llm-tldr` wins `5-0`
- lane5 (`llm-tldr`) vs `contextplus`: `llm-tldr` wins `5-0`
- `rg-native` vs `contextplus`: `rg-native` wins `5-0`
- `rg-native` vs `llm-tldr` baseline: `rg-native` wins `5-0`

### Lane5 deterministic artifact (most recent lane)
Artifact:
- `benchmark/runs/20260303-001504Z-navigate-cluster-django-lane5-b2000.json`

Summary values:
- `n=180`
- `cluster_coverage_rate_mean=1.0`
- `determinism_assignment_digest_match_rate=1.0`
- `query_cluster_recall@1_mean=0.7193`
- `query_cluster_recall@3_mean=0.9825`
- `latency_ms_p50=302.423`
- `payload_tokens_median=29.0`
- `error_rate=0.0`
- `partial_rate=0.0`

### Stability caveat (important)
Single-run segment asserts can pass strict run-level gates while still failing overall due `stability.two_of_three` if only one run was executed.

## 009 Model-Variant Snapshot (BGE Default vs Jina Opt-In)

This addendum keeps the same-tool model decision in the 008 operator handoff rather than leaving it only in the migration plan. Source of truth for the run log and report inventory: `implementations/009-migrate-bge-to-jina-code-0.5b_IMPLEMENTATION_PLAN.md`.

| Scenario | BGE | Jina | Comparator context | Operational takeaway |
| --- | --- | --- | --- | --- |
| Common lane `hybrid_rrf` retrieval | `mrr=0.8684`, `r@5=0.9649`, `r@10=1.0000`, `p@5=0.1930` | `mrr=0.8686`, `r@5=0.9825`, `r@10=1.0000`, `p@5=0.1965` | 008 snapshot: `rg=0.820`, `contextplus=0.216` MRR | Near tie. Jina has only a marginal lane1 edge. |
| Common lane `hybrid_lane2` retrieval | `mrr=0.8741`, `r@5=0.8772`, `r@10=0.9649`, `p@5=0.1754` | `mrr=0.8417`, `r@5=0.9123`, `r@10=1.0000`, `p@5=0.1825` | Gate-like lane row | Keep BGE default. Jina loses the MRR-heavy gate row. |
| Pure semantic concept retrieval | `mrr=0.6022`, `r@5=0.7719`, `r@10=0.7895`, `p@5=0.1544` | `mrr=0.7023`, `r@5=0.8596`, `r@10=0.8772`, `p@5=0.1719` | Concept lookup only | Jina is the better semantic-only model on current evidence. |
| Token-efficiency retrieval at `1000` | `semantic=0.6124`, `hybrid_rrf=0.8597` | `semantic=0.7048`, `hybrid_rrf=0.8647` | Budget-constrained retrieval | Jina helps semantic quality under tighter token budgets; hybrid stays nearly tied. |
| Structured exact-definition retrieval | `f1=0.1602`, `p50=142.1ms` | `f1=0.1520`, `p50=140.3ms` | `rg-native=0.9841`, `p50=84.8ms` | Keep exact symbol lookup lexical. Neither embedding model should replace `rg-native` here. |
| Structured behavior retrieval (semantic / hybrid) | `semantic f1=0.1458`, `hybrid f1=0.1584` | `semantic f1=0.1053`, `hybrid f1=0.1188` | `rg-native=0.0217` on the same suite | Semantic and hybrid help on concept-style targets, but BGE beats Jina in both modes. |
| Compound semantic+impact | `tte_p50_ratio=1.0250`, overlap/Jaccard parity `=1.0` | `tte_p50_ratio=1.1361`, overlap/Jaccard parity `=1.0` | No direct `contextplus` or `rg-native` equivalent | Jina does not regress correctness here, but it is less efficient than BGE on the product path. |
| Steady-state daemon semantic latency | `p50=150.2ms`, `p95=250.2ms` | `p50=166.5ms`, `p95=286.4ms` | Query latency only | BGE is still about `10-14%` faster. |
| Semantic build / peak RSS | parity rebuild still pending | `build_s=692.57`, `peak_rss=3360.1MB` | One-time index build | Jina is operationally heavier. |

Decision summary:
- Keep `bge-large-en-v1.5` as the default model.
- Keep `jina-code-0.5b` opt-in for pure semantic concept retrieval and budget-constrained semantic retrieval experiments.
- Keep `rg-native` as the first tool for exact identifier and definition lookup.

## Reproducible Operator Runbook

### 1) Fetch pinned corpus
```bash
uv run python scripts/bench_fetch_corpora.py --corpus django
```

### 2) Preflight (recommended before any h2h predict)
```bash
uv run tldrf semantic index --cache-root benchmark/cache-root --index repo:django --lang python --rebuild benchmark/corpora/django
uv run tldrf warm --cache-root benchmark/cache-root --index repo:django --lang python --rebuild benchmark/corpora/django
uv run tldrf semantic search "Where is CsrfViewMiddleware implemented?" --cache-root benchmark/cache-root --index repo:django --path benchmark/corpora/django --k 5 --lang python
uv run tldrf impact "CsrfViewMiddleware.process_view" benchmark/corpora/django --cache-root benchmark/cache-root --index repo:django --file django/middleware/csrf.py --lang python
```

### 3) Validate suite and materialize tasks
```bash
uv run python scripts/bench_head_to_head.py validate-suite \
  --suite benchmark/runs/h2h-suite-segment-retrieval-b2000.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/contextplus.v1.json

uv run python scripts/bench_head_to_head.py materialize-tasks \
  --suite benchmark/runs/h2h-suite-segment-retrieval-b2000.v1.json \
  --corpus-root benchmark/corpora/django \
  --out benchmark/runs/h2h-task-manifest-segment-retrieval.json
```

### 4) Predict for any tool profile (example: lane5)
```bash
uv run python scripts/bench_h2h_predict.py \
  --suite benchmark/runs/h2h-suite-segment-retrieval-b2000.v1.json \
  --tasks benchmark/runs/h2h-task-manifest-segment-retrieval.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.navigate_cluster_lane5.v1.json \
  --category retrieval \
  --out benchmark/runs/h2h-llm-tldr-predictions-run1-navigate-cluster-lane5-retrieval-b2000-t123-segment.json \
  --classification-out benchmark/runs/h2h-failure-classification-run1-llm-tldr-navigate-cluster-lane5-retrieval-b2000-t123.json \
  --run-metadata-out benchmark/runs/h2h-run-metadata-run1-llm-tldr-navigate-cluster-lane5-retrieval-b2000-t123.json
```

### 4a) Using daemon mode (recommended for llm-tldr profiles)

Adding `--use-daemon` routes queries through the llm-tldr daemon's Unix socket instead of spawning a subprocess per query. This eliminates ~500-1000ms startup + model-load overhead on each prediction and achieves a **19.5x p50 latency speedup** (5776ms -> 297ms on lane1 retrieval).

The daemon auto-detects MPS GPU when available. Results are byte-identical to subprocess mode.

```bash
# Daemon mode (llm-tldr profiles only, tool_id must be 'llm-tldr'):
uv run python scripts/bench_h2h_predict.py \
  --suite benchmark/runs/h2h-suite-segment-retrieval-b2000.v1.json \
  --tasks benchmark/runs/h2h-task-manifest-segment-retrieval.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.navigate_cluster_lane5.v1.json \
  --use-daemon \
  --category retrieval \
  --out benchmark/runs/h2h-llm-tldr-predictions-daemon.json \
  --run-metadata-out benchmark/runs/h2h-run-metadata-daemon.json

# Add --daemon-keep-alive to leave the daemon running between invocations
# (useful when running multiple profiles back-to-back).
```

Notes:
- `--use-daemon` is only supported for `tool_id='llm-tldr'` profiles. Non-daemon templates (contextplus, rg-native) automatically fall back to subprocess.
- The daemon is started automatically before the prediction loop and stopped after (unless `--daemon-keep-alive` is set).
- Run metadata records `"execution_mode": "daemon"` for auditability.

### 5) Score, compare, assert
```bash
uv run python scripts/bench_head_to_head.py score \
  --suite benchmark/runs/h2h-suite-segment-retrieval-b2000.v1.json \
  --tasks benchmark/runs/h2h-task-manifest-segment-retrieval.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.navigate_cluster_lane5.v1.json \
  --predictions benchmark/runs/h2h-llm-tldr-predictions-run1-navigate-cluster-lane5-retrieval-b2000-t123-segment.json \
  --out benchmark/runs/h2h-llm-tldr-score-run1-navigate-cluster-lane5-retrieval-b2000-t123-segment.json

uv run python scripts/bench_head_to_head.py compare \
  --suite benchmark/runs/h2h-suite-segment-retrieval-b2000.v1.json \
  --score-a benchmark/runs/h2h-llm-tldr-score-run1-navigate-cluster-lane5-retrieval-b2000-t123-segment.json \
  --score-b benchmark/runs/h2h-contextplus-score-run1-segment-retrieval-b2000.json \
  --label-a llm-tldr \
  --label-b contextplus \
  --out benchmark/runs/h2h-compare-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.json

uv run python scripts/bench_h2h_assert.py \
  --suite benchmark/runs/h2h-suite-segment-retrieval-b2000.v1.json \
  --score-a benchmark/runs/h2h-llm-tldr-score-run1-navigate-cluster-lane5-retrieval-b2000-t123-segment.json \
  --score-b benchmark/runs/h2h-contextplus-score-run1-segment-retrieval-b2000.json \
  --compare benchmark/runs/h2h-compare-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.json \
  --label-a llm-tldr \
  --label-b contextplus \
  --strict-gates benchmarks/head_to_head/gates.strict.v1.json \
  --out benchmark/runs/h2h-assert-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.json
```

### 6) Export canonical matrix rows
```bash
uv run python scripts/bench_h2h_export_matrix_run1.py \
  --label-a llm-tldr \
  --label-b contextplus \
  --score-a benchmark/runs/h2h-llm-tldr-score-run1-navigate-cluster-lane5-retrieval-b2000-t123-segment.json \
  --score-b benchmark/runs/h2h-contextplus-score-run1-segment-retrieval-b2000.json \
  --compare benchmark/runs/h2h-compare-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.json \
  --assert-report benchmark/runs/h2h-assert-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.json \
  --meta-a benchmark/runs/h2h-run-metadata-run1-llm-tldr-navigate-cluster-lane5-retrieval-b2000-t123.json \
  --profile-a benchmarks/head_to_head/tool_profiles/llm_tldr.navigate_cluster_lane5.v1.json \
  --profile-b benchmarks/head_to_head/tool_profiles/contextplus.v1.json \
  --run-id-a run1-navigate-cluster-lane5-retrieval-b2000-t123-segment \
  --run-id-b run1-segment-retrieval-b2000 \
  --feature-set-a feature.navigate-cluster.v1 \
  --feature-set-b baseline.run1 \
  --embedding-backend-a sentence-transformers \
  --embedding-model-a profile_unpinned \
  --embedding-backend-b unknown \
  --embedding-model-b unknown \
  --budgets 2000 \
  --out-json benchmark/runs/matrix/h2h-matrix-long-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.json \
  --out-csv benchmark/runs/matrix/h2h-matrix-long-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.csv \
  --no-markdown
```

## Running Competitors (including rg-native)

### rg-native baseline
Use profile:
- `benchmarks/head_to_head/tool_profiles/rg_native.v1.json`

Then run the same pipeline:
- validate-suite -> predict -> score -> compare -> assert -> export matrix.

### Add a new competitor
1. Add a tool profile under `benchmarks/head_to_head/tool_profiles/<tool>.v1.json`.
2. Declare true/false capabilities explicitly.
3. Provide deterministic command templates per supported category.
4. Add version capture command.
5. Use the same suite, task manifest, budgets, and seeds used by other tools.
6. Run the same pipeline and compare on identical artifacts.

## File-Level Guide

| File | Responsibility |
| --- | --- |
| `tldr/semantic.py` | retrieval + lane behavior core (`hybrid`, `abstain/rerank`, `budget-aware`, `compound`, `navigate-cluster`) |
| `tldr/cli.py` | CLI exposure and routing for lane controls |
| `tldr/daemon/core.py` | daemon command plumbing for semantic/lane controls |
| `tldr/mcp_server.py` | MCP semantic tool forwarding for lane controls |
| `scripts/bench_h2h_predict.py` | canonical row generation (`--use-daemon` for 19.5x latency speedup) |
| `scripts/bench_h2h_stitch.py` | deterministic stitcher + audit |
| `scripts/bench_h2h_assert.py` | strict gates and stability checks |
| `scripts/bench_h2h_export_matrix_run1.py` | long-format matrix export |
| `scripts/bench_retrieval_quality.py` | deterministic retrieval quality metrics |
| `scripts/bench_compound_semantic_impact.py` | lane4 deterministic benchmark |
| `scripts/bench_navigate_cluster.py` | lane5 deterministic benchmark |
| `implementations/008-beat-contextplus_IMPLEMENTATION_PLAN.md` | authoritative implementation board |
| `implementations/008-canonical-matrix-lane-decisions.md` | keep/rollback log and quantitative lane comparisons |

## Remaining Work (post lanes 1-5)
- ~~Consolidated Gate B structural sweep across lanes 1-5~~ — structural commands (impact/slice/cfg/dfg) use identical templates across all lane profiles, so per-lane structural quality is invariant. Verified with daemon mode.
- ~~Daemon `_handle_impact` canonical `"function"` key migration~~ — done: normalizer + parser tolerance + 24 regression tests.
- ~~Daemon/index operational artifact~~ — resolved: all lanes 1-5 rerun with `--use-daemon`, 16.9-18.3x retrieval speedup, 15-93x structural speedup, MPS GPU confirmed.
- Resolve pending full-product rows (`impact -> context -> rg` context metric, contextplus concept path parity).
- Lane6 optional Ollama backend loop (kept non-gating for current comparison cycle).
