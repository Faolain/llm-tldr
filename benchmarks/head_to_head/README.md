# Head-to-Head Benchmark (llm-tldr vs contextplus)

This suite is a neutral, capability-aware benchmark contract for comparing `llm-tldr` and `contextplus` on the same pinned corpus and query sets.

It explicitly separates:
- common lane (`retrieval`): required for all tools
- optional lane (`impact`, `slice`, `complexity`, `data_flow`): scored only when a tool declares native support

## Winner Views

1. Shared-capability winner (this suite):
   - Compares only lanes that both tools support under this contract.
   - Use this for strict apples-to-apples quality gating.
2. Full-product workflow winner (canonical matrix board):
   - Tracks real workflows across all lanes/capabilities.
   - `unsupported`/`N/A` rows count as explicit losses.
   - Current board: `implementations/008-canonical-matrix-run1-snapshot.md` and `implementations/008-canonical-matrix-run1-pivot-by-budget.md`.

## Inputs

- Suite contract: `benchmarks/head_to_head/suite.v1.json`
- Tool profiles:
  - `benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json`
  - `benchmarks/head_to_head/tool_profiles/llm_tldr.compound_semantic_impact_lane4.v1.json`
  - `benchmarks/head_to_head/tool_profiles/llm_tldr.navigate_cluster_lane5.v1.json`
  - `benchmarks/head_to_head/tool_profiles/contextplus.v1.json`
  - `benchmarks/head_to_head/tool_profiles/rg_native.v1.json` (retrieval-only lexical baseline)
- Query datasets:
  - `benchmarks/retrieval/django_queries.json`
  - `benchmarks/python/django_structural_queries.json`

## Reproducible Workflow

1. Fetch pinned corpus:

```bash
uv run python scripts/bench_fetch_corpora.py --corpus django
```

2. Validate suite + tool profiles:

```bash
uv run python scripts/bench_head_to_head.py validate-suite \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/contextplus.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/rg_native.v1.json
```

3. Materialize canonical task manifest (frozen task IDs + ground truth + hashes):

```bash
uv run python scripts/bench_head_to_head.py materialize-tasks \
  --suite benchmarks/head_to_head/suite.v1.json \
  --corpus-root benchmark/corpora/django \
  --out benchmark/runs/h2h-task-manifest.json
```

4. Collect predictions for each tool (adapter-owned step):
- required output format: `benchmarks/head_to_head/examples/predictions.v1.example.json`
- required file per tool, e.g.:
  - `benchmark/runs/h2h-llm-tldr-predictions.json`
  - `benchmark/runs/h2h-contextplus-predictions.json`

### Optional: Native `rg`/`grep` Retrieval Baseline (No `tldrf`)

Use the `rg_native` profile to benchmark a pure lexical retrieval baseline. This profile executes only `scripts/rg_search_adapter.py`, which shells out to `rg` (or `grep` fallback) and never calls `tldrf`.

```bash
uv run python scripts/bench_h2h_predict.py \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/rg_native.v1.json \
  --category retrieval \
  --out benchmark/runs/h2h-rg-native-predictions.json

uv run python scripts/bench_head_to_head.py score \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/rg_native.v1.json \
  --predictions benchmark/runs/h2h-rg-native-predictions.json \
  --out benchmark/runs/h2h-rg-native-score.json
```

### Optional: Daemon Mode for llm-tldr Profiles

Adding `--use-daemon` routes queries through the llm-tldr daemon instead of spawning a subprocess per query, eliminating startup + model-load overhead. Achieves **19.5x p50 latency speedup** with byte-identical results. Requires `tool_id='llm-tldr'` in the profile; non-daemon templates fall back to subprocess automatically.

```bash
uv run python scripts/bench_h2h_predict.py \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.hybrid_lane1.v1.json \
  --use-daemon \
  --out benchmark/runs/h2h-llm-tldr-predictions-daemon.json

# --daemon-keep-alive: leave daemon running between invocations
```

5. Score each tool independently:

```bash
uv run python scripts/bench_head_to_head.py score \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --predictions benchmark/runs/h2h-llm-tldr-predictions.json \
  --out benchmark/runs/h2h-llm-tldr-score.json

uv run python scripts/bench_head_to_head.py score \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/contextplus.v1.json \
  --predictions benchmark/runs/h2h-contextplus-predictions.json \
  --out benchmark/runs/h2h-contextplus-score.json
```

6. Compare both score reports with the suite winner rule:

```bash
uv run python scripts/bench_head_to_head.py compare \
  --suite benchmarks/head_to_head/suite.v1.json \
  --score-a benchmark/runs/h2h-llm-tldr-score.json \
  --score-b benchmark/runs/h2h-contextplus-score.json \
  --label-a llm-tldr \
  --label-b contextplus \
  --out benchmark/runs/h2h-compare.json
```

7. Assert strict superiority gates (winner + margins + validity + efficiency + stability):

```bash
# Single-run assertion (stability gate will fail unless strict config allows 1 run).
uv run python scripts/bench_h2h_assert.py \
  --suite benchmarks/head_to_head/suite.v1.json \
  --score-a benchmark/runs/h2h-llm-tldr-score.json \
  --score-b benchmark/runs/h2h-contextplus-score.json \
  --compare benchmark/runs/h2h-compare.json \
  --label-a llm-tldr \
  --label-b contextplus \
  --strict-gates benchmarks/head_to_head/gates.strict.v1.json \
  --out benchmark/runs/h2h-assert-strict.json

# Multi-run assertion (repeat --score-a/--score-b/--compare for run1..run3).
uv run python scripts/bench_h2h_assert.py \
  --suite benchmarks/head_to_head/suite.v1.json \
  --score-a benchmark/runs/h2h-llm-tldr-score-run1.json \
  --score-a benchmark/runs/h2h-llm-tldr-score-run2.json \
  --score-a benchmark/runs/h2h-llm-tldr-score-run3.json \
  --score-b benchmark/runs/h2h-contextplus-score-run1.json \
  --score-b benchmark/runs/h2h-contextplus-score-run2.json \
  --score-b benchmark/runs/h2h-contextplus-score-run3.json \
  --compare benchmark/runs/h2h-compare-run1.json \
  --compare benchmark/runs/h2h-compare-run2.json \
  --compare benchmark/runs/h2h-compare-run3.json \
  --label-a llm-tldr \
  --label-b contextplus \
  --strict-gates benchmarks/head_to_head/gates.strict.v1.json \
  --out benchmark/runs/h2h-assert-strict-3runs.json
```

## Required Reproducibility Artifacts

- `task_manifest.json` generated by `materialize-tasks`
- `tool_profile.json` per tool
- `predictions.json` per tool (canonical format)
- `score.json` per tool (`metrics` + `gate_checks`)
- `compare.json` final head-to-head decision
- raw logs per query/budget/trial under `benchmark/runs/raw_logs/`

## Notes

- The scorer enforces hard budget violations (`payload_tokens > budget_tokens`) and reports violation rate.
- Unsupported capabilities are allowed only when the tool profile marks them as unsupported.
- Use the same tokenizer (`cl100k_base`), corpus SHA, budgets, seeds, and trial count for both tools.
