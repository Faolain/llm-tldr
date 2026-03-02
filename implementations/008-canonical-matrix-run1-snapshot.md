# 008 Canonical Matrix Snapshot (Run1 Baseline)

- Generated: 2026-03-02
- Scope: Existing run1 artifacts only (no new LLM execution)
- Canonical matrix artifacts: `benchmark/runs/matrix/h2h-matrix-long-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment-20260302T185207Z.csv` and `benchmark/runs/matrix/h2h-matrix-long-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment-20260302T185207Z.json` plus `rg-native` retrieval segment addendum artifacts listed below
- Canonical identity axes: `tool`, `tool_version`, `feature_set_id`, `embedding_backend`, `embedding_model`, `budget_tokens`, `run_id`

## Source Artifacts

- llm-tldr score: `benchmark/runs/h2h-llm-tldr-score-run1-abstain-rerank-lane2-retrieval-b2000-t123-segment.json`
- contextplus score: `benchmark/runs/h2h-contextplus-score-run1-segment-retrieval-b2000.json`
- compare: `benchmark/runs/h2h-compare-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment.json`
- assert: `benchmark/runs/h2h-assert-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment.json`
- run metadata (llm-tldr): `benchmark/runs/h2h-run-metadata-run1-llm-tldr-abstain-rerank-lane2-retrieval-b2000-t123.json`
- run metadata (contextplus): `benchmark/runs/h2h-run-metadata-run1-contextplus.json`
- tool profile (llm-tldr): `benchmarks/head_to_head/tool_profiles/llm_tldr.abstain_rerank_lane2.v1.json`
- tool profile (contextplus): `benchmarks/head_to_head/tool_profiles/contextplus.v1.json`
- rg-native score: `benchmark/runs/h2h-rg-native-score-run1-retrieval-b2000-t123-segment.json`
- compare (rg-native vs contextplus): `benchmark/runs/h2h-compare-run1-rg-native-retrieval-b2000-t123-segment-vs-contextplus-run1-segment.json`
- compare (rg-native vs llm-tldr baseline): `benchmark/runs/h2h-compare-run1-rg-native-retrieval-b2000-t123-segment-vs-llm-tldr-baseline-segment.json`
- run metadata (rg-native): `benchmark/runs/h2h-run-metadata-run1-rg-native-retrieval-b2000-t123-segment.json`
- tool profile (rg-native): `benchmarks/head_to_head/tool_profiles/rg_native.v1.json`

## Important Caveat

This snapshot combines:
- llm-tldr from `run1-fixed` stitched allowlist artifacts
- contextplus from `run1` artifacts
- rg-native from retrieval-only run1 segment artifacts (`retrieval@2000`, `trials=1..3`)

## Canonical Row Identity (Budget 2000 Rows)

| tool | tool_version | feature_set_id | embedding_backend | embedding_model | run_id |
| --- | --- | --- | --- | --- | --- |
| llm-tldr | `0ead1a11739004a2b12b1d439f10a29a03c64296` | `feature.abstain-rerank.v1` | `sentence-transformers` | `profile_unpinned` | `run1-abstain-rerank-lane2-retrieval-b2000-t123-segment` |
| contextplus | `4d7a6c37847c698c850d4b412ddb603dfc47257e` | `baseline.run1` | `unknown` | `unknown` | `run1-segment-retrieval-b2000` |
| rg-native | `unknown` | `baseline.native-rg.v1` | `unknown` | `unknown` | `run1-rg-native-retrieval-b2000-t123-segment` |

## Budget Row Policy

| budget_tokens | policy | row_scope |
| --- | --- | --- |
| 2000 | required | `required_primary_budget` |

## Retrieval Metrics By Budget

| tool | budget_tokens | mrr_mean | recall@5_mean | recall@10_mean | precision@5_mean | fpr@5_mean | payload_tokens_median | latency_ms_p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm-tldr | 2000 | 0.8741228070175439 | 0.8771929824561403 | 0.9649122807017544 | 0.1754385964912281 | 0.0 | 78.0 | 4989.021500000001 |
| contextplus | 2000 | 0.21564327485380116 | 0.2982456140350877 | 0.3333333333333333 | 0.05964912280701755 | 1.0 | 329.0 | 7717.107 |
| rg-native | 2000 | 0.8126218323586745 | 0.8771929824561403 | 0.9473684210526315 | 0.1754385964912281 | 0.0 | 12.0 | 216.22050000000002 |

## Holistic Metrics At Budget 2000 (Required Rows)

| tool | impact_f1_mean | slice_recall_mean | slice_noise_reduction_mean | data_flow_origin_accuracy_mean | data_flow_flow_completeness_mean | complexity_mae | timeout_rate | error_rate | budget_violation_rate | parse_errors_count | result_shape_counters_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm-tldr | null | null | null | null | null | null | 0.0 | 0.0 | 0.0 | 0 | 0 |
| contextplus | null | null | null | null | null | null | 0.0 | 0.0 | 0.0 | 0 | 0 |
| rg-native | null | null | null | null | null | null | 0.0 | 0.0 | 0.0 | 0 | 0 |

## Compare / Assert Snapshot (Budget 2000)

- llm-tldr lane2 vs contextplus:
  - winner: `llm-tldr`
  - wins: `llm-tldr=5`, `contextplus=0`
  - deltas: `mrr +0.6585`, `recall@5 +0.5789`, `precision@5 +0.1158`
  - strict assert overall: `false` (failing gate: `stability.two_of_three`)
- rg-native vs contextplus:
  - winner: `rg-native`
  - wins: `rg-native=5`, `contextplus=0`
  - deltas: `mrr +0.5970`, `recall@5 +0.5789`, `precision@5 +0.1158`
  - strict assert: not run in this addendum (compare + score only)
- rg-native vs llm-tldr baseline:
  - winner: `rg-native`
  - wins: `rg-native=5`, `llm-tldr-baseline=0`
  - deltas: `mrr +0.2008`, `recall@5 +0.0877`, `precision@5 +0.0175`
