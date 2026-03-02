# 008 Canonical Matrix Snapshot (Run1 Baseline)

- Generated: 2026-03-02
- Scope: Existing run1 artifacts only (no new LLM execution)
- Canonical matrix artifacts: `benchmark/runs/matrix/h2h-matrix-long-run1-fixed-stitched-allowlist-20260302T062602Z.csv` and `benchmark/runs/matrix/h2h-matrix-long-run1-fixed-stitched-allowlist-20260302T062602Z.json`
- Canonical identity axes: `tool`, `tool_version`, `feature_set_id`, `embedding_backend`, `embedding_model`, `budget_tokens`, `run_id`

## Source Artifacts

- llm-tldr score: `benchmark/runs/h2h-llm-tldr-score-run1-fixed-stitched-allowlist-20260302T062602Z.json`
- contextplus score: `benchmark/runs/h2h-contextplus-score-run1.json`
- compare: `benchmark/runs/h2h-compare-run1-fixed-stitched-allowlist-20260302T062602Z.json`
- assert: `benchmark/runs/h2h-assert-run1-fixed-stitched-allowlist-20260302T062602Z.json`
- run metadata (llm-tldr): `benchmark/runs/h2h-run-metadata-run1-llm-tldr-fixed.json`
- run metadata (contextplus): `benchmark/runs/h2h-run-metadata-run1-contextplus.json`
- tool profile (llm-tldr): `benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json`
- tool profile (contextplus): `benchmarks/head_to_head/tool_profiles/contextplus.v1.json`

## Important Caveat

This snapshot combines:
- llm-tldr from `run1-fixed` stitched allowlist artifacts
- contextplus from `run1` artifacts

## Canonical Row Identity (Budget 2000 Rows)

| tool | tool_version | feature_set_id | embedding_backend | embedding_model | run_id |
| --- | --- | --- | --- | --- | --- |
| llm-tldr | `bbfee65bc8cc5d5051edb447d689e7ebed987a7c` | `baseline.run1.fixed.stitched.allowlist` | `sentence-transformers` | `profile_unpinned` | `run1-fixed-stitched-allowlist-20260302T062602Z` |
| contextplus | `b42853d7c2a2018f2d4376c664db30d65ea1af23` | `baseline.run1` | `unknown` | `unknown` | `run1` |

## Budget Row Policy

| budget_tokens | policy | row_scope |
| --- | --- | --- |
| 500 | optional | `optional_budget_sensitivity` |
| 1000 | optional | `optional_budget_sensitivity` |
| 2000 | required | `required_primary_budget` |
| 5000 | optional | `optional_budget_sensitivity` |

## Retrieval Metrics By Budget

| tool | budget_tokens | mrr_mean | recall@5_mean | recall@10_mean | precision@5_mean | fpr@5_mean | payload_tokens_median | latency_ms_p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm-tldr | 500 | 0.6118629908103592 | 0.7894736842105263 | 0.8070175438596491 | 0.15789473684210528 | 0.0 | 53.5 | 5041.5785 |
| llm-tldr | 1000 | 0.6118629908103592 | 0.7894736842105263 | 0.8070175438596491 | 0.15789473684210528 | 0.0 | 53.5 | 4960.0355 |
| llm-tldr | 2000 | 0.6118629908103592 | 0.7894736842105263 | 0.8070175438596491 | 0.15789473684210528 | 0.0 | 53.5 | 5021.415000000001 |
| llm-tldr | 5000 | 0.6118629908103592 | 0.7894736842105263 | 0.8070175438596491 | 0.15789473684210528 | 0.0 | 53.5 | 4962.6404999999995 |
| contextplus | 500 | 0.21691176470588236 | 0.3 | 0.3352941176470588 | 0.060000000000000005 | 1.0 | 330.0 | 7728.097 |
| contextplus | 1000 | 0.21564327485380116 | 0.2982456140350877 | 0.3333333333333333 | 0.05964912280701755 | 1.0 | 329.0 | 7701.1345 |
| contextplus | 2000 | 0.21564327485380116 | 0.2982456140350877 | 0.3333333333333333 | 0.05964912280701755 | 1.0 | 329.0 | 7717.107 |
| contextplus | 5000 | 0.21691176470588236 | 0.3 | 0.3352941176470588 | 0.060000000000000005 | 1.0 | 328.0 | 7740.205 |

## Holistic Metrics At Budget 2000 (Required Rows)

| tool | impact_f1_mean | slice_recall_mean | slice_noise_reduction_mean | data_flow_origin_accuracy_mean | data_flow_flow_completeness_mean | complexity_mae | timeout_rate | error_rate | budget_violation_rate | parse_errors_count | result_shape_counters_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm-tldr | 0.8476190476190476 | 0.8835714285714286 | 0.6572387849083454 | 1.0 | 1.0 | 1.8 | 0.0 | 0.0 | 0.0 | 0 | 0 |
| contextplus | null | null | null | null | null | null | 0.0 | 0.4301587301587302 | 0.0 | 0 | 0 |

## Compare / Assert Snapshot (Budget 2000)

- compare winner: `llm-tldr`
- compare wins: `llm-tldr=5`, `contextplus=0`
- deltas at budget 2000:
  - `mrr_mean_delta = 0.396219715956558`
  - `recall@5_mean_delta = 0.4912280701754386`
  - `precision@5_mean_delta = 0.09824561403508773`
- strict assert overall: `false`
- failed strict gate(s): `validity.contextplus.error_rate;stability.two_of_three`
- stability gate: `stability.two_of_three = false` (`insufficient_runs_for_stability_check`)
