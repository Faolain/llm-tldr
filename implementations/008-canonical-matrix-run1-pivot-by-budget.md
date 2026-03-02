# 008 Canonical Matrix (Pivot View by Budget)

- Generated: 2026-03-02
- View style: one table per budget token, columns are tool/version variants
- Source matrix artifact: `benchmark/runs/matrix/h2h-matrix-long-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment-20260302T185207Z.json`

## Column Keys

- `llm-tldr@0ead1a1 (feature.abstain-rerank.v1)`
- `contextplus@4d7a6c3 (baseline.run1)`

## Caveat

This pivot uses mixed run artifacts currently in plan baseline:
- `llm-tldr`: run1-fixed-stitched-allowlist-20260302T062602Z artifact family
- `contextplus`: run1 artifact family

## Budget 2000

| Metric | llm-tldr@0ead1a1 (feature.abstain-rerank.v1) | contextplus@4d7a6c3 (baseline.run1) |
| --- | --- | --- |
| row_policy | required | required |
| mrr_mean | 0.8741228070175439 | 0.21564327485380116 |
| recall@5_mean | 0.8771929824561403 | 0.2982456140350877 |
| recall@10_mean | 0.9649122807017544 | 0.3333333333333333 |
| precision@5_mean | 0.1754385964912281 | 0.05964912280701755 |
| fpr@5_mean | 0.0 | 1.0 |
| payload_tokens_median | 78.0 | 329.0 |
| latency_ms_p50 | 4989.021500000001 | 7717.107 |

## Global Reliability Snapshot (Run-Level)

| Metric | llm-tldr@0ead1a1 (feature.abstain-rerank.v1) | contextplus@4d7a6c3 (baseline.run1) |
| --- | --- | --- |
| timeout_rate | 0.0 | 0.0 |
| error_rate | 0.0 | 0.0 |
| unsupported_rate | 0.0 | 0.0 |
| budget_violation_rate | 0.0 | 0.0 |
| common_lane_coverage | 1.0 | 1.0 |
| capability_coverage | 1.0 | 1.0 |

## Structural Quality Snapshot (Budget 2000)

| Metric | llm-tldr@0ead1a1 (feature.abstain-rerank.v1) | contextplus@4d7a6c3 (baseline.run1) |
| --- | --- | --- |
| impact_f1_mean | null | null |
| slice_recall_mean | null | null |
| slice_noise_reduction_mean | null | null |
| data_flow_origin_accuracy_mean | null | null |
| data_flow_flow_completeness_mean | null | null |
| complexity_mae | null | null |
| complexity_kendall_tau_b | null | null |

## Primary-Gate Summary (Budget 2000)

| Check | Result |
| --- | --- |
| winner | `llm-tldr` |
| wins (`>=3/5` required) | `llm-tldr=5`, `contextplus=0` |
| mrr_mean_delta | `0.6584795321637427` |
| recall@5_mean_delta | `0.5789473684210527` |
| precision@5_mean_delta | `0.11578947368421054` |
| strict assert overall | `false` |
| failing strict gate(s) | `stability.two_of_three` |
| stability gate | `stability.two_of_three=false` (`insufficient_runs_for_stability_check`) |
