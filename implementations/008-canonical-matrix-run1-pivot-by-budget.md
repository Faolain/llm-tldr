# 008 Canonical Matrix (Pivot View by Budget)

- Generated: 2026-03-02
- View style: one table per budget token, columns are tool/version variants
- Source matrix artifact: `benchmark/runs/matrix/h2h-matrix-long-run1-fixed-stitched-allowlist-20260302T062602Z.json`

## Column Keys

- `llm-tldr@bbfee65 (baseline.run1.fixed.stitched.allowlist)`
- `contextplus@b42853d (baseline.run1)`

## Caveat

This pivot uses mixed run artifacts currently in plan baseline:
- `llm-tldr`: run1-fixed-stitched-allowlist-20260302T062602Z artifact family
- `contextplus`: run1 artifact family

## Budget 500

| Metric | llm-tldr@bbfee65 (baseline.run1.fixed.stitched.allowlist) | contextplus@b42853d (baseline.run1) |
| --- | --- | --- |
| row_policy | optional | optional |
| mrr_mean | 0.6118629908103592 | 0.21691176470588236 |
| recall@5_mean | 0.7894736842105263 | 0.3 |
| recall@10_mean | 0.8070175438596491 | 0.3352941176470588 |
| precision@5_mean | 0.15789473684210528 | 0.060000000000000005 |
| fpr@5_mean | 0.0 | 1.0 |
| payload_tokens_median | 53.5 | 330.0 |
| latency_ms_p50 | 5041.5785 | 7728.097 |

## Budget 1000

| Metric | llm-tldr@bbfee65 (baseline.run1.fixed.stitched.allowlist) | contextplus@b42853d (baseline.run1) |
| --- | --- | --- |
| row_policy | optional | optional |
| mrr_mean | 0.6118629908103592 | 0.21564327485380116 |
| recall@5_mean | 0.7894736842105263 | 0.2982456140350877 |
| recall@10_mean | 0.8070175438596491 | 0.3333333333333333 |
| precision@5_mean | 0.15789473684210528 | 0.05964912280701755 |
| fpr@5_mean | 0.0 | 1.0 |
| payload_tokens_median | 53.5 | 329.0 |
| latency_ms_p50 | 4960.0355 | 7701.1345 |

## Budget 2000

| Metric | llm-tldr@bbfee65 (baseline.run1.fixed.stitched.allowlist) | contextplus@b42853d (baseline.run1) |
| --- | --- | --- |
| row_policy | required | required |
| mrr_mean | 0.6118629908103592 | 0.21564327485380116 |
| recall@5_mean | 0.7894736842105263 | 0.2982456140350877 |
| recall@10_mean | 0.8070175438596491 | 0.3333333333333333 |
| precision@5_mean | 0.15789473684210528 | 0.05964912280701755 |
| fpr@5_mean | 0.0 | 1.0 |
| payload_tokens_median | 53.5 | 329.0 |
| latency_ms_p50 | 5021.415000000001 | 7717.107 |

## Budget 5000

| Metric | llm-tldr@bbfee65 (baseline.run1.fixed.stitched.allowlist) | contextplus@b42853d (baseline.run1) |
| --- | --- | --- |
| row_policy | optional | optional |
| mrr_mean | 0.6118629908103592 | 0.21691176470588236 |
| recall@5_mean | 0.7894736842105263 | 0.3 |
| recall@10_mean | 0.8070175438596491 | 0.3352941176470588 |
| precision@5_mean | 0.15789473684210528 | 0.060000000000000005 |
| fpr@5_mean | 0.0 | 1.0 |
| payload_tokens_median | 53.5 | 328.0 |
| latency_ms_p50 | 4962.6404999999995 | 7740.205 |

## Global Reliability Snapshot (Run-Level)

| Metric | llm-tldr@bbfee65 (baseline.run1.fixed.stitched.allowlist) | contextplus@b42853d (baseline.run1) |
| --- | --- | --- |
| timeout_rate | 0.0 | 0.0 |
| error_rate | 0.0 | 0.4301587301587302 |
| unsupported_rate | 0.0 | 0.0 |
| budget_violation_rate | 0.0 | 0.0 |
| common_lane_coverage | 1.0 | 0.9972222222222222 |
| capability_coverage | 1.0 | 0.5698412698412698 |

## Structural Quality Snapshot (Budget 2000)

| Metric | llm-tldr@bbfee65 (baseline.run1.fixed.stitched.allowlist) | contextplus@b42853d (baseline.run1) |
| --- | --- | --- |
| impact_f1_mean | 0.8476190476190476 | null |
| slice_recall_mean | 0.8835714285714286 | null |
| slice_noise_reduction_mean | 0.6572387849083454 | null |
| data_flow_origin_accuracy_mean | 1.0 | null |
| data_flow_flow_completeness_mean | 1.0 | null |
| complexity_mae | 1.8 | null |
| complexity_kendall_tau_b | null | null |

## Primary-Gate Summary (Budget 2000)

| Check | Result |
| --- | --- |
| winner | `llm-tldr` |
| wins (`>=3/5` required) | `llm-tldr=5`, `contextplus=0` |
| mrr_mean_delta | `0.396219715956558` |
| recall@5_mean_delta | `0.4912280701754386` |
| precision@5_mean_delta | `0.09824561403508773` |
| strict assert overall | `false` |
| failing strict gate(s) | `validity.contextplus.error_rate;stability.two_of_three` |
| stability gate | `stability.two_of_three=false` (`insufficient_runs_for_stability_check`) |
