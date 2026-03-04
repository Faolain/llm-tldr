# 008 Canonical Matrix (Pivot View by Budget)

- Generated: 2026-03-02
- View style: one table per budget token, columns are tool/version variants
- Source artifacts:
  - `benchmark/runs/matrix/h2h-matrix-long-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment-20260302T185207Z.json`
  - `benchmark/runs/h2h-rg-native-score-run1-retrieval-b2000-t123-segment.json`

## Column Keys

- `llm-tldr@0ead1a1 (feature.abstain-rerank.v1)`
- `contextplus@4d7a6c3 (baseline.run1)`
- `rg-native@unknown (baseline.native-rg.v1)`

## Caveat

This pivot uses mixed run artifacts currently in plan baseline:
- `llm-tldr`: run1-fixed-stitched-allowlist-20260302T062602Z artifact family
- `contextplus`: run1 artifact family
- `rg-native`: run1 retrieval-only segment artifact family

## Budget 2000

| Metric | llm-tldr@0ead1a1 (feature.abstain-rerank.v1) | contextplus@4d7a6c3 (baseline.run1) | rg-native@unknown (baseline.native-rg.v1) |
| --- | --- | --- | --- |
| row_policy | required | required | required |
| mrr_mean | 0.8741228070175439 | 0.21564327485380116 | 0.8126218323586745 |
| recall@5_mean | 0.8771929824561403 | 0.2982456140350877 | 0.8771929824561403 |
| recall@10_mean | 0.9649122807017544 | 0.3333333333333333 | 0.9473684210526315 |
| precision@5_mean | 0.1754385964912281 | 0.05964912280701755 | 0.1754385964912281 |
| fpr@5_mean | 0.0 | 1.0 | 0.0 |
| payload_tokens_median | 78.0 | 329.0 | 12.0 |
| latency_ms_p50 | 4989.021500000001 | 7717.107 | 216.22050000000002 |

## Global Reliability Snapshot (Run-Level)

| Metric | llm-tldr@0ead1a1 (feature.abstain-rerank.v1) | contextplus@4d7a6c3 (baseline.run1) | rg-native@unknown (baseline.native-rg.v1) |
| --- | --- | --- | --- |
| timeout_rate | 0.0 | 0.0 | 0.0 |
| error_rate | 0.0 | 0.0 | 0.0 |
| unsupported_rate | 0.0 | 0.0 | 0.0 |
| budget_violation_rate | 0.0 | 0.0 | 0.0 |
| common_lane_coverage | 1.0 | 1.0 | 1.0 |
| capability_coverage | 1.0 | 1.0 | 1.0 |

## Structural Quality Snapshot (Budget 2000)

| Metric | llm-tldr@0ead1a1 (feature.abstain-rerank.v1) | contextplus@4d7a6c3 (baseline.run1) | rg-native@unknown (baseline.native-rg.v1) |
| --- | --- | --- | --- |
| impact_f1_mean | null | null | null |
| slice_recall_mean | null | null | null |
| slice_noise_reduction_mean | null | null | null |
| data_flow_origin_accuracy_mean | null | null | null |
| data_flow_flow_completeness_mean | null | null | null |
| complexity_mae | null | null | null |
| complexity_kendall_tau_b | null | null | null |

## Primary-Gate Summary (Budget 2000)

| Check | Result |
| --- | --- |
| llm-tldr lane2 vs contextplus winner | `llm-tldr` (`5-0`) |
| rg-native vs contextplus winner | `rg-native` (`5-0`) |
| rg-native vs llm-tldr baseline winner | `rg-native` (`5-0`) |
| strict assert overall (lane2 vs contextplus) | `false` |
| failing strict gate(s) (lane2 vs contextplus) | `stability.two_of_three` |
| stability gate (lane2 vs contextplus) | `stability.two_of_three=false` (`insufficient_runs_for_stability_check`) |

## Full-Product Workflow Gate (N/A = Loss)

| Workflow row | llm-tldr@0ead1a1 | contextplus@4d7a6c3 | rg-native@unknown |
| --- | --- | --- | --- |
| Retrieval (common lane, 2000) | `mrr=0.874`, `r@5=0.877`, `p@5=0.175`, `fpr@5=0.000`, `p50=4989.022ms`, `tok=78` | `mrr=0.216`, `r@5=0.298`, `p@5=0.060`, `fpr@5=1.000`, `p50=7717.107ms`, `tok=329` | `mrr=0.813`, `r@5=0.877`, `p@5=0.175`, `fpr@5=0.000`, `p50=216.221ms`, `tok=12` |
| `impact -> context -> rg` (refactor path) | `impact f1=0.848 (P=0.739,R=0.933), p50=191.951ms, tok=26; context pending` | `N/A` | `N/A` |
| `slice (+anchor) -> dfg` (debug path) | `slice f1=0.919, recall=0.884, noise=0.657; dfg origin=1.000, flow=1.000` | `N/A` | `N/A` |
| Semantic search (concept path) | `semantic mrr=0.247, r@5=0.456, p@5=0.091, fpr@5=0.000` | `pending` | `N/A` |
| `cfg` / complexity | `accuracy=0.600, mae=1.800, p50=151.115ms, tok=8` | `N/A` | `N/A` |
| Daemon/index operational metrics | `build_s=1.231, patch_s=0.815, rebuild_s=1.070; daemon-vs-cli pending` | `N/A` | `N/A` |

Notes:
- This gate is workflow-level and intentionally not limited to shared lanes.
- `unsupported/N/A` is scored as loss on resolved rows.
- `pending` rows are excluded until their benchmark contract artifacts are produced.
