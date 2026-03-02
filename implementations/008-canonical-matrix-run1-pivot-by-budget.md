# 008 Canonical Matrix (Pivot View by Budget)

- Generated: 2026-03-02
- View style: one table per budget token, columns are tool/version variants
- Source snapshot: `implementations/008-canonical-matrix-run1-snapshot.md`

## Column Keys

- `llm-tldr@bbfee65 (baseline.run1.fixed.stitched.allowlist)`
- `contextplus@b42853d (baseline.run1)`

## Caveat

This pivot uses mixed run artifacts currently in plan baseline:
- `llm-tldr`: run1-fixed stitched allowlist artifact family
- `contextplus`: run1 artifact family

## Budget 500

| Metric | llm-tldr@bbfee65 | contextplus@b42853d |
| --- | ---: | ---: |
| mrr_mean | 0.612 | 0.217 |
| recall@5_mean | 0.789 | 0.300 |
| recall@10_mean | 0.807 | 0.335 |
| precision@5_mean | 0.158 | 0.060 |
| fpr@5_mean | 0.000 | 1.000 |
| payload_tokens_median | 53.500 | 330.000 |
| latency_ms_p50 | 5041.579 | 7728.097 |

## Budget 1000

| Metric | llm-tldr@bbfee65 | contextplus@b42853d |
| --- | ---: | ---: |
| mrr_mean | 0.612 | 0.216 |
| recall@5_mean | 0.789 | 0.298 |
| recall@10_mean | 0.807 | 0.333 |
| precision@5_mean | 0.158 | 0.060 |
| fpr@5_mean | 0.000 | 1.000 |
| payload_tokens_median | 53.500 | 329.000 |
| latency_ms_p50 | 4960.036 | 7701.135 |

## Budget 2000

| Metric | llm-tldr@bbfee65 | contextplus@b42853d |
| --- | ---: | ---: |
| mrr_mean | 0.612 | 0.216 |
| recall@5_mean | 0.789 | 0.298 |
| recall@10_mean | 0.807 | 0.333 |
| precision@5_mean | 0.158 | 0.060 |
| fpr@5_mean | 0.000 | 1.000 |
| payload_tokens_median | 53.500 | 329.000 |
| latency_ms_p50 | 5021.415 | 7717.107 |

## Budget 5000

| Metric | llm-tldr@bbfee65 | contextplus@b42853d |
| --- | ---: | ---: |
| mrr_mean | 0.612 | 0.217 |
| recall@5_mean | 0.789 | 0.300 |
| recall@10_mean | 0.807 | 0.335 |
| precision@5_mean | 0.158 | 0.060 |
| fpr@5_mean | 0.000 | 1.000 |
| payload_tokens_median | 53.500 | 328.000 |
| latency_ms_p50 | 4962.640 | 7740.205 |

## Global Reliability Snapshot (Run-Level)

| Metric | llm-tldr@bbfee65 | contextplus@b42853d |
| --- | ---: | ---: |
| timeout_rate | 0.000 | 0.000 |
| error_rate | 0.000 | 0.430 |
| budget_violation_rate | 0.000 | 0.000 |

## Primary-Gate Summary (Budget 2000)

| Check | Result |
| --- | --- |
| winner | `llm-tldr` |
| wins (`>=3/5` required) | `llm-tldr=5`, `contextplus=0` |
| mrr_mean_delta | `+0.396` |
| recall@5_mean_delta | `+0.491` |
| precision@5_mean_delta | `+0.098` |
| strict assert overall | `false` |
| failing strict gate(s) | `validity.contextplus.error_rate` |
| stability gate | `stability.two_of_three=false` (`insufficient_runs_for_stability_check`) |
