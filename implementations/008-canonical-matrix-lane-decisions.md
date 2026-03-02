# 008 Canonical Matrix Lane Decisions

This log records keep/rollback outcomes for feature lanes using pinned matrix row IDs and artifacts.

## Lane1: Hybrid Retrieval (Phase 5, 2026-03-02)

- Outcome: `KEEP` (comparison-first track; provisional until full `stability.two_of_three` sign-off runs are completed).
- Before row IDs:
  - `llm-tldr|bbfee65bc8cc5d5051edb447d689e7ebed987a7c|baseline.run1.fixed.stitched.allowlist|sentence-transformers|profile_unpinned|2000|run1-fixed-stitched-allowlist-20260302T062602Z`
  - `contextplus|b42853d7c2a2018f2d4376c664db30d65ea1af23|baseline.run1|unknown|unknown|2000|run1`
- After row IDs:
  - `llm-tldr|4d7a6c37847c698c850d4b412ddb603dfc47257e|feature.hybrid.v1|sentence-transformers|profile_unpinned|2000|run1-hybrid-lane1-retrieval-b2000-t123-segment`
  - `contextplus|4d7a6c37847c698c850d4b412ddb603dfc47257e|baseline.run1|unknown|unknown|2000|run1-segment-retrieval-b2000`
- Decision evidence artifacts:
  - `benchmark/runs/h2h-suite-segment-retrieval-b2000.v1.json`
  - `benchmark/runs/h2h-task-manifest-segment-retrieval.json`
  - `benchmark/runs/h2h-llm-tldr-score-run1-hybrid-lane1-retrieval-b2000-t123-segment.json`
  - `benchmark/runs/h2h-llm-tldr-score-run1-baseline-retrieval-b2000-t123-segment.json`
  - `benchmark/runs/h2h-compare-run1-llm-tldr-hybrid-vs-baseline-retrieval-b2000-t123-segment.json`
  - `benchmark/runs/h2h-contextplus-score-run1-segment-retrieval-b2000.json`
  - `benchmark/runs/h2h-compare-run1-hybrid-lane1-retrieval-b2000-t123-vs-contextplus-run1-segment.json`
  - `benchmark/runs/h2h-assert-run1-hybrid-lane1-retrieval-b2000-t123-vs-contextplus-run1-segment.json`
  - `benchmark/runs/matrix/h2h-matrix-long-run1-hybrid-lane1-retrieval-b2000-t123-vs-contextplus-run1-segment-20260302T122740Z.json`
  - `benchmark/runs/matrix/h2h-matrix-long-run1-hybrid-lane1-retrieval-b2000-t123-vs-contextplus-run1-segment-20260302T122740Z.csv`
- Gate interpretation:
  - Per-run strict gates passed: `runs[0].strict_gates_passed=true`.
  - Overall assert remains false only because `stability.two_of_three=false` with reason `insufficient_runs_for_stability_check`.

### Lane1 Comparison Table (Retrieval Segment, Budget 2000, Trials 1..3)

| Metric | llm-tldr Hybrid (lane1) | llm-tldr Baseline | contextplus Baseline | Hybrid - Baseline | Hybrid - contextplus |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mrr_mean` | 0.8563 | 0.6119 | 0.2156 | +0.2444 | +0.6407 |
| `recall@5_mean` | 0.9298 | 0.7895 | 0.2982 | +0.1404 | +0.6316 |
| `precision@5_mean` | 0.1860 | 0.1579 | 0.0596 | +0.0281 | +0.1263 |
| `fpr@5_mean` | 0.0000 | 0.0000 | 1.0000 | +0.0000 | -1.0000 |
| `payload_tokens_median` | 78.0 | 53.5 | 329.0 | +24.5 | -251.0 |
| `latency_ms_p50` | 5426.209 | 5021.415 | 7717.107 | +404.794 | -2290.898 |
| Winner (5 primary metrics) | `llm-tldr-hybrid-lane1` vs `contextplus` | - | `llm-tldr-hybrid-lane1` vs `contextplus` | `llm-tldr-hybrid-lane1` wins `3/5` vs baseline | `llm-tldr-hybrid-lane1` wins `5/5` vs contextplus |

- Rationale summary:
  - Quality: lane1 wins all primary metrics (`5/5`) with large margins (`mrr +0.6407`, `recall@5 +0.6316`, `precision@5 +0.1263`).
  - Intra-tool A/B: lane1 hybrid wins `3/5` primary metrics over llm-tldr baseline (`mrr +0.2444`, `recall@5 +0.1404`, `precision@5 +0.0281`), with `fpr@5` unchanged at `0.0`.
  - Latency: improved (`5426.209ms` vs `7717.107ms`, ratio `0.7031x`).
  - Payload: improved (`78` vs `329`, ratio `0.2371x`).
  - Tradeoff vs llm-tldr baseline: payload `+24.5` tokens median and latency `+404.794ms` p50.
  - Reliability context: segment-scoped run-validity is clean for both tools (`timeout/error/budget_violation=0`).
- Baseline integrity note:
  - The run1 snapshot/pivot markdowns now track the latest exported matrix view for active comparison work.
  - Lane1 baseline evidence remains pinned by its own run-stamped matrix artifacts and row IDs in this log.

## Lane2 Handoff (Active Next Loop)

1. Lock lane2 profile identity (`feature.abstain-rerank.v1`) and retrieval command contract.
2. Add failing tests for abstention threshold behavior, optional rerank behavior, and bounded latency/payload regression.
3. Implement minimal confidence abstention plus optional rerank switches with default behavior unchanged when disabled.
4. Run segment-scoped `retrieval@2000,trials=1..3` score/compare/assert and export a new matrix row under `benchmark/runs/matrix/`.
5. Append lane2 keep/rollback decision in this log.

## Lane2: Confidence Abstention + Optional Rerank (Phase 5, 2026-03-02)

- Outcome: `KEEP` (comparison-first track; provisional until full `stability.two_of_three` sign-off runs are completed).
- Contract identity:
  - `feature_set_id`: `feature.abstain-rerank.v1`
  - profile: `benchmarks/head_to_head/tool_profiles/llm_tldr.abstain_rerank_lane2.v1.json`
- Before row IDs:
  - `llm-tldr|bbfee65bc8cc5d5051edb447d689e7ebed987a7c|baseline.run1.fixed.stitched.allowlist|sentence-transformers|profile_unpinned|2000|run1-fixed-stitched-allowlist-20260302T062602Z`
  - `llm-tldr|4d7a6c37847c698c850d4b412ddb603dfc47257e|feature.hybrid.v1|sentence-transformers|profile_unpinned|2000|run1-hybrid-lane1-retrieval-b2000-t123-segment`
  - `contextplus|4d7a6c37847c698c850d4b412ddb603dfc47257e|baseline.run1|unknown|unknown|2000|run1-segment-retrieval-b2000`
- After row IDs:
  - `llm-tldr|0ead1a11739004a2b12b1d439f10a29a03c64296|feature.abstain-rerank.v1|sentence-transformers|profile_unpinned|2000|run1-abstain-rerank-lane2-retrieval-b2000-t123-segment`
  - `contextplus|4d7a6c37847c698c850d4b412ddb603dfc47257e|baseline.run1|unknown|unknown|2000|run1-segment-retrieval-b2000`
- Decision evidence artifacts:
  - `benchmark/runs/20260302-183458Z-retrieval-django-lane2.json`
  - `benchmark/runs/h2h-llm-tldr-predictions-run1-abstain-rerank-lane2-retrieval-b2000-t123-segment.json`
  - `benchmark/runs/h2h-failure-classification-run1-llm-tldr-abstain-rerank-lane2-retrieval-b2000-t123.json`
  - `benchmark/runs/h2h-run-metadata-run1-llm-tldr-abstain-rerank-lane2-retrieval-b2000-t123.json`
  - `benchmark/runs/h2h-llm-tldr-score-run1-abstain-rerank-lane2-retrieval-b2000-t123-segment.json`
  - `benchmark/runs/h2h-compare-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment.json`
  - `benchmark/runs/h2h-compare-run1-llm-tldr-abstain-rerank-lane2-vs-baseline-retrieval-b2000-t123-segment.json`
  - `benchmark/runs/h2h-compare-run1-llm-tldr-abstain-rerank-lane2-vs-hybrid-lane1-retrieval-b2000-t123-segment.json`
  - `benchmark/runs/h2h-assert-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment.json`
  - `benchmark/runs/matrix/h2h-matrix-long-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment-20260302T185207Z.json`
  - `benchmark/runs/matrix/h2h-matrix-long-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment-20260302T185207Z.csv`
- Gate interpretation:
  - Per-run strict gates passed: `runs[0].strict_gates_passed=true`.
  - Overall assert remains false only because `stability.two_of_three=false` with reason `insufficient_runs_for_stability_check`.

### Lane2 Comparison Table (Retrieval Segment, Budget 2000, Trials 1..3)

| Metric | llm-tldr lane2 | llm-tldr lane1 | llm-tldr baseline | contextplus baseline | lane2 - lane1 | lane2 - baseline | lane2 - contextplus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mrr_mean` | 0.8741 | 0.8563 | 0.6119 | 0.2156 | +0.0178 | +0.2623 | +0.6585 |
| `recall@5_mean` | 0.8772 | 0.9298 | 0.7895 | 0.2982 | -0.0526 | +0.0877 | +0.5789 |
| `precision@5_mean` | 0.1754 | 0.1860 | 0.1579 | 0.0596 | -0.0105 | +0.0175 | +0.1158 |
| `fpr@5_mean` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | +0.0000 | +0.0000 | -1.0000 |
| `payload_tokens_median` | 78.0 | 78.0 | 53.5 | 329.0 | +0.0 | +24.5 | -251.0 |
| `latency_ms_p50` | 4989.022 | 5426.209 | 5021.415 | 7717.107 | -437.188 | -32.394 | -2728.085 |

- Rationale summary:
  - Versus `contextplus`: lane2 wins all primary retrieval metrics (`5/5`) with clean run-validity (`timeout/error/budget_violation=0`).
  - Versus llm baseline: lane2 wins `4/5` primary retrieval metrics (all except payload), while keeping `fpr@5=0`.
  - Versus lane1: lane2 improves `mrr_mean` and latency, with flat payload and `fpr@5`, but gives up some `recall@5` and `precision@5`.
  - Keep decision basis: 008 canonical gate surface is h2h segment at budget `2000`; lane2 remains superior to baseline/contextplus and passes run-level strict gates.
  - Drawback to carry forward into lane3: lane2 quality tradeoff against lane1 (`recall/precision` drop) must be monitored when budget-aware behavior is added.

## Lane3 Handoff (Active Next Loop)

1. Lock lane3 profile identity (`feature.budget-aware.v1`) and retrieval command contract with budget-variant behavior explicit in profile/template flags.
2. Add red tests for budget-driven retrieval packing behavior (non-decreasing quality across budgets + payload scaling + no violations).
3. Implement budget-aware retrieval path behind opt-in controls with default behavior unchanged.
4. Run deterministic loop: retrieval-quality (multi-budget) + retrieval segment h2h at budget `2000` + matrix export.
5. Append lane3 keep/rollback decision in this log with before/after row IDs and drawback accounting.
