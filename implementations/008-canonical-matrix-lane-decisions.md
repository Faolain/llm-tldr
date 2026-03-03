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

## Lane2 Handoff (Historical)

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

## Lane3: Budget-Aware Retrieval (Phase 5, 2026-03-02)

- Outcome: `KEEP` (comparison-first track; provisional until full `stability.two_of_three` sign-off runs are completed).
- Contract identity:
  - `feature_set_id`: `feature.budget-aware.v1`
  - profile: `benchmarks/head_to_head/tool_profiles/llm_tldr.budget_aware_lane3.v1.json`
  - default behavior unchanged unless `--budget-tokens` is provided.
- Before row IDs:
  - `llm-tldr|bbfee65bc8cc5d5051edb447d689e7ebed987a7c|baseline.run1.fixed.stitched.allowlist|sentence-transformers|profile_unpinned|2000|run1-fixed-stitched-allowlist-20260302T062602Z`
  - `llm-tldr|0ead1a11739004a2b12b1d439f10a29a03c64296|feature.abstain-rerank.v1|sentence-transformers|profile_unpinned|2000|run1-abstain-rerank-lane2-retrieval-b2000-t123-segment`
  - `contextplus|4d7a6c37847c698c850d4b412ddb603dfc47257e|baseline.run1|unknown|unknown|2000|run1-segment-retrieval-b2000`
- After row IDs:
  - `llm-tldr|8d5f6e9d9b30eb7bdeed9075b7897fc8ae0a4036|feature.budget-aware.v1|sentence-transformers|profile_unpinned|2000|run1-budget-aware-lane3-retrieval-b2000-t123-segment`
  - `contextplus|4d7a6c37847c698c850d4b412ddb603dfc47257e|baseline.run1|unknown|unknown|2000|run1-segment-retrieval-b2000`
- Decision evidence artifacts:
  - retrieval-quality multi-budget sweep:
    - `benchmark/runs/20260302-195057Z-retrieval-django-lane3-b500.json`
    - `benchmark/runs/20260302-195057Z-retrieval-django-lane3-b1000.json`
    - `benchmark/runs/20260302-195057Z-retrieval-django-lane3-b2000.json`
    - `benchmark/runs/20260302-195057Z-retrieval-django-lane3-b5000.json`
  - h2h segment predictions + scoring:
    - `benchmark/runs/h2h-llm-tldr-predictions-run1-budget-aware-lane3-retrieval-b2000-t123-segment.json`
    - `benchmark/runs/h2h-failure-classification-run1-llm-tldr-budget-aware-lane3-retrieval-b2000-t123.json`
    - `benchmark/runs/h2h-run-metadata-run1-llm-tldr-budget-aware-lane3-retrieval-b2000-t123.json`
    - `benchmark/runs/h2h-llm-tldr-score-run1-budget-aware-lane3-retrieval-b2000-t123-segment.json`
  - compares:
    - `benchmark/runs/h2h-compare-run1-budget-aware-lane3-retrieval-b2000-t123-vs-contextplus-run1-segment-normalized-labels.json`
    - `benchmark/runs/h2h-compare-run1-llm-tldr-budget-aware-lane3-vs-baseline-retrieval-b2000-t123-segment.json`
    - `benchmark/runs/h2h-compare-run1-llm-tldr-budget-aware-lane3-vs-hybrid-lane1-retrieval-b2000-t123-segment.json`
    - `benchmark/runs/h2h-compare-run1-llm-tldr-budget-aware-lane3-vs-abstain-rerank-lane2-retrieval-b2000-t123-segment.json`
    - `benchmark/runs/h2h-compare-run1-rg-native-retrieval-b2000-t123-segment-vs-contextplus-run1-segment.json`
    - `benchmark/runs/h2h-compare-run1-rg-native-retrieval-b2000-t123-segment-vs-llm-tldr-baseline-segment.json`
  - strict assert:
    - `benchmark/runs/h2h-assert-run1-budget-aware-lane3-retrieval-b2000-t123-vs-contextplus-run1-segment-normalized-labels.json`
  - native lexical baseline:
    - `benchmark/runs/h2h-rg-native-score-run1-retrieval-b2000-t123-segment.json`
    - `benchmark/runs/h2h-run-metadata-run1-rg-native-retrieval-b2000-t123-segment.json`
    - `benchmarks/head_to_head/tool_profiles/rg_native.v1.json`
  - matrix export:
    - `benchmark/runs/matrix/h2h-matrix-long-run1-budget-aware-lane3-retrieval-b2000-t123-vs-contextplus-run1-segment-20260302-202648Z.json`
    - `benchmark/runs/matrix/h2h-matrix-long-run1-budget-aware-lane3-retrieval-b2000-t123-vs-contextplus-run1-segment-20260302-202648Z.csv`
    - `benchmark/runs/matrix/008-canonical-matrix-run1-lane3-segment-snapshot-20260302-202648Z.md`
    - `benchmark/runs/matrix/008-canonical-matrix-run1-lane3-segment-pivot-by-budget-20260302-202648Z.md`
- Gate interpretation:
  - Per-run strict gates passed: `runs[0].strict_gates_passed=true`.
  - Overall assert remains false only because `stability.two_of_three=false` with reason `insufficient_runs_for_stability_check`.

### Lane3 Comparison Table (Retrieval Segment, Budget 2000, Trials 1..3)

| Metric | llm-tldr lane3 | llm-tldr lane2 | llm-tldr lane1 | llm-tldr baseline | contextplus baseline | rg-native baseline | lane3 - lane2 | lane3 - lane1 | lane3 - baseline | lane3 - contextplus | lane3 - rg-native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mrr_mean` | 0.8741 | 0.8741 | 0.8563 | 0.6119 | 0.2156 | 0.8126 | +0.0000 | +0.0178 | +0.2623 | +0.6585 | +0.0615 |
| `recall@5_mean` | 0.8772 | 0.8772 | 0.9298 | 0.7895 | 0.2982 | 0.8772 | +0.0000 | -0.0526 | +0.0877 | +0.5789 | +0.0000 |
| `precision@5_mean` | 0.1754 | 0.1754 | 0.1860 | 0.1579 | 0.0596 | 0.1754 | +0.0000 | -0.0105 | +0.0175 | +0.1158 | +0.0000 |
| `fpr@5_mean` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | +0.0000 | +0.0000 | +0.0000 | -1.0000 | +0.0000 |
| `payload_tokens_median` | 78.0 | 78.0 | 78.0 | 53.5 | 329.0 | 12.0 | +0.0 | +0.0 | +24.5 | -251.0 | +66.0 |
| `latency_ms_p50` | 4912.824 | 4989.022 | 5426.209 | 5021.415 | 7717.107 | 216.221 | -76.198 | -513.385 | -108.591 | -2804.283 | +4696.604 |

- Rationale summary:
  - Versus `contextplus`: lane3 wins all primary retrieval metrics (`5/5`) with clean run-validity (`timeout/error/budget_violation=0`).
  - Versus llm baseline: lane3 wins `4/5` primary metrics (all except payload), while keeping `fpr@5=0`.
  - Versus lane1: lane3 improves `mrr_mean` and latency, with flat payload/`fpr@5`, but gives up `recall@5` and `precision@5`.
  - Versus lane2: lane3 is quality/payload-equivalent at budget `2000` and improves latency (`-76.198ms`).
  - Versus `rg-native`: lane3 improves `mrr_mean` while matching `recall@5`, `precision@5`, and `fpr@5`, but it is much slower and carries higher payload.
  - Budget-sensitivity diagnostic: lane3 retrieval-quality run showed budget-varying `effective_k` (`500->3`, `1000->5`, `2000->10`, `5000->25`) with `fpr@5=0.0` across budgets.
  - Drawback observed: matrix export currently depends on compare/assert label alignment with snapshot defaults (`llm-tldr`/`contextplus`), so a normalized-label compare/assert artifact was required.

## Lane4: Compound Semantic+Impact (Phase 5, 2026-03-02)

- Outcome: `KEEP` (workflow lane; provisional until full `stability.two_of_three` sign-off runs are completed).
- Contract identity:
  - `feature_set_id`: `feature.compound-semantic-impact.v1`
  - profile: `benchmarks/head_to_head/tool_profiles/llm_tldr.compound_semantic_impact_lane4.v1.json`
  - output schema contract implemented in `tldr.semantic.compound_semantic_impact_search(...)`.
- Before row IDs:
  - `llm-tldr|bbfee65bc8cc5d5051edb447d689e7ebed987a7c|baseline.run1.fixed.stitched.allowlist|sentence-transformers|profile_unpinned|2000|run1-fixed-stitched-allowlist-20260302T062602Z`
  - `contextplus|4d7a6c37847c698c850d4b412ddb603dfc47257e|baseline.run1|unknown|unknown|2000|run1-segment-retrieval-b2000`
- After row IDs:
  - `llm-tldr|working-tree|feature.compound-semantic-impact.v1|sentence-transformers|profile_unpinned|2000|run1-compound-semantic-impact-lane4-retrieval-b2000-t1-r01-r12-segment`
  - `contextplus|4d7a6c37847c698c850d4b412ddb603dfc47257e|baseline.run1|unknown|unknown|2000|run1-retrieval-b2000-t1-r01-r12-segment`
- Decision evidence artifacts:
  - compound benchmark:
    - `benchmark/runs/20260302-215311Z-compound-semantic-impact-django-lane4-b2000.json`
  - retrieval-quality regression:
    - `benchmark/runs/20260302-214803Z-retrieval-django-lane4-b2000.json`
  - bounded h2h subset (`R01..R12`, `budget=2000`, `trial=1`):
    - `benchmark/runs/h2h-suite-segment-retrieval-b2000-t1-r01-r12.v1.json`
    - `benchmark/runs/h2h-task-manifest-segment-retrieval-r01-r12.json`
    - `benchmark/runs/h2h-llm-tldr-score-run1-compound-semantic-impact-lane4-retrieval-b2000-t1-r01-r12-segment.json`
    - `benchmark/runs/h2h-llm-tldr-score-run1-baseline-retrieval-b2000-t1-r01-r12-segment.json`
    - `benchmark/runs/h2h-contextplus-score-run1-retrieval-b2000-t1-r01-r12-segment.json`
    - `benchmark/runs/h2h-compare-run1-compound-semantic-impact-lane4-vs-baseline-retrieval-b2000-t1-r01-r12-segment.json`
    - `benchmark/runs/h2h-compare-run1-compound-semantic-impact-lane4-vs-contextplus-retrieval-b2000-t1-r01-r12-segment.json`
    - `benchmark/runs/h2h-assert-run1-compound-semantic-impact-lane4-vs-contextplus-retrieval-b2000-t1-r01-r12-segment.json`
    - `benchmark/runs/matrix/h2h-matrix-long-run1-compound-semantic-impact-lane4-retrieval-b2000-t1-r01-r12-vs-contextplus.json`
    - `benchmark/runs/matrix/h2h-matrix-long-run1-compound-semantic-impact-lane4-retrieval-b2000-t1-r01-r12-vs-contextplus.csv`
- Gate interpretation:
  - Per-run strict gates passed on bounded lane4 subset: `runs[0].strict_gates_passed=true`.
  - Overall assert remains false only because `stability.two_of_three=false` with reason `insufficient_runs_for_stability_check`.

### Lane4 Comparison Table (Bounded Retrieval Segment `R01..R12`, Budget 2000, Trial 1)

| Metric | llm-tldr lane4 | llm-tldr baseline | contextplus baseline | lane4 - baseline | lane4 - contextplus |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mrr_mean` | 0.7455 | 0.6394 | 0.2727 | +0.1061 | +0.4727 |
| `recall@5_mean` | 0.8182 | 0.8182 | 0.3636 | +0.0000 | +0.4545 |
| `precision@5_mean` | 0.1636 | 0.1636 | 0.0727 | +0.0000 | +0.0909 |
| `fpr@5_mean` | 0.0000 | 0.0000 | 1.0000 | +0.0000 | -1.0000 |
| `payload_tokens_median` | 78.5 | 50.0 | 334.0 | +28.5 | -255.5 |
| `latency_ms_p50` | 5161.943 | 5321.418 | 7654.634 | -159.475 | -2492.692 |

- Rationale summary:
  - Versus `contextplus`: lane4 wins all primary retrieval metrics (`5/5`) on this bounded subset.
  - Versus llm baseline: lane4 improves `mrr` and latency while recall/precision/FPR remain flat; payload increases.
  - Compound benchmark result: `time_to_evidence` is near parity with sequential baseline (`217.036ms` vs `215.425ms`) while payload is lower on this workload (`48.0` vs `1164.5` median tokens).
  - Drawback to carry forward into lane5: lane4 retrieval payload is higher than llm baseline on this subset and needs full-lane confirmation.

## Lane5: Semantic Navigation/Clustering (Phase 5, 2026-03-03)

- Outcome: `KEEP` (workflow lane; provisional until full `stability.two_of_three` sign-off runs are completed).
- Contract identity:
  - `feature_set_id`: `feature.navigate-cluster.v1`
  - profile: `benchmarks/head_to_head/tool_profiles/llm_tldr.navigate_cluster_lane5.v1.json`
- Before row IDs:
  - `llm-tldr|bbfee65bc8cc5d5051edb447d689e7ebed987a7c|baseline.run1.fixed.stitched.allowlist|sentence-transformers|profile_unpinned|2000|run1-fixed-stitched-allowlist-20260302T062602Z`
  - `contextplus|4d7a6c37847c698c850d4b412ddb603dfc47257e|baseline.run1|unknown|unknown|2000|run1-segment-retrieval-b2000`
- After row IDs:
  - `llm-tldr|working-tree|feature.navigate-cluster.v1|sentence-transformers|profile_unpinned|2000|run1-navigate-cluster-lane5-retrieval-b2000-t123-segment`
  - `contextplus|4d7a6c37847c698c850d4b412ddb603dfc47257e|baseline.run1|unknown|unknown|2000|run1-segment-retrieval-b2000`
- Decision evidence artifacts:
  - lane5 deterministic benchmark:
    - `benchmark/runs/20260303-001504Z-navigate-cluster-django-lane5-b2000.json`
  - retrieval-quality regression:
    - `benchmark/runs/20260303-001634Z-retrieval-django-lane5-b2000.json`
  - h2h retrieval segment (`budget=2000`, `trials=1..3`):
    - `benchmark/runs/h2h-llm-tldr-predictions-run1-navigate-cluster-lane5-retrieval-b2000-t123-segment.json`
    - `benchmark/runs/h2h-failure-classification-run1-llm-tldr-navigate-cluster-lane5-retrieval-b2000-t123.json`
    - `benchmark/runs/h2h-run-metadata-run1-llm-tldr-navigate-cluster-lane5-retrieval-b2000-t123.json`
    - `benchmark/runs/h2h-llm-tldr-score-run1-navigate-cluster-lane5-retrieval-b2000-t123-segment.json`
    - `benchmark/runs/h2h-compare-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.json`
    - `benchmark/runs/h2h-compare-run1-llm-tldr-navigate-cluster-lane5-vs-baseline-retrieval-b2000-t123-segment.json`
    - `benchmark/runs/h2h-assert-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.json`
    - `benchmark/runs/matrix/h2h-matrix-long-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.json`
    - `benchmark/runs/matrix/h2h-matrix-long-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.csv`
- Gate interpretation:
  - Per-run strict gates passed: `runs[0].strict_gates_passed=true`.
  - Overall assert remains false only because `stability.two_of_three=false` with reason `insufficient_runs_for_stability_check`.

### Lane5 Comparison Table (Retrieval Segment, Budget 2000, Trials 1..3)

| Metric | llm-tldr lane5 | llm-tldr baseline | contextplus baseline | lane5 - baseline | lane5 - contextplus |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mrr_mean` | 0.8741 | 0.6119 | 0.2156 | +0.2623 | +0.6585 |
| `recall@5_mean` | 0.8772 | 0.7895 | 0.2982 | +0.0877 | +0.5789 |
| `precision@5_mean` | 0.1754 | 0.1579 | 0.0596 | +0.0175 | +0.1158 |
| `fpr@5_mean` | 0.0000 | 0.0000 | 1.0000 | +0.0000 | -1.0000 |
| `payload_tokens_median` | 78.0 | 53.5 | 329.0 | +24.5 | -251.0 |
| `latency_ms_p50` | 5170.403 | 5021.415 | 7717.107 | +148.988 | -2546.704 |

- Rationale summary:
  - Versus `contextplus`: lane5 wins all primary retrieval metrics (`5/5`) with strict run-level gates passing.
  - Versus llm baseline: lane5 wins quality (`mrr`, `recall@5`, `precision@5`) while `payload` and `latency` regress.
  - Lane5 deterministic artifact proves stable clustering behavior (`assignment_digest_match_rate=1.0`) and high query-cluster recall (`@3=0.9825`) for positive queries.
  - Drawback to carry forward into lane6: lane5 adds workflow capability but not retrieval efficiency gains versus lane2/lane3 path.

## Program Rollup (Cross-Lane, As Of 2026-03-03)

This section is the single summary view for "where we stand now" across lanes and tools.

### Gate A: Shared-Capability Winner (Retrieval Common Lane @ 2000, Trials 1..3)

| Matchup | Winner | Wins | Evidence |
| --- | --- | --- | --- |
| lane1 (`llm-tldr`) vs `contextplus` | `llm-tldr` | `5-0` | `benchmark/runs/h2h-compare-run1-hybrid-lane1-retrieval-b2000-t123-vs-contextplus-run1-segment.json` |
| lane2 (`llm-tldr`) vs `contextplus` | `llm-tldr` | `5-0` | `benchmark/runs/h2h-compare-run1-abstain-rerank-lane2-retrieval-b2000-t123-vs-contextplus-run1-segment.json` |
| lane3 (`llm-tldr`) vs `contextplus` | `llm-tldr` | `5-0` | `benchmark/runs/h2h-compare-run1-budget-aware-lane3-retrieval-b2000-t123-vs-contextplus-run1-segment-normalized-labels.json` |
| lane4 (`llm-tldr`, bounded `R01..R12`) vs `contextplus` | `llm-tldr` | `5-0` | `benchmark/runs/h2h-compare-run1-compound-semantic-impact-lane4-vs-contextplus-retrieval-b2000-t1-r01-r12-segment.json` |
| lane5 (`llm-tldr`) vs `contextplus` | `llm-tldr` | `5-0` | `benchmark/runs/h2h-compare-run1-navigate-cluster-lane5-retrieval-b2000-t123-vs-contextplus-run1-segment.json` |
| `rg-native` vs `contextplus` | `rg-native` | `5-0` | `benchmark/runs/h2h-compare-run1-rg-native-retrieval-b2000-t123-segment-vs-contextplus-run1-segment.json` |
| `rg-native` vs `llm-tldr` baseline | `rg-native` | `5-0` | `benchmark/runs/h2h-compare-run1-rg-native-retrieval-b2000-t123-segment-vs-llm-tldr-baseline-segment.json` |

### Gate B: Full-Product Workflow Winner (N/A = Loss)

| Workflow row | llm-tldr (quantitative) | contextplus (quantitative) | rg-native (quantitative) | Status |
| --- | --- | --- | --- | --- |
| Retrieval (common lane) | `mrr=0.874`, `r@5=0.877`, `p@5=0.175`, `fpr@5=0.000`, `p50=4989.022ms`, `tok=78` | `mrr=0.216`, `r@5=0.298`, `p@5=0.060`, `fpr@5=1.000`, `p50=7717.107ms`, `tok=329` | `mrr=0.813`, `r@5=0.877`, `p@5=0.175`, `fpr@5=0.000`, `p50=216.221ms`, `tok=12` | resolved |
| `impact -> context -> rg` (refactor path) | `impact f1=0.848 (P=0.739,R=0.933), p50=191.951ms, tok=26; context metric pending` | `N/A` | `N/A` | partial |
| `slice (+anchor) -> dfg` (debug path) | `slice f1=0.919, recall=0.884, noise=0.657; dfg origin=1.000, flow=1.000` | `N/A` | `N/A` | resolved |
| Semantic search (concept path) | `semantic mrr=0.247, r@5=0.456, p@5=0.091, fpr@5=0.000` | pending | `N/A` | partial |
| `cfg` / complexity | `accuracy=0.600, mae=1.800, p50=151.115ms, tok=8` | `N/A` | `N/A` | resolved |
| Daemon/index operational metrics | `build_s=1.231, patch_s=0.815, rebuild_s=1.070; daemon-vs-cli pending` | `N/A` | `N/A` | pending |

Resolved-row interpretation:
- `llm-tldr` is currently the provisional full-product winner on resolved workflow rows.
- `contextplus` and `rg-native` remain strong retrieval baselines but lose structural workflow rows via `N/A`.
- Final full-product gate close requires resolving the pending workflow rows (explicit `context` row contract, semantic row parity status for `contextplus`, and daemon/index operational artifact).

## Lane6 Handoff (Active Next Loop)

1. Run one consolidated Gate B structural sweep (`impact/slice/dfg/cfg`) across lanes 1-5 and append updated quantitative rows.
2. Decide lane6 scope (`feature.ollama-backend.v1`) as optional/non-gating for this cycle and lock provider-selection contract.
3. If lane6 proceeds, run the same deterministic loop: red tests -> implementation behind opt-in -> retrieval regression at `2000` -> retrieval segment h2h compare + canonical row export.

### * Remaining TODOs (Explicit)

- [ ] * The consolidated "across lanes 1-5" Gate B structural sweep is still pending.
- [ ] * lane6 (`feature.ollama-backend.v1`) remains pending (optional/non-gating for this cycle).
