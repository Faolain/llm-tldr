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

## 009 Model Variant Addendum (BGE vs Jina)

These rows complement the canonical run1 snapshot with the daemon-mode Django migration results from `implementations/009-migrate-bge-to-jina-code-0.5b_IMPLEMENTATION_PLAN.md`. They are same-tool model comparisons, not replacements for the original 008 h2h rows.

| Evaluation surface | BGE | Jina | Comparator context | What to take away |
| --- | --- | --- | --- | --- |
| Common lane `hybrid_rrf` quality | `mrr=0.8684`, `r@5=0.9649`, `r@10=1.0000`, `p@5=0.1930` | `mrr=0.8686`, `r@5=0.9825`, `r@10=1.0000`, `p@5=0.1965` | `contextplus` run1 remains far lower; `rg-native` remains much faster | Jina is roughly tied with BGE on lane1 quality, with only a marginal edge. |
| Common lane `hybrid_lane2` quality | `mrr=0.8741`, `r@5=0.8772`, `r@10=0.9649`, `p@5=0.1754` | `mrr=0.8417`, `r@5=0.9123`, `r@10=1.0000`, `p@5=0.1825` | This is the closest model-variant analogue to the run1 retrieval board | BGE still wins the canonical gate row on MRR, so Jina does not displace the current default lane. |
| Pure semantic concept-path quality | `mrr=0.6022`, `r@5=0.7719`, `r@10=0.7895`, `p@5=0.1544` | `mrr=0.7023`, `r@5=0.8596`, `r@10=0.8772`, `p@5=0.1719` | Both are stronger than the current `contextplus` concept-path row; `rg-native` is not the right comparator here | This is the main place Jina is genuinely better: pure semantic concept retrieval. |
| Token-efficiency retrieval @ `1000` | `semantic=0.6124`, `hybrid_rrf=0.8597` | `semantic=0.7048`, `hybrid_rrf=0.8647` | Budgeted retrieval quality, not h2h run1 | Jina improves semantic quality under token pressure; hybrid gain is small. |
| Compound semantic+impact | `tte_p50_ratio=1.0250`, correctness parity on overlap/Jaccard | `tte_p50_ratio=1.1361`, correctness parity on overlap/Jaccard | No direct `contextplus` / `rg-native` equivalent | Jina gives up efficiency on a product-path lane even though correctness is tied. |
| Structured exact-definition retrieval | `f1=0.1602`, `p50=142.1ms` | `f1=0.1520`, `p50=140.3ms` | `rg-native=0.9841`, `p50=84.8ms` | Exact definition lookup remains a lexical problem; neither embedding model should replace `rg-native` here. |
| Structured behavior retrieval (semantic) | `f1=0.1458`, `p50=169.5ms` | `f1=0.1053`, `p50=185.9ms` | `rg-native=0.0217`, `p50=87.3ms` | Semantic retrieval adds real value on concept-style target recovery, but BGE still beats Jina. |
| Structured behavior retrieval (hybrid) | `f1=0.1584`, `p50=422.9ms` | `f1=0.1188`, `p50=438.5ms` | Same suite; both beat `rg-native` on quality | Hybrid helps concept-style recovery, but BGE still wins and this harness includes file-to-symbol projection overhead. |
| Structured behavior retrieval (hybrid + `rg_empty`) | `f1=0.0000`, `p50=358.0ms` | `f1=0.0000`, `p50=358.3ms` | Negative query fixed; all positive hybrid queries suppressed | Strict lexical guards are too aggressive for the current behavior-query labels. |
| Steady-state daemon semantic latency | `p50=150.2ms`, `p95=250.2ms` | `p50=166.5ms`, `p95=286.4ms` | Query latency only, not build cost | BGE is still faster at steady-state query time by about `10-14%`. |
| Semantic build / memory cost | parity rebuild still pending | `build_s=692.57`, `peak_rss=3360.1MB` | Operational, not retrieval quality | Jina is operationally heavier, which is part of why the current decision remains `KEEP_OPT_IN`. |

Bottom line:
- Jina is useful over BGE when the task is pure semantic concept retrieval and semantic quality under budget is the priority.
- BGE remains the better default model for the full product surface because lane2, compound efficiency, structured retrieval, and operational cost still favor it.
- `rg-native` remains the correct first tool for exact symbol and definition lookup regardless of embedding model.

## Daemon-Mode Latency By Lane (Budget 2000, Retrieval, MPS GPU)

All lanes rerun with `--use-daemon` on 2026-03-03. Results are byte-identical to subprocess mode. MPS GPU auto-detected.

| Lane | Feature | Subprocess p50 | Daemon p50 | Speedup | Daemon mean | Daemon p90 | Parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| lane1 | hybrid | 5426.2 ms | 296.9 ms | **18.3x** | 373.8 ms | 319.7 ms | 60/60 |
| lane2 | abstain/rerank | 4989.0 ms | 292.5 ms | **17.1x** | 412.9 ms | 314.7 ms | 60/60 |
| lane3 | budget-aware | 4912.8 ms | 290.1 ms | **16.9x** | 286.7 ms | 300.1 ms | 60/60 |
| lane4 | compound | 5161.9 ms | 295.7 ms | **17.5x** | 295.9 ms | 321.5 ms | 12/12 |
| lane5 | navigate-cluster | 5170.4 ms | 293.1 ms | **17.6x** | 288.7 ms | 301.0 ms | 60/60 |
| contextplus | baseline | 7717.1 ms | N/A | - | N/A | N/A | - |
| rg-native | baseline | 216.2 ms | N/A | - | N/A | N/A | - |

Daemon-mode llm-tldr is within 1.37x of rg-native p50 (was 23-25x in subprocess mode).

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

## Full-Product Workflow Gate Snapshot (N/A = Loss)

This board is separate from the shared-capability retrieval gate. It evaluates end-user workflows and treats `unsupported/N/A` as explicit losses on resolved rows.

| Workflow row | llm-tldr (quantitative) | contextplus (quantitative) | rg-native (quantitative) | Evidence |
| --- | --- | --- | --- | --- |
| Retrieval (common lane, budget `2000`) | `mrr=0.874`, `r@5=0.877`, `p@5=0.175`, `fpr@5=0.000`, `p50=4989.022ms`, `tok=78` | `mrr=0.216`, `r@5=0.298`, `p@5=0.060`, `fpr@5=1.000`, `p50=7717.107ms`, `tok=329` | `mrr=0.813`, `r@5=0.877`, `p@5=0.175`, `fpr@5=0.000`, `p50=216.221ms`, `tok=12` | `benchmark/runs/h2h-llm-tldr-score-run1-abstain-rerank-lane2-retrieval-b2000-t123-segment.json`; `benchmark/runs/h2h-contextplus-score-run1-segment-retrieval-b2000.json`; `benchmark/runs/h2h-rg-native-score-run1-retrieval-b2000-t123-segment.json` |
| `impact -> context -> rg` (refactor path) | `impact: f1=0.848 (P=0.739,R=0.933), p50=191.951ms, tok=26`; `context: f1=0.880 (P=0.800,R=1.000), p50=0.191ms daemon (15516ms subprocess), tok=128`; `rg: mrr=0.813, r@5=0.877, p@5=0.175, fpr@5=0.000` | `N/A` | `N/A` | `benchmark/runs/h2h-llm-tldr-score-run1-fixed-stitched-allowlist-20260302T062602Z.json`; `benchmark/runs/h2h-rg-native-score-run1-retrieval-b2000-t123-segment.json`; `/tmp/ctx-daemon-preds.json` |
| `slice (+anchor) -> dfg` (debug path) | `slice: f1=0.919, recall=0.884, noise_reduction=0.657, p50=145.627ms, tok=11`; `dfg: origin=1.000, flow=1.000, p50=148.132ms, tok=13` | `N/A` | `N/A` | `benchmark/runs/h2h-llm-tldr-score-run1-fixed-stitched-allowlist-20260302T062602Z.json` |
| Semantic search (concept path) | `semantic strategy: mrr=0.247, r@5=0.456, p@5=0.091, fpr@5=0.000` | `mrr=0.216, r@5=0.298, p@5=0.060, fpr@5=1.000` | `N/A` | `benchmark/runs/20260302-195057Z-retrieval-django-lane3-b2000.json`; `benchmark/runs/h2h-contextplus-score-run1-segment-retrieval-b2000.json` |
| `cfg` / complexity | `accuracy=0.600`, `mae=1.800`, `p50=151.115ms`, `tok=8` | `N/A` | `N/A` | `benchmark/runs/h2h-llm-tldr-score-run1-fixed-stitched-allowlist-20260302T062602Z.json` |
| Daemon/index operational metrics | `build_s=1.231`, `patch_s=0.815`, `rebuild_s=1.070`; daemon retrieval lanes 1-5 all `p50=290-297ms` vs subprocess `p50=4912-5426ms` (`16.9-18.3x` speedup, MPS GPU); structural daemon `p50=1.7-13.9ms` (`15-93x`); context daemon `p50=0.191ms` vs subprocess `p50=15516ms` (`~82000x` warm, cached call graph); all results identical | `N/A` | `N/A` | `/tmp/bench_lane{1..5}_daemon_predictions.json`; `/tmp/bench_structural_daemon_predictions.json`; `/tmp/ctx-daemon-preds.json` |

Resolved-row provisional picture:
- `llm-tldr` leads on resolved full-product workflow rows.
- `contextplus` and `rg-native` are competitive on retrieval but lose rows where capabilities are `N/A`.
- Daemon/index operational row now resolved: `--use-daemon` eliminates subprocess startup overhead across all lanes (`16.9-18.3x` retrieval speedup, `15-93x` structural speedup, MPS GPU confirmed). All results byte-identical to subprocess mode.
