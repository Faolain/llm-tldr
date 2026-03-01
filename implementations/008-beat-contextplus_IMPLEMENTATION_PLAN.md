# llm-tldr Measurable Superiority Over contextplus Implementation Plan

- Status: In Progress (implementation complete for Phases 0-6; benchmark reruns pending)
- Owner: TBD
- Last updated: 2026-03-01
- Related spec: `specs/008-head-to-head-benchmark-llm-tldr-vs-contextplus.md`

## Objective

Make `llm-tldr` measurably better than `contextplus` using the existing neutral head-to-head harness, with hard pass/fail criteria and reproducible artifacts.

## Next Steps Checklist (008 Implementation Start)

- [x] Create artifact directories for deterministic reruns and audit outputs:
  - `mkdir -p benchmark/runs benchmark/logs benchmark/runs/stitch_audits`
- [x] Write and run failing reliability-policy tests before implementation changes:
  - `uv run pytest tests/test_bench_head_to_head_predict_helpers.py tests/test_bench_head_to_head_assert.py`
- [x] Add explicit failure classification output for each run:
  - `benchmark/runs/h2h-failure-classification-<run>.json`
- [x] Add partial rerun stitch output + audit artifacts for each tool/run:
  - `benchmark/runs/h2h-<tool>-predictions-<run>-stitched.json`
  - `benchmark/runs/stitch_audits/h2h-<tool>-stitch-audit-<run>.json`
- [ ] Re-run baseline (`run1..run3`) using unchanged suite seeds and persist run metadata sidecar:
  - `benchmark/runs/h2h-run-metadata-<run>.json`
- [x] Keep strict quality/effectiveness thresholds unchanged and enforce them on completed judgments after deterministic stitching.
- [ ] Run end-to-end 3-run benchmark artifacts and gate assertions using new tooling:
  - `scripts/bench_h2h_predict.py`, `scripts/bench_h2h_stitch.py`, `scripts/bench_h2h_baseline.py`, `scripts/bench_h2h_assert.py`
- [ ] Enable nightly full job with secrets/feature flag:
  - set `H2H_NIGHTLY_ENABLED=1` in CI environment before relying on nightly full gating.

## Implementation Progress (2026-03-01)

### Completed Delivery (Code + Tests)

- Reliability + stitching tooling:
  - Added `scripts/bench_h2h_stitch.py` with deterministic first-non-provider candidate replacement and stitch audit output.
  - Added `tests/test_bench_head_to_head_stitch_helpers.py` (4 tests).
- Phase 0:
  - Added schema/pin/materialization tests and deterministic fixtures:
    - `tests/test_bench_head_to_head_materialize_helpers.py`
    - `tests/test_bench_head_to_head_materialize_tasks.py`
    - updates in `tests/test_bench_head_to_head_suite_schema.py`
    - updates in `tests/test_bench_llm_open_ended_tasks_schema.py`
- Phase 1:
  - Added `scripts/bench_h2h_predict.py` (prediction runner, timeout mapping, raw logs, duplicate-row guard, optional classification/run-metadata sidecars).
  - Added `tests/test_bench_head_to_head_predict_helpers.py`
  - Added `tests/test_bench_head_to_head_predict_schema.py`
  - Added explicit plan-named compatibility tests in `tests/test_bench_llm_ab_run_helpers.py`.
- Phase 2:
  - Added `scripts/bench_h2h_baseline.py` (manifest consistency, `2/3` run-validity requirement, 2000-budget variance summary).
  - Added `tests/test_bench_head_to_head_baseline_helpers.py`.
- Phase 3-5:
  - Added helper validation suites:
    - `tests/test_bench_retrieval_quality_helpers.py`
    - `tests/test_bench_head_to_head_score_helpers.py`
    - `tests/test_bench_structural_analysis_helpers.py`
    - `tests/test_bench_ts_curated_recall_helpers.py`
    - `tests/test_bench_perf_daemon_vs_cli_helpers.py`
  - Updated `scripts/bench_token_efficiency.py` to deduplicate duplicate chunks in `_apply_budget`.
  - Updated `scripts/bench_perf_daemon_vs_cli.py` to compute speedup using `p50` latency (gate-aligned).
- Phase 6:
  - Added `scripts/bench_h2h_assert.py` with strict gate assertions (winner, margin, validity, efficiency, stability).
  - Added `benchmarks/head_to_head/gates.strict.v1.json`.
  - Added `tests/test_bench_head_to_head_assert.py`.
  - Updated `benchmarks/head_to_head/README.md` with strict-gate workflow.
  - Added CI workflows:
    - `.github/workflows/h2h-pr-smoke.yml`
    - `.github/workflows/h2h-nightly-full.yml`

### Validation Evidence

- Consolidated 008 implementation validation command (run on 2026-03-01):

```bash
uv run pytest \
  tests/test_bench_head_to_head_suite_schema.py \
  tests/test_bench_head_to_head_materialize_helpers.py \
  tests/test_bench_head_to_head_materialize_tasks.py \
  tests/test_bench_llm_open_ended_tasks_schema.py \
  tests/test_bench_head_to_head_predict_helpers.py \
  tests/test_bench_head_to_head_predict_schema.py \
  tests/test_bench_llm_ab_run_helpers.py \
  tests/test_bench_head_to_head_baseline_helpers.py \
  tests/test_bench_head_to_head_score_counters.py \
  tests/test_bench_retrieval_quality_helpers.py \
  tests/test_bench_token_efficiency_helpers.py \
  tests/test_bench_head_to_head_score_helpers.py \
  tests/test_bench_structural_analysis_helpers.py \
  tests/test_bench_ts_curated_recall_helpers.py \
  tests/test_bench_perf_daemon_vs_cli_helpers.py \
  tests/test_bench_head_to_head_assert.py \
  tests/test_bench_head_to_head_stitch_helpers.py
```

- Result:
  - `58 passed`

### Gotchas / Learnings Logged During Implementation

- `runpy.run_path()` monkeypatching:
  - Patch function `__globals__` for imported symbols (for example `count_tokens`), not just the module dict returned by `runpy`.
- Token packing:
  - Deduplicating repeated chunks before budgeting prevents silent budget waste and improves deterministic behavior.
- Perf gate alignment:
  - Speedup gate semantics should match `p50` latency (not mean) to align with phase pass/fail thresholds.
- Markdown lint command caveat:
  - `ruff check` should be run on Python files only; passing `README.md` to ruff treats Markdown as Python input.

## Definition Of Done (Program-Level)

A release is considered successful only if all conditions below are true:

1. Run validity gates pass for both tools on completed judgments after deterministic stitching:
   - `timeout_rate <= 0.02`
   - `error_rate <= 0.01`
   - `budget_violation_rate == 0.0`
2. Provider operational failures are non-gating for sign-off only when all are true:
   - Affected rows are explicitly classified as `provider_transport_runtime` (for example: Claude transport timeout, upstream `5xx`, connection reset, provider runtime cancellation) in `benchmark/runs/h2h-failure-classification-<run>.json`.
   - Eligible reruns exist and are stitchable under the deterministic merge protocol in this plan.
   - Stitch audit artifacts are present under `benchmark/runs/stitch_audits/`.
3. Head-to-head winner at primary budget (`2000`) is `llm-tldr` using suite winner rule (win at least 3 common-lane primary metrics).
4. Additional margin gates at budget `2000`:
   - `mrr_mean_delta (llm-tldr - contextplus) >= +0.05`
   - `recall@5_mean_delta >= +0.08`
   - `precision@5_mean_delta >= +0.05`
5. Efficiency safety gates at budget `2000`:
   - `payload_tokens_median(llm-tldr) <= 0.90 * payload_tokens_median(contextplus)`
   - `latency_ms_p50(llm-tldr) <= 1.10 * latency_ms_p50(contextplus)`
6. Stability gate:
   - Criteria 1-5 pass in at least `2/3` full reruns with suite seeds unchanged.
   - The same deterministic stitch rules are used in each rerun.

## Program Delivery Mode: Test-First With Benchmark Confirmation

This plan uses a hybrid method:

1. Test-first (`red -> green`) for deterministic logic, schema contracts, counters, and packing/ranking behavior.
2. Benchmark runs for system-level acceptance and superiority evidence.

Phase rule: implementation changes for a phase start only after that phase's "Tests To Write First" list exists and fails for the intended behavior change.

## Reliability Policy: Product Failures vs Provider Operational Failures

### Failure Classes

- `product_failure` (gating):
  - Any failure attributable to local runner/harness/tool behavior, including invalid output parsing, bad schema rows, budget violations, local crashes, or deterministic logic defects.
- `provider_transport_runtime` (non-gating if stitched per policy):
  - Upstream provider operational failures such as transport timeout, provider-side runtime timeout/cancel, transient `5xx`, or connection reset while calling hosted models (for example, Claude).
- `unclassified` (treated as gating):
  - Any row without explicit classification defaults to `product_failure` for gating.

### Gating Semantics

- Quality and effectiveness thresholds remain strict and unchanged.
- Only `provider_transport_runtime` rows may be excluded from blocking release gates, and only after deterministic rerun stitching artifacts are produced.
- `product_failure` and `unclassified` rows always count in run-validity gate math and can block sign-off.

### Required Reliability Artifacts

- `benchmark/runs/h2h-failure-classification-<run>.json`:
  - Per row key (`tool`, `task_id`, `budget`, `trial`) with `failure_class`, `reason`, and raw-log pointer.
- `benchmark/runs/h2h-run-metadata-<run>.json`:
  - Immutable run identity (`suite_hash`, `task_manifest_hash`, `tool_profile_hash`, `seed`, `model_id`).
- `benchmark/runs/stitch_audits/h2h-<tool>-stitch-audit-<run>.json`:
  - Deterministic merge decisions for every replaced or unresolved row.

## Stitching Protocol For Partial Reruns

### Eligibility

A rerun row is eligible to replace a base row only if all identity fields match:

- `suite_hash`
- `task_manifest_hash`
- `tool_profile_hash`
- `tool`
- `task_id`
- `budget`
- `trial`
- `seed`
- prompt hash and provider `model_id`

Only base rows classified as `provider_transport_runtime` are eligible for replacement.

### Artifact Requirements

- Base predictions file:
  - `benchmark/runs/h2h-<tool>-predictions-<run>.json`
- Partial rerun predictions file(s):
  - `benchmark/runs/h2h-<tool>-predictions-<run>-rerun<N>.json`
- Classification file:
  - `benchmark/runs/h2h-failure-classification-<run>.json`
- Output merged predictions file:
  - `benchmark/runs/h2h-<tool>-predictions-<run>-stitched.json`
- Output audit file:
  - `benchmark/runs/stitch_audits/h2h-<tool>-stitch-audit-<run>.json`

### Deterministic Merge Rules

For each key (`tool`, `task_id`, `budget`, `trial`):

1. Keep base row unless base row is classified `provider_transport_runtime`.
2. For eligible keys, sort rerun candidates by `(rerun_index asc, completed_at_utc asc)`.
3. Select the first candidate whose classification is not `provider_transport_runtime`.
4. If no such candidate exists, keep base row and mark unresolved provider operational failure.
5. Do not merge by best score/quality; merge is identity + ordering based only.

### Audit Trail

- Every replacement must record:
  - `row_key`
  - `base_artifact`
  - `replacement_artifact`
  - `base_failure_class`
  - `replacement_failure_class`
  - `decision_rule`
  - `decision_timestamp_utc`
- Every unresolved provider operational failure must record:
  - `row_key`
  - attempted rerun artifact list
  - final unresolved reason
- Audit files are immutable artifacts and must be archived with compare outputs.

### Example Commands

```bash
uv run python scripts/bench_h2h_predict.py \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --out benchmark/runs/h2h-llm-tldr-predictions-run1-rerun1.json

uv run python scripts/bench_h2h_stitch.py \
  --base benchmark/runs/h2h-llm-tldr-predictions-run1.json \
  --rerun benchmark/runs/h2h-llm-tldr-predictions-run1-rerun1.json \
  --classification benchmark/runs/h2h-failure-classification-run1.json \
  --run-metadata benchmark/runs/h2h-run-metadata-run1.json \
  --out benchmark/runs/h2h-llm-tldr-predictions-run1-stitched.json \
  --audit benchmark/runs/stitch_audits/h2h-llm-tldr-stitch-audit-run1.json
```

## Phase 0: Freeze Contract And Reproducible Baseline Inputs

### Goals

- Lock benchmark contract and inputs so future wins are credible and auditable.
- Replace `contextplus` profile template with a runnable, versioned profile.

### Deliverables

- `benchmarks/head_to_head/tool_profiles/contextplus.v1.json` (real command templates, no placeholders).
- `benchmarks/head_to_head/suite.v1.json` and query files confirmed pinned to Django `5.1.13` SHA `c04a09ddb3bb1fe8157292fcd902b35cad9a5e10`.
- Optional stricter gate config file for this program:
  - `benchmarks/head_to_head/gates.strict.v1.json`.

### Tests To Write First (Before Implementation)

- `tests/test_bench_head_to_head_tool_profiles_schema.py::test_contextplus_profile_is_real_profile_not_template`
- `tests/test_bench_head_to_head_tool_profiles_schema.py::test_contextplus_retrieval_template_has_no_placeholder_text`
- `tests/test_bench_head_to_head_suite_schema.py::test_head_to_head_suite_django_pin_matches_corpora_manifest`
- `tests/test_bench_head_to_head_materialize_helpers.py::test_materialize_tasks_is_deterministic_for_identical_inputs`
- `tests/test_bench_llm_open_ended_tasks_schema.py::test_open_ended_task_query_alignment_and_anchor_consistency`
- `tests/test_bench_llm_open_ended_tasks_schema.py::test_oe08_regression_guard_maps_to_b10_configure`
- `tests/test_bench_head_to_head_materialize_tasks.py::test_materialize_tasks_valid_fixture_has_zero_warnings_and_stable_hash` (new)

### Commands

```bash
uv run pytest \
  tests/test_bench_head_to_head_suite_schema.py \
  tests/test_bench_head_to_head_tool_profiles_schema.py \
  tests/test_bench_django_retrieval_queries_schema.py \
  tests/test_bench_django_structural_queries_schema.py

uv run python scripts/bench_fetch_corpora.py --corpus django

uv run python scripts/bench_head_to_head.py validate-suite \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/contextplus.v1.json

uv run python scripts/bench_head_to_head.py materialize-tasks \
  --suite benchmarks/head_to_head/suite.v1.json \
  --corpus-root benchmark/corpora/django \
  --out benchmark/runs/h2h-task-manifest.json
```

### Pass/Fail Thresholds

- `validate-suite` exits `0` with no schema or pin mismatch.
- `materialize-tasks` output SHA is stable across 3 consecutive runs.
- All listed pytest checks pass.

## Phase 1: Build Prediction Runners (Execution Gap Closure)

### Goals

- Close the current gap where head-to-head harness scores results but does not execute both tools.
- Standardize prediction generation and raw log capture.

### Deliverables

- New runner script:
  - `scripts/bench_h2h_predict.py`
- Required capabilities:
  - Read suite + task manifest + tool profile.
  - Execute command templates per `(task_id, trial, budget)`.
  - Emit canonical predictions JSON format.
  - Emit raw logs under `benchmark/runs/raw_logs/<tool>/<trial>/<task>.log`.
  - Enforce per-query timeout and status mapping (`ok|unsupported|timeout|error`).
- Tests:
  - `tests/test_bench_head_to_head_predict_helpers.py`
  - `tests/test_bench_head_to_head_predict_schema.py`

### Tests To Write First (Before Implementation)

- `tests/test_bench_head_to_head_predict_helpers.py::test_render_command_template_raises_on_missing_placeholder`
- `tests/test_bench_head_to_head_predict_helpers.py::test_timeout_maps_to_timeout_status_not_error`
- `tests/test_bench_head_to_head_predict_helpers.py::test_raw_log_path_is_tool_trial_task_layout`
- `tests/test_bench_head_to_head_predict_schema.py::test_predictions_schema_rejects_duplicate_task_budget_trial_rows`
- `tests/test_bench_llm_ab_run_helpers.py::test_structured_failure_classes_mutually_exclusive`
- `tests/test_bench_llm_ab_run_helpers.py::test_structured_bad_json_reconciliation`
- `tests/test_bench_llm_ab_run_helpers.py::test_judge_bad_json_reconciliation`

### Commands

```bash
uv run pytest \
  tests/test_bench_head_to_head_predict_helpers.py \
  tests/test_bench_head_to_head_predict_schema.py

uv run python scripts/bench_h2h_predict.py \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --out benchmark/runs/h2h-llm-tldr-predictions.json

uv run python scripts/bench_h2h_predict.py \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/contextplus.v1.json \
  --out benchmark/runs/h2h-contextplus-predictions.json
```

### Pass/Fail Thresholds

- Prediction row coverage per tool `>= 0.99` of expected supported tasks.
- No malformed prediction entries (scorer accepts file without schema errors).
- Timeout/error rates remain within suite run-validity limits after scoring.

## Phase 2: Establish True Baseline (Before Optimizing)

### Goals

- Capture current head-to-head performance with repeatability and variance.
- Create a baseline report that all future phases are compared against.

### Deliverables

- Three full baseline runs (same machine class, same seeds, same suite).
- Baseline summary artifact:
  - `benchmark/runs/h2h-baseline-summary.json`

### Tests To Write First (Before Implementation)

- `tests/test_bench_head_to_head_baseline_helpers.py::test_baseline_summary_rejects_mixed_task_manifest_hashes`
- `tests/test_bench_head_to_head_baseline_helpers.py::test_baseline_summary_requires_two_of_three_valid_runs`
- `tests/test_bench_head_to_head_baseline_helpers.py::test_baseline_variance_uses_budget_2000_mrr_and_recall5`
- `tests/test_bench_head_to_head_score_counters.py::test_score_emits_typed_parse_diagnostics_without_gate_math_drift` (new)

### Commands

```bash
# Repeat 3 times (run1, run2, run3)
uv run python scripts/bench_head_to_head.py score \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --predictions benchmark/runs/h2h-llm-tldr-predictions.json \
  --out benchmark/runs/h2h-llm-tldr-score-run1.json

uv run python scripts/bench_head_to_head.py score \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/contextplus.v1.json \
  --predictions benchmark/runs/h2h-contextplus-predictions.json \
  --out benchmark/runs/h2h-contextplus-score-run1.json

uv run python scripts/bench_head_to_head.py compare \
  --suite benchmarks/head_to_head/suite.v1.json \
  --score-a benchmark/runs/h2h-llm-tldr-score-run1.json \
  --score-b benchmark/runs/h2h-contextplus-score-run1.json \
  --label-a llm-tldr \
  --label-b contextplus \
  --out benchmark/runs/h2h-compare-run1.json
```

### Pass/Fail Thresholds

- Each run passes fairness gates (`task_manifest_hash`, tokenizer, budgets).
- At least `2/3` runs have both tools pass run-validity gates on completed judgments after deterministic stitching.
- Baseline variance at budget `2000` is bounded:
  - `stdev(mrr_mean) <= 0.02`
  - `stdev(recall@5_mean) <= 0.03`

## Phase 3: Retrieval Quality Win (Common Lane)

### Goals

- Improve retrieval relevance enough to win common-lane metrics versus contextplus.
- Avoid wins caused by over-budget payload or instability.

### Deliverables

- Retrieval stack upgrades (expected areas):
  - Better hybrid ranking strategy (`semantic + lexical` fusion tuning).
  - Query rewrite/normalization improvements for symbol and intent queries.
  - Deterministic no-result handling aligned with benchmark guard policy.
- Tests:
  - Extend `tests/test_bench_token_efficiency_helpers.py` (budget packing invariants).
  - Add retrieval ranking helper tests for any new scoring logic.

### Tests To Write First (Before Implementation)

- `tests/test_bench_retrieval_quality_helpers.py::test_rrf_fuse_boosts_docs_supported_by_multiple_rankers`
- `tests/test_bench_retrieval_quality_helpers.py::test_rrf_fuse_tie_break_is_deterministic`
- `tests/test_bench_retrieval_quality_helpers.py::test_no_result_guard_rg_empty_forces_empty_semantic_and_hybrid`
- `tests/test_bench_token_efficiency_helpers.py::test_apply_budget_keeps_anchor_context_before_optional_windows`

### Commands

```bash
uv run pytest tests/test_bench_token_efficiency_helpers.py

uv run python scripts/bench_retrieval_quality.py \
  --corpus django \
  --cache-root benchmark/cache-root \
  --index repo:django-h2h \
  --ks 1,5,10 \
  --no-result-guard rg_empty \
  --out benchmark/runs/phase3-retrieval-quality.json

uv run python scripts/bench_head_to_head.py score \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --predictions benchmark/runs/h2h-llm-tldr-predictions.json \
  --out benchmark/runs/h2h-llm-tldr-score-phase3.json

uv run python scripts/bench_head_to_head.py score \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/contextplus.v1.json \
  --predictions benchmark/runs/h2h-contextplus-predictions.json \
  --out benchmark/runs/h2h-contextplus-score-phase3.json

uv run python scripts/bench_head_to_head.py compare \
  --suite benchmarks/head_to_head/suite.v1.json \
  --score-a benchmark/runs/h2h-llm-tldr-score-phase3.json \
  --score-b benchmark/runs/h2h-contextplus-score-phase3.json \
  --label-a llm-tldr \
  --label-b contextplus \
  --out benchmark/runs/h2h-compare-phase3.json
```

### Pass/Fail Thresholds

- Budget `2000` compare result: winner is `llm-tldr`.
- `llm-tldr` wins `>= 3/5` primary common-lane metrics.
- Margin gates at budget `2000`:
  - `mrr_mean_delta >= +0.05`
  - `recall@5_mean_delta >= +0.08`
  - `precision@5_mean_delta >= +0.05`

## Phase 4: Token-Efficiency Dominance (Low Budgets)

### Goals

- Make llm-tldr clearly better under constrained budgets (`500`, `1000`) where retrieval strategy quality matters most.

### Deliverables

- Budget-aware context selection/packing improvements.
- De-duplication and chunk prioritization updates to reduce wasted tokens.
- Guardrails ensuring no hard budget overruns.

### Tests To Write First (Before Implementation)

- `tests/test_bench_token_efficiency_helpers.py::test_apply_budget_never_exceeds_budget_including_join_separators`
- `tests/test_bench_token_efficiency_helpers.py::test_apply_budget_deduplicates_duplicate_chunks`
- `tests/test_bench_head_to_head_score_helpers.py::test_score_budget_violation_rate_counts_over_budget_ok_predictions`
- `tests/test_bench_llm_ab_prompts_slice_packing.py::test_slice_open_ended_context_metadata_contract_and_determinism`
- `tests/test_bench_llm_ab_prompts_data_flow_packing.py::test_data_flow_budget_hard_cap_and_deterministic_drop_order` (new)

### Commands

```bash
uv run pytest tests/test_bench_token_efficiency_helpers.py

uv run python scripts/bench_token_efficiency.py \
  --corpus django \
  --mode retrieval \
  --budgets 500,1000,2000 \
  --cache-root benchmark/cache-root \
  --index repo:django-h2h \
  --no-result-guard rg_empty \
  --out benchmark/runs/phase4-token-efficiency.json

# Re-run predictions + score + compare after implementation changes.
```

### Pass/Fail Thresholds

- `budget_violation_rate == 0.0` for llm-tldr.
- At budget `500`:
  - `recall@5_mean_delta >= +0.10` vs contextplus.
- At budget `1000`:
  - `recall@5_mean_delta >= +0.08` vs contextplus.
- At both `500` and `1000`:
  - `payload_tokens_median(llm-tldr) <= 0.90 * payload_tokens_median(contextplus)`.

## Phase 5: Structural + Runtime Differentiation (Defensible Product Advantage)

### Goals

- Strengthen llm-tldr capabilities contextplus does not natively provide.
- Ensure structural strengths do not regress while retrieval is being tuned.

### Deliverables

- Structural quality improvements for `impact`, `slice`, `data_flow`, `complexity`.
- TS callgraph quality checks on pinned curated corpora.
- Daemon-vs-CLI latency improvements for repeated-query workflows.

### Tests To Write First (Before Implementation)

- `tests/test_bench_structural_analysis_helpers.py::test_data_flow_origin_accuracy_requires_exact_origin_line`
- `tests/test_bench_structural_analysis_helpers.py::test_python_function_span_resolves_class_method_names`
- `tests/test_bench_ts_curated_recall_helpers.py::test_load_graph_cache_rejects_language_mismatch`
- `tests/test_bench_perf_daemon_vs_cli_helpers.py::test_speedup_metric_uses_p50_latency_for_gate_alignment`

### Commands

```bash
uv run pytest \
  tests/test_python_slice_behavior.py \
  tests/test_ts_callgraph_fixture.py \
  tests/test_ts_callgraph_multi_tsconfig.py \
  tests/test_daemon_identity.py \
  tests/test_daemon_index_mode.py \
  tests/test_daemon_stats.py

uv run python scripts/bench_structural_analysis.py \
  --corpus django \
  --cache-root benchmark/cache-root \
  --index repo:django-h2h \
  --out benchmark/runs/phase5-structural-django.json

uv run python scripts/bench_ts_curated_recall.py \
  --repo-root benchmark/corpora/peerbit \
  --curated benchmarks/ts/peerbit_curated_edges.json \
  --cache-root benchmark/cache-root \
  --index repo:peerbit-h2h \
  --mode both \
  --fail-under-edge-recall 0.90 \
  --fail-under-impact-recall 0.90 \
  --out benchmark/runs/phase5-ts-peerbit.json

uv run python scripts/bench_ts_curated_recall.py \
  --repo-root benchmark/corpora/nextjs \
  --curated benchmarks/ts/nextjs_curated_edges.json \
  --cache-root benchmark/cache-root \
  --index repo:nextjs-h2h \
  --mode both \
  --fail-under-edge-recall 0.85 \
  --fail-under-impact-recall 0.80 \
  --out benchmark/runs/phase5-ts-nextjs.json

uv run python scripts/bench_perf_daemon_vs_cli.py \
  --corpus django \
  --cache-root benchmark/cache-root \
  --index repo:django-h2h \
  --include-semantic \
  --iterations 10 \
  --out benchmark/runs/phase5-daemon-vs-cli.json
```

### Pass/Fail Thresholds

- Structural quality at budget `2000`:
  - `impact_f1 >= 0.70`
  - `slice_recall >= 0.80`
  - `data_flow_origin_accuracy >= 0.75`
  - `complexity_mae <= 2.5`
- TS curated recall thresholds pass (commands enforce non-zero failure if below thresholds).
- Daemon warm-start p50 latency for `impact` and `semantic` is at least `30%` faster than CLI (`daemon_p50 <= 0.70 * cli_p50`).

## Phase 6: CI Gates And Release Policy

### Goals

- Make superiority durable by gating regressions in PR and nightly workflows.

### Deliverables

- New assert tool:
  - `scripts/bench_h2h_assert.py` (fails with non-zero exit when strict superiority gates fail).
- CI workflows:
  - PR smoke: schema + lightweight h2h subset.
  - Nightly full: full suite, both tools, all budgets.
- Documentation update:
  - `benchmarks/head_to_head/README.md` with strict-gate workflow.

### Tests To Write First (Before Implementation)

- `tests/test_bench_head_to_head_assert.py::test_assert_fails_when_compare_winner_is_not_llm_tldr`
- `tests/test_bench_head_to_head_assert.py::test_assert_fails_when_margin_gates_are_below_threshold`
- `tests/test_bench_head_to_head_assert.py::test_assert_fails_on_validity_or_efficiency_gate_failure`
- `tests/test_bench_head_to_head_assert.py::test_assert_requires_stability_two_of_three_runs`
- `tests/test_bench_head_to_head_assert.py::test_assert_passes_only_when_all_strict_gates_pass`

### Commands

```bash
uv run pytest \
  tests/test_bench_head_to_head_suite_schema.py \
  tests/test_bench_head_to_head_tool_profiles_schema.py \
  tests/test_bench_head_to_head_predict_helpers.py \
  tests/test_bench_head_to_head_predict_schema.py \
  tests/test_bench_head_to_head_assert.py

uv run python scripts/bench_h2h_assert.py \
  --suite benchmarks/head_to_head/suite.v1.json \
  --score-a benchmark/runs/h2h-llm-tldr-score.json \
  --score-b benchmark/runs/h2h-contextplus-score.json \
  --compare benchmark/runs/h2h-compare.json \
  --label-a llm-tldr \
  --label-b contextplus \
  --strict-gates benchmarks/head_to_head/gates.strict.v1.json
```

### Pass/Fail Thresholds

- PR smoke runtime `<= 20 minutes` and must pass on default branch before merge.
- Nightly full run must pass strict superiority gates for `7` consecutive nights before claiming completion (using deterministic stitched artifacts when provider operational failures occur).
- Any nightly failure classified as `product_failure` or `unclassified` opens a blocking regression issue with attached run artifacts.
- Nightly failures classified exclusively as `provider_transport_runtime` require rerun + stitch + audit artifacts, and remain non-blocking unless classification or stitch requirements are missing.

## Long-Running Execution Notes

For full multi-budget runs, use `tmux` + log tee so jobs survive session disruptions and remain auditable:

```bash
tmux new-session -d -s h2h-full \
  'cd /Users/aristotle/Documents/Projects/llm-tldr && \
   PYTHONUNBUFFERED=1 NO_COLOR=1 uv run python scripts/bench_h2h_predict.py ... 2>&1 | tee benchmark/logs/h2h-full.log'
```

## Final Exit Checklist

- [ ] Phase 0-6 deliverables merged.
- [ ] All phase pass/fail thresholds met on completed judgments after deterministic stitching.
- [ ] `llm-tldr` wins head-to-head by suite rule and strict margin gates.
- [ ] Results reproduced in at least 2 of 3 full reruns with identical stitch rules.
- [ ] All provider operational failures (if any) are explicitly classified as `provider_transport_runtime` and have rerun + stitch audit artifacts.
- [ ] No `unclassified` failures remain in release-candidate runs.
- [ ] Benchmark artifacts archived under `benchmark/runs/` with hashes, including stitch audit files.
