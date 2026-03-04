# Spec 007: Retrieval Optimization and Reliability Hardening

- Status: Proposed
- Owner: TBD
- Last updated: 2026-03-01
- Type: Tactical feeder spec
- Normative keywords: `MUST`, `MUST NOT`, `SHOULD`, `MAY`

## 1. Purpose and Role

Spec 007 defines tactical work that improves retrieval quality and benchmark reliability so Spec 008 release gates can be passed reproducibly.
Spec 007 focuses on three areas:

1. Task integrity for benchmark and open-ended task mapping.
2. Output integrity via typed counter instrumentation and parse diagnostics.
3. Open-ended context rendering quality for `slice` and `data_flow`.

Spec 007 is an enabler. It is not the release decision authority.

## 2. Non-Goals and Non-Authority

Spec 007 `MUST NOT`:

1. Change Spec 008 gate thresholds, winner rule, budgets, tokenizer, corpus pinning, or anti-bias controls.
2. Declare `llm-tldr` head-to-head victory.
3. Replace capability gating with proxy substitutions.
4. Introduce per-query hand-tuning after observing outcomes.

Any ship/no-ship decision remains under Spec 008 authority.

## 2.1 Delivery Method: Test-First, Then Benchmark

Spec 007 work `MUST` be delivered in test-first order:

1. Write failing tests that encode the invariant or contract being changed.
2. Implement the minimum code change to pass those tests.
3. Run benchmark/judge workflows only after invariant tests pass.

Benchmark outcomes are acceptance evidence, not a substitute for invariant tests.

## 3. Scope of Work

### 3.1 Task Integrity

1. Enforce open-ended task to structural query alignment (including OE08-class mismatch prevention).
2. Validate category and anchor consistency across task definitions.
3. Require zero `materialize-tasks` warnings for benchmark manifests.

### 3.2 Output Counter Reliability

1. Add typed counters for empty vs malformed outputs in predictor/judge paths.
2. Add typed parse/result diagnostics in scorer paths.
3. Preserve backward compatibility with legacy aggregate keys.

### 3.3 Open-Ended Context Rendering

1. Improve `slice` and `data_flow` rendering using contiguous windows and branch/body bridging.
2. Emit deterministic context metadata (`strategy`, coverage/truncation fields, included lines).
3. Enforce deterministic budget arbitration and drop order; no hard budget overruns.

### 3.4 Test-First Backlog (Write Before Implementation)

1. Task integrity (maps primarily to 008 Phase 0)
- `tests/test_bench_llm_open_ended_tasks_schema.py::test_open_ended_task_query_alignment_and_anchor_consistency`
- `tests/test_bench_llm_open_ended_tasks_schema.py::test_oe08_regression_guard_maps_to_b10_configure`
- `tests/test_bench_head_to_head_materialize_tasks.py::test_materialize_tasks_valid_fixture_has_zero_warnings_and_stable_hash` (new)
- `tests/test_bench_head_to_head_tool_profiles_schema.py::test_contextplus_profile_is_real_contract_input`

2. Output integrity (maps primarily to 008 Phases 1, 2, and 6)
- `tests/test_bench_llm_ab_run_helpers.py::test_structured_failure_classes_mutually_exclusive`
- `tests/test_bench_llm_ab_run_helpers.py::test_structured_bad_json_reconciliation`
- `tests/test_bench_llm_ab_run_helpers.py::test_judge_bad_json_reconciliation`
- `tests/test_bench_head_to_head_score_counters.py::test_score_emits_typed_parse_diagnostics_without_gate_math_drift` (new)

3. Open-ended rendering integrity (maps primarily to 008 Phases 4, 5, and 6)
- `tests/test_bench_llm_ab_prompts_slice_packing.py::test_slice_open_ended_context_metadata_contract_and_determinism`
- `tests/test_bench_llm_ab_prompts_data_flow_packing.py::test_data_flow_budget_hard_cap_and_deterministic_drop_order` (new)

## 4. Required Invariants and Acceptance Signals

All invariants below are mandatory for 007 readiness and feed 008 phases, but do not replace 008 gates.

| Area | Required invariants (`MUST`) | Acceptance signals (readiness) | Feeds 008 |
|---|---|---|---|
| Task integrity | Open-ended tasks map to valid structural queries with matching category and anchors; `materialize-tasks` warnings are empty | Integrity/schema tests pass; `h2h-task-manifest.json` has `warnings == []` | Contract freeze |
| Output counters | Counter classes are mutually exclusive (`empty`, `malformed`, `transport/status`); `bad_json = empty_output_total + malformed_output_total`; `judge_bad_json = judge_empty_verdict_total + judge_malformed_verdict_total` | Counter tests pass; score/run reports include typed counters and reconciliation invariants | Run-validity diagnostics |
| Open-ended rendering | Context packets include deterministic rendering metadata; budget limits are never exceeded; rendering order is deterministic | Full open-ended judge run is valid with `tasks_judged == task_count`, `answer_errors_total == 0`, `judge_errors_total == 0`, `judge_bad_json == 0`, overall win rate `>= 0.65`, category means: `impact >= 0.65`, `slice >= 0.55`, `data_flow >= 0.60`, sustained in `>= 2` independent reruns | Context-quality hardening |

Readiness signals are tactical quality checks only. Final release pass/fail remains Spec 008 gates.

## 5. Cross-Document Authority Model

Precedence order for conflicts:

1. Canonical 008 implementation authority: `implementations/008-beat-contextplus_IMPLEMENTATION_PLAN.md`.
2. Canonical benchmark contract authority: `specs/008-head-to-head-benchmark-llm-tldr-vs-contextplus.md` and `benchmarks/head_to_head/suite.v1.json`.
3. Spec 007 (this document): tactical optimization and reliability hardening constraints.
4. `implementations/008-head-to-head-benchmark-llm-tldr-vs-contextplus_SUPPORTING_PLAN.md`: supporting/legacy execution reference unless merged into canonical 008 authority.

If guidance conflicts, higher-precedence documents govern.

## 6. Interface/Contract Expectations for 007 Outputs Consumed by 008

007 deliverables `MUST` be machine-consumable and stable:

1. `007.task_integrity.v1`
- Artifact: `benchmark/runs/h2h-task-manifest.json`
- Contract: deterministic task manifest plus explicit `warnings` array.
- Rule: non-empty warnings mark the run non-comparable for 008 input freeze steps.

2. `007.output_integrity.v1`
- Artifacts: `bench_llm_ab_run` and `bench_head_to_head score` reports.
- Contract keys: typed counters for empty/malformed classes and legacy compatibility keys.
- Rule: invariant violations fail 007 readiness and block 008 diagnostic confidence.

3. `007.open_ended_context.v1`
- Artifacts: prompt packet/report and judge run report under `benchmark/llm/` and `benchmark/runs/`.
- Contract: context metadata present, deterministic packing behavior, tactical thresholds reported.
- Rule: missing metadata or budget violations fail readiness; no silent fallback.

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Overfitting to one corpus/task wording | Local gains do not generalize | Keep deterministic rendering rules, avoid manual per-query tuning, validate on repeated runs |
| Token/latency regression from richer context | Weak efficiency and instability | Enforce hard budget limits and deterministic drop order |
| Counter-schema drift | Loss of diagnostic comparability | Keep legacy aggregates and enforce reconciliation tests |
| Authority confusion between 007 and 008 | Incorrect release claims | Enforce precedence model in docs and CI checks |
| Interrupted long-running evaluations | Incomplete evidence bundle | Use durable execution (`tmux`) and persist logs/artifacts per run |

## 8. Exit Criteria for 007 Readiness

Spec 007 is ready when all conditions are true:

1. Task-integrity tests pass and mismatch count is zero.
2. `validate-suite` and `materialize-tasks` pass with zero warnings.
3. Typed output counters exist and reconciliation invariants pass.
4. Open-ended context rendering tests pass with no budget violations.
5. Open-ended judge tactical thresholds are met and sustained in at least two independent reruns.
6. Required artifacts are produced under `benchmark/runs/` and `benchmark/llm/` with reproducible identifiers.
7. No 007 change modifies 008 gate math, winner logic, or benchmark contract.
8. Handoff package for 008 tactical inputs is complete and auditable.
