# Spec 007: Retrieval Optimization Implementation Plan (Feeds Spec 008 Tactical Inputs)

- Status: Planned
- Owner: TBD
- Last updated: 2026-03-01
- Canonical 008 implementation authority: [implementations/008-beat-contextplus_IMPLEMENTATION_PLAN.md](./008-beat-contextplus_IMPLEMENTATION_PLAN.md).
- Benchmark contract authority: [specs/008-head-to-head-benchmark-llm-tldr-vs-contextplus.md](../specs/008-head-to-head-benchmark-llm-tldr-vs-contextplus.md) and `benchmarks/head_to_head/suite.v1.json`.
- Supporting/legacy 008 execution reference: [implementations/008-head-to-head-benchmark-llm-tldr-vs-contextplus_SUPPORTING_PLAN.md](./008-head-to-head-benchmark-llm-tldr-vs-contextplus_SUPPORTING_PLAN.md).
- Role of this spec: tactical implementation plan that hardens input integrity, output integrity, and open-ended context quality so Spec 008 gates can be passed reproducibly.

## Objective

Turn 007 from diagnostics into execution by implementing and validating:

1. Task/suite integrity fixes first (including OE08-class mismatch prevention).
2. Typed empty-vs-malformed counters in predictor/scorer reporting.
3. Better context rendering for open-ended `slice` and `data_flow` tasks (contiguous windows, branch/body bridging, explanation scaffold).
4. Handoff artifacts that feed Spec 008 tactical inputs (contract integrity, reliability diagnostics, and context-quality improvements) without changing 008 gate math.

## Scope

In scope:

1. Open-ended task integrity/schema checks.
2. `scripts/bench_llm_ab_run.py` reliability instrumentation.
3. `scripts/bench_head_to_head.py` parse/result diagnostics.
4. `scripts/bench_llm_ab_prompts.py` context packaging improvements for `slice` and `data_flow`.
5. Verifiable acceptance criteria with commands.

Out of scope:

1. Changing Spec 008 gate thresholds/winner rule.
2. Changing corpus pinning, budgets, tokenizer, or anti-bias controls defined in 008.
3. Declaring final release victory (final verdict stays in 008 compare + strict gates).

## Non-Goals

1. No per-task prompt hand-tuning after seeing outcomes.
2. No proxy substitutions for unsupported tool capabilities.
3. No unrelated benchmark framework rewrites.
4. No git history operations in this plan.

## Alignment With The 007 vs 008 Comparison

This plan explicitly incorporates the prior comparison summary:

1. Overlap retained: retrieval outcomes, token-efficiency outcomes, benchmark validity.
2. Canonical 008 remains governance/release-gate source of truth: DoD, strict deltas, rerun stability, final winner rule.
3. 007 is tactical depth: root-cause fixes, context-packing tactics, task-integrity and output-quality hardening.
4. Tensions resolved in order:
   - Fix suite/task integrity first (OE08-class issue).
   - Add empty/malformed counters in predictor/scorer.
   - Implement richer context-rendering tactics with budget controls.
   - Use 008 gates for final ship/no-ship judgment.

## Baseline Issues

| Issue | Evidence | Why it matters |
|---|---|---|
| Historical OE08-class task-integrity issue (now fixed) | Prior analysis found a mismatch; current task maps `OE08 -> B10` and references `configure` in `django/conf/__init__.py`. Keep this as a permanent regression guard. | One invalid task can distort category outcomes and undermine fairness claims. |
| Open-ended underperformance was mainly packaging | [benchmark/runs/20260210-053918Z-llm-ab-run-judge.json](../benchmark/runs/20260210-053918Z-llm-ab-run-judge.json): overall `0.3056`; derived per-category means from `per_task`: impact `0.6667`, slice `0.2083`, data_flow `0.0417`; `judge_bad_json=6`. | Signals were useful, but sparse/non-contiguous context hurt narrative answers. |
| Packaging changes can materially improve judge results | [benchmark/runs/20260210-205924Z-llm-ab-run-judge-open-ended-t3.json](../benchmark/runs/20260210-205924Z-llm-ab-run-judge-open-ended-t3.json): overall `0.6944`; derived means: impact `0.7778`, slice `0.5952`, data_flow `0.7333`; `judge_bad_json=0`. | Confirms tactical rendering/integrity work is high leverage. |
| Output-integrity counters are coarse | `bench_llm_ab_run.py` tracks `bad_json`/`judge_bad_json` but not explicit empty vs malformed splits. | Root-cause diagnosis and regressions are harder to isolate. |
| Scorer parse diagnostics are not fully typed by failure class | `bench_head_to_head.py` has status and parse errors but lacks dedicated empty/malformed payload class accounting. | Harder to distinguish transport/status failures from payload-shape failures. |

## Phased Implementation (Execution-Ready)

This plan provides tactical inputs to 008 governance gates. It does not define release-gate semantics.

### Phase 0: Suite/Task Integrity First

Implementation tasks:

1. Make OE08-style alignment a testable invariant.
2. Extend [tests/test_bench_llm_open_ended_tasks_schema.py](../tests/test_bench_llm_open_ended_tasks_schema.py) with new invariants:
   - Question references expected function/file for mapped query.
   - Category anchors exist (for example `target_line` for `slice`, variable anchor for `data_flow`).
   - Note: `query_id` existence and category matching are already enforced.
3. Keep [benchmarks/llm/open_ended_tasks.json](../benchmarks/llm/open_ended_tasks.json) synchronized with [benchmarks/python/django_structural_queries.json](../benchmarks/python/django_structural_queries.json).
4. Require `materialize-tasks` warnings to remain zero.

Acceptance criteria:

1. `open_ended_tasks` schema/integrity tests pass with zero mismatches.
2. `validate-suite` passes for all participating tool profiles.
3. `materialize-tasks` produces zero warnings.
4. `benchmarks/head_to_head/tool_profiles/contextplus.v1.json` exists (created per canonical 008 Phase 0 contract freeze).

Verification:

```bash
set -euo pipefail

uv run python scripts/bench_fetch_corpora.py --corpus django

uv run pytest \
  tests/test_bench_llm_open_ended_tasks_schema.py \
  tests/test_bench_django_structural_queries_schema.py

uv run python scripts/bench_head_to_head.py validate-suite \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/contextplus.v1.json

uv run python scripts/bench_head_to_head.py materialize-tasks \
  --suite benchmarks/head_to_head/suite.v1.json \
  --corpus-root benchmark/corpora/django \
  --out benchmark/runs/h2h-task-manifest.json

uv run python - <<'PY'
import json
d = json.load(open("benchmark/runs/h2h-task-manifest.json"))
warnings = d.get("warnings")
assert isinstance(warnings, list), type(warnings)
assert warnings == [], warnings
print("PASS: materialized task warnings = 0")
PY
```

### Phase 3: Predictor/Scorer Output Integrity Counters (Feeds 008 Phase 3)

Implementation tasks:

1. Extend [scripts/bench_llm_ab_run.py](../scripts/bench_llm_ab_run.py) reporting with explicit counters:
   - Structured mode: `empty_output_total`, `malformed_output_total`, plus per-source splits.
   - Judge mode: `judge_empty_verdict_total`, `judge_malformed_verdict_total`.
2. Preserve backward compatibility:
   - `bad_json = empty_output_total + malformed_output_total`
   - `judge_bad_json = judge_empty_verdict_total + judge_malformed_verdict_total`
3. Extend [scripts/bench_head_to_head.py](../scripts/bench_head_to_head.py) scoring diagnostics with typed parse/result counters (for example: empty result object, non-object result, category-shape mismatch).
4. Add/update tests:
   - [tests/test_bench_llm_ab_run_helpers.py](../tests/test_bench_llm_ab_run_helpers.py) for invariants.
   - New `tests/test_bench_head_to_head_score_counters.py` for scorer classification.
5. Add malformed fixture coverage:
   - New `tests/fixtures/head_to_head/predictions.malformed.json` with at least: empty result object, non-object result, category-shape mismatch.

Classification policy (mutually exclusive):

- `empty`: payload/verdict is blank after trim.
- `malformed`: non-empty payload that fails JSON parse or required-shape validation.
- `transport/status failure`: timeout/provider/runner exception (tracked separately; never counted as empty/malformed).

Acceptance criteria:

1. New counter keys are present in run reports.
2. Counter invariants reconcile exactly with legacy `bad_json` fields.
3. Counter-focused tests pass.

Verification:

```bash
set -euo pipefail

uv run pytest \
  tests/test_bench_llm_ab_run_helpers.py \
  tests/test_bench_head_to_head_suite_schema.py \
  tests/test_bench_head_to_head_tool_profiles_schema.py \
  tests/test_bench_head_to_head_score_counters.py

test -f tests/fixtures/head_to_head/predictions.malformed.json

uv run python scripts/bench_head_to_head.py score \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --predictions tests/fixtures/head_to_head/predictions.malformed.json \
  --out benchmark/runs/h2h-score-malformed-smoke.json

uv run python - <<'PY'
import json
report = json.load(open("benchmark/runs/h2h-score-malformed-smoke.json"))
status_counts = report.get("status_counts", {})
assert isinstance(status_counts, dict), type(status_counts)
assert status_counts.get("error", 0) > 0 or report.get("parse_errors"), "expected malformed payload classification"
print("PASS: malformed payload classification is enforced")
PY
```

### Phase 4: Context Rendering Tactics (Feeds 008 Phase 4)

Prerequisites (must pass before running Phase 4 judge verification):

1. `validate-suite` passes for all participating tool profiles.
2. `materialize-tasks` passes with `warnings == []`.
3. Corpus pin matches suite SHA (`c04a09ddb3bb1fe8157292fcd902b35cad9a5e10`).
4. Comparable provider/model stack is pinned:
   - answer: `codex / gpt-5.3-codex`
   - judge: `claude_sdk / claude-sonnet-4-5-20250929`
5. Verification blocks use strict shell mode and do not mask failures.

Implementation tasks:

1. Treat contiguous slice packing as an existing baseline in `scripts/bench_llm_ab_prompts.py`; preserve current behavior with regression tests (no reimplementation).
2. Extend open-ended `data_flow` packaging to parity with slice-style contiguous windows plus branch/body bridging.
3. Define prompt-contract metadata semantics:
   - required keys when code is emitted: `strategy`, `included_lines`, `function_span_lines`, `truncated`
   - `semantic_roles`: list of `{line:int, role:str, rationale:str}` with role in `input|predicate|transform|use|return`
4. Define deterministic budget arbitration and drop order:
   - keep order: anchor/target window -> bridge lines -> def/use windows -> related definitions -> extra windows
   - if over budget, drop in reverse order until within budget
5. Extend tests:
   - [tests/test_bench_llm_ab_prompts_slice_packing.py](../tests/test_bench_llm_ab_prompts_slice_packing.py) regression baseline
   - new data-flow bridging tests
   - new prompt-contract/role-schema tests

Acceptance criteria:

1. Prompt-packet tests pass for windowing, bridging, and scaffold presence.
2. Generated contexts include strategy/scaffold/coverage metadata where applicable.
3. No budget overruns in generated payloads.
4. Open-ended judge run at budget `2000` is valid and meets tactical thresholds on the full open-ended task set:
   - `tasks_judged == open_ended_tasks_count`
   - per-category counts in `results.per_task` match task-file category counts
   - `answer_errors_total == 0`
   - `judge_errors_total == 0`
   - `judge_bad_json == 0`
   - overall `win_rate_tldr_over_rg >= 0.65`
   - impact mean `>= 0.65`
   - slice mean `>= 0.55`
   - data_flow mean `>= 0.60`
5. Sustained check: criterion (4) passes in at least 2 independent reruns with identical config.

Important: Phase 4 judge-based tactical thresholds are non-gating and MUST NOT be used as a substitute for Spec 008 common-lane winner gates.

Verification:

```bash
set -euo pipefail

uv run pytest \
  tests/test_bench_llm_ab_prompts_slice_packing.py \
  tests/test_bench_llm_ab_run_helpers.py

TS="$(date -u +%Y%m%d-%H%M%SZ)"
SEED=0
ANSWER_PROVIDER="codex"
ANSWER_MODEL="gpt-5.3-codex"
JUDGE_PROVIDER="claude_sdk"
JUDGE_MODEL="claude-sonnet-4-5-20250929"

PROMPTS="benchmark/llm/${TS}-llm-ab-django-open-ended.jsonl"
PROMPTS_REPORT="benchmark/runs/${TS}-llm-ab-prompts-django-open-ended.json"
JUDGE_REPORT="benchmark/runs/${TS}-llm-ab-run-judge-open-ended.json"

uv run python scripts/bench_llm_ab_prompts.py \
  --corpus django \
  --tasks benchmarks/llm/open_ended_tasks.json \
  --structural-queries benchmarks/python/django_structural_queries.json \
  --retrieval-queries benchmarks/retrieval/django_queries.json \
  --budget-tokens 2000 \
  --seed "${SEED}" \
  --prompts-out "${PROMPTS}" \
  --out "${PROMPTS_REPORT}"

uv run python scripts/bench_llm_ab_run.py \
  --prompts "${PROMPTS}" \
  --mode judge \
  --provider "${ANSWER_PROVIDER}" \
  --model "${ANSWER_MODEL}" \
  --judge-provider "${JUDGE_PROVIDER}" \
  --judge-model "${JUDGE_MODEL}" \
  --trials 3 \
  --timeout-s 180 \
  --judge-timeout-s 180 \
  --enforce-json-schema \
  --out "${JUDGE_REPORT}"

uv run python - "${JUDGE_REPORT}" "benchmarks/llm/open_ended_tasks.json" <<'PY'
import collections
import json
import sys

run = json.load(open(sys.argv[1]))
tasks_doc = json.load(open(sys.argv[2]))
tasks = tasks_doc.get("tasks", tasks_doc if isinstance(tasks_doc, list) else [])
expected = [t for t in tasks if t.get("task_type") == "open_ended"]
expected_counts = collections.Counter(t.get("category") for t in expected)

results = run.get("results", {})
per_task = results.get("per_task", [])
got_counts = collections.Counter(t.get("category") for t in per_task if isinstance(t, dict))

assert results.get("tasks_judged") == len(expected), (results.get("tasks_judged"), len(expected))
assert len(per_task) == len(expected), (len(per_task), len(expected))
assert got_counts == expected_counts, (got_counts, expected_counts)
assert results.get("answer_errors_total") == 0, results.get("answer_errors_total")
assert results.get("judge_errors_total") == 0, results.get("judge_errors_total")
assert results.get("judge_bad_json") == 0, results.get("judge_bad_json")
assert float(results.get("win_rate_tldr_over_rg") or 0.0) >= 0.65, results.get("win_rate_tldr_over_rg")

by = {"impact": [], "slice": [], "data_flow": []}
for t in per_task:
    c = t.get("category")
    if c in by:
        by[c].append(float(t.get("win_mean_tldr_over_rg", 0.0)))
means = {k: (sum(v) / len(v) if v else None) for k, v in by.items()}

assert means["impact"] is not None and means["impact"] >= 0.65, means
assert means["slice"] is not None and means["slice"] >= 0.55, means
assert means["data_flow"] is not None and means["data_flow"] >= 0.60, means
print("PASS", {"overall": results.get("win_rate_tldr_over_rg"), "means": means, "counts": dict(got_counts)})
PY
```

## Tactical Metrics

Final ship/no-ship remains Spec 008. These metrics track readiness and tactical progress for 007.

| Metric | Baseline snapshot | Tactical target |
|---|---|---|
| Open-ended judge win rate (overall) | `0.3056` (`20260210-053918`) | `>= 0.65` sustained |
| Impact mean (derived from `per_task`) | `0.6667` | `>= 0.65` (no regression) |
| Slice mean (derived from `per_task`) | `0.2083` | `>= 0.55` |
| Data-flow mean (derived from `per_task`) | `0.0417` | `>= 0.60` |
| `judge_bad_json` | `6` in baseline, `0` in later run | `0` |
| Open-ended task integrity mismatches | Historical OE08-class issue | `0` mismatches |
| Typed parse/output counters | Missing/coarse | Present and invariant-checked |
| Judge stack comparability | implicit | exact match to baseline stack, or run labeled non-comparable |

## Risks And Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Overfitting to Django phrasing | Local gains may not generalize | Keep deterministic rendering rules; validate on additional subsets after stabilization. |
| Token inflation from richer windows | Budget/latency regressions | Enforce strict packing and deterministic drop order. |
| Counter drift across report formats | Inconsistent diagnostics | Preserve legacy keys and enforce invariant tests. |
| Confusing tactical thresholds with release gates | Incorrect ship/no-ship decisions | Keep precedence explicit: 008 gates decide release. |
| Long-running judge runs interrupted | Lost artifacts | Use `tmux` + log tee for runs >5 minutes (example command below). |

Long-run command pattern (when needed):

```bash
tmux new-session -d -s phase4-judge \
  'cd /Users/aristotle/Documents/Projects/llm-tldr && \
   PYTHONUNBUFFERED=1 NO_COLOR=1 uv run python scripts/bench_llm_ab_run.py ... 2>&1 | tee benchmark/runs/phase4-judge.log'
```

## Validation Checklist

1. [ ] Task integrity tests pass (`open_ended_tasks` + structural schema).
2. [ ] `validate-suite` and `materialize-tasks` pass with zero warnings.
3. [ ] Typed predictor/scorer counters are implemented and tested.
4. [ ] Invariants hold (`legacy == empty + malformed` forms).
5. [ ] Context rendering tests pass for contiguous windows + bridging + scaffold.
6. [ ] Budget constraints hold in generated prompt payloads.
7. [ ] Open-ended judge run at `2000` meets full tactical thresholds.
8. [ ] Artifacts are timestamped (or deterministically named) under `benchmark/runs/` and `benchmark/llm/`.
9. [ ] No edits in 007 altered 008 gates or winner logic.

## Handoff To Spec 008

007 is a feeder plan; 008 is release-gate authority.

| 007 output | Consumed by 008 capability | Gate relevance |
|---|---|---|
| Task integrity checks + zero-warning manifest | Contract/input freeze | Protects fairness and input validity. |
| Empty-vs-malformed counter instrumentation | Run-validity diagnostics | Improves reliability diagnosis for validity gates. |
| Context rendering and scaffold improvements | Retrieval/context quality hardening | Improves quality under constrained budgets without changing gate math. |

Mandatory release-gate commands and final pass/fail decisions remain those in Spec 008.
