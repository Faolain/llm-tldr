# Agentic Benchmark Plan: Kimi With TLDRF vs Native Tools

- Status: Proposed
- Last updated: 2026-03-06

## Goal

Measure whether a smaller, cost-efficient answer model, starting with Kimi CLI, performs better on real repository work when equipped with `tldrf` than when limited to native lexical tools such as `rg`, `grep`, file reads, and tests.

The benchmark must answer the stronger version of the README claim:

- not just "better context can improve answers"
- but "a smaller model using `tldrf` as a tool solves repo work better, with fewer tokens, turns, and time, than the same model using native tools alone"

## Primary Hypotheses

1. `tldrf` improves final task outcomes for the same answer model by reducing search space and context pollution.
2. The best practical policy is not `tldrf`-only. It is:
   - `rg` first for exact lexical lookup
   - `tldrf` for structure, blast radius, slices, data flow, and concept lookup
3. The right evaluation is agentic and end-to-end:
   - task -> tool choice -> evidence gathering -> answer or patch -> test or oracle
   - not just retrieval metrics or prebuilt context packets

## What The Repo Already Proves

Existing benchmark infrastructure already measures a meaningful part of the value proposition:

- [benchmarks/README.md](../benchmarks/README.md)
- [scripts/bench_llm_ab_run.py](../scripts/bench_llm_ab_run.py)
- [scripts/bench_llm_ab_prompts.py](../scripts/bench_llm_ab_prompts.py)
- [benchmarks/llm/tasks.json](../benchmarks/llm/tasks.json)
- [benchmarks/llm/open_ended_tasks.json](../benchmarks/llm/open_ended_tasks.json)
- [benchmarks/head_to_head/README.md](../benchmarks/head_to_head/README.md)

Current proof surface:

- `tldrf` context can beat `rg` context on Django structural tasks.
- Open-ended judge runs already support "better context produces better answers."
- Deterministic retrieval, impact, slice, DFG, and token-efficiency families already exist.

Current missing proof:

- agent-chosen tool use
- full-run token and turn accounting across the whole task
- wall-clock time to completion
- solve rate and time-to-first-passing-test
- changed-file recall / precision

## Practical Model Stack

Preferred benchmark stack for the first real agentic run:

- Answer model / "model user": Kimi CLI using the existing logged-in local session
- Optional later fallback model: DeepSeek
- Judge: Claude Sonnet medium via the existing Claude Code CLI subscription

Important implementation constraint:

- today [scripts/bench_llm_ab_run.py](../scripts/bench_llm_ab_run.py) supports `codex`, `claude_sdk`, `claude_cli`, and `anthropic`
- when Claude is needed for this benchmark, standardize on `claude_cli`, not `claude_sdk`
- use the tested CLI shape `claude -p --model sonnet --effort medium <prompt>` as the base invocation
- using Kimi in the harness requires adding a Kimi CLI provider adapter or wrapper path first

Verification note:

- tested locally on 2026-03-06: `claude -p --model sonnet --effort medium "Reply with exactly: OK"` returned `OK`

Cost note:

- target cost is primarily subscription-backed for the answer model because Kimi CLI is already logged in locally
- incremental cost should mostly come from the Claude judge path and larger external benchmark sweeps

Model-comparison note:

- the first version of this benchmark does **not** compare Kimi against a stronger model
- it compares the same model in two arms:
  - Kimi with native tools only
  - Kimi with native tools plus `tldrf`
- once that baseline is stable, the same harness can be rerun later with additional models

Instruction-surface note:

- keep the harness / system prompt fixed across both benchmark arms
- do **not** treat system-prompt editing as a tuning lever for this benchmark
- tune exactly one canonical instruction document that is intended to graduate into normal everyday usage as well as eval usage
- explanatory docs may mirror that policy, but they are not independent tuning surfaces

## Benchmark Source Strategy

This plan intentionally uses two benchmark layers:

1. Local repo-native evaluation first
   - fastest iteration
   - reuses existing Django suites and scoring harnesses
   - proves the benchmark design before expensive external runs
2. External validation second
   - use SWE-bench Verified first
   - expand to larger SWE-bench runs only after the local harness and policy are stable

Why SWE-bench Verified is the right external anchor:

- real GitHub issue tasks
- strong fit for debugging and patching
- existing hidden-test style task framing
- directly supports comparing:
  - native tools only
  - native tools + `tldrf`

Useful external references for later phases:

- SWE-bench: https://github.com/SWE-bench/SWE-bench
- RepoBench: https://github.com/Leolty/RepoBench
- CodeEditorBench: https://github.com/DeepSoftwareAnalytics/CodeEditorBench
- Debug-gym: https://github.com/microsoft/debug-gym
- RefactorBench: https://refactorbench.github.io/

Lower priority:

- `ai-coding-lang-bench` is acceptable as a cheap model-capability smoke test, but it is not the right primary benchmark for `tldrf` because it is not centered on repository structure, retrieval policy, or deterministic graph-assisted workflows

## Benchmark Arms

Every agentic benchmark phase should compare at least these two arms:

- Baseline arm:
  - `rg`
  - `grep`
  - direct file reads
  - tests
- Augmented arm:
  - same native tools
  - plus `tldrf`
  - plus the single canonical instruction policy

Canonical instruction policy:

- one authoritative instruction document only
- for now, treat [AGENTS.md](../AGENTS.md) as the canonical source unless and until it is intentionally replaced by a dedicated `skill.md`
- [docs/llm-tldr-reference-card.md](../docs/llm-tldr-reference-card.md) remains explanatory and should mirror the policy, not diverge from it
- evals and normal usage should both consume the same canonical instruction policy

The augmented arm is not allowed to replace `rg` for exact lexical lookup.

## Metrics That Matter

Primary metrics:

- solve rate
- time to first passing test
- wall-clock completion time
- total input tokens
- total output tokens
- turn count
- tool-call count
- changed-file recall / precision
- test-selection accuracy
- tool-choice accuracy

Secondary metrics:

- judge score for open-ended explanations or plans
- answer quality on structured Django packet tasks
- localization accuracy
- transcript length and context growth

Judge policy:

- use deterministic tests or scripted oracles wherever possible
- use Claude Sonnet medium as judge only when deterministic scoring is not feasible

Pinned judge configuration (create `benchmarks/agentic/judge_config.json`):

```json
{
  "judge_provider": "claude_cli",
  "judge_model": "sonnet",
  "judge_effort": "medium",
  "judge_temperature": 0.0,
  "judge_max_tokens": 800,
  "judge_retries": 1,
  "enforce_json_schema": true
}
```

All benchmark scripts must read judge configuration from this file. If the model changes, it changes in one place. Every run report must include a `judge_config_hash` (SHA-256 of the judge config JSON). The phase gate script must compare the judge config hash between runs; if they differ, the gate fails. The Claude runtime path should invoke `claude -p --model sonnet --effort medium` plus any required harness flags (`--output-format`, `--json-schema`, tool disabling, and permission settings).

Metric collection notes:

- The existing `bench_llm_ab_run.py` harness is single-turn (one prompt, one answer). It cannot collect turn count, tool-call count, transcript data, or workflow compliance. Ten of the fourteen primary/secondary metrics require multi-turn instrumentation.
- `bench_agent_tasks.py` is a genuinely new multi-turn agent loop program, not a mode flag on the existing harness. It must manage conversation turns, capture structured transcripts, and post-process them.
- Token usage fallback: if Kimi CLI does not report token counts, estimate via `tldr.stats.count_tokens()` (tiktoken `cl100k_base`) applied to the raw transcript text.
- Wall-clock time is automatically captured by the existing harness pattern (`time.monotonic()` around each call).
- Token aggregation: the existing run reports do not aggregate total input/output tokens across a run. The new scripts must add `total_input_tokens` and `total_output_tokens` to the report summary.

Statistical significance:

- Add `scipy` as a dev dependency for significance testing.
- For solve rate comparisons (binary paired outcomes, n >= 20): use McNemar's test.
- For continuous paired metrics (F1, tokens, time): use Wilcoxon signed-rank test.
- At the pilot size of 20-50 tasks, acknowledge that statistical power is limited. Pilot results are directional; confirmatory power requires n >= 50.
- Phase gates should report p-values alongside point estimates but should not hard-gate on significance until Phase F (n >= 200).

## Phase Plan

### Phase A: Reuse Existing `llm_ab` Harness

Purpose:

- cheaply test whether better `tldrf` context helps Kimi relative to `rg`-only context

Work:

- rerun existing Django structured and open-ended suites from:
  - [benchmarks/llm/tasks.json](../benchmarks/llm/tasks.json)
  - [benchmarks/llm/open_ended_tasks.json](../benchmarks/llm/open_ended_tasks.json)
- run the same Kimi model in two conditions:
  - baseline context arm built from native-tool packets
  - augmented context arm built from `tldrf` packets
- keep the judge fixed as Claude Sonnet medium via `claude_cli`

Acceptance gate (`benchmarks/agentic/phase_a_gates.json`):

```json
{
  "win_rate_tldr_over_rg_min": 0.55,
  "f1_mean_tldr_min_delta": 0.05,
  "max_error_rate": 0.05,
  "min_tasks_completed": 20
}
```

Verification command:

```bash
uv run python scripts/bench_phase_gate.py \
  --report benchmark/runs/<ts>-llm-ab-run-structured.json \
  --gates benchmarks/agentic/phase_a_gates.json
```

Exit 0 = pass (proceed to Phase B). Exit 2 = fail (investigate before continuing).

### Phase B: Tool-Usage Preflight and Tool-Choice Evaluation

Purpose:

- validate that the agent actually uses `tldrf` correctly before any expensive or large-scale benchmark run
- test whether the agent uses the correct workflow before measuring patch success

Gold policy source:

- canonical instruction source: [AGENTS.md](../AGENTS.md)
- explanatory mirror only: [docs/llm-tldr-reference-card.md](../docs/llm-tldr-reference-card.md)

Required preflight gate before full runs:

- run a small live pilot on representative tasks before Phase C or later phases
- record full transcripts plus per-step tool calls
- automated transcript validation via `scripts/bench_preflight_validate.py` replaces manual review (see below)
- if the agent skips `tldrf` on tasks where it should use it, or misuses it repeatedly, stop and tune the instruction surface before continuing
- do not record or publish full Phase D / Phase E outcome numbers until this gate passes

Automated transcript validation (`scripts/bench_preflight_validate.py`):

- parses per-step tool calls from the transcript JSONL
- compares first tool call against the task's `expected_first_tool` field
- flags tasks where agent made > 3 consecutive calls without invoking any expected-workflow tool
- detects systematic failure: if > 2 tasks in the same workflow class fail identically, sets `systematic_failure_detected: true`
- emits structured JSON report with `systematic_failure_detected`, `failure_patterns`, and `recommendation` ("pass" or "tune_and_rerun")
- exit 0 on pass, exit 2 on failure

Initial pilot shape:

- 8-12 representative tasks
- at least 2 exact-lookup tasks where `rg-first` should win
- at least 2 concept lookup tasks where `semantic/hybrid` should appear
- at least 2 refactor tasks where `impact -> context -> rg` should appear
- at least 2 debugging tasks where `slice -> dfg` should appear

Initial labeled workflow classes:

- exact symbol / definition -> `rg-first`
- concept lookup -> `semantic/hybrid -> context`
- refactor blast radius -> `impact -> context -> rg`
- line-level debugging -> `slice -> dfg`
- repeated queries -> daemon or MCP

Each task in `benchmarks/agentic/preflight_tasks.json` must include machine-readable policy fields:

```json
{
  "id": "PF01",
  "question": "Where is the Django ORM query compilation entry point?",
  "workflow_class": "concept_lookup",
  "target_repo": "django",
  "expected_first_tool": "tldrf_semantic_search",
  "expected_tool_set": ["tldrf_semantic_search", "tldrf_context"],
  "forbidden_first_tool": ["rg"],
  "max_allowed_dead_end_turns": 1
}
```

"Skips tldrf" = the agent never invokes any tool in `expected_tool_set` for that task.
"Repeatedly" = 20% or more of tldrf-required tasks skip tldrf entirely.

Metrics:

- workflow-compliance rate
- correct-first-tool rate
- `tldrf`-usage rate on tasks where `tldrf` is expected
- `rg-first` compliance on exact lexical tasks
- tool-choice accuracy
- unnecessary tool-call rate
- dead-end-turn rate
- recovery-after-wrong-first-tool rate
- median turns before first appropriate tool use

If the preflight gate fails, tune in this order:

1. the single canonical instruction document
2. task examples / few-shot demonstrations that are bundled under that same instruction policy
3. tool wrappers or clearer tool descriptions if the model is still confused

Do not tune:

- the harness / system prompt as a benchmark-specific workaround
- multiple competing instruction documents at the same time

Failed preflight runs are invalidation signals, not product evidence:

- they mean the instruction surface or tool ergonomics are not ready
- they should trigger tuning and rerun, not be counted as wins/losses for `tldrf`

Provisional preflight acceptance gate (`benchmarks/agentic/preflight_gates.json`):

```json
{
  "correct_first_tool_min": 0.80,
  "workflow_compliance_min": 0.80,
  "tldrf_usage_on_required_min": 0.90,
  "rg_first_on_exact_min": 0.90,
  "median_dead_end_turns_max": 1,
  "systematic_failure_detected_must_be": false
}
```

Verification commands:

```bash
# Run the preflight suite
uv run python scripts/bench_tool_choice.py \
  --tasks benchmarks/agentic/preflight_tasks.json \
  --provider kimi_cli \
  --model <kimi-model-id> \
  --instruction-source AGENTS.md \
  --max-turns 20 \
  --timeout-s 300 \
  --trials 1 \
  --out benchmark/runs/<ts>-preflight.json

# Validate transcripts automatically (replaces manual review)
uv run python scripts/bench_preflight_validate.py \
  --report benchmark/runs/<ts>-preflight.json \
  --gates benchmarks/agentic/preflight_gates.json
```

Tuning iteration cap:

- maximum 3 tuning iterations before escalating to human review
- each iteration: run preflight, check gate, produce structured diagnostic identifying which workflow class fails and what tool the agent used instead
- if all 3 iterations fail, the diagnostic report must be reviewed before continuing (this is the only remaining human gate)

Failed preflight quarantine:

- the preflight runner tags output with `"preflight_status": "passed"` or `"failed"`
- Phase C/D/E runners must refuse to start if the most recent preflight report has `preflight_status: "failed"`

Acceptance:

- tool policy is stable enough that benchmark noise is not dominated by bad workflow selection
- instruction tuning happens before, not after, the first expensive full run
- failed preflight traces are excluded from benchmark result summaries

### Phase C: Deterministic Task Families That Reflect TLDRF Strengths

Purpose:

- prove where `tldrf` helps and where it should not be used

Priority task families:

- concept lookup when the symbol name is unknown
- impact-aware refactor planning
- line-level debugging and value provenance
- affected-test selection
- compound "find code and who calls it"
- exact identifier lookup tasks where `rg` should win

Corpora:

- start with Django
- then add `requests`
- then add `urllib3`

Acceptance gate (`benchmarks/agentic/phase_c_gates.json`):

- within each declared category (`rg-wins`, `tldrf-wins`, `mixed`), the expected winning arm must achieve `win_rate >= 0.70` within that category
- at least 5 tasks per category minimum
- overall error rate <= 0.10

```bash
uv run python scripts/bench_phase_gate.py \
  --report benchmark/runs/<ts>-phase-c.json \
  --gates benchmarks/agentic/phase_c_gates.json
```

### Phase D: Local End-to-End Patch/Test Tasks

Purpose:

- measure the real edit loop instead of only answer quality

Task shape:

- debugging fixes
- small refactors with blast radius
- multi-file change-impact tasks
- tests that should be selected or updated

Scoring:

- hidden tests or scripted oracles first
- judge only for non-deterministic outputs like explanations or refactor plans

Acceptance gate (`benchmarks/agentic/phase_d_gates.json`):

- `solve_rate_delta_min`: 0.0 (augmented must be at least equal)
- if solve rates are within 0.05 of each other, require at least one of:
  - `token_reduction_min_pct`: 15
  - `turn_reduction_min_pct`: 20
  - `time_reduction_min_pct`: 15
- `min_tasks_completed`: 10

```bash
uv run python scripts/bench_phase_gate.py \
  --report benchmark/runs/<ts>-phase-d.json \
  --gates benchmarks/agentic/phase_d_gates.json
```

### Phase E: SWE-bench Verified Pilot

Purpose:

- validate the local findings on a recognized external benchmark

Initial run:

- 20-50 Django-heavy SWE-bench Verified tasks
- baseline arm vs augmented arm
- Kimi CLI as the answer model
- Claude Sonnet medium as the judge only where a secondary qualitative read is useful

Record:

- success rate
- turns
- tokens
- time
- patch quality notes

Acceptance gate (`benchmarks/agentic/phase_e_gates.json`):

- `solve_rate_augmented >= solve_rate_baseline + 0.05` OR
- `(solve_rate_delta >= -0.02 AND token_reduction >= 20%)`
- if neither holds, the gate fails and Phase F does not proceed
- report p-values (McNemar's for solve rate, Wilcoxon for tokens/time) but do not hard-gate on significance at pilot size

```bash
uv run python scripts/bench_phase_gate.py \
  --report benchmark/runs/<ts>-phase-e.json \
  --gates benchmarks/agentic/phase_e_gates.json
```

### Phase F: External Scale-Out

Purpose:

- turn the pilot into a credible public result

Run:

- 200+ SWE-bench Verified tasks if pilot signal is positive
- optional follow-up on full SWE-bench
- optional later validation on RepoBench / RefactorBench / Debug-gym depending on where the local signal is strongest

Acceptance gate (`benchmarks/agentic/phase_f_gates.json`):

- results must hold across 3 independent runs with `min_pass_runs: 2` (replicating the stability semantics from `bench_h2h_assert.py`)
- McNemar's test p < 0.05 for solve rate OR Wilcoxon p < 0.05 for token/time reduction
- minimum 200 tasks completed

```bash
uv run python scripts/bench_phase_gate.py \
  --report benchmark/runs/<ts>-phase-f.json \
  --gates benchmarks/agentic/phase_f_gates.json
```

### Phase G: Optional Model-Matrix Expansion

Purpose:

- rerun the now-stable benchmark with additional answer models after the Kimi baseline is proven

Possible follow-ups:

- DeepSeek as a cheaper comparison model
- one stronger already-supported model in the current harness
- future Kimi revisions

Acceptance:

- model-to-model comparisons are built on top of a stable same-model baseline rather than mixed into the first benchmark pass

## Implementation Work Needed In This Repo

### Existing files to modify

- [scripts/bench_llm_ab_run.py](../scripts/bench_llm_ab_run.py)
  - add `_kimi_cli_call()` following `_claude_cli_call()` pattern (line ~521)
  - add `'kimi_cli'` to `--provider` choices (line ~668) and `--judge-provider` choices (line ~676)
  - add answer dispatcher branch (line ~939) and judge dispatcher branch (line ~1034)
  - add `--answer-retries N` parameter (default 1) for empty/malformed responses, analogous to `--judge-retries`
  - add `total_input_tokens` and `total_output_tokens` to report summary
- [benchmarks/README.md](../benchmarks/README.md)
  - publish the new benchmark family and results

### New scripts to create

- `scripts/bench_tool_choice.py`
  - multi-turn agent loop that runs tasks from a task JSON, captures structured transcripts
  - records per-turn tool calls, normalizes tool names to canonical vocabulary
  - emits JSONL transcripts + summary report JSON
  - arguments: `--tasks`, `--provider`, `--model`, `--instruction-source`, `--max-turns`, `--timeout-s`, `--trials`, `--out`
- `scripts/bench_agent_tasks.py`
  - end-to-end patch/test task runner
  - manages agent conversation, captures `git diff --name-only` after each run
  - computes changed-file recall/precision against gold patch
  - arguments: `--tasks`, `--provider`, `--model`, `--arm baseline|augmented`, `--max-turns`, `--timeout-s`, `--out`
  - circuit breaker: `--max-consecutive-errors 5`, `--max-error-rate-abort 0.30`
  - checkpoint/resume: `--resume <answers-jsonl>` to skip completed task IDs on restart
- `scripts/bench_preflight_validate.py`
  - reads preflight report + transcripts, performs automated pattern detection
  - replaces manual transcript review
  - arguments: `--report`, `--gates`
  - exit 0 on pass, exit 2 on failure
- `scripts/bench_phase_gate.py`
  - generic gate checker following `bench_h2h_assert.py` pattern (`_add_gate` / `_evaluate_run` / exit-code)
  - accepts `--report <path>` and `--gates <path>`
  - evaluates thresholds, writes diagnostic JSON with per-gate pass/fail
  - exit 0 on pass, exit 2 on failure
- `scripts/bench_agentic_orchestrate.py`
  - top-level orchestrator that chains: Phase A -> Gate A -> Phase B -> Gate B -> ... -> Phase F
  - each phase is a subprocess call; if any gate exits nonzero, orchestrator stops
  - writes summary JSON with `stopped_at_phase`, `reason`, and gate diagnostic path
  - arguments: `--start-from-phase A`, `--end-at-phase F`, `--kimi-model <id>`
  - optional `--notify-webhook URL` for completion/failure notifications
- `scripts/bench_curate_swebench.py`
  - filters SWE-bench Verified for `repo == "django/django"`, resolved tasks
  - sorts by instance_id for deterministic ordering, selects first N tasks
  - validates each task has a runnable test command
  - writes `benchmarks/agentic/swebench_subset.json`
  - arguments: `--source <swebench-path>`, `--count 30`, `--out`

### New data files to create

- `benchmarks/agentic/judge_config.json` — pinned judge model, temperature, provider
- `benchmarks/agentic/preflight_tasks.json` — 10 tasks (2 per workflow class), with machine-readable policy fields
- `benchmarks/agentic/tool_choice_tasks.json` — 30-50 tasks across all workflow classes (superset of preflight)
- `benchmarks/agentic/patch_tasks.json` — 10 tasks with `hidden_test_command`, `expected_test_result`, `expected_changed_files`, `max_turns`, `timeout_s`
- `benchmarks/agentic/swebench_subset.json` — 30 Django SWE-bench Verified tasks with `instance_id`, `repo`, `base_commit`, `patch`, `test_cmd`, `timeout_s`
- `benchmarks/agentic/phase_a_gates.json`
- `benchmarks/agentic/preflight_gates.json`
- `benchmarks/agentic/phase_c_gates.json`
- `benchmarks/agentic/phase_d_gates.json`
- `benchmarks/agentic/phase_e_gates.json`
- `benchmarks/agentic/phase_f_gates.json`

### Extended report schema for agentic runs

The report envelope (`bench_util.make_report()`) must additionally include:

- `arm`: `"baseline"` | `"augmented"`
- `instruction_surface`: filename + SHA-256 of the canonical instruction document
- `tools_available`: list of tool names available in this arm
- `task_suite_sha256`: hash of the task JSON for reproducibility
- `judge_config_hash`: SHA-256 of the judge config JSON

Each per-task result entry must additionally include:

- `transcript_path`: path to separate JSONL transcript file
- `solve_rate`: 0 or 1
- `turn_count`: int
- `tool_call_count`: int
- `tool_calls`: list of normalized tool names in order
- `changed_files`: list of file paths
- `wall_clock_s`: float
- `first_pass_time_s`: float (time to first passing test, if applicable)

### Kimi CLI provider adapter specification

```python
def _kimi_cli_call(
    *,
    model: str,
    prompt: str,
    timeout_s: float,
    json_schema: dict[str, Any] | None,
    env: dict[str, str] | None,
) -> tuple[str, dict[str, Any]]:
    """
    Invoke Kimi CLI via subprocess, following _claude_cli_call() pattern.
    - Pass prompt via stdin
    - Parse stdout for response text
    - Extract token counts if Kimi CLI reports them; otherwise return {}
    - Raise RuntimeError on non-zero exit, timeout, or auth expiry
    """
```

Session health: before each batch of N tasks, run a lightweight Kimi CLI no-op to verify the session is alive. If expired, emit structured error and stop (do not silently produce empty results).

### Canonical instruction surface to maintain

- [AGENTS.md](../AGENTS.md) during the initial benchmark program
- optional future replacement: a single dedicated `skill.md`
- if that replacement happens, it must become the sole source of truth for both eval and normal usage

## Immediate Next Steps

### Step 1: Implement Kimi CLI provider adapter

Add `_kimi_cli_call()` to `scripts/bench_llm_ab_run.py` following the `_claude_cli_call()` pattern at line ~521. Add `'kimi_cli'` to `--provider` choices at line ~668.

Verify:
```bash
uv run python scripts/bench_llm_ab_run.py \
  --provider kimi_cli --model <kimi-model-id> \
  --prompts benchmark/llm/<test-packet>.jsonl \
  --limit 1 --dry-run
```

### Step 2: Run smallest possible Phase A pilot

```bash
# Generate prompt packets (if not already available)
uv run python scripts/bench_llm_ab_prompts.py \
  --corpus django --budget-tokens 2000

# Structured tasks — Kimi with both context arms
uv run python scripts/bench_llm_ab_run.py \
  --prompts benchmark/llm/<ts>-llm-ab-django.jsonl \
  --provider kimi_cli \
  --model <kimi-model-id> \
  --mode structured --trials 3 --limit 5

# Check Phase A gate (smoke test)
uv run python scripts/bench_phase_gate.py \
  --report benchmark/runs/<ts>-llm-ab-run-structured.json \
  --gates benchmarks/agentic/phase_a_gates.json
```

### Step 3: Create preflight task suite

Create `benchmarks/agentic/preflight_tasks.json` with 10 tasks (2 per workflow class). Each task must include: `id`, `workflow_class`, `expected_first_tool`, `expected_tool_set`, `forbidden_first_tool`, `question`, `target_repo`, `max_allowed_dead_end_turns`. Draw from existing Django queries in `benchmarks/python/django_structural_queries.json` and `benchmarks/retrieval/django_queries.json`.

Optionally automate with:
```bash
uv run python scripts/bench_curate_preflight.py \
  --structural benchmarks/python/django_structural_queries.json \
  --retrieval benchmarks/retrieval/django_queries.json \
  --out benchmarks/agentic/preflight_tasks.json
```

### Step 4: Run preflight gate (max 3 tuning iterations)

```bash
uv run python scripts/bench_tool_choice.py \
  --tasks benchmarks/agentic/preflight_tasks.json \
  --provider kimi_cli \
  --model <kimi-model-id> \
  --instruction-source AGENTS.md \
  --max-turns 20 --timeout-s 300 --trials 1 \
  --out benchmark/runs/<ts>-preflight.json

uv run python scripts/bench_preflight_validate.py \
  --report benchmark/runs/<ts>-preflight.json \
  --gates benchmarks/agentic/preflight_gates.json
```

If gate fails: review diagnostic JSON, tune AGENTS.md, rerun. Maximum 3 iterations.

### Step 5: Create full tool-choice suite

Create `benchmarks/agentic/tool_choice_tasks.json` with 30-50 tasks across all workflow classes (superset of preflight). Same schema as preflight tasks. Rerun with same verification commands as step 4.

### Step 6: Create local patch/test tasks

Create `benchmarks/agentic/patch_tasks.json` with 10 tasks. Each must include: `id`, `category`, `repo`, `issue_description`, `hidden_test_command`, `expected_test_result`, `expected_changed_files`, `max_turns`, `timeout_s`. Selection criteria: every task must have a deterministic test command (not judge-only).

### Step 7: Select SWE-bench Verified tasks

```bash
uv run python scripts/bench_curate_swebench.py \
  --source <swebench-verified-path> \
  --repo django/django \
  --count 30 \
  --out benchmarks/agentic/swebench_subset.json
```

### Step 8: Create gate threshold files

Create all six gate JSON files under `benchmarks/agentic/` with the thresholds specified in each phase's acceptance section above.

### Step 9: Create orchestrator

Implement `scripts/bench_agentic_orchestrate.py` that chains all phases with gate checks. This is the single entry point for hands-off execution:

```bash
uv run python scripts/bench_agentic_orchestrate.py \
  --kimi-model <kimi-model-id> \
  --start-from-phase A \
  --end-at-phase F
```

## Running Log

- 2026-03-06: Created this plan to combine the existing local benchmark stack with a new agentic evaluation goal focused on Kimi-with-`tldrf` vs Kimi-without-`tldrf`.
- 2026-03-06: Locked the preferred initial model stack as:
  - answer model: Kimi CLI
  - optional later fallback model: DeepSeek
  - judge: Claude Sonnet medium
- 2026-03-06: Narrowed the first benchmark pass to a same-model comparison only:
  - Kimi with native tools
  - Kimi with native tools plus `tldrf`
- 2026-03-06: Deferred cross-model comparisons until after the first same-model baseline is working and trusted.
- 2026-03-06: Locked SWE-bench Verified as the preferred first external validation source, but only after local harness reuse and tool-choice validation.
- 2026-03-06: Added an explicit preflight gate before full agentic runs:
  - small live task set
  - transcript review
  - workflow-compliance metrics
  - instruction tuning if the agent does not use `tldrf` appropriately
- 2026-03-06: Locked the instruction-tuning policy:
  - keep harness/system prompts fixed
  - tune one canonical instruction document only
  - treat mirror docs as explanatory, not independent benchmark levers
- 2026-03-06: Autonomy review — applied 33 fixes across 10 categories to make the plan hands-off:
  - pinned judge config to exact Claude CLI settings (`claude_cli`, `sonnet`, `medium`) with temperature, provider, and hash enforcement
  - replaced all vague acceptance criteria with numeric phase gate thresholds in JSON files
  - replaced "manually review transcripts" with automated `bench_preflight_validate.py`
  - added machine-readable task schema with `expected_first_tool`, `workflow_class`, `forbidden_first_tool`
  - added `bench_phase_gate.py` (generic gate checker following `bench_h2h_assert.py` pattern)
  - added `bench_agentic_orchestrate.py` (top-level orchestrator with stop-on-failure)
  - added circuit breaker (`--max-consecutive-errors`, `--max-error-rate-abort`) to agent task runner
  - added checkpoint/resume (`--resume <answers-jsonl>`) for crash recovery
  - added session health check for Kimi CLI to handle auth expiry during long runs
  - added tuning iteration cap (max 3) before escalating to human review
  - added preflight quarantine: downstream phases refuse to start if preflight status is "failed"
  - added statistical significance testing plan (McNemar's + Wilcoxon, scipy dev dependency)
  - converted all 7 immediate next steps into 9 concrete steps with verbatim CLI commands
  - added Kimi CLI provider adapter specification with exact function signature
  - added extended report schema for agentic runs (arm, instruction hash, transcript paths)
  - noted CI blocker: Kimi session-based auth prevents GitHub Actions automation; orchestrator is local-only

## Gotchas / Learnings

- Existing `llm_ab` results are valuable, but they measure prebuilt context packets, not tool-using agents.
- A strong external benchmark does not remove the need for a local benchmark. Local suites are still the fastest way to tune tool policy, packing, and harness behavior.
- If the model does not know when to use `tldrf`, the benchmark mostly measures prompt quality, not tool utility. Full runs should be blocked until the preflight gate passes.
- System-prompt changes would contaminate the comparison. The benchmark should vary tool availability and one canonical instruction policy, not hidden harness behavior.
- The most realistic winning setup is expected to be policy-combined, not tool-exclusive:
  - `rg` for exact lookup
  - `tldrf` for structure and concept search
- Kimi CLI avoids forcing an OpenRouter dependency into the first benchmark pass, which keeps the initial harness closer to the real local workflow we actually want to evaluate.
- Judge configuration should be pinned exactly once selected. "Claude Sonnet medium" should not drift between pilot and scale-out runs. Use `benchmarks/agentic/judge_config.json` with provider `claude_cli`, model `sonnet`, and effort `medium`, and invoke Claude via `claude -p`.
- The existing `bench_llm_ab_run.py` is fundamentally single-turn (one prompt, one answer). It cannot collect turn count, tool-call count, or transcript data. `bench_agent_tasks.py` is a genuinely new multi-turn program, not a mode flag.
- Token usage is inconsistent across providers: `codex` returns `{}`, `claude_cli` returns full usage. For Kimi, include a tiktoken fallback estimation path.
- No existing statistical significance testing exists in the codebase. `scipy` must be added as a dev dependency. At pilot sizes (20-50 tasks), results are directional, not confirmatory.
- Kimi CLI uses a logged-in local session. Session tokens can expire during multi-hour runs (Phase E/F). The provider adapter must include a session health check before each batch.
- The orchestrator is the single most important file for making the benchmark autonomous. Without `bench_agentic_orchestrate.py`, "hands-off" execution is impossible.
- Corpus expansion to `requests` and `urllib3` (Phase C) requires adding entries to `benchmarks/corpora.json` with pinned git refs. The existing `scripts/bench_fetch_corpora.py` handles fetching.
- CI automation is blocked by Kimi CLI's session-based auth (no injectable API key). Accept that the orchestrator is local-only unless Kimi provides token-based auth injection.
