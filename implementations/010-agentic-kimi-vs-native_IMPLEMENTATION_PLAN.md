# Agentic Benchmark Plan: Kimi With TLDRF vs Native Tools

- Status: Proposed
- Owner: TBD
- Last updated: 2026-03-06

## Goal

Measure whether a smaller, cost-efficient answer model, starting with Kimi on OpenRouter, performs better on real repository work when equipped with `tldrf` than when limited to native lexical tools such as `rg`, `grep`, file reads, and tests.

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

- Answer model / "model user": Kimi on OpenRouter
- Fallback cheap answer model: DeepSeek on OpenRouter
- Judge: Claude Sonnet via existing Claude subscription or API

Important implementation constraint:

- today [scripts/bench_llm_ab_run.py](../scripts/bench_llm_ab_run.py) supports `codex`, `claude_sdk`, `claude_cli`, and `anthropic`
- using Kimi or DeepSeek in the harness requires adding an OpenAI-compatible / OpenRouter provider adapter first

Cost note:

- target budget is low double digits for pilot runs
- exact spend must be verified against current provider pricing before full sweeps

Model-comparison note:

- the first version of this benchmark does **not** compare Kimi against a stronger model
- it compares the same model in two arms:
  - Kimi with native tools only
  - Kimi with native tools plus `tldrf`
- once that baseline is stable, the same harness can be rerun later with additional models

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
  - plus the repo usage policy from [AGENTS.md](../AGENTS.md) and [docs/llm-tldr-reference-card.md](../docs/llm-tldr-reference-card.md)

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
- use Claude Sonnet as judge only when deterministic scoring is not feasible

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
- keep the judge fixed and stronger

Acceptance:

- we can quantify whether `tldrf`-over-`rg` uplift exists for Kimi before adding more models

### Phase B: Tool-Usage Preflight and Tool-Choice Evaluation

Purpose:

- validate that the agent actually uses `tldrf` correctly before any expensive or large-scale benchmark run
- test whether the agent uses the correct workflow before measuring patch success

Gold policy source:

- [AGENTS.md](../AGENTS.md)
- [docs/llm-tldr-reference-card.md](../docs/llm-tldr-reference-card.md)

Required preflight gate before full runs:

- run a small live pilot on representative tasks before Phase C or later phases
- record full transcripts plus per-step tool calls
- manually review a small sample before trusting the aggregate score
- if the agent skips `tldrf` on tasks where it should use it, or misuses it repeatedly, stop and tune the instruction surface before continuing
- do not record or publish full Phase D / Phase E outcome numbers until this gate passes

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

1. benchmark instructions / system prompt
2. future `skill.md` or `AGENTS.md` guidance
3. task examples / few-shot demonstrations
4. tool wrappers or clearer tool descriptions if the model is still confused

Failed preflight runs are invalidation signals, not product evidence:

- they mean the instruction surface or tool ergonomics are not ready
- they should trigger tuning and rerun, not be counted as wins/losses for `tldrf`

Provisional preflight acceptance gate:

- correct-first-tool rate >= 0.80
- workflow-compliance rate >= 0.80
- `tldrf`-usage rate >= 0.90 on `tldrf`-required tasks
- `rg-first` compliance >= 0.90 on exact-lookup tasks
- median dead-end turns <= 1
- manual transcript review confirms no obvious repeated failure mode

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

Acceptance:

- task categories clearly separate:
  - `rg`-wins tasks
  - `tldrf`-wins tasks
  - mixed-policy tasks where the combo is strongest

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

Acceptance:

- augmented arm shows better solve rate, or equal solve rate with materially lower time/tokens/turns

### Phase E: SWE-bench Verified Pilot

Purpose:

- validate the local findings on a recognized external benchmark

Initial run:

- 20-50 Django-heavy SWE-bench Verified tasks
- baseline arm vs augmented arm
- Kimi as the answer model
- Claude Sonnet as the judge only where a secondary qualitative read is useful

Record:

- success rate
- turns
- tokens
- time
- patch quality notes

Acceptance:

- the pilot is strong enough to justify scale-out

### Phase F: External Scale-Out

Purpose:

- turn the pilot into a credible public result

Run:

- 200+ SWE-bench Verified tasks if pilot signal is positive
- optional follow-up on full SWE-bench
- optional later validation on RepoBench / RefactorBench / Debug-gym depending on where the local signal is strongest

Acceptance:

- a stable result that answers whether Kimi with `tldrf` is meaningfully better than Kimi without it

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

Likely files to change:

- [scripts/bench_llm_ab_run.py](../scripts/bench_llm_ab_run.py)
  - add OpenRouter / OpenAI-compatible provider adapter
  - extend full-run usage accounting
- New: `scripts/bench_tool_choice.py`
  - add a preflight mode that emits transcript-level workflow-compliance reports
  - evaluate workflow selection quality
- New: `scripts/bench_agent_tasks.py`
  - run end-to-end tool-using tasks
- New: `benchmarks/agentic/preflight_tasks.json`
- New: `benchmarks/agentic/tool_choice_tasks.json`
- New: `benchmarks/agentic/patch_tasks.json`
- New: `benchmarks/agentic/swebench_subset.json`
- [benchmarks/README.md](../benchmarks/README.md)
  - publish the new benchmark family and results

## Immediate Next Steps

1. Create the OpenRouter provider adapter so Kimi can run inside the existing answer-model harness.
2. Reuse the current Django `llm_ab` packets and run the smallest possible pilot:
   - Kimi baseline arm
   - Kimi plus `tldrf` arm
3. Define the preflight tool-usage suite and transcript review gate from `AGENTS.md` and the reference card.
4. Run the preflight gate and tune the instruction surface until workflow compliance is acceptable.
5. Define the full tool-choice gold policy suite.
6. Curate a first batch of local patch/test tasks on Django.
7. Select the first 20-50 Django-heavy SWE-bench Verified tasks for the external pilot.

## Running Log

- 2026-03-06: Created this plan to combine the existing local benchmark stack with a new agentic evaluation goal focused on Kimi-with-`tldrf` vs Kimi-without-`tldrf`.
- 2026-03-06: Locked the preferred initial model stack as:
  - answer model: Kimi on OpenRouter
  - fallback cheap answer model: DeepSeek on OpenRouter
  - judge: Claude Sonnet
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

## Gotchas / Learnings

- Existing `llm_ab` results are valuable, but they measure prebuilt context packets, not tool-using agents.
- A strong external benchmark does not remove the need for a local benchmark. Local suites are still the fastest way to tune tool policy, packing, and harness behavior.
- If the model does not know when to use `tldrf`, the benchmark mostly measures prompt quality, not tool utility. Full runs should be blocked until the preflight gate passes.
- The most realistic winning setup is expected to be policy-combined, not tool-exclusive:
  - `rg` for exact lookup
  - `tldrf` for structure and concept search
- Cost claims must be treated as variable until provider pricing is rechecked at execution time.
