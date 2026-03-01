# Head-to-Head Benchmark vs Contextplus Implementation Plan

- Status note: Supporting/legacy execution reference.
- Canonical 008 implementation authority: `implementations/008-beat-contextplus_IMPLEMENTATION_PLAN.md`.

- Status: Planned
- Owner: TBD
- Last updated: 2026-03-01
- Source: `specs/008-head-to-head-benchmark-llm-tldr-vs-contextplus.md`

## Objective

Make `llm-tldr` measurably better than `contextplus` under a neutral, reproducible benchmark protocol and publish defensible evidence.

## Success Criteria (Program-Level)

1. Head-to-head suite completes with valid artifacts for both tools.
2. `llm-tldr` wins at least 3 primary common-lane metrics at budget `2000`.
3. `llm-tldr` passes all run-validity and tool-quality gates in the suite.
4. Public claims in docs are aligned to pinned run artifacts (no contradictory headline numbers).

## Phase 0: Benchmark Contract Completion and Tool Adapters

### Goal
Finish the neutral benchmark execution path end-to-end for both tools.

### Deliverables
- Final `contextplus` tool profile:
  - `benchmarks/head_to_head/tool_profiles/contextplus.v1.json`
- Adapter workflow that generates canonical predictions JSON for each tool.
- First full `compare.json` produced from real runs.

### Tasks
- [ ] Replace placeholders in `contextplus.v1.template.json` with executable commands.
- [ ] Implement/validate prediction generation scripts for both tools.
- [ ] Produce `env.json`, `score.json` (both tools), and `compare.json`.

### Acceptance (Verifiable)
- [ ] `uv run python scripts/bench_head_to_head.py validate-suite --suite benchmarks/head_to_head/suite.v1.json --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json --tool-profile benchmarks/head_to_head/tool_profiles/contextplus.v1.json` exits `0`.
- [ ] `uv run python scripts/bench_head_to_head.py materialize-tasks --suite benchmarks/head_to_head/suite.v1.json --corpus-root benchmark/corpora/django --out benchmark/runs/h2h-task-manifest.json` exits `0`.
- [ ] `score` command exits `0` for both tools with no schema/gate-format errors.
- [ ] `compare` command exits `0` and writes `benchmark/runs/h2h-compare.json`.

## Phase 1: Claim Hygiene and Metrics Canonicalization

### Goal
Remove ambiguous/contradictory performance claims and establish one canonical source for metrics.

### Deliverables
- Updated benchmark-reporting policy in docs.
- Updated README/docs values to reference exact run artifacts.
- Regression checks for claim consistency.

### Tasks
- [ ] Replace inconsistent latency/token headlines with artifact-backed values.
- [ ] Add a small test/lint rule to ensure benchmark claims link to concrete report paths.
- [ ] Reconcile language-support messaging (`17 labels` vs `16 bundled grammars`).

### Acceptance (Verifiable)
- [ ] `rg -n "100ms|30s|300x|89-99%" README.md docs/TLDR.md benchmarks/README.md` only returns statements with explicit report references or documented caveats.
- [ ] Added/updated tests pass:
  - `uv run pytest tests/test_bench_*schema.py`
  - any new claim-consistency test file(s).
- [ ] Docs reference at least one pinned report path for each top-level performance claim.

## Phase 2: Production Hybrid Retrieval (Not Benchmark-Only)

### Goal
Promote hybrid fusion (lexical + semantic) into product retrieval path.

### Deliverables
- New production command/API path for hybrid retrieval.
- Configurable fusion strategy and stable output schema.

### Tasks
- [ ] Add `tldrf hybrid search` (or equivalent `semantic search --mode hybrid`).
- [ ] Implement fusion in `tldr/` runtime code (not only in `scripts/bench_*`).
- [ ] Add unit and CLI tests for ranking behavior and deterministic output ordering.

### Acceptance (Verifiable)
- [ ] `uv run tldrf --help` includes new hybrid mode/command.
- [ ] New tests pass:
  - `uv run pytest tests/test_semantic*.py tests/test_cli*.py` (including new hybrid tests).
- [ ] On `scripts/bench_retrieval_quality.py`, hybrid meets:
  - `MRR_hybrid >= max(MRR_rg, MRR_semantic) - 0.01`
  - `FPR@5_hybrid <= min(FPR@5_rg, FPR@5_semantic) + 0.01`

## Phase 3: Semantic Abstention and Reranking

### Goal
Reduce false positives while preserving retrieval quality.

### Deliverables
- Threshold/abstention policy for semantic/hybrid outputs.
- Optional reranking stage (configurable).

### Tasks
- [ ] Add score-threshold gating and explicit “no confident result” handling.
- [ ] Add rerank stage flag/path.
- [ ] Add negative-query tests and benchmark checks.

### Acceptance (Verifiable)
- [ ] Negative-query regression tests pass (no ungrounded top-k spillover).
- [ ] Retrieval benchmark at budget `2000` meets:
  - `FPR@5 <= 0.05`
  - `MRR drop vs pre-rerank <= 0.02`
- [ ] `uv run pytest` includes dedicated abstention/rerank tests and passes.

## Phase 4: Compound Semantic+Impact Query

### Goal
Provide single-call workflow that returns semantic matches with caller/use-site evidence.

### Deliverables
- New compound command/API (e.g., `semantic impact`).
- Output schema containing both match and caller sets.

### Tasks
- [ ] Implement compound orchestration over existing semantic + impact components.
- [ ] Ensure predictable token-bounded payload composition.
- [ ] Add fixture tests for correctness and shape.

### Acceptance (Verifiable)
- [ ] New command appears in CLI help and MCP mapping docs.
- [ ] Fixture tests validate callers for known symbols:
  - `uv run pytest tests/test_*impact*.py tests/test_*semantic*.py` (including new compound tests).
- [ ] Median latency of compound mode is <= `1.5x` median latency of sequential `semantic + impact` on a fixed sample set.

## Phase 5: Semantic Navigation/Clustering Feature

### Goal
Add a semantic-navigation mode comparable to Context+ clustering while preserving deterministic core outputs.

### Deliverables
- New navigation command (e.g., `tldrf navigate`).
- Cluster output schema with stable IDs and file membership.

### Tasks
- [ ] Implement embedding-space clustering pipeline.
- [ ] Make labeling optional or clearly separated from deterministic cluster generation.
- [ ] Add synthetic clustering tests and basic integration coverage.

### Acceptance (Verifiable)
- [ ] Deterministic tests pass for cluster assignment on synthetic fixtures.
- [ ] Command returns cluster coverage >= `95%` of indexed files for representative repos (excluding ignored files).
- [ ] `uv run pytest` includes new navigation tests and passes.

## Phase 6: Optional Ollama Embedding Backend

### Goal
Reduce onboarding friction by offering an alternative local embedding provider.

### Deliverables
- Provider abstraction with `sentence-transformers` and `ollama` backends.
- Config/env flags to choose provider.

### Tasks
- [ ] Introduce provider interface and refactor existing semantic code path.
- [ ] Implement Ollama backend with robust error handling.
- [ ] Add provider-selection tests and fallback behavior tests.

### Acceptance (Verifiable)
- [ ] `tldrf semantic index` works with both providers under documented config.
- [ ] Integration tests pass for provider selection and failure paths.
- [ ] Retrieval sanity check: overlap@5 between providers is reported and tracked (no hard parity gate required initially).

## Phase 7: Final Head-to-Head Run and Release Gate

### Goal
Generate final evidence package and enforce a release-quality decision gate.

### Deliverables
- Final artifact set:
  - `task_manifest.json`
  - `predictions.json` (both tools)
  - `score.json` (both tools)
  - `compare.json`
  - `env.json`
  - raw logs
- Decision report summarizing winners and deltas.

### Tasks
- [ ] Run full head-to-head across all budget tiers.
- [ ] Confirm gates and winner rule from suite.
- [ ] Publish summary in `benchmarks/README.md` and source artifact paths.

### Acceptance (Verifiable)
- [ ] `compare.json` shows `llm-tldr` winning at least 3 primary common-lane metrics at budget `2000`.
- [ ] `llm-tldr` meets suite run-validity/tool-quality gates:
  - `timeout_rate <= 0.02`
  - `error_rate <= 0.01`
  - `budget_violation_rate == 0.0`
  - `retrieval_mrr >= 0.55`
  - `retrieval_fpr@5 <= 0.05`
- [ ] All referenced report paths exist and are reproducible with documented commands.

## Verification Commands (Core)

```bash
uv run ruff check scripts/ tests/
uv run pytest

uv run python scripts/bench_head_to_head.py validate-suite \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/contextplus.v1.json

uv run python scripts/bench_head_to_head.py materialize-tasks \
  --suite benchmarks/head_to_head/suite.v1.json \
  --corpus-root benchmark/corpora/django \
  --out benchmark/runs/h2h-task-manifest.json

uv run python scripts/bench_head_to_head.py score ...
uv run python scripts/bench_head_to_head.py compare ...
```

## Risks and Mitigations

- Risk: benchmark adapter mismatch across tools.
  - Mitigation: strict tool profile schema + canonical predictions schema + validate-suite gating.
- Risk: overfitting to one corpus.
  - Mitigation: keep Django as primary, add secondary corpus in v2 once baseline is stable.
- Risk: headline claims drift from measurements.
  - Mitigation: enforce report-path references for any public performance statement.
