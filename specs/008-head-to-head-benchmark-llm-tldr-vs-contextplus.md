# Spec 008: Neutral Head-to-Head Benchmark (llm-tldr vs contextplus)

## Goal

Provide a neutral, reproducible, implementation-ready benchmark contract that compares `llm-tldr` and `contextplus` on identical tasks, budgets, and environment constraints.

## Benchmark Philosophy

- Compare each tool on tasks it can natively and realistically perform.
- Keep a required common lane (`retrieval`) so both tools are always directly comparable.
- Keep advanced structural lanes (`impact`, `slice`, `complexity`, `data_flow`) capability-gated and optional.
- Record unsupported categories as `N/A` via explicit capability declaration, not proxy substitutions.

## Canonical Config + Artifacts

- Suite contract: `benchmarks/head_to_head/suite.v1.json`
- Tool profiles:
  - `benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json`
  - `benchmarks/head_to_head/tool_profiles/contextplus.v1.template.json`
- Prediction format example: `benchmarks/head_to_head/examples/predictions.v1.example.json`
- Harness script: `scripts/bench_head_to_head.py`
- Runbook: `benchmarks/head_to_head/README.md`

## Tasks

### Lane A (Required): Retrieval

- Source dataset: `benchmarks/retrieval/django_queries.json`
- Task shape:
  - input: natural-language query
  - output: ranked file list (`ranked_files`)
  - ground truth: `relevant_files`

### Lane B (Capability-Gated): Structural Analysis

- Source dataset: `benchmarks/python/django_structural_queries.json`
- Categories:
  - `impact`: caller set for target function
  - `slice`: backward line slice for `(file, function, target_line)`
  - `complexity`: cyclomatic complexity for `(file, function)`
  - `data_flow`: origin + flow lines for `(file, function, variable)`

## Dataset + Pinning

- Corpus: Django
- Required ref: `5.1.13`
- Required SHA: `c04a09ddb3bb1fe8157292fcd902b35cad9a5e10`
- Manifest source: `benchmarks/corpora.json`

The harness fails materialization if corpus SHA does not match.

## Metrics

### Retrieval

- MRR
- Recall@K (`K` from suite budgets)
- Precision@K
- FPR@K (negative queries)
- Latency p50 (ms)
- Payload tokens median

### Impact

- Precision / Recall / F1 on caller-set match
- Latency p50
- Payload tokens median

### Slice

- Precision / Recall / F1 on line-set match
- Noise reduction mean (`1 - predicted_lines/total_function_lines`)
- Latency p50

### Complexity

- Accuracy vs radon ground truth
- MAE vs radon
- Kendall tau-b vs radon ranking

### Data Flow

- Origin accuracy
- Flow completeness mean
- Latency p50

### Reliability / Fairness

- Timeout rate
- Error rate
- Unsupported rate
- Budget violation rate
- Common-lane coverage
- Capability coverage

## Protocol

- Trials: `3`
- Seeds: `[11, 29, 47]`
- Query order: seeded shuffle
- Warmup per category: `2`
- Cold start iterations: `1`
- Warm start iterations: `3`
- Timeout per query: `30s`
- Timeout per full run: `7200s`
- Retry on timeout: `1`

## Budget Controls

- Tokenizer: `cl100k_base`
- Budget tiers: `500, 1000, 2000, 5000`
- Hard payload max tokens: `5000`
- Hard payload max bytes: `65536`
- Retrieval top-k: `10`
- Retrieval metric Ks: `1, 5, 10`

The scorer records hard budget violations when `payload_tokens > budget_tokens`.

## Pass/Fail Gates

### Run Validity

- timeout_rate <= `0.02`
- error_rate <= `0.01`
- budget_violation_rate <= `0.0`

### Tool Quality

At budget `2000`:
- common_lane_coverage >= `1.0`
- retrieval_mrr >= `0.55`
- retrieval_fpr@5 <= `0.05`
- if supported:
  - impact_f1 >= `0.50`
  - slice_recall >= `0.70`
  - data_flow_origin_accuracy >= `0.70`
  - complexity_mae <= `3.0`

### Head-to-Head Winner Rule

- Primary budget: `2000`
- Winner must win at least 3 common-lane primary metrics:
  - `mrr_mean`, `recall@5_mean`, `precision@5_mean`, `latency_ms_p50`, `payload_tokens_median`
- Tie-breakers:
  - lower timeout_rate
  - lower payload_tokens_median
  - lower latency_ms_p50

## Anti-Bias Controls

- Freeze suite config, dataset hashes, and gate thresholds before first run.
- Require same corpus SHA and tokenizer for both tools.
- Require identical budget set and trial count.
- Keep tool capabilities explicit in profile files (no hidden proxies).
- Blind scoring outputs (`tool_a`, `tool_b`) until comparison complete.
- Include negative retrieval queries and all failures in final score.
- Persist raw logs per `(tool, task, budget, trial)` for audit.

## Required Scripts

- `scripts/bench_head_to_head.py validate-suite`
- `scripts/bench_head_to_head.py materialize-tasks`
- `scripts/bench_head_to_head.py score`
- `scripts/bench_head_to_head.py compare`

## Required Reproducibility Artifacts

- `task_manifest.json`
- `tool_profile.json` (one per tool)
- `predictions.json` (one per tool)
- `score.json` (one per tool)
- `compare.json`
- `env.json`
- raw query logs

## Minimal End-to-End Commands

```bash
uv run python scripts/bench_fetch_corpora.py --corpus django

uv run python scripts/bench_head_to_head.py validate-suite \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json

uv run python scripts/bench_head_to_head.py materialize-tasks \
  --suite benchmarks/head_to_head/suite.v1.json \
  --corpus-root benchmark/corpora/django \
  --out benchmark/runs/h2h-task-manifest.json

# Produce predictions.json for each tool using tool-specific adapters.

uv run python scripts/bench_head_to_head.py score \
  --suite benchmarks/head_to_head/suite.v1.json \
  --tasks benchmark/runs/h2h-task-manifest.json \
  --tool-profile benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json \
  --predictions benchmark/runs/h2h-llm-tldr-predictions.json

uv run python scripts/bench_head_to_head.py compare \
  --suite benchmarks/head_to_head/suite.v1.json \
  --score-a benchmark/runs/h2h-llm-tldr-score.json \
  --score-b benchmark/runs/h2h-contextplus-score.json
```

## Addendum (2026-03-01): Peer Review Audit + Corrections

This addendum captures a claim-by-claim audit of an external peer review, validates or corrects those claims against repository evidence, and defines concrete win conditions for making `llm-tldr` measurably better than `contextplus`.

### A. Claim Verification (Audited)

#### A1. `llm-tldr` claims

| Claim | Verdict | Evidence | Correction |
|---|---|---|---|
| Deep structural stack (CFG/DFG/PDG) exists and is a major differentiator | True | `tldr/api.py` (`get_cfg_context`, `get_dfg_context`, `get_slice`) | Keep claim. |
| Daemon uses memoization and incremental invalidation | Partial | `tldr/daemon/cached_queries.py`, `tldr/daemon/core.py` | True for many handlers; not all paths are Salsa-memoized (`impact`/`semantic` are special paths). |
| Hybrid retrieval is a production feature | False | `tldr/cli.py` commands vs `scripts/bench_retrieval_quality.py` | Hybrid RRF currently exists in benchmark/eval tooling, not production CLI/API retrieval. |
| Semantic indexing/search uses FAISS | True | `tldr/semantic.py`, `tldr/indexing/index.py`, `pyproject.toml` | Keep claim. |
| Token-efficiency numbers in docs are universally current and externally validated | Partial | `benchmarks/README.md`, `docs/TLDR.md` | Treat as internal benchmark evidence; use pinned run artifacts as canonical source. |
| Language support count is consistently `16` | False | `tldr/semantic.py`, `tldr/cli.py`, `tests/test_language_wiring.py`, `docs/TLDR.md` | Reconcile docs/code: there are 17 language labels in code paths, with 16 bundled/tested tree-sitter grammars. |

#### A2. `contextplus` claims

| Claim | Verdict | Evidence | Correction |
|---|---|---|---|
| MCP-only interface | Partial | `src/index.ts`, `package.json` | MCP stdio server is primary runtime interface, but there is non-MCP CLI behavior (`init`, root argument). |
| Embedding backend is Ollama-based | True | `src/core/embeddings.ts`, `src/tools/semantic-navigate.ts` | Keep claim. |
| Vector retrieval is brute-force over cached vectors | True | `src/core/embeddings.ts` (`SearchIndex`) | Keep claim (no FAISS/ANN index layer). |
| Supports “43 grammars/languages” | Partial | `src/core/tree-sitter.ts`, `README.md` | Code maps 43 file extensions; comments/docs also mention 36 languages. Counts are inconsistent and should not be overstated. |
| Spectral clustering navigation exists | True | `src/core/clustering.ts`, `src/tools/semantic-navigate.ts` | Keep claim. |
| `propose_commit` is strict reject-gate for style violations | False | `src/tools/propose-commit.ts`, `test/main/propose-commit.test.mjs` | Most violations are warning-only; hard reject path is narrow. |
| Benchmark framework supports “99% accuracy” claim | False | `package.json`, `README.md`, `test/demo/embeddings-proof.demo.mjs` | No formal reproducible benchmark harness with dataset + metrics + thresholds. |
| `get_blast_radius` is semantic dependency tracing | Partial | `src/tools/blast-radius.ts`, `src/core/walker.ts` | Useful regex heuristic, not semantic reference/call graph. |

### B. Corrected Numeric Claims

| Claim | Status | Corrected value/evidence |
|---|---|---|
| `hybrid_rrf MRR = 0.868` | Verified (rounded) | Exact value in run artifact: `0.8684419381787802`; rounded in docs table. |
| `100ms vs 30s` daemon headline | Not cleanly verified as a single canonical metric | Repo docs contain multiple latency narratives (`300x` claim and a `~155x` measured table). Use pinned run artifacts as single source of truth. |
| `89–99% token savings` | Partially true, range is selective | Detailed tables include wider variation by workflow/budget. Report per-scenario values and avoid single headline without context. |
| `tldrf = 17 languages` | Partially true | 17 language labels in code paths; 16 bundled/tested tree-sitter grammars. |
| `contextplus = 36 languages / 43` | Partially true but inconsistent | 43 extension mappings are explicit; “36 languages” appears in comments/docs. Report both as extension coverage vs grammar count to avoid inflation. |

### C. Comparative Positioning (Audited)

#### C1. Where `llm-tldr` is stronger today

- Structural analysis depth and task coverage (`impact`, `slice`, `cfg`, `dfg`, callgraph-based reasoning).
- Benchmark rigor and reproducibility infrastructure (`benchmarks/`, scripts, schemas, reports).
- Operational architecture (daemon + index isolation + benchmarkable workflows).

#### C2. Where `contextplus` is stronger today

- Production retrieval ergonomics: built-in semantic+keyword blend with thresholds.
- Semantic navigation/clustering UX for initial codebase orientation.
- Lower dependency friction for users already standardized on Ollama.

#### C3. Where both are weak

- Cross-language end-to-end reasoning remains limited.
- Open-ended context packing still needs disciplined, benchmark-driven iteration.
- Large-repo scaling pressure exists in both systems (different bottlenecks).

### D. Measurable Win Definition (Head-to-Head)

Use `benchmarks/head_to_head/suite.v1.json` as the canonical contract.  
Primary decision rule at budget `2000`:

- `llm-tldr` must win at least 3 common-lane primary metrics in `compare.json`.
- Gate requirements must pass:
  - `timeout_rate <= 0.02`
  - `error_rate <= 0.01`
  - `budget_violation_rate == 0.0`
  - `retrieval_mrr >= 0.55`
  - `retrieval_fpr@5 <= 0.05`

### E. Benchmarks Required To Prove Superiority

1. Neutral head-to-head suite:
   - `scripts/bench_head_to_head.py` (`validate-suite`, `materialize-tasks`, `score`, `compare`)
   - `benchmarks/head_to_head/suite.v1.json`
2. Retrieval quality:
   - `scripts/bench_retrieval_quality.py`
3. Token efficiency:
   - `scripts/bench_token_efficiency.py`
4. Structural quality:
   - `scripts/bench_structural_analysis.py`
5. LLM task-based A/B where applicable:
   - `scripts/bench_llm_ab_prompts.py`
   - `scripts/bench_llm_ab_run.py`

### F. Current Status of Head-to-Head Harness (This Turn)

- Added benchmark contract/runbook/tool profiles/example predictions:
  - `benchmarks/head_to_head/suite.v1.json`
  - `benchmarks/head_to_head/README.md`
  - `benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json`
  - `benchmarks/head_to_head/tool_profiles/contextplus.v1.template.json`
  - `benchmarks/head_to_head/examples/predictions.v1.example.json`
- Added harness + tests:
  - `scripts/bench_head_to_head.py`
  - `tests/test_bench_head_to_head_suite_schema.py`
  - `tests/test_bench_head_to_head_tool_profiles_schema.py`
- Verified:
  - `uv run ruff check ...` passes for new harness/tests.
  - `uv run pytest ...` passes for new harness/tests.
  - `validate-suite` and `materialize-tasks` executed successfully.
- Materialized task manifest currently includes `105` tasks:
  - retrieval `60`, impact `15`, slice `10`, complexity `10`, data_flow `10`.

### G. Required Next Action

Implement the phased execution plan in:

- `implementations/008-head-to-head-benchmark-llm-tldr-vs-contextplus_SUPPORTING_PLAN.md`

That plan defines concrete engineering phases with verifiable acceptance criteria and benchmark pass/fail gates.

## Appendix H (Normative): Formal Benchmark Methodology Contract

This appendix is normative for v1 runs and is aligned to:

- `benchmarks/head_to_head/suite.v1.json`
- `scripts/bench_head_to_head.py`

If any earlier prose conflicts with this appendix, this appendix wins.

### H1. Required Lanes and Capability Gating

- Lane `common_retrieval` is mandatory for every tool.
  - `required_for_all_tools = true`
  - categories: `retrieval`
- Lane `structural_optional` is capability-gated.
  - `required_for_all_tools = false`
  - categories: `impact`, `slice`, `complexity`, `data_flow`
- The harness-required category set is computed as:
  - all categories from lanes where `required_for_all_tools == true`, plus
  - categories from optional lanes where `tool_profile.capabilities[category] == true`
- Unsupported categories must remain `N/A` via `capabilities.<category> = false`; no proxy substitutions are permitted.

### H2. Dataset and Pinning Rules

- Canonical corpus pin (from suite):
  - `corpus_id = django`
  - `required_ref = 5.1.13`
  - `required_git_sha = c04a09ddb3bb1fe8157292fcd902b35cad9a5e10`
  - `corpus_manifest = benchmarks/corpora.json`
- Query sources:
  - `benchmarks/retrieval/django_queries.json`
  - `benchmarks/python/django_structural_queries.json`
- `materialize-tasks` must fail when corpus `HEAD` does not match the required SHA (after tag-object SHA resolution).
- `materialize-tasks` emits:
  - `source_hashes` for both query source files
  - `task_manifest_sha256 = sha256(tasks)` (canonical hash for scoring fairness)
- `score` must fail on:
  - `suite_id` mismatch between suite/tasks/predictions
  - edited task manifest (`task_manifest_sha256` mismatch)
  - predictions manifest hash mismatch when `predictions.task_manifest_sha256` is present and not equal

### H3. Protocol, Budgets, and Prediction Completeness

- Protocol values (suite v1):
  - `trials = 3`
  - `seeds = [11, 29, 47]`
  - `query_order = shuffle_with_seed`
  - `warmup_per_category = 2`
  - `cold_start_iterations = 1`
  - `warm_start_iterations = 3`
  - `timeout_s_per_query = 30`
  - `timeout_s_per_run = 7200`
  - `retry_on_timeout = 1`
- Budget values (suite v1):
  - `tokenizer = cl100k_base`
  - `token_budgets = [500, 1000, 2000, 5000]`
  - `retrieval_top_k = 10`
  - `retrieval_ks = [1, 5, 10]`
  - `max_payload_tokens_hard = 5000`
  - `max_payload_bytes_hard = 65536`
- Required prediction keyspace:
  - one unique prediction per `(task_id, budget_tokens, trial)` for every required category
  - duplicates are counted (`duplicate`) and ignored after first occurrence
  - missing required keys count toward error accounting
- Allowed prediction `status` values:
  - `ok`, `unsupported`, `timeout`, `error`, `pending`
  - unknown status values are coerced to `error`
- Hard budget violation semantics in scorer:
  - violation iff `payload_tokens > budget_tokens` for an `ok` prediction
  - `payload_bytes` is collected for telemetry but not currently used in gate computation

### H4. Metric and Gate Semantics

- Retrieval scoring:
  - positive queries (`relevant_files != []`): `mrr`, `recall@k`, `precision@k`
  - negative queries (`relevant_files == []`): `fpr@k` where prediction at top-k implies `1.0`, else `0.0`
- Structural scoring:
  - `impact`: caller-set precision/recall/F1
  - `slice`: line-set precision/recall/F1 + `noise_reduction = 1 - predicted_lines/total_function_lines` (when denominator exists)
  - `complexity`: exact-match accuracy, absolute error, kendall tau-b
  - `data_flow`: origin accuracy, flow completeness
- Secondary rates:
  - `timeout_rate = timeout / expected_total`
  - `error_rate = (error + missing) / expected_total`
  - `unsupported_rate = unsupported / expected_total`
  - `budget_violation_rate = budget_violations / expected_total`
  - `common_lane_coverage = retrieval_ok / retrieval_expected`
  - `capability_coverage = ok / expected_total`
- Gate logic (from suite `gates`):
  - run validity:
    - `timeout_rate <= 0.02`
    - `error_rate <= 0.01`
    - `budget_violation_rate <= 0.0`
  - fairness:
    - tokenizer must match `cl100k_base`
    - budget set in predictions must exactly match `[500, 1000, 2000, 5000]`
    - identical task manifest hash is required
  - tool quality at primary budget `2000`:
    - `common_lane_coverage >= 1.0`
    - `retrieval mrr_mean >= 0.55`
    - `retrieval fpr@5_mean <= 0.05` (vacuously passes if metric absent; therefore negatives must remain in dataset)
    - if capability enabled:
      - `impact f1_mean >= 0.5`
      - `slice recall_mean >= 0.7`
      - `data_flow origin_accuracy_mean >= 0.7`
      - `complexity mae <= 3.0`
    - if capability disabled, corresponding structural gate is marked `skipped` and does not fail run
    - if capability enabled but metric missing/non-numeric, that gate fails
- Head-to-head winner rule at primary budget `2000`:
  - compare common-lane primary metrics:
    - `mrr_mean`, `recall@5_mean`, `precision@5_mean`, `latency_ms_p50` (lower is better), `payload_tokens_median` (lower is better)
  - winner must win at least 3 of 5 metrics
  - tie-breakers in order:
    - lower `timeout_rate`
    - lower `payload_tokens_median`
    - lower `latency_ms_p50`

### H5. Anti-Bias Controls

- Freeze suite JSON, query source hashes, and gate thresholds before either tool is run.
- Enforce same corpus SHA, tokenizer, budgets, trials, and seed list for both tools.
- Keep capability declarations explicit in tool profiles; unsupported categories remain `N/A`.
- Do not hand-tune prompts per query/trial after seeing interim results.
- Score and compare blind (`tool_a` / `tool_b`) until final decision artifact is generated.
- Include negative retrieval queries and all failures (`timeout`, `error`, `missing`) in final metrics.
- Persist raw outputs and command lines per query for audit.

### H6. Required Artifacts (Run Bundle)

For each benchmark run, publish a single run directory with:

- `validate.json` (`validate-suite` output)
- `task_manifest.json` (`materialize-tasks` output)
- `tool_a.profile.json` and `tool_b.profile.json` (profiles actually used)
- `tool_a.predictions.json` and `tool_b.predictions.json` (canonical schema)
- `tool_a.score.json` and `tool_b.score.json` (`score` outputs)
- `compare.blind.json` (labels `tool_a`, `tool_b`)
- `compare.revealed.json` (optional relabeling after blind decision)
- `env.json` (OS/CPU/Python/Node/tool versions)
- `raw_logs/<tool>/<trial>/<task>.log` (or stricter equivalent)

### H7. Exact Run Sequence (Normative CLI Flow)

```bash
set -euo pipefail

RUN_ID="$(date -u +%Y%m%d-%H%M%SZ)"
RUN_ROOT="benchmark/runs/h2h-${RUN_ID}"
export RUN_ROOT
mkdir -p "${RUN_ROOT}/raw_logs/tool_a" "${RUN_ROOT}/raw_logs/tool_b"

SUITE="benchmarks/head_to_head/suite.v1.json"
TOOL_A_PROFILE="benchmarks/head_to_head/tool_profiles/llm_tldr.v1.json"
TOOL_B_PROFILE="benchmarks/head_to_head/tool_profiles/contextplus.v1.json" # instantiate from template first

cp "${TOOL_A_PROFILE}" "${RUN_ROOT}/tool_a.profile.json"
cp "${TOOL_B_PROFILE}" "${RUN_ROOT}/tool_b.profile.json"

uv run python scripts/bench_fetch_corpora.py --corpus django

uv run python scripts/bench_head_to_head.py validate-suite \
  --suite "${SUITE}" \
  --tool-profile "${TOOL_A_PROFILE}" \
  --tool-profile "${TOOL_B_PROFILE}" \
  --out "${RUN_ROOT}/validate.json"

uv run python scripts/bench_head_to_head.py materialize-tasks \
  --suite "${SUITE}" \
  --corpus-root benchmark/corpora/django \
  --out "${RUN_ROOT}/task_manifest.json"

# Adapter-owned execution step:
# Produce canonical predictions for each tool using identical task manifest,
# budget list, trials, seeds, and tokenizer from the suite.
# Required output files:
#   ${RUN_ROOT}/tool_a.predictions.json
#   ${RUN_ROOT}/tool_b.predictions.json

uv run python scripts/bench_head_to_head.py score \
  --suite "${SUITE}" \
  --tasks "${RUN_ROOT}/task_manifest.json" \
  --tool-profile "${TOOL_A_PROFILE}" \
  --predictions "${RUN_ROOT}/tool_a.predictions.json" \
  --out "${RUN_ROOT}/tool_a.score.json"

uv run python scripts/bench_head_to_head.py score \
  --suite "${SUITE}" \
  --tasks "${RUN_ROOT}/task_manifest.json" \
  --tool-profile "${TOOL_B_PROFILE}" \
  --predictions "${RUN_ROOT}/tool_b.predictions.json" \
  --out "${RUN_ROOT}/tool_b.score.json"

uv run python scripts/bench_head_to_head.py compare \
  --suite "${SUITE}" \
  --score-a "${RUN_ROOT}/tool_a.score.json" \
  --score-b "${RUN_ROOT}/tool_b.score.json" \
  --label-a tool_a \
  --label-b tool_b \
  --out "${RUN_ROOT}/compare.blind.json"

uv run python - <<'PY'
import json, os, pathlib
run_root = pathlib.Path(os.environ["RUN_ROOT"])
a = json.loads((run_root / "tool_a.score.json").read_text())
b = json.loads((run_root / "tool_b.score.json").read_text())
env = {"tool_a_meta": a.get("meta"), "tool_b_meta": b.get("meta")}
(run_root / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
PY
```

## Peer-Review Claim Truth Audit (2026-03-01)

### 1) Claim Audit (True / Partial / False)

| Claim | Verdict | Corrected statement | Evidence |
|---|---|---|---|
| "Head-to-head winner is established (`llm-tldr` vs `contextplus`)." | **False** | Current `h2h` compare artifact is example-only and compares the same score file against itself (`score_a == score_b`), so it cannot establish a real winner. | `benchmark/runs/h2h-example-compare.json` (`inputs.score_a`, `inputs.score_b`, identical SHA); `winner: "tie"` |
| "Head-to-head MRR is 1.0." | **False** | `MRR=1.0` in the example score is from a tiny partial sample (5/1260 expected records). It is not a valid full-suite head-to-head result. | `benchmark/runs/h2h-example-score.json` (`status_counts.ok=5`, `expected_total=1260`, `missing=1255`, `rates.error_rate=0.9960`) |
| "llm-tldr is universally ~10ms / 155x faster." | **Partial** | Latency is workload-dependent. Structural benchmark shows very low TLDR latency (p50 `4.384ms`), but retrieval semantic timing is materially higher (derived p50 `348.3ms` from per-query timings). | `benchmark/runs/20260210-005452Z-phase4-python-structural-django.json` (`results.timing.tldr.p50_s=0.004384`); `benchmark/runs/20260210-001934Z-retrieval-django-bge-guard-rg-empty.json` (`results.per_query[*].semantic.time_s`) |
| "95% token savings is the benchmarked baseline." | **Partial** | Token reduction is strong for structural tasks, but not universal. In retrieval at budget 2000, higher-MRR hybrid uses more tokens than `rg`. | `docs/TLDR.md` token table (`187,760 -> 21,580` total); `benchmark/runs/20260210-001934Z-token-efficiency-retrieval-django-bge-guard-rg-empty.json` (`results.retrieval.summary`) |
| "Support is 16 languages." | **Partial (outdated doc)** | Current code-level language list is 17 (adds `luau`), while docs table still lists 16. | `docs/TLDR.md` ("one interface, 16 languages"); `tldr/semantic.py` (`ALL_LANGUAGES` includes `luau`) |

### 2) Evidence-Backed Numerical Corrections

| Metric | Prior/quoted number | Corrected number | Source |
|---|---:|---:|---|
| Retrieval MRR (head-to-head) | `1.0` | Not valid as head-to-head: sample coverage `0.3968%` (5/1260). Full retrieval benchmark (no-result guard): `hybrid_rrf=0.8605`, `rg=0.8199`, `semantic=0.6124` at budget 2000. | `benchmark/runs/h2h-example-score.json`; `benchmark/runs/20260210-001934Z-token-efficiency-retrieval-django-bge-guard-rg-empty.json` (`results.retrieval.summary`) |
| Latency | "~10ms" / "155x faster" | Structural phase4: TLDR `p50=4.384ms`, `p95=17.569ms`; grep baseline `p50=72.488ms`, `p95=92.952ms`. Retrieval (derived from per-query times): `rg p50=82.8ms`, `semantic p50=348.3ms`. | `benchmark/runs/20260210-005452Z-phase4-python-structural-django.json` (`results.timing`); `benchmark/runs/20260210-001934Z-retrieval-django-bge-guard-rg-empty.json` (`results.per_query[*].time_s`) |
| Token savings | "95% fewer tokens" (headline) | Docs-table total is `88.51%` (`187,760 -> 21,580`). Structural impact (budget 2000): TLDR structured payload mean `70.53` vs `rg_window_function 812.13` (`91.3%` lower). Retrieval (budget 2000): hybrid payload `504.67` vs `rg 220.45` (`+128.9%`, not savings). | `docs/TLDR.md` token table; `benchmark/runs/20260210-214935Z-token-efficiency-structural-django-multistep.json` (`results.structural.impact.summary`); `benchmark/runs/20260210-001934Z-token-efficiency-retrieval-django-bge-guard-rg-empty.json` (`results.retrieval.summary`) |
| Language count | `16` | `17` in code (`python, typescript, javascript, go, rust, java, c, cpp, ruby, php, kotlin, swift, csharp, scala, lua, luau, elixir`). | `docs/TLDR.md`; `tldr/semantic.py` (`ALL_LANGUAGES`) |

### 3) Positioning Implications (llm-tldr vs contextplus)

1. Do **not** claim a completed head-to-head winner yet: repository evidence contains validation + task materialization + example scoring, but no real `contextplus` predictions/score artifacts.
2. Position `llm-tldr` on demonstrated structural signal (impact/slice/data-flow quality-per-token), not blanket retrieval token-savings claims.
3. For common-lane retrieval positioning, use tradeoff language: higher MRR can require higher payload tokens; avoid universal "faster and smaller" claims.
4. For publishable `llm-tldr vs contextplus` claims, require a full real run with both tools across all expected entries (`1260`) and passing suite validity gates.
