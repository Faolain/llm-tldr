# BGE -> Jina Code Embeddings 0.5B Migration Implementation Plan

- Status: In Progress
- Owner: TBD
- Last updated: 2026-03-06
- Source: user-provided migration outline ("BGE-1.5 -> Jina Code Embeddings 0.5B Migration Plan"), plus local repo paths `tldr/semantic.py`, `tldr/cli.py`, `tldr/daemon/core.py`, and benchmark runners under `scripts/`

## Next Steps (Recommended Order)

1. Finish the active Jina Django semantic build and capture its `/usr/bin/time -l` output.
2. Run the remaining daemon-mode Jina query benchmarks against `repo:django-jina05b` and compare them to the archived BGE daemon baseline.
3. Run the CLI-only comparison exceptions that daemon mode cannot cover yet: fresh build-time/RSS and dependency-scoped semantic benchmarks.
4. Fill Phase 6 and decide whether Jina stays opt-in, becomes default, or is rejected.

## Decisions & Assumptions (locked for this plan)

- This migration is **test-first**. Deterministic behavior changes ship only after `red -> green`; benchmark runs are acceptance evidence, not a substitute for invariant tests.
- The migration target is **opt-in Jina support first**, not an immediate default-model swap.
- `bge-large-en-v1.5` remains the default through the decision gate because:
  - Existing indexes are 1024-dimensional and cannot be reused with an 896-dimensional model.
- Local repo reality:
  - `tldr/semantic.py` currently hardcodes the BGE query instruction `Represent this code search query: ...`.
  - `build_semantic_index()` currently embeds `build_embedding_text(unit)` directly, so there is no document-side instruction/prefix.
  - `get_model()` currently instantiates `SentenceTransformer(hf_name, device=device)` with no model-specific tokenizer kwargs.
  - Most benchmark runners do **not** take an embedding-model flag; they consume whatever semantic index already exists for `--index`. Only `scripts/bench_dep_indexes.py` takes `--model` directly.
- Benchmark comparability therefore uses **separate index ids per model**, not rebuild-in-place:
  - Example: `repo:django-bge15`
  - Example: `repo:django-jina05b`
- Prefer the existing `sentence-transformers` load path first. If Jina cannot be made correct with explicit tokenizer/pooling configuration through that API, add a minimal dedicated adapter behind the same internal interface rather than changing the public CLI.
- Dependency floor is already sufficient in the local repo:
  - `sentence-transformers>=5.2.0`
  - `transformers==4.57.3` in `uv.lock`
  - `torch==2.9.1` in `uv.lock`

## Non-goals (MVP)

- Reworking lane2/lane4/lane5 retrieval semantics beyond what is required to keep model-specific query embeddings correct.
- Making Jina the default before the benchmark and license gates pass.
- Using Matryoshka truncation in the initial implementation. That is a fallback only if CPU latency regressions fail the gate.
- Preserving binary compatibility with old semantic indexes. Rebuild is mandatory when switching dimensions.

## High-level Architecture

- Extend the semantic model registry in `tldr/semantic.py` from a lightweight `SUPPORTED_MODELS` mapping into a model profile contract that can answer:
  - Hugging Face id
  - embedding dimension
  - query instruction/prefix
  - document instruction/prefix
  - tokenizer kwargs
  - any model-specific loader requirements
- Route embedding construction through explicit helpers:
  - query-time helper for `semantic_search()`
  - document-time helper for `build_semantic_index()`
- Keep model metadata persisted in semantic index metadata and continue to reject model/dimension mismatches at read time.
- Update CLI, daemon config defaults/help, and docs to expose Jina as a selectable model while keeping BGE as the default until rollout.

## Program Delivery Mode: Test-First With Benchmark Confirmation

This plan uses a hybrid method:

1. Deterministic tests first for model metadata, instruction routing, loader kwargs, cache behavior, and mismatch guards.
2. Benchmark runs second for retrieval quality, token efficiency, LLM answer quality, and performance.

Phase rule: implementation work for a phase starts only after that phase's `Tests To Write First (Red)` list exists and fails for the intended behavior change.

## Canonical Benchmark Matrix For This Migration

The attached migration plan named the right benchmark families, but the local repo's scripts work slightly differently. This is the canonical local mapping:

| Tier | Goal | Local runner | Notes |
| --- | --- | --- | --- |
| 1 | Direct semantic retrieval quality | `scripts/bench_retrieval_quality.py` | Local script emits `rg`, `semantic`, and `hybrid_rrf` in one report. `hybrid_lane2` appears only when lane2 flags are enabled. |
| 1 | Dependency-scoped semantic quality | `scripts/bench_dep_indexes.py` | Already supports `--model` directly; use it for `requests` and `urllib3`. |
| 2 | Retrieval quality under token budgets | `scripts/bench_token_efficiency.py --mode retrieval` | Use per-model `--index` ids; compare retrieval rows only. |
| 2 | Compound semantic + impact | `scripts/bench_compound_semantic_impact.py` | Verifies retrieval-stage changes do not break lane4 effectiveness/latency. |
| 2 | End-to-end LLM answer quality | `scripts/bench_llm_ab_prompts.py` + `scripts/bench_llm_ab_run.py` | Prompt generation is separate from answer-model execution in this repo. |
| 3 | Index build time + peak RSS | manual `/usr/bin/time -l uv run tldrf semantic index ...` | Use separate index ids; record wall time and max RSS. |
| 3 | Query latency p50/p95 | `scripts/bench_perf_daemon_vs_cli.py --include-semantic` | Point at a corpus/index that already has semantic artifacts. |
| 4 | Optional head-to-head suite | `scripts/bench_head_to_head.py` | Nice to have only after earlier gates pass; requires model-specific tool/profile wiring or paired prediction artifacts. |

## Benchmark Identity / Artifact Protocol

- Use `benchmark/cache-root` as the shared cache root for all migration runs.
- Use model-specific index ids everywhere a runner accepts `--index`:
  - `repo:django-bge` for the current reusable BGE daemon/query baseline that already has valid semantic artifacts
  - `repo:django-bge15` for a fresh BGE rebuild when build-time / peak-RSS parity is required
  - `repo:django-jina05b` for the Jina candidate
- Archive JSON outputs under `benchmark/runs/`.
- For long runs, use `tmux` and tee logs under `benchmark/logs/` per `AGENTS.md`.

Suggested log-friendly session naming:

- `jina-bench-baseline`
- `jina-bench-token`
- `jina-bench-llm`

## Phase 0: Lock The Migration Contract + Baseline Protocol

**Goals**
- Turn the model swap into an explicit contract instead of a hidden string replacement.
- Capture the current BGE baseline using the local benchmark harness and artifact naming conventions.

### Running Log (Phase 0)
- 2026-03-06: Verified that `tldr/semantic.py` currently has one hardcoded BGE query prefix and no document-side prefixing.
- 2026-03-06: Verified that `get_model()` currently has no model-specific `tokenizer_kwargs`, so Jina left-padding is not represented anywhere yet.
- 2026-03-06: Verified that `scripts/bench_retrieval_quality.py`, `scripts/bench_token_efficiency.py`, `scripts/bench_compound_semantic_impact.py`, `scripts/bench_perf_daemon_vs_cli.py`, and `scripts/bench_llm_ab_prompts.py` all use index metadata from `--index` rather than a direct `--model` flag.
- 2026-03-06: Verified that `scripts/bench_dep_indexes.py` already supports `--model`, so dependency-scoped evaluation does not require extra harness work.
- 2026-03-06: Verified the repo keeps tracked benchmark inputs under `benchmarks/` and untracked run artifacts under `benchmark/`; the benchmark phases should continue to use `benchmark/cache-root`, `benchmark/runs/`, and `benchmark/corpora/`.

### Immediate Next Steps (Phase 0)
- Record the Jina license decision before considering any default-model change.
- Keep using the existing `repo:django-bge` semantic index for daemon-mode query benchmarks because it already identifies as `BAAI/bge-large-en-v1.5` on the Django corpus.
- Rebuild the BGE Django baseline under `repo:django-bge15` only for fresh build-time / peak-RSS comparisons or if the daemon-side quality deltas are close enough that we need stricter rerun parity.

**Deliverables**
- Locked model/index naming convention for BGE vs Jina benchmark runs.
- BGE baseline reports archived under `benchmark/runs/`.
- Explicit license/default policy recorded in this plan.

**Key tasks**
- [ ] Decide and record the license outcome:
  - `blocked_default`
  - `opt_in_only`
  - `commercially_licensed`
- [ ] Build and time the BGE Django semantic index:
  ```bash
  /usr/bin/time -l uv run tldrf semantic index \
    --cache-root benchmark/cache-root \
    --index repo:django-bge15 \
    --lang python \
    --model bge-large-en-v1.5 \
    --rebuild \
    benchmark/corpora/django
  ```
- [ ] Run the BGE retrieval-quality baseline:
  ```bash
  uv run python scripts/bench_retrieval_quality.py \
    --corpus django \
    --cache-root benchmark/cache-root \
    --index repo:django-bge15 \
    --ks 5,10
  ```
- [ ] Run the BGE lane2 retrieval baseline:
  ```bash
  uv run python scripts/bench_retrieval_quality.py \
    --corpus django \
    --cache-root benchmark/cache-root \
    --index repo:django-bge15 \
    --ks 5,10 \
    --lane2-abstain-threshold 0.35 \
    --lane2-rerank \
    --lane2-rerank-top-n 8 \
    --lane2-max-latency-ms-p50-ratio 1.10 \
    --lane2-max-payload-tokens-median-ratio 0.90
  ```

**Acceptance**
- Baseline BGE artifacts exist for index build time, retrieval quality, and lane2 retrieval quality.
- License status is explicit before any default-model decision is considered.

## Phase 1: Model Profile Contract (Registry + Metadata)

**Goals**
- Make Jina-specific behavior representable in code before touching embedding flow.

### Tests To Write First (Red)

- [ ] Add `tests/test_semantic_model_profiles.py`:
  - assert `SUPPORTED_MODELS["jina-code-0.5b"]` exists
  - assert `hf_name == "jinaai/jina-code-embeddings-0.5b"`
  - assert `dimension == 896`
  - assert query prefix/instruction matches the locked nl2code query prompt
  - assert document prefix/instruction matches `Candidate code snippet:\n`
  - assert `tokenizer_kwargs["padding_side"] == "left"`
- [ ] Add `tests/test_semantic_model_registry_helpers.py`:
  - `_resolve_hf_model_name("jina-code-0.5b")` resolves correctly
  - `_canonical_model_id(...)` round-trips both key and HF id
  - `_model_dimension("jina-code-0.5b") == 896`
  - BGE still resolves to 1024 and keeps its current query prefix contract

### Green Implementation

- [ ] Extend `SUPPORTED_MODELS` in `tldr/semantic.py` with a richer model profile for:
  - `bge-large-en-v1.5`
  - `all-MiniLM-L6-v2`
  - `jina-code-0.5b`
- [ ] Add model-profile helpers, for example:
  - `_model_profile(model_name)`
  - `_query_prefix_for_model(model_name)`
  - `_document_prefix_for_model(model_name)`
  - `_sentence_transformer_kwargs_for_model(model_name)`
- [ ] Keep metadata fields backward compatible:
  - `model`
  - `dimension`
  - `count`

**Acceptance**
- Jina is selectable by model key and canonical HF id.
- Existing BGE and MiniLM lookups continue to work unchanged.

### Running Log (Phase 1)
- 2026-03-06: Added a richer `SUPPORTED_MODELS` contract in `tldr/semantic.py` for BGE, MiniLM, and Jina, including query/document prefixes plus model-specific tokenizer kwargs.
- 2026-03-06: Added registry/helper coverage for canonical HF ids, dimensions, and the locked BGE/Jina prefix contracts.

### Gotchas / Learnings (Phase 1)
- Backward compatibility is easiest when metadata continues to store canonical HF ids; keyed aliases stay an input concern only.
- The new registry contract can stay dict-based for now; a dataclass refactor is not required to express model-specific behavior.

## Phase 2: Query/Document Instruction Routing In The Embedding Pipeline

**Goals**
- Apply the right instruction at the right stage:
  - query prefix at search time
  - passage prefix at index time

### Running Log (Phase 2)
- 2026-03-06: `compute_embedding()` currently treats every text as a generic embedding input and `_semantic_unit_search()` hardcodes the BGE query prefix at one callsite.
- 2026-03-06: `build_semantic_index()` currently batches `build_embedding_text(unit)` directly, so Jina would silently miss the required document instruction without this phase.
- 2026-03-06: Added `build_document_embedding_text()` and `build_query_embedding_text()` and wired `build_semantic_index()` / `_semantic_unit_search()` through them.
- 2026-03-06: Extended isolated semantic-index coverage so the Jina path exercises a real 896-dimensional dummy embedding shape.

### Gotchas / Learnings (Phase 2)
- Keeping `compute_embedding()` generic avoids hiding query/document semantics inside the model loader.
- Search-time mismatch checks should happen before FAISS search and should fail on both metadata-vs-model and query-vector-vs-index dimension mismatches.

### Tests To Write First (Red)

- [ ] Add `tests/test_semantic_instruction_routing.py`:
  - Jina document embedding prepends `Candidate code snippet:\n`
  - Jina query embedding prepends `Find the most relevant code snippet given the following query:\n`
  - BGE query embedding still prepends `Represent this code search query: `
  - BGE document embedding remains unprefixed
- [ ] Extend `tests/test_semantic_index_isolated.py`:
  - replace the fixed 3-dimensional `DummyModel` with a configurable dummy so the Jina path is exercised with `dim=896`
  - assert metadata dimension follows the produced embedding shape
  - assert search/index mismatch errors still fire when metadata model differs
- [ ] Add `tests/test_semantic_index_rebuild_guards.py`:
  - indexing BGE artifacts and then attempting Jina without `--rebuild` raises the expected model mismatch
  - querying a BGE-built index with Jina raises the expected model/dimension mismatch

### Green Implementation

- [ ] Split embedding text preparation into explicit helpers, for example:
  - `build_document_embedding_text(unit, model_name)`
  - `build_query_embedding_text(query, model_name)`
- [ ] Update `build_semantic_index()` to batch document-prefixed text per model profile.
- [ ] Update `_semantic_unit_search()` / query embedding path to use model-specific query text.
- [ ] Keep FAISS dimension validation intact so old indexes fail fast and clearly.

**Acceptance**
- Jina indexes store 896-dimensional metadata and use the correct document/query instructions.
- BGE behavior remains unchanged.

## Phase 3: Loader Semantics (Left Padding, Caching, Fallback)

**Goals**
- Ensure the Jina model is loaded in a way that matches its decoder/last-token pooling requirements.

### Tests To Write First (Red)

- [ ] Add `tests/test_semantic_model_loading.py`:
  - monkeypatch `sentence_transformers.SentenceTransformer` and assert Jina load calls include `tokenizer_kwargs={"padding_side": "left"}`
  - assert BGE and MiniLM do not force left-padding
  - assert model cache invalidates correctly when switching `(model, device)`
- [ ] Add `tests/test_semantic_model_cache_keys.py` if the current global cache variables become insufficient once model kwargs differ by profile.
- [ ] If the sentence-transformers path proves insufficient, add tests first for a tiny adapter contract:
  - `.encode(str | list[str], normalize_embeddings=True, batch_size=...) -> np.ndarray`

### Green Implementation

- [ ] Update `get_model()` so model-specific kwargs are passed into `SentenceTransformer(...)`.
- [ ] Keep existing download confirmation/device behavior intact.
- [ ] If runtime testing shows sentence-transformers cannot make Jina correct, add a minimal internal adapter using `transformers` while preserving the existing `get_model(...).encode(...)` call pattern used elsewhere.

**Acceptance**
- Jina load path is explicit, tested, and not silently sharing a stale BGE-loaded model instance.

### Running Log (Phase 3)
- 2026-03-06: Verified the local `sentence-transformers` constructor already accepts `tokenizer_kwargs`, so Jina left-padding can stay on the existing load path.
- 2026-03-06: Updated `get_model()` cache identity to include model-specific loader kwargs in addition to `(hf_name, device)`.

### Gotchas / Learnings (Phase 3)
- Caching only on `(hf_name, device)` is too weak once loader kwargs differ by profile.
- No dedicated adapter is needed yet; keep that as a fallback only if runtime behavior disproves the `SentenceTransformer(..., tokenizer_kwargs=...)` path.

## Phase 4: CLI, Daemon, and Documentation Wiring

**Goals**
- Make Jina discoverable and configurable without making it the default yet.

### Running Log (Phase 4)
- 2026-03-06: Updated CLI help text to advertise Jina as an opt-in model and warn that switching models requires a rebuild.
- 2026-03-06: Wired daemon-triggered semantic indexing to honor `semantic.model` config while leaving search index-driven by default to avoid accidental config/index mismatches.
- 2026-03-06: Updated `README.md`, `docs/TLDR.md`, and `benchmarks/README.md` with opt-in Jina guidance, rebuild requirements, and the non-commercial license caveat.

### Gotchas / Learnings (Phase 4)
- Auto-applying daemon config `model` to search would create surprising failures against existing BGE-built indexes; config should drive indexing, not override search unless explicitly requested.
- Argparse wraps long model names across lines, so help-text tests need normalization before asserting on `jina-code-0.5b`.

### Tests To Write First (Red)

- [ ] Add `tests/test_cli_semantic_model_flags.py`:
  - `semantic index --model jina-code-0.5b` parses successfully
  - help text includes the Jina model key
- [ ] Add `tests/test_daemon_semantic_model_config.py`:
  - `_load_semantic_config()` accepts `model: "jina-code-0.5b"`
  - daemon defaults remain `bge-large-en-v1.5` before the rollout phase
- [ ] Add or extend docs-smoke tests only if the repo already has a pattern for them; otherwise keep docs validation manual.

### Green Implementation

- [ ] Update `tldr/cli.py` help text to list Jina alongside BGE and MiniLM.
- [ ] Update `tldr/daemon/core.py` semantic config comments/defaults to support Jina selection without flipping the default yet.
- [ ] Update docs:
  - `README.md`
  - `docs/TLDR.md`
  - `benchmarks/README.md`
- [ ] Add an explicit migration note:
  - Jina requires full semantic reindex due to dimension change
  - license is non-commercial unless separately licensed

**Acceptance**
- Users can select Jina from CLI/config.
- Docs state the reindex requirement and license caveat.

## Phase 5: Required Benchmark Runs (BGE Baseline vs Jina Candidate)

**Goals**
- Prove or disprove the migration with the local benchmark harness, not just unit tests.
- Run repo-scale comparison workloads through the daemon wherever the harness supports it so the evidence matches the intended production path.

### Immediate Next Steps (Phase 5)
- Run the deterministic red/green suite before launching long benchmark sessions so benchmark failures do not mask contract regressions.
- Use `repo:django-bge` as the current daemon/query baseline and `repo:django-jina05b` as the Jina candidate; reserve `repo:django-bge15` for fresh rebuild timing and strict rerun parity only.
- Treat daemon mode as the canonical protocol for repo-scale query runs (`retrieval`, `token efficiency`, `compound semantic+impact`, `perf`).
- Treat index-build timing/RSS and dependency-scoped semantic comparison as explicit CLI-only exceptions until daemon-backed equivalents exist.
- Run perf as a semantic-focused microbench (`--commands semantic_search --include-semantic`) rather than the broad default command set so the latency row reflects the model swap instead of unrelated `tree`/`structure` work.
- Use `scripts/bench_structured_retrieval.py` for deterministic exact-target tie-breaker runs against `rg`, `repo:django-bge`, and `repo:django-jina05b`; this is the canonical path for the optional `Structured retrieval F1` row.

### Gotchas / Learnings (Phase 5)
- Most benchmark runners do not take `--model`; the semantic model is whatever is already materialized behind the chosen `--index`.
- `scripts/bench_dep_indexes.py` is the exception and can compare BGE vs Jina directly without prebuilding separate repo indexes.
- `scripts/bench_retrieval_quality.py`, `scripts/bench_token_efficiency.py`, and `scripts/bench_compound_semantic_impact.py` now need `--use-daemon` for the migration's canonical evidence path.
- `scripts/bench_dep_indexes.py` is still CLI-oriented today; treat it as supplementary until it is daemonized or replaced with an equivalent daemon-backed dependency benchmark.
- Index build timing is intentionally CLI-only. There is no meaningful daemon-mode substitute for `semantic index`, so `/usr/bin/time -l uv run tldrf semantic index ...` remains the source of truth for build-time and peak-RSS evidence.
- Jina benchmarking should prewarm the model cache or set `TLDR_AUTO_DOWNLOAD=1`; the first unattended run failed at the interactive model-download prompt.
- Benchmark daemon startup must use a fresh foreground subprocess. The previous helper path (`start_daemon(... foreground=False)`) produced `Empty response from daemon` during semantic search, while a foreground daemon launched in its own process handled the same requests correctly.
- The retrieval benchmark had a loop-scoping regression during daemonization; it was fixed before collecting migration evidence so per-query aggregation now covers the full query set instead of only the final query.
- The existing `repo:django-bge` semantic index is good enough for the daemon-side BGE query baseline because its metadata already locks the corpus/model pair. Keep `repo:django-bge15` for fresh-build comparisons instead of blocking the whole matrix on a redundant rebuild.
- On Django, Jina is clearly better in plain semantic retrieval but not uniformly better once lane2/rerank heuristics are enabled. The gate cannot rely on the semantic-only win alone.
- The broad perf microbench was dominated by slow non-semantic commands (`tree` in particular), so it is not a good default-model latency proxy for this migration. Use the new semantic-only command filter for the canonical perf row.
- The new structured-retrieval harness resolves exact symbol targets from the existing Django retrieval suite deterministically, but the first real run only yielded `45` scoreable symbol queries plus `3` negatives; `7` queries were ambiguous and `5` were unsupported module-level targets.
- Structured retrieval F1 strongly favors exact-match backends like native `rg`. It is useful as a capability-gap metric, but it should not override the higher-level semantic retrieval gains when the product question is concept search rather than exact definition lookup.

### Running Log (Phase 5)
- 2026-03-06: Added `--use-daemon` support to the retrieval, token-efficiency, and compound benchmark runners and switched the perf runner onto the same benchmark daemon helper.
- 2026-03-06: Replaced the benchmark daemon helper with a foreground subprocess launcher after reproducing `semantic search` failures on the old background/fork path.
- 2026-03-06: Verified daemon-mode smokes for retrieval, token efficiency, and compound against the existing `repo:django` semantic index before launching the migration matrix.
- 2026-03-06: Confirmed `repo:django-bge15` only had structural artifacts (`call_graph.json`) and required a fresh semantic rebuild before any BGE-vs-Jina comparison.
- 2026-03-06: Archived the daemon-mode BGE query baseline from `repo:django-bge`:
  - retrieval: `benchmark/runs/20260306-101803Z-retrieval-django-bge-daemon.json`
  - lane2 retrieval: `benchmark/runs/20260306-101803Z-retrieval-django-bge-lane2-daemon.json`
  - token efficiency: `benchmark/runs/20260306-101949Z-token-efficiency-django-bge-daemon.json`
  - compound semantic+impact: `benchmark/runs/20260306-102247Z-compound-semantic-impact-django-bge-daemon.json`
- 2026-03-06: Recorded current BGE daemon baseline metrics:
  - semantic MRR@10 `0.6022`, Recall@5 `0.7719`, Recall@10 `0.7895`, Precision@5 `0.1544`
  - hybrid_rrf MRR@10 `0.8684`, Recall@10 `1.0000`
  - lane2 MRR@10 `0.8741`
  - token-efficiency MRR @1000: semantic `0.6124`, hybrid_rrf `0.8597`
  - compound TTE p50 ratio `1.0250`
- 2026-03-06: Started the Jina Django semantic build in `tmux` session `jina-bench-jina05b-build`; artifact writes are expected only after the in-memory embedding pass completes.
- 2026-03-06: Completed the Jina Django semantic build:
  - log: `benchmark/logs/20260306-101553Z-django-jina05b-semantic.log`
  - artifacts: `repo:django-jina05b`, model `jinaai/jina-code-embeddings-0.5b`, dimension `896`, count `35712`
  - wall time `692.57s`
  - max RSS `3360096256` bytes
- 2026-03-06: Archived Jina daemon retrieval artifacts:
  - retrieval: `benchmark/runs/20260306-103455Z-retrieval-django-jina05b-daemon.json`
  - lane2 retrieval: `benchmark/runs/20260306-103542Z-retrieval-django-jina05b-lane2-daemon.json`
  - token efficiency: `benchmark/runs/20260306-103645Z-token-efficiency-django-jina05b-daemon.json`
- 2026-03-06: Recorded current Jina daemon metrics:
  - semantic MRR@10 `0.7023`, Recall@5 `0.8596`, Recall@10 `0.8772`, Precision@5 `0.1719`
  - hybrid_rrf MRR@10 `0.8686`, Recall@10 `1.0000`
  - lane2 MRR@10 `0.8417`
  - token-efficiency MRR @1000: semantic `0.7048`, hybrid_rrf `0.8647`
- 2026-03-06: Archived Jina compound semantic+impact artifact `benchmark/runs/20260306-103725Z-compound-semantic-impact-django-jina05b-daemon.json` with:
  - compound TTE p50 ratio `1.1361`
  - same `partial_rate` as BGE (`0.9167`)
  - identical retrieval overlap / impact caller Jaccard (`1.0`), so the regression is latency/payload rather than correctness drift
- 2026-03-06: Archived semantic-only BGE perf baselines:
  - `benchmark/runs/20260306-103913Z-perf-daemon-vs-cli-django-bge-semantic.json` (`iterations=10`)
  - `benchmark/runs/20260306-104233Z-perf-daemon-vs-cli-django-bge-semantic-i3.json` (`iterations=3`)
  - current concise reference row: daemon p50 `135.7ms`, daemon p95 `138.8ms`, CLI p50 `6929.8ms`, speedup `51.1x`
- 2026-03-06: Abandoned the original all-command perf run for `repo:django-bge` because the `tree` command dominated runtime and did not isolate model-driven latency. The canonical perf rerun should use `scripts/bench_perf_daemon_vs_cli.py --commands semantic_search --include-semantic`.
- 2026-03-06: Patched the compound/perf benchmark reports to persist index/model provenance before collecting the remaining Jina artifacts.
- 2026-03-06: Attempted matching semantic-only Jina perf runs at `iterations=10` and `iterations=3`, but both remained materially slower/heavier than the BGE baseline and were stopped instead of spending more benchmark time on repeated CLI model reloads. Keep this as a CLI/startup operational warning rather than a daemon-query conclusion.
- 2026-03-06: Closed the daemon-query latency gap with `benchmark/runs/20260306-adhoc-daemon-semantic-latency-bge-vs-jina.json` (`iterations=10`, query `entry implementation`):
  - BGE daemon semantic search: mean `159.8ms`, p50 `150.2ms`, p95 `250.2ms`
  - Jina daemon semantic search: mean `174.6ms`, p50 `166.5ms`, p95 `286.4ms`
  - steady-state daemon delta: Jina is about `+16.2ms` / `+10.8%` at p50 and `+36.2ms` / `+14.4%` at p95, which is modest compared with the larger startup/build penalties
- 2026-03-06: Added `scripts/bench_structured_retrieval.py` and targeted tests to score exact structured retrieval across `rg` and daemon-backed semantic indexes.
- 2026-03-06: Ran the first Django structured-retrieval comparison artifact `benchmark/runs/20260306-182147Z-structured-retrieval-django.json` with:
  - query set: `32` total (`31` positive exact-definition targets, `1` negative), daemon-backed for `repo:django-bge` and `repo:django-jina05b`
  - `rg-native` micro F1 `0.9841` (`tp=31`, `fp=1`, `fn=0`), p50 latency `84.8ms`
  - BGE micro F1 `0.1602` (`tp=27`, `fp=279`, `fn=4`), p50 latency `142.1ms`
  - Jina micro F1 `0.1520` (`tp=26`, `fp=285`, `fn=5`), p50 latency `140.3ms`
  - interpretation: this exact-match suite is strongly lexical-friendly, so `rg-native` dominates. Within the semantic backends, BGE slightly beats Jina on micro F1 while Jina is only marginally faster on p50.
- 2026-03-06: Extended `scripts/bench_structured_retrieval.py` with opt-in hybrid backends (`--include-hybrid`) that project hybrid file hits back to structured unit predictions via semantic-unit rows from the same files.
- 2026-03-06: Ran the behavior-oriented Django structured-retrieval comparison artifact `benchmark/runs/20260306-185707Z-structured-retrieval-django.json` with:
  - query set: `17` total (`16` positive behavior-oriented targets, `1` negative), daemon-backed for `repo:django-bge` and `repo:django-jina05b`, `top_k=5`, `--include-hybrid`
  - `rg-native` micro F1 `0.0217` (`tp=1`, `fp=75`, `fn=15`), p50 latency `87.3ms`
  - BGE semantic micro F1 `0.1458` (`tp=7`, `fp=73`, `fn=9`), p50 latency `169.5ms`
  - Jina semantic micro F1 `0.1053` (`tp=5`, `fp=74`, `fn=11`), p50 latency `185.9ms`
  - BGE hybrid-projected micro F1 `0.1584` (`tp=8`, `fp=77`, `fn=8`), p50 latency `422.9ms`
  - Jina hybrid-projected micro F1 `0.1188` (`tp=6`, `fp=79`, `fn=10`), p50 latency `438.5ms`
  - interpretation: for concept-style exact-target recovery, `rg-native` is not competitive. Semantic and hybrid both beat lexical `rg` by a wide margin, and hybrid gives both models a small lift over semantic-only, but BGE still beats Jina on this stricter structured row.
  - concrete examples:
    - BGE hybrid recovered `SB13` (`django.contrib.auth.decorators.py.login_required`) while Jina hybrid still missed it, reinforcing the earlier ranking-behavior difference rather than suggesting a migration bug.
    - Jina hybrid recovered behavior targets that semantic-only missed, including `SB03` (`technical_500_response`), `SB09` (`atomic`), and `SB16` (`QuerySet`).
  - gotcha: these hybrid latencies are benchmark-harness latencies for `hybrid file retrieval + semantic unit projection`, not raw hybrid file-search latency.
- 2026-03-06: Reran the same behavior-oriented structured suite with `--hybrid-no-result-guard rg_empty` in artifact `benchmark/runs/20260306-190233Z-structured-retrieval-django.json`:
  - semantic-only rows stayed effectively unchanged
  - both hybrid rows collapsed to empty predictions for every positive query: BGE hybrid micro F1 `0.0000` (`tp=0`, `fp=0`, `fn=16`), Jina hybrid micro F1 `0.0000` (`tp=0`, `fp=0`, `fn=16`)
  - negative-query suppression did work: `SB17` hybrid false positives dropped from `5` to `0` for both BGE and Jina
  - interpretation: `rg_empty` is too aggressive for this behavior-oriented suite because the current lexical `rg_pattern`s are not reliable guards for concept-style queries. Do not use this guarded artifact as the main hybrid quality signal.
  - immediate next step if we want a production-leaning guarded hybrid row: either curate guard-quality `rg_pattern`s per behavior query or evaluate a softer hybrid guard than strict `rg_empty`.

### Tier 1: Direct Embedding Quality (Must Run)

- [ ] Build the Jina Django semantic index:
  ```bash
  /usr/bin/time -l uv run tldrf semantic index \
    --cache-root benchmark/cache-root \
    --index repo:django-jina05b \
    --lang python \
    --model jina-code-0.5b \
    --rebuild \
    benchmark/corpora/django
  ```
- [ ] Run retrieval quality against BGE and Jina indexes:
  ```bash
  uv run python scripts/bench_retrieval_quality.py \
    --corpus django \
    --cache-root benchmark/cache-root \
    --index repo:django-bge \
    --ks 5,10 \
    --use-daemon

  uv run python scripts/bench_retrieval_quality.py \
    --corpus django \
    --cache-root benchmark/cache-root \
    --index repo:django-jina05b \
    --ks 5,10 \
    --use-daemon
  ```
  Read from the same report:
  - `results.agg_positive.semantic`
  - `results.agg_positive.hybrid_rrf`
  - `results.agg_negative.*` for false-positive behavior
- [ ] Run lane2 retrieval for both models:
  ```bash
  uv run python scripts/bench_retrieval_quality.py \
    --corpus django \
    --cache-root benchmark/cache-root \
    --index repo:django-bge \
    --ks 5,10 \
    --use-daemon \
    --lane2-abstain-threshold 0.35 \
    --lane2-rerank \
    --lane2-rerank-top-n 8 \
    --lane2-max-latency-ms-p50-ratio 1.10 \
    --lane2-max-payload-tokens-median-ratio 0.90

  uv run python scripts/bench_retrieval_quality.py \
    --corpus django \
    --cache-root benchmark/cache-root \
    --index repo:django-jina05b \
    --ks 5,10 \
    --use-daemon \
    --lane2-abstain-threshold 0.35 \
    --lane2-rerank \
    --lane2-rerank-top-n 8 \
    --lane2-max-latency-ms-p50-ratio 1.10 \
    --lane2-max-payload-tokens-median-ratio 0.90
  ```
- [ ] Run dependency benchmarks for both models:
  ```bash
  uv run python scripts/bench_dep_indexes.py \
    --dep requests,urllib3 \
    --lang python \
    --model bge-large-en-v1.5

  uv run python scripts/bench_dep_indexes.py \
    --dep requests,urllib3 \
    --lang python \
    --model jina-code-0.5b
  ```

### Tier 2: Downstream Impact (Should Run)

- [ ] Run token-efficiency retrieval benchmarks for both models:
  ```bash
  uv run python scripts/bench_token_efficiency.py \
    --corpus django \
    --mode retrieval \
    --cache-root benchmark/cache-root \
    --index repo:django-bge \
    --budgets 500,1000,2000,5000 \
    --use-daemon

  uv run python scripts/bench_token_efficiency.py \
    --corpus django \
    --mode retrieval \
    --cache-root benchmark/cache-root \
    --index repo:django-jina05b \
    --budgets 500,1000,2000,5000 \
    --use-daemon
  ```
- [ ] Run compound semantic+impact for both models:
  ```bash
  uv run python scripts/bench_compound_semantic_impact.py \
    --corpus django \
    --cache-root benchmark/cache-root \
    --index repo:django-bge \
    --budget-tokens 2000 \
    --k 8 \
    --query-limit 12 \
    --use-daemon \
    --retrieval-mode hybrid \
    --no-result-guard rg_empty \
    --impact-depth 3 \
    --impact-limit 3 \
    --impact-language python \
    --lane2-abstain-threshold 0.35 \
    --lane2-rerank \
    --lane2-rerank-top-n 8

  uv run python scripts/bench_compound_semantic_impact.py \
    --corpus django \
    --cache-root benchmark/cache-root \
    --index repo:django-jina05b \
    --budget-tokens 2000 \
    --k 8 \
    --query-limit 12 \
    --use-daemon \
    --retrieval-mode hybrid \
    --no-result-guard rg_empty \
    --impact-depth 3 \
    --impact-limit 3 \
    --impact-language python \
    --lane2-abstain-threshold 0.35 \
    --lane2-rerank \
    --lane2-rerank-top-n 8
  ```
### Tier 4: Optional LLM Validation (Last Phase Only)

- [ ] Only after deterministic retrieval/perf evidence is in place, optionally run prompt generation for each model/index:
  ```bash
  uv run python scripts/bench_llm_ab_prompts.py \
    --corpus django \
    --cache-root benchmark/cache-root \
    --index repo:django-bge \
    --budget-tokens 2000

  uv run python scripts/bench_llm_ab_prompts.py \
    --corpus django \
    --cache-root benchmark/cache-root \
    --index repo:django-jina05b \
    --budget-tokens 2000
  ```
- [ ] Optionally run structured answer scoring for both prompt packets:
  ```bash
  uv run python scripts/bench_llm_ab_run.py \
    --mode structured \
    --prompts <bge-prompts.jsonl> \
    --provider <provider> \
    --model <answer-model>

  uv run python scripts/bench_llm_ab_run.py \
    --mode structured \
    --prompts <jina-prompts.jsonl> \
    --provider <provider> \
    --model <answer-model>
  ```
- [ ] Leave judge-mode A/B as the final optional tie-breaker only if earlier evidence is still ambiguous:
  ```bash
  uv run python scripts/bench_llm_ab_run.py \
    --mode judge \
    --prompts <bge-prompts.jsonl> \
    --provider <provider> \
    --model <answer-model> \
    --judge-provider <judge-provider> \
    --judge-model <judge-model>

  uv run python scripts/bench_llm_ab_run.py \
    --mode judge \
    --prompts <jina-prompts.jsonl> \
    --provider <provider> \
    --model <answer-model> \
    --judge-provider <judge-provider> \
    --judge-model <judge-model>
  ```

### Tier 3: Performance Regression (Must Run)

- [ ] Record index build wall time and max RSS from `/usr/bin/time -l` for both model builds.
- [ ] Run semantic latency microbench for both indexes:
  ```bash
  uv run python scripts/bench_perf_daemon_vs_cli.py \
    --corpus django \
    --lang python \
    --cache-root benchmark/cache-root \
    --index repo:django-bge \
    --include-semantic \
    --commands semantic_search

  uv run python scripts/bench_perf_daemon_vs_cli.py \
    --corpus django \
    --lang python \
    --cache-root benchmark/cache-root \
    --index repo:django-jina05b \
    --include-semantic \
    --commands semantic_search
  ```
- [ ] Record:
  - semantic search p50/p95
  - daemon vs CLI speedup
  - index artifact sizes

### Tier 5: Optional Head-to-Head Suite (Nice To Have)

- [ ] Only after Tier 1-3 look promising, add model-specific tool/profile wiring or paired prediction artifacts and run `scripts/bench_head_to_head.py`.
- [ ] Keep this out of the critical path for the initial opt-in implementation.

**Acceptance**
- All required benchmark artifacts exist for both BGE and Jina.
- Reports are comparable by corpus, index id, budget, and answer-model configuration.

## Phase 6: Decision Gate

**Goals**
- Decide whether Jina should remain opt-in, become the default, or be rejected.

### Immediate Next Steps (Phase 6)
- Do not finalize the comparison table until both BGE and Jina reports exist for the same corpus and semantic-model identity. The current BGE daemon baseline may come from `repo:django-bge`; use `repo:django-bge15` only where fresh rebuild timing/RSS parity matters.
- Treat the current code changes as enabling work only; the gate is still pending benchmark evidence plus the license decision.
- `Structured retrieval F1` is now available as an optional deterministic tie-breaker row. `Judge win rate` remains the final optional phase only if earlier evidence is still ambiguous.
- Current benchmark signal is already sufficient to block `PROCEED_TO_DEFAULT`: Jina improves plain semantic retrieval, but lane2 and compound regress, and operational cost is clearly higher. At best, the current evidence supports `KEEP_OPT_IN`.
- Immediate next steps for the structured lane if we need tighter product fidelity:
  - if we want to keep `rg_empty` in the structured hybrid evaluation, first curate behavior-query guard patterns that actually hit the intended implementation files; the current broad concept patterns suppress nearly all positive queries
  - if structured hybrid scoring becomes a gating metric, replace the current semantic-unit projection resolver with a per-file symbol resolver so the metric is closer to what a file-first hybrid UX would actually surface

### Required Comparison Table

Fill this table from the archived reports:

| Metric | BGE baseline | Jina | Delta | Decision note |
| --- | --- | --- | --- | --- |
| MRR@10 (semantic) | 0.6022 | 0.7023 | +0.1001 | `benchmark/runs/20260306-101803Z-retrieval-django-bge-daemon.json`, `benchmark/runs/20260306-103455Z-retrieval-django-jina05b-daemon.json` |
| Recall@5 (semantic) | 0.7719 | 0.8596 | +0.0877 | `benchmark/runs/20260306-101803Z-retrieval-django-bge-daemon.json`, `benchmark/runs/20260306-103455Z-retrieval-django-jina05b-daemon.json` |
| Recall@10 (semantic) | 0.7895 | 0.8772 | +0.0877 | `benchmark/runs/20260306-101803Z-retrieval-django-bge-daemon.json`, `benchmark/runs/20260306-103455Z-retrieval-django-jina05b-daemon.json` |
| Precision@5 (semantic) | 0.1544 | 0.1719 | +0.0175 | `benchmark/runs/20260306-101803Z-retrieval-django-bge-daemon.json`, `benchmark/runs/20260306-103455Z-retrieval-django-jina05b-daemon.json` |
| MRR@10 (hybrid_rrf) | 0.8684 | 0.8686 | +0.0001 | `benchmark/runs/20260306-101803Z-retrieval-django-bge-daemon.json`, `benchmark/runs/20260306-103455Z-retrieval-django-jina05b-daemon.json` |
| Recall@10 (hybrid_rrf) | 1.0000 | 1.0000 | +0.0000 | `benchmark/runs/20260306-101803Z-retrieval-django-bge-daemon.json`, `benchmark/runs/20260306-103455Z-retrieval-django-jina05b-daemon.json` |
| Lane2 MRR@10 | 0.8741 | 0.8417 | -0.0324 | `benchmark/runs/20260306-101803Z-retrieval-django-bge-lane2-daemon.json`, `benchmark/runs/20260306-103542Z-retrieval-django-jina05b-lane2-daemon.json` |
| Token-efficiency retrieval MRR @1000 | semantic 0.6124 / hybrid_rrf 0.8597 | semantic 0.7048 / hybrid_rrf 0.8647 | +0.0924 / +0.0050 | `benchmark/runs/20260306-101949Z-token-efficiency-django-bge-daemon.json`, `benchmark/runs/20260306-103645Z-token-efficiency-django-jina05b-daemon.json` |
| Compound TTE p50 ratio | 1.0250 | 1.1361 | +0.1111 | `benchmark/runs/20260306-102247Z-compound-semantic-impact-django-bge-daemon.json`, `benchmark/runs/20260306-103725Z-compound-semantic-impact-django-jina05b-daemon.json` |
| Structured retrieval F1 | 0.1602 | 0.1520 | -0.0082 | `benchmark/runs/20260306-182147Z-structured-retrieval-django.json`; `rg-native` on the same run is `0.9841`, so this row is useful for cross-tool sanity checking but is too lexical-friendly to override the primary semantic-retrieval decision |
| Structured retrieval F1 (behavior suite, semantic) | 0.1458 | 0.1053 | -0.0406 | `benchmark/runs/20260306-185707Z-structured-retrieval-django.json`; on this concept-oriented suite `rg-native` collapses to `0.0217`, so semantic retrieval is clearly buying something real even though BGE still leads Jina |
| Structured retrieval F1 (behavior suite, hybrid) | 0.1584 | 0.1188 | -0.0396 | `benchmark/runs/20260306-185707Z-structured-retrieval-django.json`; hybrid lifts both models over their semantic-only rows on the same suite, but BGE still wins and the measured latency here includes file-to-symbol projection inside the benchmark harness |
| Structured retrieval F1 (behavior suite, hybrid, `rg_empty` guard) | 0.0000 | 0.0000 | +0.0000 | `benchmark/runs/20260306-190233Z-structured-retrieval-django.json`; useful only as a guard-sensitivity check. The strict lexical guard suppressed all positive hybrid queries while fixing the negative query, so this should not be treated as the representative hybrid quality row |
| Judge win rate |  |  |  |  |
| Dependency MRR (`requests`) |  |  |  |  |
| Dependency MRR (`urllib3`) |  |  |  |  |
| Index build time (s) |  | 692.57 |  | Jina build log `benchmark/logs/20260306-101553Z-django-jina05b-semantic.log`; comparable BGE rebuild still pending |
| Query latency p50 (ms) | 150.2 | 166.5 | +16.2 | `benchmark/runs/20260306-adhoc-daemon-semantic-latency-bge-vs-jina.json`; use this as the current apples-to-apples daemon semantic-search latency row |
| Query latency p95 (ms) | 250.2 | 286.4 | +36.2 | `benchmark/runs/20260306-adhoc-daemon-semantic-latency-bge-vs-jina.json`; the formal CLI-heavy Jina perf rerun remains unnecessary for the daemon-first decision |
| Peak RSS (MB) |  | 3360.1 |  | Converted from `3360096256` bytes max RSS in the Jina build log; comparable BGE rebuild still pending |

### Cross-Tool Structured Summary

Use this table when the question is not just "BGE vs Jina?" but "which retrieval mode is actually the right tool for this job?"

**Structured Retrieval Micro F1**

| Suite | `rg-native` | BGE semantic | Jina semantic | BGE hybrid | Jina hybrid | What it says |
| --- | --- | --- | --- | --- | --- | --- |
| Exact-definition suite (`benchmark/runs/20260306-182147Z-structured-retrieval-django.json`) | 0.9841 | 0.1602 | 0.1520 |  |  | Exact symbol/definition lookup is a lexical problem first. Use `rg-native`; neither semantic backend is close, and Jina does not beat BGE. |
| Behavior suite (`benchmark/runs/20260306-185707Z-structured-retrieval-django.json`) | 0.0217 | 0.1458 | 0.1053 | 0.1584 | 0.1188 | Concept-style exact-target recovery is where semantic/hybrid adds value over `rg-native`. Jina is useful relative to `rg-native`, but BGE still wins across both semantic and hybrid. |
| Behavior suite with hybrid `rg_empty` guard (`benchmark/runs/20260306-190233Z-structured-retrieval-django.json`) | 0.0217 | 0.1458 | 0.1053 | 0.0000 | 0.0000 | The current behavior-query lexical guards are too weak for strict `rg_empty`. This run is a guard-sensitivity check, not a representative hybrid-quality row. |

**Structured Retrieval p50 Latency (ms)**

| Suite | `rg-native` | BGE semantic | Jina semantic | BGE hybrid | Jina hybrid | What it says |
| --- | --- | --- | --- | --- | --- | --- |
| Exact-definition suite (`benchmark/runs/20260306-182147Z-structured-retrieval-django.json`) | 84.8 | 142.1 | 140.3 |  |  | `rg-native` is the fastest and best on exact lookup. Jina is only marginally faster than BGE here and still worse on quality. |
| Behavior suite (`benchmark/runs/20260306-185707Z-structured-retrieval-django.json`) | 87.3 | 169.5 | 185.9 | 422.9 | 438.5 | Semantic beats `rg-native` on quality for concept lookup, but hybrid is substantially slower in this harness because it includes deterministic file-to-symbol projection. |
| Behavior suite with hybrid `rg_empty` guard (`benchmark/runs/20260306-190233Z-structured-retrieval-django.json`) | 85.2 | 186.5 | 189.8 | 358.0 | 358.3 | The guard reduces hybrid work, but it only looks "faster" because it suppresses nearly all positive results. Do not interpret this as a real hybrid latency win. |

**Cross-Tool Decision Summary**

- `rg-native` is the best tool for exact identifier and definition lookup.
- BGE semantic or hybrid is the strongest option we have for concept-style structured retrieval on the current Django evidence.
- Jina is useful relative to `rg-native` on concept-style retrieval, but the current evidence does not show a case where Jina beats BGE on these structured rows.
- Strict `rg_empty` hybrid guards should not be used with broad behavior-query lexical patterns unless those guard patterns are curated to hit the intended implementation files.

### Defining Structured Retrieval F1

- Treat this as an optional deterministic tie-breaker for retrieval quality, not a replacement for MRR/Recall.
- Goal: measure whether the retrieved rows resolve to the correct structured code targets, not just whether a relevant file appeared somewhere in the ranking.
- Score only the top-`k` normalized retrieval rows after deduplication.
- Record the table row as the micro-averaged F1 over all scored queries. Keep per-query precision, recall, and F1 in the artifact for debugging.

**Normalized Prediction Schema**

Use this normalized object shape for every retrieved row before scoring:

```json
{
  "file": "repo/relative/path.py",
  "qualified_symbol": "package.module.Class.method",
  "symbol_kind": "function|method|class|module|unknown",
  "start_line": 123,
  "end_line": 145,
  "rank": 1
}
```

- `file` is required and must be repo-relative and normalized to `/`.
- `qualified_symbol` is the primary identity field when present.
- `symbol_kind`, `start_line`, and `end_line` are supporting fields for debugging and fallback matching.
- `rank` is not part of the identity key; it is only preserved for auditability.

**Gold Target Schema**

Curated benchmark labels should use the same identity surface, minus rank:

```json
{
  "query_id": "django-q001",
  "targets": [
    {
      "file": "django/contrib/admin/options.py",
      "qualified_symbol": "django.contrib.admin.options.ModelAdmin.get_search_results",
      "symbol_kind": "method",
      "start_line": 1181,
      "end_line": 1207
    }
  ]
}
```

**Identity Key And Matching**

- Primary key: `(file, qualified_symbol)` when `qualified_symbol` exists in both gold and prediction.
- Fallback key: `(file, start_line, end_line)` only when symbol identity is unavailable on one or both sides.
- If neither `qualified_symbol` nor a valid span is available, the row is unscorable and should be excluded from this metric while incrementing an `unscorable_predictions` or `unscorable_gold_targets` counter in the artifact.
- Deduplicate predictions and gold targets by identity key before counting matches.

**Counting Rules**

- `tp`: predicted identity keys that exactly match a gold identity key for that query.
- `fp`: predicted identity keys in the scored top-`k` set that do not appear in the gold set.
- `fn`: gold identity keys not recovered by the scored top-`k` prediction set.
- Per-query metrics:
  - `precision = tp / (tp + fp)`
  - `recall = tp / (tp + fn)`
  - `f1 = 2 * precision * recall / (precision + recall)`
- Table metric:
  - micro-average by summing `tp`, `fp`, and `fn` across all queries first, then computing precision, recall, and F1 from those totals
  - keep macro-average as a secondary debug row if needed, but do not use it as the Phase 6 comparison row

**Scoring Notes**

- This metric is intentionally stricter than file-level retrieval metrics. Returning the correct file but the wrong function should count as a miss when symbol identity is available.
- For queries with multiple valid answers, include every acceptable target in the gold `targets` array.
- If a query is intentionally file-level only, omit `qualified_symbol` from the gold label and force span-based matching for that query.
- Do not use LLM judgment in this row; this is a deterministic set-overlap metric.
- Current first-pass Django suite is intentionally definition-oriented so `rg-native` can participate fairly without AST heuristics. Treat it as a cross-tool exact-match sanity check, not as the primary signal for the BGE-vs-Jina semantic default decision.
- Second-pass behavior-oriented Django suite uses broader natural-language queries and broader `rg_pattern`s to test concept search against exact structured targets. Treat this as the more representative cross-tool check for whether semantic or hybrid retrieval adds product value over native `rg`.
- Hybrid structured runs currently score a deterministic projection:
  - retrieve top files with `retrieval_mode=hybrid`
  - run semantic unit retrieval for the same query
  - keep unit rows whose files appear in the hybrid top-file set
  - sort by hybrid file rank first, semantic unit rank second, then score the resulting structured rows
- This makes the hybrid row useful today, but it is still an approximation of a file-first UX. If the hybrid structured row becomes gating, replace the projection with a per-file symbol resolver.
- `rg_empty` guard is only as good as the lexical guard pattern. On the current behavior-oriented suite, enabling `rg_empty` suppressed both hybrid backends to zero because the broad concept `rg_pattern`s rarely produced lexical matches even when the semantic answer was recoverable. Use this guard only with query labels whose lexical patterns are known-good.
- Immediate next step if this row is implemented: add a dedicated benchmark artifact that persists `per_query`, `micro_totals`, `micro_precision`, `micro_recall`, `micro_f1`, and unscorable-count diagnostics for both BGE and Jina daemon runs.

### Interpreting Compound TTE

- `tte_ms_p50_ratio` is `compound median time-to-evidence / sequential median time-to-evidence`.
- `1.0` means the compound path is equally efficient to running `semantic_search + bounded per-row impact_analysis` sequentially.
- `>1.0` means the compound feature adds overhead relative to the sequential baseline; `<1.0` means it is actually buying back time.
- BGE at `1.0250` means compound is about `2.5%` slower than sequential on the median query. Jina at `1.1361` means compound is about `13.6%` slower than sequential on the median query.
- This metric is a relative efficiency check, not an absolute latency comparison across models. Jina compound p50 is still lower in absolute terms than BGE compound p50, but it is a worse trade relative to Jina's own faster sequential baseline.
- Product rule: do not let this metric alone block the default-model decision unless `compound-impact` is part of the default user path. Use it as a secondary gate: block the default flip only if compound shows correctness drift or materially worse user-visible latency for a feature we intend to make first-class by default.
- Current interpretation: this is an efficiency regression, not a correctness regression, because retrieval overlap and impact-caller Jaccard both stayed at `1.0`. It supports `KEEP_OPT_IN` for Jina and argues against advertising compound as a default-path win, but lane2 regression and operational/build cost remain the stronger blockers for any default flip.

### Gate Rules

- `PROCEED_TO_DEFAULT` only if all are true:
  - license is not a blocker
  - semantic MRR improves by at least 5 percent relative or hybrid quality improves materially with no major regression elsewhere
  - query latency does not regress by more than 2x on CPU/MPS
  - no correctness regressions in deterministic tests
- `KEEP_OPT_IN` if:
  - quality is better or neutral
  - performance is acceptable
  - but license or rollout risk still blocks default change
- `ROLL_BACK_JINA` if any are true:
  - deterministic correctness remains flaky
  - semantic/hybrid quality regresses materially
  - CPU latency regresses by more than 2x and Matryoshka/fallback work is not yet justified
  - license blocks intended usage

### Fallback Option

- If quality is clearly better but CPU performance fails the gate:
  - run a follow-up spike on Matryoshka truncation
  - do not change the default in the same PR/implementation wave

**Acceptance**
- One explicit decision is recorded: `PROCEED_TO_DEFAULT`, `KEEP_OPT_IN`, or `ROLL_BACK_JINA`.

## Phase 7: Rollout (Only If Gate Passes)

**Goals**
- Safely expose the chosen default and communicate the rebuild requirement.

### Immediate Next Steps (Phase 7)
- Leave `DEFAULT_MODEL` and daemon defaults untouched until Phase 6 explicitly records `PROCEED_TO_DEFAULT`.
- If the gate stalls on license or performance, ship the current work as opt-in support and keep the rollout phase open.

### Tests To Write First (Red)

- [ ] Add/extend tests that assert the chosen default model if and only if the gate has passed and the rollout PR intends to flip it.
- [ ] Add docs/manual validation checklist for fresh-index rebuild behavior.

### Green Implementation

- [ ] Update `DEFAULT_MODEL` in `tldr/semantic.py` only if Phase 6 says `PROCEED_TO_DEFAULT`.
- [ ] Update daemon default config in `tldr/daemon/core.py`.
- [ ] Update user-facing docs and examples to match the final decision.
- [ ] Add a migration note everywhere semantic indexing is documented:
  - old semantic indexes must be rebuilt
  - example command:
    ```bash
    uv run tldrf semantic index --cache-root <cache-root> --index <index-id> --lang <lang> --model <model> --rebuild <path>
    ```

**Acceptance**
- Default selection, docs, and rebuild guidance are all aligned.

## Acceptance Criteria (Summary)

- Deterministic test coverage exists for:
  - Jina model registry entry
  - query/document instruction routing
  - left-padding loader behavior
  - model/dimension mismatch guards
  - CLI/daemon model wiring
- `uv run pytest` passes with the new and updated tests.
- Benchmark artifacts exist for all required benchmark families named in the migration plan:
  - retrieval quality
  - dependency indexes
  - token efficiency
  - compound semantic+impact
  - performance regression
- Optional last-phase artifacts may also exist for:
  - LLM A/B
- The decision gate is explicitly recorded before any default-model flip.
- If the gate does not pass, Jina support may still ship as opt-in, but BGE remains the default.
