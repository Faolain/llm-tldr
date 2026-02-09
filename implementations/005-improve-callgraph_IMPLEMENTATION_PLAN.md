# TypeScript Compiler-Grade Callgraph Implementation Plan

- Status: Implemented (Phases 0-4, 6). Remaining: Phase 5.
- Owner: TBD
- Last updated: 2026-02-09
- Source: `specs/005-improve-callgraph.md` (Conclusion + FutureWork)

## Decisions & Assumptions (locked for this plan)

- Primary goal is **TypeScript compiler-grade resolution** for call graphs (workspace/module resolution + symbol/type binding), not syntax-only heuristics.
- First target is **TypeScript/JavaScript** call graph quality; other language call graphs remain unchanged except for shared infrastructure (caching/trace UX).
- Edges are **conservative**: only emit a call edge when resolution is confident; do not guess for dynamic patterns.
- Outputs are **stable and deterministic** across runs (canonical paths, stable node IDs, stable sort order).
- When TS resolution is unavailable (no Node, no TypeScript, ambiguous tsconfig), `tldrf` must **fall back cleanly** to the current syntax-only TS call graph and mark results as incomplete.
- MVP will implement TS resolution via the **TypeScript compiler API / language service** (no long-lived tsserver process). tsserver remains a contingency only if needed.

## Approach Options (tsserver/LSP vs Compiler API)

This plan documents two approaches, but the MVP is **definitively Option A**. Option B is only a fallback if Option A cannot meet the fixture + curated-edge success criteria.

- Option A: TypeScript compiler API / language service (no tsserver process)
  - How: run a small Node helper that loads the repo tsconfig(s), builds a `Program` + `TypeChecker` (or `LanguageService`), and resolves each callsite to a concrete definition.
  - Pros: fewer moving parts; direct control; easier to make deterministic.
  - Cons: you must correctly handle project references/tsconfig selection and keep performance acceptable.
- Option B: tsserver (LSP-ish)
  - How: run `tsserver` for the workspace, then ask for definition/signature info per callsite; convert those results into call edges.
  - Pros: battle-tested project + module resolution (paths, project references, pnpm workspace layout).
  - Cons: orchestration/caching complexity; request volume/perf; still need logic to turn “definitions” into edges.

Fallback switch criteria (to justify implementing tsserver):
- Option A cannot reach the fixture acceptance criteria without introducing unsound edges.
- Option A cannot meet the curated real-repo recall target (80-90%) due to project loading/module resolution gaps that tsserver resolves.

## Non-goals (MVP)

- Perfect resolution for dynamic JS/TS patterns (untyped `any`, reflection, `obj[method]()`).
- IDE-grade “find all references” fidelity in every repo shape.
- Cross-repo call graphs spanning external dependencies (beyond optionally recording edges into workspace boundary).

## High-level Architecture

- Add a TS-resolved call graph builder that uses the **TypeScript compiler API / language service** to resolve each callsite to a concrete definition.
- Replace the current TS path in `tldr/cross_file_calls.py` (`_build_typescript_call_graph` + `_extract_ts_file_calls`) with a resolved builder that returns edges as:
  - caller: (file, qualified symbol id)
  - callee: (file, qualified symbol id)
- Add an opt-in **trace mode** that explains missing/filtered edges:
  - callsite location -> resolved symbol (or failure reason) -> definition location
- Ensure the improved call graph becomes a practical adjunct to `rg/grep` by:
  - improving `calls`/`impact`/`change-impact` recall on TS monorepos
  - providing deterministic, grep-friendly locations in output
  - providing trace output when results are unexpectedly empty

## Phase 0: Define Success Criteria + Fixture (Before Code)

**Goals**
- Make “it works” objectively testable before implementing the builder.

### Running Log (Phase 0)
- 2026-02-08: Read existing TS call graph (`tldr/cross_file_calls.py`). Current approach is tree-sitter + naive module resolution (`Path(stem)` join), which misses monorepo imports and re-exports; also relies on tree-sitter being installed (otherwise TS call graph is effectively empty).
- 2026-02-08: Aha: current incremental patching (`tldr/patch.py`) only patches **intra-file** edges for TypeScript, so cached TS graphs can silently degrade after edits. For TS-resolved graphs, we should either disable patching or implement a TS-aware patch strategy.
- 2026-02-09: Added fixture `tests/fixtures/ts-monorepo/` (paths-based `@scope/a`, `@lib/*`, barrel re-exports, default export, interface dispatch, supported vs unsupported callback patterns) + golden edge list `tests/fixtures/ts-monorepo/expected_edges.json`.
- 2026-02-09: Added fixture compile check (`tsc -p tests/fixtures/ts-monorepo --noEmit`) as part of `tests/test_ts_callgraph_fixture.py`.

**Deliverables**
- TS monorepo fixture with known ground-truth edges.
- Golden expected edge list for the fixture.
- Explicit “not resolvable” cases and their expected behavior.

**Key tasks**
- [x] Add `tests/fixtures/ts-monorepo/` with these required patterns:
  - packages/a exports `foo` via `src/index.ts` re-export from `src/foo.ts`
  - packages/b imports via workspace package import: `import { foo } from "@scope/a"`
  - alias import: `import { foo as f } from "@scope/a"`
  - default export import: `import FooDefault from "@scope/a"`
  - `compilerOptions.paths` alias case: `@lib/* -> packages/a/src/*`
  - class/interface dispatch case:
    - `interface I { m(): void }`, `class C implements I { m(){} }`, `const x: I = new C(); x.m()` must resolve to `C.m`
  - callback/ref patterns with explicit support policy:
    - supported: `handlers = { onFoo }` then `handlers.onFoo()`
    - not supported: `handlers: Record<string, Function>`, then `handlers[name]()`
- [x] Add `tests/fixtures/ts-monorepo/expected_edges.json` (or `.yaml`) with canonical edge entries:
  - `caller: { file, symbol, line, col }`
  - `callee: { file, symbol, line, col }`
  - `callsite: { file, line, col }` (optional but strongly recommended)
- [x] Add a fixture build check (either via a test or pre-test script):
  - `tsc -p tests/fixtures/ts-monorepo` must pass.

**Acceptance**
- Fixture compiles cleanly.
- Golden edge list is stable and reviewed (no ambiguous naming).

## Phase 1: TS-Resolved Builder (Compiler/Language Service)

**Goals**
- Build a call graph for TS that correctly resolves symbols across a monorepo shape (paths, project references, barrels).

### Running Log (Phase 1)
- 2026-02-08: Question: `const x: I = new C(); x.m()` does not naturally resolve to `C.m` via the TS checker because `x` is typed as `I`. Proposed “sound” workaround for fixture: only emit `x.m -> C.m` when `x` is a `const` with a `new C()` initializer (no reassignments), using the initializer’s concrete type to resolve the method.
- 2026-02-08: Aha: to resolve re-exports and renamed imports, we must aggressively de-alias symbols via `checker.getAliasedSymbol` and handle `export { … } from` and barrel `index.ts` patterns.
- 2026-02-09: Implemented TS-resolved call graph via Node helper + TS compiler API:
  - Node: `tldr/ts/ts_callgraph_node.js` builds a `Program` + `TypeChecker`, walks `CallExpression`s, resolves to concrete declarations, and emits edges + (optional) skipped-callsite reasons.
  - Python glue: `tldr/ts/ts_callgraph.py` runs the helper, parses JSON, and returns deterministic edge ordering + metadata.
- 2026-02-09: Toolchain discovery supports:
  - `node` in PATH
  - TypeScript module resolution from (1) project `node_modules`, else (2) global `npm root -g`, else (3) ambient Node resolution
  - tsconfig selection: prefer `<root>/tsconfig.json`, else require exactly one `tsconfig.json` under root (otherwise mark ambiguous and fall back).
- 2026-02-09: Soundness/skip policy implemented:
  - skip `obj[name]()` (element access) and `any` dispatch
  - skip interface-only signatures unless the “const + new” narrowing rule can resolve an implementing method
  - skip callees outside workspace (node_modules, `.d.ts`, outside root)
- 2026-02-09: Known gap: overload disambiguation is not encoded into the symbol string yet (we currently include declaration `line/col` metadata, but `ProjectCallGraph` still keys edges by `(file, symbol)`).
- 2026-02-09: Real-world tsconfig gotcha found on Peerbit: root `tsconfig.json` exists but has **no project inputs** (it extends `aegir` config and inherits `include` pointing at aegir internals). Our resolver currently treats root tsconfig as primary, which yields `processed_files=0` and thus an empty graph. Next: detect “no inputs” and fall back to per-package tsconfigs (multi-project build).
- 2026-02-09: Peerbit validation found many exported callables are `export const foo = (...) => {}` / `export const foo = function (...) {}`. The TS checker often resolves call signatures to the arrow/function expression node itself; our resolver treated these as `callee_unnamed` and dropped edges. Fixed by deriving a stable symbol id from the containing `VariableDeclaration` / `PropertyDeclaration` (and added fixture coverage).
- 2026-02-09: Implemented multi-tsconfig build fallback (`graph_source=ts-resolved-multi`) when the chosen tsconfig yields no workspace inputs (common in TS monorepos). Added a focused fixture `tests/fixtures/ts-multi-tsconfig/` + test `tests/test_ts_callgraph_multi_tsconfig.py` and a conservative `dist/src/*.d.ts -> src/*.ts` mapping (only when the source file exists) so workspace edges don't point at ignored build outputs.
- 2026-02-09: Aha: `checker.getResolvedSignature(callExpr).getDeclaration()` can point at type-level signature nodes (e.g. `FunctionTypeNode` / call signatures) that do not have a stable symbol name. Fix: only prefer `sigDecl` when it is nameable; otherwise fall back to the value symbol declaration. Added fixture coverage for contextually typed function values (`tests/fixtures/ts-monorepo/packages/a/src/typed.ts`).
- 2026-02-09: Aha: the Node helper was truncating stdout around ~64KB because it called `process.exit()` synchronously after writing JSON. Fix: exit from the stdout write callback so large payloads flush fully.
- 2026-02-09: Bounded TS trace payloads at the resolver boundary (sample + count + truncated flag) to prevent huge JSON outputs when `--ts-trace` is enabled on large repos.

**Deliverables**
- TS-resolved call graph builder runnable from Python, producing edges + trace metadata.

**Key tasks**
- [x] Implement a new module (suggested):
  - `tldr/ts/ts_callgraph.py` (Python glue + data types)
  - `tldr/ts/ts_callgraph_node.js` (Node helper using TypeScript API)
- [x] TS toolchain discovery (per scanned repo):
  - detect `node` availability
  - locate `typescript` from the scanned repo (prefer `node_modules/typescript`; do not require llm-tldr to ship TypeScript)
  - locate nearest/primary `tsconfig.json` (support project references)
- [x] In Node helper:
  - load tsconfig(s) via TypeScript APIs (support project references)
  - build a `Program`/`TypeChecker` (or LanguageService) for the workspace
  - traverse `CallExpression` sites and resolve to a `Signature`/`Symbol`
  - map symbol to a definition location (file + span), and compute a stable qualified symbol id:
    - prefer `ClassName.method`, `functionName`, `namespace.function`, include overload disambiguation via location/span
- [x] Define and enforce soundness constraints:
  - only produce edges when the definition resolves to a concrete declaration node
  - do not produce edges for dynamic property access or unresolved `any` dispatch
  - optionally keep a “skipped callsites” list with reasons for trace mode
  - explicitly gate interface dispatch edges: only emit `x.m() -> C.m` when the checker can resolve the implementing declaration (no guesswork)

- [ ] (Optional fallback) Add a tsserver-backed resolver adapter:
  - implement `tldr/ts/tsserver_client.py` (or similar) to start/stop tsserver and issue definition/signature requests
  - map `definition` results to canonical `file/symbol` ids used by the call graph
  - keep this behind a feature flag (`TLDR_TS_RESOLVER=tsserver`) until it proves necessary

**Acceptance**
- On the fixture, builder produces at least all golden edges.
- Builder produces zero edges for explicitly “not resolvable” cases.
- Repeated runs produce identical edges and ordering.

## Phase 2: Integrate With Existing Graph + Queries

**Goals**
- Make `calls`, `impact`, `change-impact` reliably benefit from the new TS call graph.

### Running Log (Phase 2)
- 2026-02-08: Integration needs to remain backward compatible with existing JSON cache files (`.tldr/cache/call_graph.json`). Add optional `meta` fields rather than changing the edge tuple shape.
- 2026-02-09: Integrated TS-resolved builder into `tldr/cross_file_calls.py`:
  - `build_project_call_graph(..., ts_trace=False)` plumbs trace intent.
  - `ProjectCallGraph` now carries `meta` and a deterministic `sorted_edges()` helper.
  - `_build_typescript_call_graph` prefers ts-resolved, falls back to syntax-only, and records `ts_resolution_errors` + `incomplete` when falling back.
- 2026-02-09: Updated cache + CLI/daemon wiring:
  - `tldr/cli.py` and `tldr/daemon/core.py` now persist `meta` into `call_graph.json` (backward compatible) and serialize edges in stable order.
  - Trace payloads are excluded from cache (to avoid huge cache files).
  - Added CLI flags `--ts-trace` for `calls` and `impact`.
- 2026-02-09: Packaging: included `tldr/ts/*.js` as package data so installed builds can run the Node helper.

**Deliverables**
- TS edges included in `ProjectCallGraph` and consumed by analysis layers.
- Query output indicates whether TS resolution was used or a fallback was applied.

**Key tasks**
- [x] Extend `tldr/cross_file_calls.py`:
  - replace `_build_typescript_call_graph` with a resolved builder path
  - keep the current syntax-only builder as fallback
- [x] Ensure call graph caching works in both CLI and daemon paths:
  - `tldr/cli.py` cache path(s)
  - `tldr/daemon/core.py` cache + incremental rebuild path
- [x] Add metadata to `ProjectCallGraph` (or adjacent result) such as:
  - `graph_source: ts-resolved | ts-syntax-only | <other-lang>`
  - `ts_resolution_errors: [...]` (summarized)
- [x] Add a trace flag and plumbing:
  - env `TLDR_TS_TRACE=1` or a CLI flag (preferred) to emit resolution traces for missing edges

**Acceptance**
- `tldrf calls --language typescript` (or equivalent path) uses ts-resolved when available.
- `tldrf impact <symbol>` returns correct fixture callers.
- When ts-resolved is not available, output is clearly marked as incomplete and does not crash.

## Phase 3: Automated Tests (Must-Have)

**Goals**
- Prevent regressions; ensure we never “think it works” while missing key TS edges.

### Running Log (Phase 3)
- 2026-02-08: CI environments may not have Node/TypeScript available. Tests for TS-resolved mode should skip cleanly when toolchain is missing, while still testing the syntax-only fallback path.
- 2026-02-09: Added fixture-backed tests in `tests/test_ts_callgraph_fixture.py`:
  - `tsc` compile check for the fixture (skips if `tsc` unavailable)
  - golden edge assertions (skips if TS-resolved toolchain unavailable)
  - explicit negative check for dynamic element access
  - determinism check (repeat build)
  - impact assertions for `foo` and `C.m`
  - forced syntax-only mode coverage via `TLDR_TS_RESOLVER=syntax`

**Deliverables**
- Fixture-backed tests for call graph build, impact queries, and import/module resolution.
- Negative tests for failure modes and determinism.

**Key tasks**
- [x] Add fixture-backed tests in `tests/test_ts_callgraph_fixture.py`:
  - build call graph and assert golden edges present
  - assert no edges for explicitly unsupported patterns (dynamic element access)
  - assert determinism (stable output ordering)
  - `impact(foo)` includes known callers
  - `impact(C.m)` includes known callers for interface dispatch case
  - cover forced syntax-only fallback via `TLDR_TS_RESOLVER=syntax`
- [ ] (Optional) Add extra failure-mode tests:
  - toolchain missing -> fallback + “incomplete” marker
  - multiple tsconfigs -> deterministic selection or explicit ambiguity error
  - path casing/symlink canonicalization stability

**Acceptance**
- `uv run pytest` passes with TS fixture tests on supported environments.

## Phase 4: Real-Repo Validation (Curated Peerbit Checks)

**Goals**
- Validate value on a real TS monorepo where rg currently wins.

### Running Log (Phase 4)
- 2026-02-08: Next step after fixture: build a small curated edge set for a real monorepo and add a harness to compute recall and emit trace reasons for misses.
- 2026-02-09: Dry-run on `/Users/aristotle/Documents/Projects/peerbit` initially revealed the TS-resolved pipeline could select a root `tsconfig.json` that yields **0 workspace inputs** (`processed_files=0`), producing an empty graph. Implemented multi-tsconfig fallback to unblock monorepos with non-workspace root tsconfigs.
- 2026-02-09: Peerbit has many `export const ... = (...) => {}` helpers (e.g. `createCache`, `createLibp2pExtended`, `resolveBootstrapAddresses`) whose edges were being dropped due to `callee_unnamed` / signature-declaration selection. Implemented naming + signature fallback fixes and added fixture coverage; this unblocked curated-edge recall checks.
- 2026-02-09: Peerbit root (`/Users/aristotle/Documents/Projects/peerbit`) now builds with `graph_source=ts-resolved-multi` and produces `edge_count=3855` across `ts_projects_ok=59` / `ts_projects_err=1` in ~91s (no daemon cache), marked `incomplete=True` due to the single error project.
- 2026-02-09: Added Phase 4 harness + curated edge set:
  - `scripts/peerbit_curated_edges.json` (31 curated edges across packages)
  - `scripts/validate_peerbit_callgraph.py` (edge + impact recall)
  - Current recall on the curated set: 100% (edges + impact).

**Deliverables**
- A curated set of ~20 known edges for Peerbit-style repos and a validation harness.

**Key tasks**
- [x] Create a small curated edge list (JSON) with caller/callee pairs confirmed via `rg` + manual inspection.
- [x] Add a manual validation script that:
  - runs `tldrf impact <symbol>` for each curated target
  - computes recall vs the curated caller set
  - prints misses with trace reasons when available

**Acceptance**
- Target recall 80-90% on curated edges (excluding explicitly dynamic patterns).
- Misses are explainable via trace output (not silent).

## Phase 5: Performance + Incremental Rebuilds

**Goals**
- Improve quality without making “warm daemon workflows” unusably slower.

### Running Log (Phase 5)
- 2026-02-08: The current incremental patcher does not preserve cross-file TS edges; for TS-resolved graphs we may need to choose between (a) full rebuild on dirty edits, or (b) a long-lived TS language service process to support real incremental updates.
- 2026-02-09: Stopgap implemented: `tldr/cli.py` now does a full rebuild for TypeScript graphs when dirty (instead of using the intra-file-only patcher), to prevent cross-file edges from being dropped silently.
- 2026-02-09: Added `scripts/bench_ts_callgraph.py` (manual benchmark runner). Current fixture baseline on this machine:
  - full build: ~0.51s (`graph_source=ts-resolved`, `edge_count=9`)
  - rebuild after touching 1 file: ~0.45s (currently a full rebuild for TS).
  - Peerbit root full build baseline: ~91s (see Phase 4 harness output).
- 2026-02-09: Implemented **TS-resolved incremental patching** for dirty TS files:
  - New patcher: `tldr/patch.py:patch_typescript_resolved_dirty_files()` recomputes outbound edges for dirty `.ts/.tsx` files via the existing TS compiler API resolver (allowlist mode), then patches the cached graph in-place.
  - Wired into cache flow: `tldr/cli.py` now prefers the incremental patch when the cached `graph_source` is `ts-resolved` / `ts-resolved-multi`, falling back to full rebuild if the resolver is disabled/unavailable.
  - Bench support: `scripts/bench_ts_callgraph.py` now reports `incremental_patch_after_touch` for the fixture.
- 2026-02-09: Updated local fixture benchmark (after incremental patch landed):
  - full build: ~0.57s
  - incremental patch (1-file): ~0.45s
  - full rebuild after touch: ~0.46s
  - Note: on the small fixture, program creation dominates so patch vs rebuild is similar; the expected win is on multi-tsconfig monorepos where patch only rebuilds the touched file’s nearest tsconfig (instead of *all* tsconfigs).
- 2026-02-09: Added warm-daemon query bench plumbing:
  - Fixed daemon impact handling to use `tldr.analysis.impact_analysis` over cached call graphs (`tldr/daemon/core.py`).
  - `scripts/bench_ts_callgraph.py --peerbit-root … --peerbit-daemon` measures daemon `warm` + 5 `impact` query latencies.
- 2026-02-09: Fixed daemon client framing: `tldr/daemon/startup.py:query_daemon()` now reads until newline (responses can exceed a single `recv()`), unblocking large `impact` responses and accurate daemon benchmarks.
- 2026-02-09: Peerbit warm-daemon timings (local):
  - `warm` (TS graph build + cache write): ~118s, `edge_count=3855` (`graph_source=ts-resolved-multi`)
  - 5 fixed `impact` queries (warm daemon, `depth=1`, with file filters): ~3.8-5.1ms/query
- 2026-02-09: Peerbit incremental TS rebuild (single file, local):
  - Patching outbound edges for `packages/clients/peerbit/src/peer.ts`: ~1.3s (vs ~118s full multi-tsconfig build).

**Deliverables**
- Benchmarks for fixture build and incremental edits.

**Key tasks**
- [x] Track:
  - full call graph build time on the fixture (see `scripts/bench_ts_callgraph.py`)
  - rebuild time after touching a single fixture file (see `scripts/bench_ts_callgraph.py`)
- [ ] Track: time to answer 5 fixed impact queries with warm daemon (Peerbit).
- [x] Add a consistent benchmark script (CI optional) and define regression thresholds (see `scripts/bench_ts_callgraph.py`).

**Acceptance**
- Incremental rebuild is noticeably faster than full build.
- No severe daemon CPU/memory spikes during repeated queries.

## Phase 6: Debuggability + UX (Adjunct to rg/grep)

**Goals**
- Make “why didn’t impact/calls show X?” actionable and fast to diagnose.

### Running Log (Phase 6)
- 2026-02-08: Trace output should be bounded (count + sampled examples) and grep-friendly (`path:line:col`) so it helps users immediately verify with `rg`.
- 2026-02-09: Implemented trace plumbing end-to-end:
  - Node resolver can emit `skipped` callsites with `reason` + `file/line/col`.
  - CLI `calls` / `impact` gained `--ts-trace`, returning a bounded `{ skipped_count, skipped_sample }` payload while keeping caches trace-free.
- 2026-02-09: Trace is now bounded at the resolver boundary (`skipped_count`, `skipped_limit`, `skipped_truncated`) so enabling trace on large repos does not crash or emit massive JSON.

**Deliverables**
- Trace output that helps the user immediately fall back to `rg/grep` (or fix TS config/tooling) with concrete locations and reasons.

**Key tasks**
- [x] Implement trace summaries:
  - missing edge shows: callsite location + resolution failure reason
  - next: include candidate symbol(s) when available and optionally print a suggested `rg` command seeded with the callee identifier
- [x] Ensure outputs are grep-friendly:
  - canonical `path:line:col` formatting for callers/callees/callsites
  - stable sort order and consistent symbol naming

**Acceptance**
- When results are empty or missing expected edges, trace output identifies whether the root cause is:
  - module/tsconfig resolution
  - symbol/type resolution
  - filtering/soundness policy
  - caching/incremental patching

## Files / Modules Likely to Change

- `tldr/cross_file_calls.py` (replace/augment TS call graph builder path)
- `tldr/analysis.py` (may need minor adjustments if TS symbol naming changes)
- `tldr/cli.py` (trace flag + metadata surfacing)
- `tldr/daemon/core.py` (cache + incremental rebuild integration)
- New: `tldr/ts/ts_callgraph.py`
- New: `tldr/ts/ts_callgraph_node.js` (or similar Node helper)

## Acceptance Criteria (Summary)

- Correctness (fixture + real repo):
  - workspace package imports (e.g. `@peerbit/*`)
  - `compilerOptions.paths` aliases
  - barrel re-exports (`index.ts`, `export * from`)
  - named/default exports + renamed imports
  - class methods (`obj.method()` when the type is known)
- Soundness:
  - no edges for dynamic property access, unresolved `any`, or other non-confident cases
- Stability:
  - identical output across repeated runs on the same checkout
- Performance:
  - incremental update after single-file edit faster than full rebuild
- Debuggability:
  - trace output provides callsite + resolution outcome for missing edges

Important: Keep a running log within this document of all learnings, ahas, answers to questions, and next steps in the appropriate sections for each phase, update it list as you go on.
