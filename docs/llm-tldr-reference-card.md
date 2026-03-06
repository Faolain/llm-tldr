# llm-tldr One-Page Usage Guide

> Use `rg-native` for exact lookup. Use `tldrf` when you need structure, blast radius, slices, or concept search. Use daemon or MCP for repeated queries.
> If you are running from this repo checkout instead of an installed CLI, prefix commands with `uv run`.

## Start Here

| If the question is... | Reach for | Why / measured evidence |
|---|---|---|
| Exact string, symbol, import, file path, or definition | `rg-native` first | Exact-definition retrieval is `F1 0.9841` for `rg-native` vs `0.1602` for BGE and `0.1520` for Jina. |
| "What breaks if I change this?" | `tldrf impact` -> `tldrf context` -> `rg-native` | Deterministic caller graphs beat grep-style baselines on impact, then `rg` closes lexical cleanup gaps. |
| "Why is this value wrong here?" | `tldrf slice` -> `tldrf dfg`; add `tldrf cfg` for branch bugs | Slice is the strongest debugging workflow: `F1 0.919`; DFG origin / flow accuracy is `1.0`. |
| "I do not know the symbol name, only the behavior" | `tldrf semantic search` with the default hybrid path | llm-tldr retrieval at `budget=2000` beats `rg-native` on ranking quality (`MRR 0.874` vs `0.813`) while adding structural follow-up tools. |
| Repeated agent or editor queries | Daemon or MCP, not direct CLI subprocesses | Daemon mode cuts warm retrieval from about `5s` to about `293ms` and structural commands by `15x-93x`. |
| Pure semantic concept lookup or semantic-only tight-budget retrieval | `jina-code-0.5b` opt-in | Jina improves semantic-only retrieval (`MRR 0.7023` vs `0.6022`) and budget-`1000` semantic retrieval (`0.7048` vs `0.6124`). |

## Default Workflows

| Workflow | Default path | When it is the right tool |
|---|---|---|
| Refactor safely | `impact` -> `context` -> `rg-native` | Use this before renaming, splitting, deleting, or changing a signature. |
| Debug precisely | `slice` -> `dfg` -> `cfg` | Use this when you know the failing line or function and want the smallest relevant context. |
| Onboard to unfamiliar code | `semantic search` -> `context` -> `impact` | Use this when you know intent but not names, then tighten to symbol-level reasoning. |
| Exact cleanup / migration sweep | `rg-native` -> direct reads | Use this for exhaustive imports, docs, configs, generated files, and module-level code. |
| Multi-query agent session | MCP or daemon-backed calls | Use this whenever an LLM or script will issue more than one query against the same repo. |

## Commands To Memorize

| Task | Command |
|---|---|
| Exact lookup | `rg -n "pattern" .` |
| Blast radius | `tldrf impact "function_name" . --file path/to/file.py` |
| Reachable context | `tldrf context "function_name" --project . --depth 2` |
| Minimal debug slice | `tldrf slice path/to/file.py function_name 42` |
| Data provenance | `tldrf dfg path/to/file.py function_name` |
| Control-flow hotspot | `tldrf cfg path/to/file.py function_name` |
| Concept search | `tldrf semantic search "natural language intent" --path . --k 8 --hybrid` |
| Affected tests | `tldrf change-impact` |
| Start daemon | `tldrf daemon start --project .` |
| Keep the index fresh | `tldrf daemon notify path/to/file.py --project .` |

## Session Setup That Actually Pays Off

Use the same cache root and index id across the whole session. That is what turns tldrf from a slow CLI command into a reusable code index.

```bash
ROOT="$(git rev-parse --show-toplevel)"
INDEX_ID="repo:<project>"

tldrf warm --cache-root "$ROOT" --index "$INDEX_ID" --lang python .
tldrf semantic index --cache-root "$ROOT" --index "$INDEX_ID" --lang python .
tldrf daemon start --cache-root "$ROOT" --index "$INDEX_ID" --scan-root .
```

- Build once, then query the warm daemon or MCP repeatedly instead of respawning the CLI.
- Reuse the same `--cache-root` and `--index` across agent turns, scripts, and editor sessions.
- Pin `--lang` on structural commands in mixed-language repos.
- Prefer explicit `--cache-root "$ROOT"` over `--cache-root git` until that shortcut has stable regression coverage.
- After edits, use `tldrf daemon notify <changed-file> --project .` or rerun `warm` when the project has drifted a lot.

## Current Gotchas To Remember

- Plain CLI commands are still fresh-process invocations. `docs/usage.md` is the accurate source for daemon vs subprocess behavior.
- `warm` does not replace `semantic index`; structural and semantic setup are separate steps.
- Exact identifier and definition lookup is still a grep problem. Do not treat semantic retrieval as a replacement for `rg-native` there.

## Recommended Defaults

| Decision | Current default | Why |
|---|---|---|
| Exact lookup tool | `rg-native` | Lexical lookup is still the right first tool for symbol / definition search. |
| Product retrieval model | `bge-large-en-v1.5` | Jina is roughly tied on common `hybrid_rrf`, but BGE still wins lane2 MRR, compound efficiency, and structured concept retrieval. |
| Semantic-only experiment model | `jina-code-0.5b` opt-in | Jina is the better pure semantic model on the current Django evidence, but it requires a semantic rebuild and separate license review. |
| Query execution mode | Daemon or MCP | Warm daemon mode is fast enough to be practical and avoids repeated model loads. |

Current product decision: keep `BGE` as the default model, keep `Jina` opt-in, and keep `rg-native` as the first tool for exact lookup.

## What The Benchmarks Actually Support

| Claim | Current signal |
|---|---|
| llm-tldr is better than `contextplus` on retrieval | Yes. Shared retrieval lanes are decisive in the 008 comparison program. |
| llm-tldr gives better final answers than `rg-native` on structured debugging / analysis tasks | Yes. Open-ended judge win rate over `rg` is `0.694` at `budget=2000`. |
| llm-tldr is worth using for refactors and debugging | Yes. Impact, slice, DFG, CFG, and token-efficiency benchmarks all show lower-noise structural context than grep-style baselines. |
| llm-tldr should replace `rg-native` for exact identifier lookup | No. Exact lexical lookup is still a grep problem. |
| Jina should replace BGE as the default model | No. Jina wins semantic-only retrieval, but not the broader product path. |

## Where tldrf Adds Value Over rg-native

- `impact`: deterministic reverse call graph for blast radius instead of grep guesses.
- `slice` and `dfg`: line-level provenance and forward flow, which `rg` cannot infer.
- `context`: call-graph-bounded summaries that reduce context rot and token waste.
- `semantic search`: concept lookup when the user knows behavior but not names.
- daemon or MCP reuse: index once, keep it warm, avoid repeated lossy full-repo search.

## Gaps To Close Next

- Add end-to-end edit-loop benchmarks with time-to-first-correct-edit, token spend, and changed-file recall against `rg-native`.
- Add edit-loop benchmarks for `daemon notify` and auto-reindex latency so "index once, keep it fresh" is measured, not assumed.
- Add task-level refactor benchmarks with gold changed-file / changed-symbol sets to prove fewer misses than `rg-native` during code modification, not just retrieval quality.
- Rerun downstream judge-mode comparisons on workflows that explicitly use `impact` + `context` and `slice` + `dfg`, not only retrieval packets.
- Complete dependency-scoped BGE vs Jina benchmarks (`requests`, `urllib3`) so model guidance is not Django-only.
- Add mixed-language and monorepo structural-quality benchmarks to prove the workflow scales beyond the current Python-heavy evidence.
- Add stronger hybrid negative-query guard benchmarks; current strict `rg_empty` behavior on the structured behavior suite is too aggressive to treat as production guidance.
- Add more user-facing daemon parity coverage so repeated CLI workflows can route through the daemon without raw JSON callsites.
- Clean up longer-form docs that still imply `warm` alone builds semantic embeddings automatically; the one-pager is the correct guidance for now.
- Add doc-and-help parity coverage for `semantic search` syntax, `warm` vs `semantic index`, `--cache-root` handling, and explicit `--lang` requirements during indexing.

## Docs Map

- This page: task selection and defaults.
- `benchmarks/README.md`: aggregate benchmark results and model/tool comparison matrix.
- `docs/usage.md`: execution modes, daemon vs subprocess behavior.
- `implementations/008-benchmark-summary.md`: operator handoff and canonical benchmark runbook.
