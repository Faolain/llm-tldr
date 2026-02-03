# Implementation Plan: 001 - Isolated Indexes

Spec: `specs/001-feat-isolated-index.md`

## Summary

Add a first-class **Index** concept so TLDR can:

- **Scan code in-place** (e.g. `node_modules/pkg`, `.venv/.../site-packages/pkg`)
- **Store all TLDR state under a chosen cache root** (e.g. your repo root)
- **Maintain multiple isolated indexes** (one per `module@version`), preventing cache collisions across deps
- Support multi-index operation in the **daemon** and **MCP server**
- Provide **index management** commands (`list/info/rm/gc`)

## Goals (Acceptance Criteria)

1. CLI supports `--scan-root`, `--cache-root`, `--index` and uses them consistently across:
   - `warm`, `semantic index/search`, `tree`, `structure`, `search`, `context`, `calls`, `impact`, `imports`, `importers`, daemon commands.
2. All caches are **namespaced per index** under `cache_root/.tldr/indexes/<index_key>/...`.
3. Semantic indexing:
   - Never “walks upward” to find the project root when explicit `--cache-root/--index` are provided.
   - Reads/writes FAISS + metadata only under the selected index directory.
   - Stores the embedding model in index metadata and rejects mismatches (or forces explicit override).
4. Ignore config is **index-scoped**:
   - Default ignore file at `cache_root/.tldr/indexes/<index_key>/.tldrignore`.
   - Add `--ignore-file`, and `--use-gitignore/--no-gitignore`.
5. Daemon/MCP:
   - Daemon socket/lock/pid identity is keyed by `(cache_root, index_id)` (one daemon per index).
   - Daemon always uses the correct scan_root + index caches.
6. Index management commands exist and are safe:
   - `tldr index list --cache-root ...`
   - `tldr index info <index_id> --cache-root ...`
   - `tldr index rm <index_id> --cache-root ...`
   - `tldr index gc --cache-root ...` (optional)
7. Tests cover:
   - Path resolution and collision avoidance across two indexes
   - Two indexes under one repo root don’t overwrite each other
   - Daemon queries for index A never read caches for index B

## Non-Goals (for this spec)

- Automatically resolving `module@version` for every ecosystem (leave to the “dep-tldr” skill / a future “auto id” helper).
- Multi-index queries (search across several indexes at once). The spec target is “one daemon per index”.
- Copying dependencies into a separate analysis directory (explicitly avoided).

## Proposed User-Facing CLI

### Global flags / env vars

Add to top-level parser (applies to all subcommands):

- `--scan-root <path>` (env `TLDR_SCAN_ROOT`)
- `--cache-root <path>` (env `TLDR_CACHE_ROOT`)
- `--index <id>` (env `TLDR_INDEX`)
- `--ignore-file <path>` (env `TLDR_IGNORE_FILE`)
- `--use-gitignore` / `--no-gitignore` (env `TLDR_USE_GITIGNORE=1|0`)

Precedence (highest → lowest):

1. CLI flags
2. env vars
3. legacy defaults (current behavior)

### Path semantics (positional vs `--scan-root`)

- For subcommands with a positional directory `path` argument (e.g. `warm/tree/structure/search`), treat that positional value as the default `scan_root`.
- `--scan-root` overrides the positional directory path.
- If both are provided and resolve to different paths, exit with an error (avoid silently indexing the wrong tree).
- For existing command-specific flags that represent a scan root (e.g. `context --project`, `semantic search --path`), keep them as aliases for `scan_root` for backward compatibility, but prefer documenting `--scan-root`.

### Examples

Index a Node dep **in-place**, store caches under repo root:

```bash
tldr warm --cache-root . --scan-root node_modules/zod --index node:zod@3.23.8 --no-gitignore
tldr semantic index --cache-root . --scan-root node_modules/zod --index node:zod@3.23.8 --lang typescript
tldr semantic search "parse schema errors" --cache-root . --scan-root node_modules/zod --index node:zod@3.23.8
```

Index a Python dep in `.venv`:

```bash
tldr warm --cache-root . --scan-root .venv/lib/python3.12/site-packages/requests --index py:requests@2.31.0 --no-gitignore
```

Default behavior (no flags): unchanged.

## Index Model

### Data structures (new)

Add a small module to centralize index concerns, e.g. `tldr/indexing/index.py`:

- `IndexConfig` (dataclass):
  - `cache_root: Path`
  - `scan_root: Path`
  - `index_id: str` (logical id; may contain `:`, `@`, `/`, etc.)
  - `index_key: str` (filesystem-safe key derived from `index_id`)
  - `languages: list[str] | None`
  - `semantic_model: str | None`
  - `ignore_file: Path`
  - `use_gitignore: bool`
  - `created_at/updated_at` (optional)

- `IndexPaths` helper:
  - `index_dir = cache_root/.tldr/indexes/<index_key>/`
  - `meta = index_dir/meta.json`
  - `call_graph = index_dir/call_graph.json`
  - `languages = index_dir/languages.json`
  - `dirty = index_dir/dirty.json` (replace `.tldr/cache/dirty.json` in index-mode)
  - `semantic_dir = index_dir/semantic/`
  - `semantic_faiss = semantic_dir/index.faiss`
  - `semantic_metadata = semantic_dir/metadata.json`

### Index ID vs on-disk key

To stay cross-platform, introduce a filesystem-safe `index_key`:

- Keep `index_id` as the stable external identifier shown to users and stored in `meta.json`.
- Derive `index_key` via a deterministic encoding (recommended):
  - `index_key = base32(sha256(index_id))[:16] + "-" + slug(index_id)[:48]`
  - Store both fields in `meta.json` so list/info can map back.

Rationale: `:` and `/` are invalid on Windows; raw `index_id` cannot safely be used as a directory name.

## Cache Layout (New)

Under `cache_root`:

```
.tldr/
  indexes/
    <index_key>/
      meta.json
      call_graph.json
      languages.json
      dirty.json
      .tldrignore
      semantic/
        index.faiss
        metadata.json
```

Keep existing single-index layout for legacy mode (no flags), at minimum:

```
.tldr/cache/call_graph.json
.tldr/languages.json
.tldr/cache/semantic/{index.faiss,metadata.json}
.tldr/cache/dirty.json
```

## Backward Compatibility Strategy

Two viable approaches; pick one early and keep it consistent:

### Option A (recommended): “Legacy mode” stays untouched

- If the user does not provide any of the index-related knobs (`--cache-root`, `--scan-root`, `--index`, `--ignore-file`, `--use-gitignore/--no-gitignore`, or env equivalents), keep current paths and behavior exactly.
- If **any** of those are provided, switch into “index mode” and use the new `indexes/<index_key>/...` layout.
- If in index mode and `--index` is omitted, auto-derive a deterministic `index_id` from `scan_root`:
  - Prefer `path:<scan_root relative to cache_root>` when `scan_root` is under `cache_root`.
  - Otherwise use `abs:<scan_root resolved>`.

Pros: no migration complexity; no test churn for existing expectations.
Cons: two layouts to support long-term.

### Option B: Always use new layout (with migration)

- Treat current state as `index_id="default"`, store in `indexes/<key-of-default>/`.
- On first run, migrate existing `.tldr/cache/*` into the default index dir (or keep reading old paths as fallback).

Pros: one layout going forward.
Cons: migration risk; requires updating multiple tests/docs in one go.

## Ignore Handling (Index-Scoped)

### Implementation changes

Update `tldr/tldrignore.py` to support:

- Loading patterns from an arbitrary ignore file path (not necessarily `scan_root/.tldrignore`).
- Allow specifying a separate `gitignore_root` (recommended: `cache_root` in index mode).

Suggested interface:

- `load_ignore_patterns_from_file(ignore_file: Path) -> PathSpec`
- `ensure_ignore_file(ignore_file: Path) -> (created: bool, msg: str)`
- `IgnoreSpec(scan_root: Path, ignore_file: Path, use_gitignore: bool, gitignore_root: Path, cli_patterns: list[str] | None)`
  - `match_file(rel_path)` uses patterns relative to `scan_root`.
  - gitignore checks use `gitignore_root` with relpaths relative to that root.

Default in index mode:

- `ignore_file = index_dir/.tldrignore` (create if missing using existing DEFAULT_TEMPLATE)
- `use_gitignore = False` if `scan_root` looks like a dependency dir (`node_modules`, `.venv`, `site-packages`) OR if user passes `--no-gitignore`
- `gitignore_root = cache_root` if `use_gitignore=True`

## Semantic Indexing Changes

### Current issue

`tldr/semantic.py` uses `_find_project_root(scan_path)` and writes to:

- `<project_root>/.tldr/cache/semantic/...`

This collides across multiple scan roots inside the same repo.

### Plan

1. Add optional `IndexConfig`/`IndexPaths` parameters to:
   - `build_semantic_index(...)`
   - `semantic_search(...)`
2. If `cache_root/index` are provided:
   - Do not call `_find_project_root()`.
   - Read/write strictly within `index_paths.semantic_dir`.
3. Persist model info:
   - `meta.json` stores the selected embedding model (HF name) and dimension.
   - `semantic/metadata.json` also stores model + dimension; enforce consistency.
4. Model mismatch policy:
   - **Search:** if the user supplies `--model` and it does not match the model recorded in `semantic/metadata.json`, error (results would be invalid and FAISS dimensions may not match). If `--model` is omitted, always use the model recorded in metadata.
   - **Index build:** if an index already exists and the requested `--model` differs, require an explicit overwrite flag (recommend `--rebuild`) to overwrite.
   - Always validate embedding **dimension** (from metadata vs FAISS index) before querying.

## Call Graph + “Warm” Cache Namespacing

### Current issue

Call graph cache and language cache are written under `<scan_root>/.tldr/...` and collide.

### Plan

1. Centralize call graph cache IO behind `IndexPaths`:
   - read/write `call_graph.json`, `languages.json`, `dirty.json` (index-mode)
2. Modify:
   - `tldr/cli.py`:
     - `_get_or_build_graph()` uses `index_paths.call_graph`.
     - `warm` writes `index_paths.call_graph` and `index_paths.languages`.
     - `resolve_language()` reads `index_paths.languages`.
   - `tldr/session_warm.py`: use `IndexPaths` for cache age checks.
   - `tldr/dirty_flag.py`: make dirty file path configurable; in index-mode store under index dir.

## Daemon + MCP Multi-Index Support

### Identity / socket naming

Update `tldr/daemon/startup.py` and `tldr/daemon/core.py`:

- Replace “hash(project_path)” with “hash(cache_root + index_id)”.
- Keep tmp files:
  - `/tmp/tldr-<hash>.sock`
  - `/tmp/tldr-<hash>.pid`
  - `/tmp/tldr-<hash>.lock`

### Daemon config

Update TLDRDaemon to be created with `IndexConfig`:

- `cache_root` determines where `.tldr/indexes/...` lives
- `scan_root` determines what the daemon scans/serves
- All query handlers use:
  - `scan_root` for filesystem scans and relative paths
  - `index_paths.*` for caches

### CLI surface

Update `tldr cli daemon {start,stop,status,query,notify}` to accept global index flags and propagate them to daemon startup/query.

### MCP server

Update `tldr/mcp_server.py`:

- Accept `--cache-root`, `--scan-root`, `--index` flags (and env vars).
- Compute socket identity from `(cache_root, index_id)`.
- When auto-starting the daemon, pass the same index flags through to `tldr cli daemon start ...`.

## Index Management Commands

Add `tldr index` subcommand tree in `tldr/cli.py`:

- `tldr index list --cache-root <path>`
  - List all `indexes/*/meta.json` entries (display `index_id`, `scan_root`, size, last_updated)
- `tldr index info <index_id> --cache-root <path>`
  - Resolve `index_id` to `index_key` by scanning meta files
  - Print meta + cache file presence and sizes
- `tldr index rm <index_id> --cache-root <path> [--force]`
  - Deletes that index directory only
  - Refuses to remove unknown/unparseable dirs unless `--force`
- `tldr index gc --cache-root <path> [--days N] [--max-total-mb M]` (optional)
  - Removes indexes not used recently (based on `meta.json` timestamps)

## Testing Plan

### Unit tests

1. `tests/test_index_paths.py`
   - `IndexPaths` resolves distinct directories for two different `index_id`s under the same `cache_root`
   - `index_key` sanitization is deterministic and collision-resistant

2. `tests/test_ignore_index_scoped.py`
   - Using `--ignore-file` affects filtering even when `scan_root` is elsewhere

### Integration tests (no daemon)

`tests/test_isolated_indexes_integration.py`:

- Create a tmp “repo” directory as `cache_root`
- Create two “deps” as separate scan roots with distinct code
- Run:
  - `build_semantic_index(scan_root=A, cache_root=repo, index=A)`
  - `build_semantic_index(scan_root=B, cache_root=repo, index=B)`
- Assert:
  - `repo/.tldr/indexes/<A>/semantic/...` and `<B>/semantic/...` both exist
  - `semantic_search(..., index=A)` never opens B’s FAISS
  - Results differ by content (each dep has a uniquely-named function)

### Integration tests (daemon)

`tests/test_daemon_isolated_indexes.py` (run on Unix and Windows if feasible):

- Start daemon for index A, query `tree/structure/search` and verify it matches scan_root A
- Start daemon for index B and verify it matches scan_root B
- Ensure sockets differ (hash differs) and stop both cleanly

## Implementation Sequence (Phased)

1. **Index core**
   - Add `IndexConfig` + `IndexPaths` + meta read/write helpers
2. **CLI plumbing**
   - Add global flags/env resolution
   - Thread `IndexConfig` into relevant CLI handlers
3. **Cache namespacing**
   - Call graph + languages + dirty + semantic cache paths via `IndexPaths`
4. **Ignore refactor**
   - Index-scoped ignore file creation + gitignore toggle
5. **Daemon/MCP**
   - Daemon keyed by `(cache_root,index_id)` and uses correct scan_root
6. **Index commands**
   - list/info/rm/gc
7. **Tests + docs**
   - Add new tests; update docs/README with examples and cache layout

## Design Decisions (Recommended)

1. **Positional path stays** for backward compatibility and convenience, and is treated as the default `scan_root`. `--scan-root` overrides it; specifying both with different resolved paths is an error.
2. **Index mode activates** when **any** of the new knobs is set (CLI or env). If `--index` is omitted in index mode, derive a deterministic `index_id` from `scan_root` (prefer `path:` relative to `cache_root`, otherwise `abs:`).
3. **Semantic model mismatch is a hard error** when searching with an explicit `--model` that differs from metadata; rebuilding with a different model requires an explicit overwrite flag (recommend `--rebuild`).
