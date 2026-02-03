# Implementation Plan: 001 - Isolated Indexes

Spec: `specs/001-feat-isolated-index.md`

## Summary

Add a first-class **Index** concept so TLDR can:

- **Scan code in-place** (e.g. `node_modules/pkg`, `.venv/.../site-packages/pkg`)
- **Store all TLDR state under a chosen cache root** (e.g. your repo root)
- **Maintain multiple isolated indexes** (one per `module@version`), preventing cache collisions across deps
- Support multi-index operation in the **daemon** and **MCP server**
- Provide **index management** commands (`list/info/rm/gc`)

## Phases (suggested delivery)

- Phase 1 (MVP: isolated index + semantic): CLI flag plumbing + per-index cache layout + semantic isolation + “no writes outside cache_root” guard + baseline tests.
- Phase 2 (Ignore + completeness): index-scoped ignore file + gitignore toggles + ensure every subcommand uses `IndexContext` consistently.
- Phase 3 (Daemon + MCP): per-index daemon identity + request routing + daemon isolation tests.
- Phase 4 (Index management): `tldr index list/info/rm/gc` (+ optional `gc`) + cleanup/UX polish.

## Goals (Acceptance Criteria)

1. CLI supports `--scan-root`, `--cache-root`, `--index` and uses them consistently across:
   - `warm`, `semantic index/search`, `tree`, `structure`, `search`, `context`, `calls`, `impact`, `imports`, `importers`, daemon commands.
   - `dead`, `arch`, `change-impact` (and any daemon/MCP wrappers that call these analyses).
   - Index flags work both **before and after** subcommands (argparse parent parser strategy; see “CLI Parsing” below).
2. All **per-scan_root persisted state** is **namespaced per index** under `cache_root/.tldr/indexes/<index_key>/...`.
   - Global config remains global under `cache_root/.tldr/config.json` and `cache_root/.claude/settings.json`.
3. Semantic indexing:
   - Never “walks upward” to find the project root when explicit `--cache-root/--index` are provided.
   - Reads/writes FAISS + metadata only under the selected index directory.
   - Stores a canonical embedding model id in index metadata and rejects mismatches (or forces explicit override).
   - Validates FAISS dimensionality matches expected model dim and stored metadata dim.
4. Ignore config is **index-scoped**:
   - Default ignore file at `cache_root/.tldr/indexes/<index_key>/.tldrignore`.
   - Add `--ignore-file`, and `--use-gitignore/--no-gitignore`.
   - Default ignore includes `.tldr/` (and scanners hard-prune `.tldr/` to prevent self-indexing).
   - Ignore configuration is persisted in `meta.json` and treated as part of the index’s stable configuration (see “Index metadata schema”).
5. Daemon/MCP:
   - Legacy daemon identity (socket/lock/pid naming) is unchanged when **not** in index mode.
   - In index mode, daemon socket/lock/pid identity is keyed by `(cache_root, index_id)` (implemented as `(resolved_cache_root, index_key)`) (one daemon per index).
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
   - Legacy mode behavior unchanged (cache layout + daemon identity)
   - Background subprocesses in index mode propagate flags and don’t write outside `cache_root` (in particular, they must not create `.tldr*` under `scan_root` when `scan_root != cache_root`)
8. In index mode, TLDR performs **no writes outside `cache_root`**:
   - All persisted TLDR state (call graph, languages, dirty, semantic, patch file hashes, dedup content index, hook stats, daemon status/config) lives under `cache_root/.tldr/...`.
   - When `scan_root != cache_root`, TLDR must not create or modify any files under `scan_root` (no `.tldr/`, no `.tldrignore`, no cache artifacts).
   - Runtime-only daemon artifacts (socket/pid/lock/port) may remain in the OS temp dir.

## Non-Goals (for this spec)

- Automatically resolving `module@version` for every ecosystem (leave to the “dep-tldr” skill / a future “auto id” helper).
- Multi-index queries (search across several indexes at once). The spec target is “one daemon per index”.
- Copying dependencies into a separate analysis directory (explicitly avoided).

## Proposed User-Facing CLI

### Global flags / env vars

#### CLI Parsing (argparse)

Implement index flags using an argparse **parent parser** attached to both the root parser and all subparsers so flags work **before and after** subcommands (e.g. `tldr --cache-root ... warm ...` and `tldr warm --cache-root ...`).

- Create `index_parent = argparse.ArgumentParser(add_help=False)` containing:
  - `--scan-root`, `--cache-root`, `--index`
  - ignore knobs: `--ignore-file`, `--use-gitignore/--no-gitignore`, plus existing `--ignore` + `--no-ignore` (+ env resolution where applicable)
  - optional cross-cutting write flags: `--force-rebind`, `--rebuild` (or attach these only to write subcommands)
- Attach it to the root parser **and** to **every** subcommand parser via `parents=[index_parent]` (and do the same for nested subparsers):
  - `semantic index/search`
  - `daemon start/stop/status/query/notify`
  - `index list/info/rm/gc`

#### Flags

Add index flags (and env vars) to all subcommands:

- `--scan-root <path>` (env `TLDR_SCAN_ROOT`)
- `--cache-root <path>` (env `TLDR_CACHE_ROOT`)
- `--index <id>` (env `TLDR_INDEX`)
- `--ignore-file <path>` (env `TLDR_IGNORE_FILE`)
- `--use-gitignore` / `--no-gitignore` (env `TLDR_USE_GITIGNORE=1|0`)
- Existing ignore knobs (already in CLI today; include in `index_parent` so they work after subcommands too):
  - `--ignore <pattern>` (repeatable; gitignore syntax)
  - `--no-ignore` (bypass ignore rules)
- `--force-rebind` (where applicable; see “Binding guard” below)
- `--rebuild` (where applicable; forces overwrite of per-index caches)

Precedence (highest → lowest):

1. CLI flags
2. env vars
3. legacy defaults (current behavior)

**Ignore flag semantics (important):**

- Legacy mode (no `--cache-root`): ignore knobs (`--ignore-file`, `--use-gitignore/--no-gitignore`, `--ignore`, `--no-ignore`) affect filtering for that invocation but do not change cache layout.
- Index mode (`--cache-root` set): ignore knobs are treated as **index configuration inputs** and are persisted in `meta.json` (via `ignore.*`).
  - Changing ignore **configuration inputs** for an existing index (ignore file path, gitignore toggle/root, `--ignore` patterns, `--no-ignore`) requires an explicit override (`--force-rebind` or a future explicit “reconfigure ignore” flow).
  - Editing the ignore file contents at the configured ignore file path invalidates caches and should trigger stale detection/rebuild; it must not require `--force-rebind`.

#### Write / overwrite flags (make explicit)

Some commands build or mutate per-index caches. Standardize overwrite semantics:

- `--force-rebind`:
  - Allowed on commands that create/overwrite index state (`warm`, `semantic index`, daemon-triggered rebuilds, etc.).
  - If an index directory already exists but is bound to a different `scan_root` (or has different ignore configuration inputs), `--force-rebind` wipes that index directory, rewrites `meta.json`, and proceeds.
- `--rebuild`:
  - Forces overwrite of existing per-index caches (e.g., semantic FAISS index + metadata; optionally call graph/languages/dirty-derived caches).
  - Should be supported at minimum by `tldr semantic index` (and optionally by `tldr warm`).

### Path semantics (positional vs `--scan-root`)

- For subcommands with a positional directory `path` argument (e.g. `warm/tree/structure/search`), treat that positional value as the default `scan_root`.
- `--scan-root` overrides the positional directory path.
- If both are provided and resolve to different paths, exit with an error (avoid silently indexing the wrong tree).
- For existing command-specific flags that represent a scan root (e.g. `context --project`, `semantic search --path`), keep them as aliases for `scan_root` for backward compatibility, but prefer documenting `--scan-root`.
- These legacy aliases only set `scan_root`; they do not activate index mode (layout stays legacy unless `--cache-root` is set).

### Optional convenience: `--scan-root` default from `meta.json` (quality-of-life)

When querying an existing index, allow omitting `--scan-root`:

- If `--cache-root` + `--index` identify an existing `meta.json`, default `scan_root` from meta for read-only commands (e.g. `semantic search`, `tree`, `search`, `context`, daemon query).
- Still enforce binding:
  - If the user provides an explicit `--scan-root` that differs from meta, error unless `--force-rebind`.
- For write commands (`warm`, `semantic index`), either:
  - require `--scan-root` explicitly, or
  - default from meta (safer UX), but still require `--rebuild` / dirty checks as appropriate.

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

Default behavior (no `--cache-root/--index`): unchanged.

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
  - `gitignore_root: Path` (derived; see “Gitignore Root” below)
  - `created_at/updated_at` (optional)

- `IndexPaths` helper (covers **all persisted state**):
  - `tldr_dir = cache_root/.tldr/` (global)
  - `tldr_config = tldr_dir/config.json` (global; used by daemon config loading in index mode)
  - `claude_settings = cache_root/.claude/settings.json` (global; used by daemon config loading in index mode)
  - `indexes_dir = tldr_dir/indexes/`
  - `index_dir = indexes_dir/<index_key>/`
  - `meta = index_dir/meta.json`
  - `ignore_file = index_dir/.tldrignore` (default; may be overridden by `--ignore-file`)
  - `cache_dir = index_dir/cache/`
  - `call_graph = cache_dir/call_graph.json`
  - `languages = index_dir/languages.json`
  - `dirty = cache_dir/dirty.json` (replaces legacy `.tldr/cache/dirty.json` in index mode)
  - `file_hashes = cache_dir/file_hashes.json` (for `tldr/patch.py`)
  - `content_index = cache_dir/content_index.json` (for `tldr/dedup.py`)
  - `stats_dir = index_dir/stats/` (for `tldr/stats.py`)
  - `hook_activity = stats_dir/hook_activity.jsonl`
  - `semantic_dir = cache_dir/semantic/`
  - `semantic_faiss = semantic_dir/index.faiss`
  - `semantic_metadata = semantic_dir/metadata.json`
  - `daemon_status = index_dir/status` (optional; stop writing `.tldr/status` under `scan_root`)

- `IndexContext` (computed once per invocation; recommended):
  - Purpose: centralize “which roots/paths/ignore rules are active” so command handlers don’t re-derive paths and accidentally write into `scan_root`.
  - Fields (suggested):
    - `mode: Literal["legacy", "index"]`
    - `scan_root: Path`
    - `cache_root: Path | None` (None in legacy)
    - `index_id: str | None`
    - `index_key: str | None`
    - `paths: IndexPaths | LegacyPaths`
    - `ignore_spec: IgnoreSpec | None`
    - `workspace_root: Path | None` (where `.claude/workspace.json` is loaded from; see below)
  - All CLI subcommands should build an `IndexContext` first and then pass it through to:
    - call graph builders / warm
    - semantic index/search
    - analysis wrappers (`impact/dead/arch/context/change-impact`)
    - daemon/MCP startup + request handling

### Index ID vs on-disk key

To stay cross-platform, introduce a filesystem-safe `index_key`:

- Keep `index_id` as the stable external identifier shown to users and stored in `meta.json`.
- Derive `index_key` via a deterministic encoding (recommended):
  - Prefer a **short hash-only** key to reduce Windows `MAX_PATH` risk:
    - `index_key = base32(sha256(index_id))[:20]`
  - Store the human-readable `index_id` in `meta.json` and use `tldr index list/info` to map keys back.

Rationale: `:` and `/` are invalid on Windows; raw `index_id` cannot safely be used as a directory name.

### Spec alignment (index_id vs on-disk key)

The spec uses `cache_root/.tldr/indexes/<index_id>/...` as a conceptual layout. Implementation should be explicit that:

- The on-disk directory name is `index_key`, not raw `index_id` (Windows path safety).
- `meta.json` is the source of truth for mapping `index_key -> index_id`.
- `tldr index list/info` is the canonical user surface for this mapping.
- The spec examples elide internal subdirectories; this plan uses `indexes/<index_key>/cache/...` for most mutable caches to keep the index dir tidy. Tests/docs should assert against the actual layout.

### Index metadata schema + binding rules

Add a small `meta.json` schema (versioned) used by CLI, daemon, and index management:

- `schema_version: int`
- `index_id: str`
- `index_key: str`
- `scan_root_abs: str` (resolved, normalized path string)
- `cache_root_abs: str` (resolved, normalized path string)
- `scan_root_rel_to_cache_root: str | null` (portable when `scan_root` is under `cache_root`)
- `created_at: str` (ISO-8601)
- `last_used_at: str` (ISO-8601; rate-limit updates to avoid constant rewrites)
- `tldr_version: str` (optional; for future migrations)
- `ignore: { ignore_file: str | null, use_gitignore: bool, gitignore_root_abs: str | null, cli_patterns: list[str], no_ignore: bool, config_fingerprint: str, fingerprint: str }` (persisted index config; see below)
- `semantic: { model: str, dim: int, lang: str | null }` (optional; if semantic index exists)

Binding guard (prevents silent cross-contamination):

- If `index_dir/meta.json` exists, validate it matches the requested identity:
  - `meta.index_id == requested index_id`
  - Compare normalized paths using `Path(...).resolve()` and `os.path.normcase()` (Windows):
    - If `scan_root_rel_to_cache_root` is present and requested `scan_root` is under requested `cache_root`, prefer comparing the **relative** value (portable across moves).
    - Otherwise compare `scan_root_abs`.
  - Ensure `index_dir` is under `cache_root/.tldr/indexes`.
  - **Do not hard-fail solely due to `cache_root_abs` mismatch.** Index dirs are already discovered under the current `cache_root`, and repos commonly move (CI temp dirs, user renames). Treat `cache_root_abs` as informational/debug and update it opportunistically (rate-limited) when opening the index.
  - Binding guard remains strict only for `index_id` + `scan_root` (ignore edits should not invalidate the index identity).
  - Ignore configuration is guarded separately:
    - Persist `ignore.config_fingerprint = sha256(ignore.ignore_file + ignore.cli_patterns + ignore.no_ignore + ignore.use_gitignore + ignore.gitignore_root_abs)` in `meta.json`.
    - Persist `ignore.fingerprint = sha256(ignore.config_fingerprint + ignore file contents)` in `meta.json`.
    - If the user passes ignore-related flags that would change `ignore.config_fingerprint`, require an explicit override:
      - either `--force-rebind` (wipe index dir, rewrite meta, rebuild as needed), or
      - a dedicated “reconfigure index ignore” flow (optional enhancement) that updates `meta.json` and invalidates caches without requiring a new `--index`.
    - If `ignore.config_fingerprint` matches but `ignore.fingerprint` changes (e.g., the ignore file contents were edited), treat caches as stale and rebuild/invalidate as appropriate (see “Write policy” and “Add index validity breadcrumbs” below).
- On mismatch: **error** with actionable next steps:
  - choose a new `--index`
  - or run `tldr index rm <index_id>`
  - or run the original command with `--force-rebind` (wipes that index directory and rewrites meta)

### Add index validity breadcrumbs (recommended)

Add cheap “belongs to this index” metadata to every persisted artifact so TLDR can detect “wrong index loaded” even if directories are moved/copied:

- Include `index_id`, `index_key`, `scan_root_abs`, `cache_root_abs` and the current `ignore.fingerprint` in:
  - `call_graph.json`
  - `languages.json`
  - `cache/semantic/metadata.json`
- On load, validate those fields match the active `IndexConfig` (or error with a clear “wrong index / stale cache” message).

### Write policy (index mode)

Make “what causes writes” explicit to avoid surprise behavior:

- **Index creation/mutation (writes required):** `tldr index init`, `tldr warm`, `tldr semantic index`, `tldr daemon start` (and any rebuild flows).
  - These commands may create `index_dir/`, write `meta.json`, create the default index-scoped `.tldrignore`, and write caches.
- **Query commands (should not create a new index):** `tree/structure/search/context/calls/impact/imports/importers/...`, `tldr daemon query/status`, MCP calls.
  - If `--cache-root/--index` are provided but `meta.json` is missing, error with next steps (`tldr index init` or `tldr warm`).
  - If required caches are missing/stale:
    - Query commands may compute best-effort in-memory results but must not write any caches (unless a write command is explicitly invoked).
    - If a query cannot run without the persisted cache (e.g. semantic search without a FAISS index), error with next steps (`tldr warm` / `tldr semantic index`).
  - Query commands may **best-effort** update `meta.last_used_at` (rate-limited) for GC purposes, but failures to lock/write must not fail the query.

## Cache Layout (New)

Under `cache_root`:

```
.tldr/
  config.json
  indexes/
    <index_key>/
      meta.json
      languages.json
      .tldrignore
      status
      cache/
        call_graph.json
        dirty.json
        file_hashes.json
        content_index.json
        semantic/
          index.faiss
          metadata.json
      stats/
        hook_activity.jsonl
```

Keep existing single-index layout for legacy mode (no `--cache-root/--index`), at minimum:

```
.tldr/cache/call_graph.json
.tldr/languages.json
.tldr/cache/semantic/{index.faiss,metadata.json}
.tldr/cache/dirty.json
```

## Backward Compatibility Strategy

Two viable approaches; pick one early and keep it consistent:

### Option A (recommended): “Legacy mode” stays untouched

- If the user does not provide any **index mode knobs** (`--cache-root`, `--index`, or env equivalents), keep current paths and behavior exactly.
- Index mode requires a cache root:
  - Index mode activates only when `--cache-root` (or env `TLDR_CACHE_ROOT`) is set.
  - `--index` (or env `TLDR_INDEX`) selects/creates a namespace under that cache root.
  - If `--index` is set but `--cache-root` is missing, exit with an error explaining that index mode requires an explicit cache root (to preserve the “no writes outside cache_root” guarantee).
  - `--scan-root` by itself is allowed but **does not** change the cache layout (still legacy).
  - Positional paths and legacy alias flags (e.g. `context --project`, `semantic search --path`) only set `scan_root`; they do **not** activate index mode.
- When index mode is active, use the new `indexes/<index_key>/...` layout.
- If in index mode and `--index` is omitted, auto-derive a deterministic `index_id` from `scan_root`:
  - Prefer `path:<scan_root relative to cache_root>` when `scan_root` is under `cache_root`.
  - Otherwise use `abs:<scan_root resolved>`.

Ignore knobs (`--ignore-file`, `--use-gitignore/--no-gitignore`) must work in **both** modes and must not implicitly switch layouts. This avoids surprising behavior like “toggled gitignore → changed all cache locations”.

Pros: no migration complexity; no test churn for existing expectations.
Cons: two layouts to support long-term.

## Ignore Handling (Index-Scoped)

### Implementation changes

Update `tldr/tldrignore.py` to support:

- Loading patterns from an arbitrary ignore file path (not necessarily `scan_root/.tldrignore`).
- Allow specifying a separate `gitignore_root` (see “Gitignore Root” below).

Suggested interface:

- `load_ignore_patterns_from_file(ignore_file: Path) -> PathSpec`
- `ensure_ignore_file(ignore_file: Path) -> (created: bool, msg: str)`
- `IgnoreSpec(scan_root: Path, ignore_file: Path, use_gitignore: bool, gitignore_root: Path, cli_patterns: list[str] | None)`
  - `match_file(rel_path)` uses patterns relative to `scan_root`.
  - gitignore checks use `gitignore_root` with relpaths relative to that root.

Default in index mode:

- `ignore_file = index_dir/.tldrignore` (create if missing using existing DEFAULT_TEMPLATE)
- Introduce two default templates to avoid “dependency index is empty” failures:
  - `DEFAULT_TEMPLATE_PROJECT` (current behavior; includes `dist/`, `build/`, etc.)
  - `DEFAULT_TEMPLATE_DEP` (minimal; **does not** ignore `dist/` / `build/` by default; still ignores nested `node_modules/`, VCS dirs, caches like `__pycache__/`, and TLDR state like `.tldr/`)
- Update project-default ignore to include:
  - `.tldr/` (prevents self-indexing when cache_root is within scan_root)
  - `.tldr/indexes/` (defense-in-depth against self-indexing if any scanner misses a prune)
  - optional: `.claude/` (often large/noisy)
  - Apply this to legacy default ignore templates too (so `.tldr/indexes/` is ignored even in non-index mode).
- Additionally, hard-prune `.tldr/` (and `indexes/`) in scanners regardless of ignore config to prevent recursion/self-indexing.
- `use_gitignore = False` if `scan_root` looks like a dependency dir (`node_modules`, `.venv`, `site-packages`) OR if user passes `--no-gitignore`
- `gitignore_root = <git toplevel for cache_root>` if `use_gitignore=True` (fallback to `cache_root` if not in a git repo)
- If `use_gitignore=True` but gitignore would ignore most/all of `scan_root` (common for `node_modules/` inside a repo), warn loudly and/or auto-disable gitignore unless explicitly forced.
- Persist ignore configuration in `meta.json`:
  - On index creation, write `ignore.ignore_file` (prefer a path relative to `cache_root` when possible), `ignore.use_gitignore`, `ignore.gitignore_root_abs`, `ignore.cli_patterns`, `ignore.no_ignore`, `ignore.config_fingerprint`, and `ignore.fingerprint`.
  - On subsequent runs, if the user does **not** pass ignore-related flags, default to the ignore config stored in meta to keep caches consistent.
  - If the user *does* pass ignore-related flags that would change `ignore.config_fingerprint`, require `--force-rebind` (or an explicit “reconfigure ignore” flow).
  - If the ignore file contents change with the same `ignore.ignore_file`, update `ignore.fingerprint` and treat persisted caches as stale (invalidate and rebuild on the next write command; query commands must not auto-write).

#### `--ignore-file` safety policy (index mode)

Avoid accidental writes into dependency trees:

- If `--ignore-file` points **inside `scan_root`** and the file is missing: hard error (refuse to create it under `scan_root`; use the default index ignore file or create it yourself).
- If `--ignore-file` is missing and is **outside `cache_root`**: hard error (refuse to create files outside `cache_root` in index mode).
- Otherwise, auto-create only when writing the default index-scoped ignore file under `index_dir` (or when the user explicitly points `--ignore-file` somewhere under `cache_root`).

### Plumbing: ensure all scanners use the same ignore rules

Thread ignore handling through all call sites that currently bypass CLI ignore config:

- `tldr/cross_file_calls.py::scan_project(...)`:
  - accept an `IgnoreSpec` (or `ignore_file + use_gitignore + gitignore_root`) instead of loading from `scan_root` internally.
- `tldr/api.py::scan_project_files(...)`:
  - pass the `IgnoreSpec` through to `cross_file_calls.scan_project(...)`.
- `tldr/semantic.py` helpers:
  - `extract_units_from_project(...)` accepts an `IgnoreSpec` (or the same knobs) so semantic indexing uses the same ignore set as `warm/search/tree/...`.
  - `_detect_project_languages(...)` uses the same ignore mechanism.
- `tldr/daemon/cached_queries.py`:
  - stop constructing `IgnoreSpec(project, use_gitignore=True)` directly; use the daemon’s `IndexConfig`/`IndexPaths`-derived ignore spec.
- Daemon startup:
  - stop creating `.tldrignore` inside `scan_root`; in index mode, create the ignore file at `IndexPaths.ignore_file` (or `--ignore-file` target).
- **Commands that bypass ignore spec today (must-fix for feature correctness):**
  - `tldr/api.py::get_relevant_context(...)`:
    - replace direct `project.rglob("*")` walking with an ignore-aware scan (either reuse `scan_project_files(...)` with an index-derived `IgnoreSpec`, or thread an `IgnoreSpec` into the function).
    - ensure the call graph build uses the same ignore rules.
  - `tldr/analysis.py::{analyze_impact, analyze_dead_code, analyze_architecture}`:
    - accept `ignore_spec: IgnoreSpec | None` (or accept `IndexContext/IndexConfig`) and pass it to call graph + structure builders.
  - `tldr/change_impact.py`:
    - must use index-scoped dirty flag path (via `IndexPaths.dirty`) and pass ignore spec into `scan_project_files(...)` and impact analysis.
  - CLI handlers `context`, `impact`, `dead`, `arch`, `change-impact` must build the ignore spec from `IndexContext` and pass it through (including in daemon cached wrappers).

### Gitignore Root

In index mode, `gitignore_root = cache_root` is not always correct (monorepos, nested working trees, or cache_root not in git at all).

Rule:

- When `use_gitignore=True`, set `gitignore_root` to the actual git toplevel for `cache_root` (e.g., via `git -C <cache_root> rev-parse --show-toplevel`).
- If that fails, treat as “not a git repo” and fall back to `cache_root` (gitignore checks become no-ops).

### Workspace config (`.claude/workspace.json`)

`cross_file_calls.build_project_call_graph(..., use_workspace_config=True)` loads workspace config from the *root it is given*. In index mode this can subtly change scope:

- If `scan_root` is a package subdir in a monorepo, `.claude/workspace.json` likely lives at repo root (`cache_root`), not at `scan_root`.
- Passing `scan_root` as the root will miss workspace config and change results compared to legacy behavior.

Policy decision (make explicit and test/document):

- In index mode, if `scan_root` is under `cache_root`, load workspace config from `cache_root` and **rebase** pattern evaluation so config stays relative to the workspace root:
  - `workspace_root = cache_root`
  - `prefix = scan_root.relative_to(workspace_root)`
  - For each scanned file: `workspace_rel = prefix / rel_to_scan_root`
  - Evaluate `workspace.json` include/exclude patterns against `workspace_rel` (not `rel_to_scan_root`).
- If `scan_root` is outside `cache_root`, disable workspace config by default (or require an explicit opt-in flag).

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
   - Normalize to a single `canonical_model_id` for persistence and comparisons (e.g. always HF name):
     - Handle both short keys (`bge-large-en-v1.5`) and HF ids (`BAAI/bge-large-en-v1.5`) without false mismatches.
   - `meta.json` stores the selected embedding model (canonical id) and dimension.
   - `semantic/metadata.json` also stores model + dimension; enforce consistency.
4. Model mismatch policy:
   - **Search:** if the user supplies `--model` and it does not match the model recorded in `semantic/metadata.json`, error (results would be invalid and FAISS dimensions may not match). If `--model` is omitted, always use the model recorded in metadata.
   - **Index build:** if an index already exists and the requested `--model` differs, require an explicit overwrite flag (recommend `--rebuild`) to overwrite.
   - Always validate embedding **dimension** (FAISS index dim == metadata dim == model dim) before querying.
5. Ignore consistency:
   - Stop calling `ensure_tldrignore(project_root)` in index mode (never create `.tldrignore` under `scan_root`).
   - Build and pass a single `IgnoreSpec` (from `IndexConfig`) into:
     - unit extraction
     - language detection
     - any file walking performed by semantic indexing

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
3. Fix existing daemon call-graph persistence inconsistency:
   - Daemon warm handler writes `.tldr/cache/call_graph.json`, but daemon `_ensure_call_graph_loaded()` currently tries `.tldr/call_graph.json`.
   - Canonicalize legacy path to `.tldr/cache/call_graph.json` everywhere.
   - In index mode, use `IndexPaths.call_graph` for both load and save.

## Other Persisted State to Isolate (Patch/Dedup/Stats/Daemon)

The current codebase writes additional state under `.tldr/...` that must be moved in index mode to avoid polluting `scan_root` and to prevent cross-index collisions:

- Patch/incremental cache (`tldr/patch.py`): move `.tldr/cache/file_hashes.json` → `IndexPaths.file_hashes`.
- Dedup content index (`tldr/dedup.py`): move `.tldr/cache/content_index.json` → `IndexPaths.content_index`.
- Hook stats (`tldr/stats.py`): move `.tldr/stats/hook_activity.jsonl` → `IndexPaths.hook_activity`.
- Daemon status/pid/config:
  - Stop writing `.tldr/daemon.pid` and `.tldr/status` under `scan_root` in index mode; write status to `IndexPaths.daemon_status` (optional) and rely on temp-dir pid/lock for process identity.
  - Load daemon config from `IndexPaths.tldr_config` (i.e., under `cache_root/.tldr/`) instead of `scan_root/.tldr/config.json` in index mode.
  - In index mode, read `.claude/settings.json` from `IndexPaths.claude_settings` (i.e., under `cache_root/.claude/`) instead of `scan_root/.claude/settings.json`.

## Atomic IO + Corruption Resistance

Index mode increases concurrent read/write scenarios (CLI + daemon + background warm + semantic rebuild). Make persistence robust:

- Introduce a shared helper (e.g. `atomic_write_text()` / `atomic_write_json()`) and update *all* persisted writers to use it:
  - `dirty_flag.mark_dirty`
  - `patch.save_file_hash_cache`
  - `dedup.save`
  - index `meta.json` writes
  - semantic metadata writes
- All JSON writes (meta, call graph, languages, dirty, file_hashes, content_index) are **atomic**: write to a temp file in the same directory, then `os.replace()`.
- FAISS index writes are **atomic**: write to a temp path then replace.
- Atomic replace prevents corruption but **not lost updates** for read-modify-write files.
  - Protect all per-index state writes with a lock (recommended simplest: reuse `index_dir/index.lock` for *any* write to per-index state).
  - At minimum, lock around updates to: `dirty.json`, `file_hashes.json`, `content_index.json`, and any “patching” updates to `call_graph.json`.
  - Optionally split into `state.lock` (all writes) vs `rebuild.lock` (long rebuilds) if contention becomes an issue.

## Daemon + MCP Multi-Index Support

### Identity / socket naming

Update `tldr/daemon/startup.py` and `tldr/daemon/core.py`:

- Legacy mode: keep existing identity (socket/lock/pid naming) keyed exactly as today (e.g. hash of resolved scan root).
- Index mode: use `hash(str(resolved_cache_root) + "\0" + index_key)` for identity.
  - `index_key` is deterministic from `index_id` and avoids long/unsafe strings; `meta.json` remains the source of truth for mapping `index_key ↔ index_id`.
- Keep tmp files:
  - `/tmp/tldr-<hash>.sock`
  - `/tmp/tldr-<hash>.pid`
  - `/tmp/tldr-<hash>.lock`
  - `/tmp/tldr-<hash>.port` (Windows only; see below)

Windows note (port collisions):

- Current deterministic port mapping can collide across multiple indexes.
- Mitigation: on bind failure, choose an ephemeral free port and persist it to `/tmp/tldr-<hash>.port` so clients can reliably discover the chosen port.
- Client discovery rule:
  - If `port_file` exists, use it.
  - Otherwise fall back to deterministic mapping.
  - Clean up the port file on daemon shutdown and on `tldr index rm/gc`.

### Daemon config

Update TLDRDaemon to be created with `IndexConfig`:

- `cache_root` determines where `.tldr/indexes/...` lives
- `scan_root` determines what the daemon scans/serves
- All query handlers use:
  - `scan_root` for filesystem scans and relative paths
  - `index_paths.*` for caches
- Config roots:
  - In index mode, daemon reads `.claude/settings.json` from `cache_root` (not `scan_root`).
  - In index mode, daemon reads `.tldr/config.json` from `cache_root/.tldr/config.json` (not `scan_root/.tldr/config.json`).
  - Legacy behavior unchanged.

### CLI surface

Update `tldr cli daemon {start,stop,status,query,notify}` to accept global index flags and propagate them to daemon startup/query.
- In index mode, daemon start must not pre-create `.tldr/` or `.tldrignore` under `scan_root`.

### Windows daemon spawn propagation (`python -m tldr.daemon`)

On Windows, daemon startup spawns a new process (not a fork). Index mode must work through this path or the feature will be partially broken.

- Update `tldr/daemon/startup.py::main` argparse to accept the same index flags/env resolution as the main CLI:
  - `--cache-root`, `--scan-root`, `--index`
  - `--ignore-file`, `--use-gitignore/--no-gitignore`, `--ignore`, `--no-ignore`
  - `--force-rebind`, `--rebuild` (as applicable)
- Update the Windows spawn invocation to pass these args explicitly (prefer args for debuggability; env as fallback).
- Ensure all identity helpers used by socket/lock/pid/port discovery are keyed by `(resolved_cache_root, index_key)` in index mode (see “State Audit Checklist”).

### MCP server

Update `tldr/mcp_server.py`:

- Accept `--cache-root`, `--scan-root`, `--index` flags (and env vars).
- Compute socket identity using the same legacy vs index-mode identity rules as the daemon.
- When auto-starting the daemon, pass the same index flags through to `tldr cli daemon start ...`.
- Enforce fixed index identity: if MCP is started with `--cache-root/--scan-root/--index`, reject or ignore per-call project args that don’t match to prevent accidental cross-index queries.

## Subprocess Flag Propagation (Must-Fix)

Any subprocess “re-exec” must preserve the active index configuration; otherwise it will silently revert to legacy layout and write to the wrong place.

- Add a helper: `build_reexec_args(index_config, base_cmd: list[str]) -> list[str]`
- Use it in:
  - `tldr warm --background` spawn
  - `tldr/session_warm.py` background warm spawn
  - Windows daemon spawn (`python -m tldr.daemon ...`)
  - daemon semantic reindex subprocess (`_trigger_background_reindex()` or equivalent)
  - MCP daemon auto-start

## Index Management Commands

Add `tldr index` subcommand tree in `tldr/cli.py`:

- `tldr index list --cache-root <path>`
  - List all `indexes/*/meta.json` entries (display `index_id`, `scan_root`, size, last_updated)
- `tldr index info <index_id> --cache-root <path>`
  - Resolve `index_id` to `index_key` by scanning meta files
  - Print meta + cache file presence and sizes
- `tldr index rm <index_id> --cache-root <path> [--force]`
  - Deletes that index directory only
  - Refuse to delete an index that appears to have a running daemon unless `--force`
  - Refuses to remove unknown/unparseable dirs unless `--force`
- `tldr index gc --cache-root <path> [--days N] [--max-total-mb M]` (optional)
  - Removes indexes not used recently (based on `meta.json` timestamps)
  - Skip or refuse indexes with a running daemon unless `--force`

### Optional enhancements

- `tldr index init`:
  - Creates `meta.json` + default ignore file without warming/indexing.
- `tldr index prune-orphans` (or fold into `gc`):
  - Removes index dirs missing/invalid `meta.json` (with `--force`).

## State Audit Checklist (Must-Do Before Coding)

Audit and update every `.tldr/...` write in the repo so in index mode all persisted state stays under `cache_root` (and `scan_root` is untouched when `scan_root != cache_root`):

- `tldr/cli.py`: call graph cache, languages cache, ignore file creation
- `tldr/cli.py`: warm `--background` subprocess args must include index flags
- `tldr/session_warm.py`: call graph cache path + background warm subprocess args must include index flags
- `tldr/dirty_flag.py`: dirty.json path
- `tldr/semantic.py`: semantic dir + ignore file creation location + `_find_project_root` bypass
- `tldr/cross_file_calls.py` / `tldr/api.py::scan_project_files`: ignore plumbing
- `tldr/api.py::get_relevant_context`: must stop bypassing ignore config via `rglob` and must honor index-scoped ignore
- `tldr/analysis.py::analyze_impact/analyze_dead_code/analyze_architecture`: pass ignore spec through wrappers
- `tldr/change_impact.py`: dirty flag path + impact/import scans must use index-scoped ignore + paths
- `tldr/patch.py`: file_hashes.json
- `tldr/dedup.py`: content_index.json
- `tldr/stats.py`: stats dir + hook_activity.jsonl
- `tldr/daemon/core.py`: config loading, pid/status writes, call graph load/save path consistency
- `tldr/daemon/core.py::_get_tmp_pid_path`: must use index identity `(resolved_cache_root, index_key)` in index mode
- `tldr/daemon/startup.py::_get_lock_path/_get_pid_path/_get_socket_path`: must use index identity `(resolved_cache_root, index_key)` in index mode
- Windows port file (`/tmp/tldr-<hash>.port`, if used): must also be keyed by index identity `(resolved_cache_root, index_key)`
- `tldr/daemon/startup.py::main` (`python -m tldr.daemon` entrypoint): must accept index flags/env resolution and honor index paths
- `tldr/daemon/startup.py`: index-mode identity + startup coordination; must not create `.tldrignore` under `scan_root`
- `tldr/daemon/core.py`: background semantic reindex subprocess args must include index flags
- `tldr/cli.py` daemon start: stop pre-creating `.tldr/` under `scan_root` in index mode
- `tldr/daemon/cached_queries.py`: ignore spec must come from `IndexConfig`/`IndexPaths`
- `tldr/mcp_server.py`: index flags, identity, and daemon auto-start must propagate index flags

## Testing Plan

### Unit tests

1. `tests/test_index_paths.py`
   - `IndexPaths` resolves distinct directories for two different `index_id`s under the same `cache_root`
   - `index_key` sanitization is deterministic and collision-resistant

2. `tests/test_ignore_index_scoped.py`
   - Using `--ignore-file` affects filtering even when `scan_root` is elsewhere
   - Editing index-local `.tldrignore` invalidates caches without requiring `--force-rebind`

3. `tests/test_cli_parsing_index_flags.py` (pure python)
   - `tldr --cache-root X warm Y --index Z` parses
   - `tldr warm Y --cache-root X --index Z` parses

4. `tests/test_workspace_config_rebase.py`
   - `.claude/workspace.json` patterns still apply when `scan_root` is a subdirectory of `cache_root`

### Integration tests (no daemon)

`tests/test_isolated_indexes_integration.py`:

- Create a tmp “repo” directory as `cache_root`
- Create two “deps” as separate scan roots with distinct code
- Run:
  - `build_semantic_index(scan_root=A, cache_root=repo, index=A, model=TEST_MODEL)`
  - `build_semantic_index(scan_root=B, cache_root=repo, index=B, model=TEST_MODEL)`
- Assert:
  - `repo/.tldr/indexes/<key(A)>/cache/semantic/...` and `repo/.tldr/indexes/<key(B)>/cache/semantic/...` both exist
  - `semantic_search(..., index=A)` never opens B’s FAISS
  - Results differ by content (each dep has a uniquely-named function)

### Integration tests (daemon)

`tests/test_daemon_isolated_indexes.py` (run on Unix and Windows if feasible):

- Start daemon for index A, query `tree/structure/search` and verify it matches scan_root A
- Start daemon for index B and verify it matches scan_root B
- Ensure sockets differ (hash differs) and stop both cleanly

### Additional regression/behavior tests (cover critical gaps)

1. **No writes outside cache_root (index mode)**
   - Run `tldr warm` (and optionally daemon start) with `scan_root=node_modules/pkg` and `cache_root=<repo>`.
   - Assert `scan_root/.tldr*` is not created (when `scan_root != cache_root`).
   - Assert writes under `cache_root/.tldr/indexes/...` are allowed/expected.
   - Run `tldr warm --background` with the same flags and assert the same.
   - Run `tldr context`, `tldr impact`, `tldr dead`, `tldr arch`, and `tldr change-impact` with the same flags and assert the same (these commands currently bypass ignore/index plumbing and are a high regression risk).

2. **Ignore plumbing consistency**
   - Ensure the same ignore rules apply to:
     - `scan_project_files` (importers)
     - `warm` scan
     - semantic unit extraction / language detection
     - `context` (`get_relevant_context`) and `impact/dead/arch` analyses
   - Add an index-scoped `.tldrignore` that excludes a known file/function and assert those commands do not report it.

3. **Index binding guard**
   - Create index `A` bound to scan_root `A`.
   - Re-run with `--index A` but scan_root `B`.
   - Expect a hard error unless `--force-rebind`.

4. **Workspace config rebase**
   - Place `.claude/workspace.json` at `cache_root` with an exclusion like `**/dist/**`.
   - Set `scan_root = cache_root/packages/foo` and ensure `packages/foo/dist/x.js` is excluded even though `rel_to_scan_root == dist/x.js`.

5. **Daemon call graph persistence regression**
   - Warm call graph, restart daemon, verify it loads from the canonical path (legacy: `.tldr/cache/call_graph.json`; index: `IndexPaths.call_graph`).

6. **Windows daemon spawn propagation (manual checklist / Windows CI)**
   - Start daemon in index mode and confirm it loads `meta.json` and uses `cache_root/.tldr/indexes/<index_key>/...` (not `scan_root/.tldr/...`).

7. **Windows port collision fallback (if implemented)**
   - Simulate two index identities mapping to the same deterministic port and ensure daemon startup still succeeds using the persisted port file.

8. **Legacy mode unchanged**
   - Run `tldr warm .` with no index knobs and assert legacy caches are written under `./.tldr/cache/...` and no `./.tldr/indexes/` directory is created.
   - Start daemon with no index knobs and assert socket identity matches legacy naming (no `(cache_root,index_id)` identity applied).

### Test performance constraints

- Avoid large model downloads in CI:
  - Use `all-MiniLM-L6-v2` (or similar small model) for integration tests via an env guard like `TLDR_TEST_MODEL`.
  - Alternatively monkeypatch embedding/model resolution in tests.

## Implementation Sequence (Phased)

1. **Index core**
   - Add `IndexConfig` + `IndexPaths` + `IndexContext` + meta read/write helpers
   - Implement binding validation + `--force-rebind`
2. **CLI plumbing**
   - Add global flags/env resolution (parent parser attached to root parser + all subcommands)
   - Implement index mode activation rule (`--cache-root` required; `--index` without `--cache-root` is an error)
   - Thread `IndexContext` into all CLI handlers (and daemon/MCP construction)
3. **State audit + namespacing**
   - Call graph + languages + dirty + semantic cache paths via `IndexPaths`
   - Patch/dedup/stats/daemon state migrated to `IndexPaths` (no writes outside `cache_root`)
   - Fix daemon call graph load/save inconsistency (canonicalize legacy path)
4. **Ignore refactor**
   - Index-scoped ignore file creation + gitignore root derivation
   - Thread ignore spec into `cross_file_calls`, `api.scan_project_files`, semantic helpers, daemon cached queries
   - Fix ignore bypasses in `api.get_relevant_context`, `analysis.py` wrappers, and `change_impact.py`
5. **Atomic IO + locking**
   - Atomic writes for all JSON + FAISS; lock around all per-index state writes
6. **Daemon/MCP**
   - Daemon keyed by `(cache_root,index_id)` and uses correct scan_root
   - Preserve legacy daemon identity when not in index mode
   - Windows port collision mitigation if using TCP
7. **Index commands**
   - list/info/rm/gc with “refuse to delete running daemon” safety checks
8. **Tests + docs**
   - Add new tests; update docs/README with examples and cache layout

## Design Decisions (Recommended)

1. **Positional path stays** for backward compatibility and convenience, and is treated as the default `scan_root`. `--scan-root` overrides it; specifying both with different resolved paths is an error.
2. **Index mode activates** only when `--cache-root` (or env `TLDR_CACHE_ROOT`) is set. `--index` (or env `TLDR_INDEX`) requires an explicit cache root and selects a namespace under it. `--scan-root` and legacy alias flags (`--project/--path`) only set `scan_root` and must not switch layouts. Ignore-only knobs (`--ignore-file`, `--use-gitignore/--no-gitignore`) must not switch layouts.
3. **Semantic model mismatch is a hard error** when searching with an explicit `--model` that differs from metadata; rebuilding with a different model requires an explicit overwrite flag (recommend `--rebuild`).
4. **Index binding is strict (but portable)**:
   - `meta.json` must match `index_id` and `scan_root` (prefer `scan_root_rel_to_cache_root` when available; otherwise `scan_root_abs`).
   - Never hard-fail solely due to `cache_root_abs` mismatch (repos move); treat it as informational and update it opportunistically.
   - Treat ignore **configuration inputs** as stable index config; changing them requires explicit `--force-rebind` (or an explicit reconfigure ignore flow). Editing ignore file contents invalidates caches and must not require a rebind.
5. **Gitignore root is debuggable**: derive gitignore root from `git rev-parse --show-toplevel` for `cache_root`; fall back to `cache_root` if not in a git repo.
6. **All writes are atomic** (temp + `os.replace()`), and long rebuilds take a per-index lock.
7. **Rate-limit meta rewrites**: update `last_used_at` at most once every N minutes to reduce unnecessary IO.

## Definition of Done

- [ ] Windows + Unix daemon works with index mode (including the Windows `python -m tldr.daemon` spawn path).
- [ ] Two indexes under one `cache_root` never collide (semantic + call graph + daemon identity).
- [ ] `scan_root` is untouched when `scan_root != cache_root` (no `.tldr*` artifacts in dependency trees).
- [ ] Workspace config works when `scan_root` is a subdirectory.
- [ ] Editing index-local `.tldrignore` triggers stale detection/rebuild without forcing a rebind.
- [ ] Legacy mode behavior unchanged, including legacy daemon identity.
