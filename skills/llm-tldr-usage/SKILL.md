---
name: llm-tldr-usage
description: Use when Codex needs to operate llm-tldr (tldr CLI) to index, search, or analyze a codebase. Covers warm/semantic/context/search/impact/slice workflows, daemon usage, .tldrignore handling, index isolation with --cache-root and --index, and index management via tldr index list/info/rm.
---

# Llm Tldr Usage

## Overview
Use this skill to drive llm-tldr CLI workflows for indexing and querying codebases. Prefer index mode with `--cache-root=git` to keep all `.tldr` data under the repo root and avoid scattered caches.

## Quick Start
1. Index the current repo (structural caches):
```bash
tldr warm --cache-root=git .
```
2. Build semantic embeddings (one-time per index):
```bash
tldr semantic index --cache-root=git .
```
3. Ask a behavioral question:
```bash
tldr semantic search "describe how auth tokens are validated" --cache-root=git --path .
```
4. Pull focused context for an edit:
```bash
tldr context login --cache-root=git --project .
```

## Choose The Right Path
- Index the current repo and search it.
- Index a dependency with an isolated index id.
- Manage existing indexes under a shared cache root.

## Index A Repo (Default Case)
- Build indexes:
```bash
tldr warm --cache-root=git /path/to/repo
```
- Build semantic embeddings:
```bash
tldr semantic index --cache-root=git /path/to/repo
```
- Search semantically:
```bash
tldr semantic search "retry logic for network calls" --cache-root=git --path /path/to/repo
```
- Use structural tools when you need precise code shape:
```bash
tldr structure src/ --lang python --cache-root=git /path/to/repo
```

## Index A Dependency (Isolated)
Use an explicit index id so dependency data does not mix with the main repo.

1. Pick an index id format that is stable and versioned, for example:
`dep:<name>@<version>:site`
2. Create or rebuild the dependency index:
```bash
tldr warm --cache-root=git --index dep:requests@2.31.0:site /path/to/site-packages/requests
```
3. Query the dependency directly:
```bash
tldr semantic search "Session.request implementation" --cache-root=git --index dep:requests@2.31.0:site --path /path/to/site-packages/requests
```

If the dependency path or version is unknown, run the dependency indexer skill helper first:
```bash
python skills/llm-tldr-dep-indexer/scripts/ensure_dep_index.py python requests
```
Use the returned `index_id` and `scan_root` for subsequent `tldr` commands.

## Example: Repo + Dependency Under One Cache Root
With `--cache-root=git`, all indexes created inside a repo live under the repo-root `.tldr/` regardless of where you run the command.

Dependency index (stored under repo root):
```bash
# Build the dependency index
uv run tldr --cache-root=git --index dep:requests \
  semantic index .venv/lib/python3.12/site-packages/requests --lang python

# Query it
uv run tldr --cache-root=git --index dep:requests \
  semantic search "HTTPAdapter.send implementation" \
  --path .venv/lib/python3.12/site-packages/requests
```

Repo index (same cache root, different index id):
```bash
# Build the repo index
uv run tldr --cache-root=git --index repo:llm-tldr \
  semantic index . --lang python

# Query it
uv run tldr --cache-root=git --index repo:llm-tldr \
  semantic search "daemon status handling" --path .
```

Where they live on disk:
```text
<repo>/.tldr/indexes/<hash-of-dep:requests>/
<repo>/.tldr/indexes/<hash-of-repo:llm-tldr>/
```

Can they both be searched? Yes, but one at a time in the current CLI. You switch scopes by changing `--index`.

## Index Management (List, Inspect, Delete)
Use these to keep a clean cache and verify what exists.

- List indexes:
```bash
tldr index list --cache-root=git
```
- Inspect one index:
```bash
tldr index info --cache-root=git dep:requests@2.31.0:site
```
- Remove an index:
```bash
tldr index rm --cache-root=git dep:requests@2.31.0:site
```

## Query Toolkit
Pick the smallest tool that answers the question.

- Quick file tree: `tldr tree <path>`
- Surface API structure: `tldr structure <path> --lang <lang>`
- Text search: `tldr search "pattern" <path>`
- Function context: `tldr context <func> --project <path>`
- Callers: `tldr impact <func> <path>`
- Program slice: `tldr slice <file> <func> <line>`
- Semantic search: `tldr semantic search "natural language query" --path <path>`

Always include `--cache-root=git` (and `--index` if used) so queries hit the intended index.

## Daemon Workflow
Use the daemon when running many queries in a session.

```bash
tldr daemon start --cache-root=git --project /path/to/repo
```
Notify the daemon on file changes to keep caches fresh:
```bash
tldr daemon notify /path/to/repo/src/auth.py --cache-root=git --project /path/to/repo
```

## Ignore Files And Scope
- Prefer `.tldrignore` for persistent exclusions.
- Use `--ignore` for ad-hoc exclusions and `--no-ignore` to override.

```bash
tldr --ignore "fixtures/" --ignore "*.generated.ts" tree .
```

## Notes And Gotchas
- `--index` requires `--cache-root`. Use `--cache-root=git` inside a git repo.
- When `--cache-root=git` is not available, pass a concrete path instead.
- Without `--index`, llm-tldr derives an index id from the scan root relative to the cache root.
- `.tldrignore` is index-scoped in index mode, so each isolated index can have its own ignore rules.

## References
Load these repo docs when you need full CLI detail or copy-ready examples:
- `README.md`
- `docs/TLDR.md`
