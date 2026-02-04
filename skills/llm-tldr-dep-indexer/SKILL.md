---
name: llm-tldr-dep-indexer
description: Resolve exact installed dependency source and build isolated llm-tldr indexes. Prefer installed artifacts and only fetch repo/registry source when needed.
---

# llm-tldr Dependency Indexer

Create accurate, isolated llm-tldr indexes for dependencies. This skill is **llm-tldr only**. It prefers installed artifacts and escalates to registry or VCS source only when the installed package is insufficient.

## Quick Start

Python:
```bash
uv run python /Users/aristotle/.agents/skills/llm-tldr-dep-indexer/scripts/resolve_python_dep.py torch
```

Node:
```bash
node /Users/aristotle/.agents/skills/llm-tldr-dep-indexer/scripts/resolve_node_dep.js lodash
```

Use the JSON output to choose a source path, then index:
```bash
tldr semantic index <SOURCE_PATH> \
  --cache-root=git \
  --index dep_<name>_<site|repo> \
  --lang <python|javascript|all> \
  --no-gitignore
```

Note: `--cache-root=git` resolves the repo root automatically and keeps a single `.tldr` cache per repo. Use a custom path if you want caches outside the repo.

### Index Helper (refresh-aware)

Use the helper to resolve the installed dependency, version it, and create/reuse the correct index automatically.

Python:
```bash
python /Users/aristotle/.agents/skills/llm-tldr-dep-indexer/scripts/ensure_dep_index.py \
  python requests
```

Node:
```bash
python /Users/aristotle/.agents/skills/llm-tldr-dep-indexer/scripts/ensure_dep_index.py \
  node lodash
```

The helper:
- Builds an index id like `dep:<name>@<version>:site` by default.
- Uses `--cache-root=git` unless you override it.
- Creates a new index when the version changes (or rebuilds if `--rebuild` is set).

## Decision Rules (Default Behavior)

1. **Local/editable install**
   - If origin is local path or editable, index that local path.
2. **Registry install**
   - Index the installed artifact first (site-packages or node_modules).
3. **Escalate only when needed**
   - If installed artifact is opaque (native binaries, minified bundles, missing sources), fetch exact source from registry or VCS and index that too.
4. **VCS install**
   - If origin is VCS, clone the exact commit/tag and index it.

## What Counts as “Opaque”

Python (site-packages):
- Many `.so/.pyd/.dylib` files and few `.py` sources.
- You only see thin wrappers around compiled cores.

JavaScript (node_modules):
- Only `dist/` output and no `src/`, or heavy bundling/minification.
- `.node` or `.wasm` present (native/WASM internals).

## Exact Source Pulling

Python (registry sdist, exact version):
```bash
uv pip download --no-binary :all: <pkg>==<ver>
# extract the tar.gz, then index the extracted folder
```

Python (VCS):
```bash
git clone <repo> <dest>
cd <dest>
git checkout <tag-or-commit>
```

Node (registry tarball, exact version):
```bash
npm pack <pkg>@<ver>
# extract the .tgz, then index the extracted folder
```

Node (VCS):
```bash
git clone <repo> <dest>
cd <dest>
git checkout <tag-or-commit>
```

## Index Naming Convention

Use consistent names for clarity:
- `dep_<name>_site` for installed artifacts
- `dep_<name>_repo` for source repos

## Cache Root Guidance

Prefer `--cache-root=git` when running inside a repo to avoid scattering `.tldr` caches across subfolders. If you want caches outside the repo, set an explicit path (or `TLDR_CACHE_ROOT`) instead.

## Outputs to Provide to User

Always return:
- Source path used for indexing
- Version and origin (local/registry/vcs)
- Index id(s) created
- A scoped search command example

## Scripts

- `scripts/resolve_python_dep.py <dist_name> [--module <import_name>]`
  - Emits JSON with version, install path, origin, and native extension indicators.
- `scripts/resolve_node_dep.js <pkg> [--from <dir>]`
  - Emits JSON with version, install path, and native/WASM indicators.
- `scripts/ensure_dep_index.py <python|node> <name>`
  - Resolves, versions, and (re)indexes dependencies using `--cache-root=git` by default.

Use these scripts for deterministic resolution and provenance capture.
