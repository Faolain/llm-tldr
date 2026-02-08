# TLDR: Code Analysis for AI Agents

[![PyPI](https://img.shields.io/pypi/v/llm-tldr)](https://pypi.org/project/llm-tldr/)
[![Python](https://img.shields.io/pypi/pyversions/llm-tldr)](https://pypi.org/project/llm-tldr/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

**Give LLMs exactly the code they need. Nothing more.**

> Fork note: this repo installs the CLI as `tldrf` (and `tldrf-mcp`) to avoid conflicts with other `tldr` tools.

```bash
# One-liner: Install, index, search
pip install llm-tldr && tldrf warm . && tldrf semantic "what you're looking for" .
```

Your codebase is 100K lines. Claude's context window is 200K tokens. Raw code won't fit—and even if it did, the LLM would drown in irrelevant details.

TLDR extracts *structure* instead of dumping *text*. The result: **95% fewer tokens** while preserving everything needed to understand and edit code correctly.

```bash
pip install llm-tldr
tldrf warm .                    # Index your project
tldrf context main --project .  # Get LLM-ready summary
```

---

## How It Works

TLDR builds 5 analysis layers, each answering different questions:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Program Dependence  → "What affects line 42?"      │
│ Layer 4: Data Flow           → "Where does this value go?"  │
│ Layer 3: Control Flow        → "How complex is this?"       │
│ Layer 2: Call Graph          → "Who calls this function?"   │
│ Layer 1: AST                 → "What functions exist?"      │
└─────────────────────────────────────────────────────────────┘
```

**Why layers?** Different tasks need different depth:
- Browsing code? Layer 1 (structure) is enough
- Refactoring? Layer 2 (call graph) shows what breaks
- Debugging null? Layer 5 (slice) shows only relevant lines

The daemon keeps indexes in memory for **100ms queries** instead of 30-second CLI spawns.

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         YOUR CODE                                │
│  src/*.py, lib/*.ts, pkg/*.go                                    │
└───────────────────────────┬──────────────────────────────────────┘
                            │ tree-sitter
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     5-LAYER ANALYSIS                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│  │   AST   │→│  Calls  │→│   CFG   │→│   DFG   │→│   PDG   │     │
│  │   L1    │ │   L2    │ │   L3    │ │   L4    │ │   L5    │     │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
└───────────────────────────┬──────────────────────────────────────┘
                            │ bge-large-en-v1.5
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    SEMANTIC INDEX                                │
│  1024-dim embeddings in FAISS  →  "find JWT validation"          │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                       DAEMON                                     │
│  In-memory indexes  •  100ms queries  •  Auto-lifecycle          │
└──────────────────────────────────────────────────────────────────┘
```

### The Semantic Layer: Search by Behavior

The real power comes from combining all 5 layers into **searchable embeddings**.

Every function gets indexed with:
- Signature + docstring (L1)
- What it calls + who calls it (L2)
- Complexity metrics (L3)
- Data flow patterns (L4)
- Dependencies (L5)
- First ~10 lines of actual code

This gets encoded into **1024-dimensional vectors** using `bge-large-en-v1.5`. The result: search by *what code does*, not just what it says.

```bash
# "validate JWT" finds verify_access_token() even without that exact text
tldrf semantic "validate JWT tokens and check expiration" .
```

**Why this works:** Traditional search finds `authentication` in variable names and comments. Semantic search understands that `verify_access_token()` *performs* JWT validation because the call graph and data flow reveal its purpose.

### Setting Up Semantic Search

```bash
# Build the semantic index (one-time, ~2 min for typical project)
tldrf warm /path/to/project

# Search by behavior
tldrf semantic "database connection pooling" .
```

Embedding dependencies (`sentence-transformers`, `faiss-cpu`) are included with `pip install llm-tldr`. The semantic index is cached under your `cache_root` in `.tldr/` (see below).

### Where TLDR Stores Things

TLDR stores two different kinds of artifacts:

- **Index artifacts (on disk):** analysis caches, embeddings, and per-index status files live under `CACHE_ROOT/.tldr/`. Typical local usage (`tldrf warm .`) creates `./.tldr/indexes/<index_key>/...`. Legacy mode (no `--cache-root/--index`) uses `./.tldr/cache/...`.
- **Daemon runtime artifacts (on disk):** the daemon's socket/pid/lock/port files live under `TLDR_DAEMON_DIR`. Default is `/tmp/tldr` on macOS/Linux and the system temp dir on Windows. Example files: `tldr-<hash>.sock`, `tldr-<hash>.pid`, `tldr-<hash>.lock`.

If you want daemon runtime artifacts somewhere else (for example, to avoid a shared `/tmp`), set `TLDR_DAEMON_DIR` to a **short** path. Unix domain sockets have path-length limits, so very long repo-local paths can fail.

### Keeping the Index Fresh

The daemon tracks dirty files and auto-rebuilds after 20 changes, but you need to notify it when files change:

```bash
# Notify daemon of a changed file
tldrf daemon notify src/auth.py --project .
```

**Integration options:**

1. **Git hook** (post-commit):
   ```bash
   git diff --name-only HEAD~1 | xargs -I{} tldrf daemon notify {} --project .
   ```

2. **Editor hook** (on save):
   ```bash
   tldrf daemon notify "$FILE" --project .
   ```

3. **Manual rebuild** (when needed):
   ```bash
   tldrf warm .  # Full rebuild
   ```

The daemon auto-rebuilds semantic embeddings in the background once the dirty threshold (default: 20 files) is reached.

---

## The Workflow

### Before Reading Code
```bash
tldrf tree src/                      # See file structure
tldrf structure src/ --lang python   # See functions/classes
```

### Before Editing
```bash
tldrf extract src/auth.py            # Full file analysis
tldrf context login --project .      # LLM-ready summary (95% savings)
```

### Before Refactoring
```bash
tldrf impact login .                 # Who calls this? (reverse call graph)
tldrf change-impact                  # Which tests need to run?
```

### Debugging
```bash
tldrf slice src/auth.py login 42     # What affects line 42?
tldrf dfg src/auth.py login          # Trace data flow
```

### Finding Code by Behavior
```bash
tldrf semantic "validate JWT tokens" .   # Natural language search
```

---

## Quick Setup

### 1. Install

```bash
pip install llm-tldr
```

### 2. Index Your Project

```bash
tldrf warm /path/to/project
```

This builds all analysis layers and starts the daemon. Takes 30-60 seconds for a typical project, then queries are instant.

### 3. Start Using

```bash
tldrf context main --project .   # Get context for a function
tldrf impact helper_func .       # See who calls it
tldrf semantic "error handling"  # Find by behavior
```

---

## Agent Skills (Included)

This repo includes a dependency-indexing skill at `skills/llm-tldr-dep-indexer/`.

It supports:
- Python dependencies from `.venv` / `site-packages`
- Node dependencies from `node_modules`

What it lets an agent do:
- Resolve the exact installed dependency source + version
- Create or reuse a **versioned, isolated** index for that dependency
- Keep caches under the repo root via `--cache-root=git`
- Query the dependency directly without mixing it with the main repo

Quick example (Python):
```bash
python skills/llm-tldr-dep-indexer/scripts/ensure_dep_index.py python requests
```

---

This repo also includes a usage skill at `skills/llm-tldr-usage/`.

It supports:
- Indexing and querying repos with `tldrf warm`, `tldrf semantic`, `tldrf context`, `tldrf slice`, and `tldrf impact`
- Isolated indexes with `--cache-root=git` and `--index`
- Index management via `tldrf index list/info/rm`
- Daemon workflows and `.tldrignore` usage

Quick example:
```bash
tldrf warm --cache-root=git .
tldrf semantic index --cache-root=git .
tldrf semantic search "token validation flow" --cache-root=git --path .
```

---

## Real Example: Why This Matters

**Scenario:** Debug why `user` is null on line 42.

**Without TLDR:**
1. Read the 150-line function
2. Trace every variable manually
3. Miss the bug because it's hidden in control flow

**With TLDR:**
```bash
tldrf slice src/auth.py login 42
```

**Output:** Only 6 lines that affect line 42:
```python
3:   user = db.get_user(username)
7:   if user is None:
12:      raise NotFound
28:  token = create_token(user)  # ← BUG: skipped null check
35:  session.token = token
42:  return session
```

The bug is obvious. Line 28 uses `user` without going through the null check path.

---

## Command Reference

### Exploration
| Command | What It Does |
|---------|--------------|
| `tldrf tree [path]` | File tree |
| `tldrf structure [path] --lang <lang>` | Functions, classes, methods |
| `tldrf search <pattern> [path]` | Text pattern search |
| `tldrf extract <file>` | Full file analysis |

### Analysis
| Command | What It Does |
|---------|--------------|
| `tldrf context <func> --project <path>` | LLM-ready summary (95% savings) |
| `tldrf cfg <file> <function>` | Control flow graph |
| `tldrf dfg <file> <function>` | Data flow graph |
| `tldrf slice <file> <func> <line>` | Program slice |

### Cross-File
| Command | What It Does |
|---------|--------------|
| `tldrf calls [path]` | Build call graph |
| `tldrf impact <func> [path]` | Find all callers (reverse call graph) |
| `tldrf dead [path]` | Find unreachable code |
| `tldrf arch [path]` | Detect architecture layers |
| `tldrf imports <file>` | Parse imports |
| `tldrf importers <module> [path]` | Find files that import a module |

### Semantic
| Command | What It Does |
|---------|--------------|
| `tldrf warm <path>` | Build all indexes (including embeddings) |
| `tldrf semantic <query> [path]` | Natural language code search |

### Diagnostics
| Command | What It Does |
|---------|--------------|
| `tldrf diagnostics <file>` | Type check + lint |
| `tldrf change-impact [files]` | Find tests affected by changes |
| `tldrf doctor` | Check/install diagnostic tools |

### Daemon
| Command | What It Does |
|---------|--------------|
| `tldrf daemon start` | Start background daemon |
| `tldrf daemon stop` | Stop daemon |
| `tldrf daemon status` | Check status |

---

## Supported Languages

Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP, C#, Kotlin, Scala, Swift, Lua, Elixir

Language is auto-detected or specify with `--lang`.

---

## MCP Integration

For AI tools (Claude Desktop, Claude Code):

**Claude Desktop** - Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "tldrf": {
      "command": "tldrf-mcp",
      "args": ["--project", "/path/to/your/project"]
    }
  }
}
```

**Claude Code** - Add to `.claude/settings.json`:
```json
{
  "mcpServers": {
    "tldrf": {
      "command": "tldrf-mcp",
      "args": ["--project", "."]
    }
  }
}
```

---

## Configuration

### `.tldrignore` - Exclude Files

TLDR respects `.tldrignore` (gitignore syntax) for all commands including `tree`, `structure`, `search`, `calls`, and semantic indexing:

```bash
# Auto-create with sensible defaults
tldrf warm .  # Creates .tldrignore if missing
```

**Default exclusions:**
- `node_modules/`, `.venv/`, `__pycache__/`
- `dist/`, `build/`, `*.egg-info/`
- Binary files (`*.so`, `*.dll`, `*.whl`)
- Security files (`.env`, `*.pem`, `*.key`)

**Customize** by editing `.tldrignore`:
```gitignore
# Add your patterns
large_test_fixtures/
vendor/
data/*.csv
```

**CLI Flags:**
```bash
# Add patterns from command line (can be repeated)
tldrf --ignore "packages/old/" --ignore "*.generated.ts" tree .

# Bypass all ignore patterns
tldrf --no-ignore tree .
```

### Settings - Daemon Behavior

Create `.tldr/config.json` for daemon settings:

```json
{
  "semantic": {
    "enabled": true,
    "auto_reindex_threshold": 20
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable semantic search |
| `auto_reindex_threshold` | `20` | Files changed before auto-rebuild |

### Monorepo Support

For monorepos, create `.claude/workspace.json` to scope indexing:

```json
{
  "active_packages": ["packages/core", "packages/api"],
  "exclude_patterns": ["**/fixtures/**"]
}
```

---

## Performance

| Metric | Raw Code | TLDR | Improvement |
|--------|----------|------|-------------|
| Tokens for function context | 21,000 | 175 | **99% savings** |
| Tokens for codebase overview | 104,000 | 12,000 | **89% savings** |
| Query latency (daemon) | 30s | 100ms | **300x faster** |

---

## Deep Dive

For the full architecture explanation, benchmarks, and advanced workflows:

**[Full Documentation](./docs/TLDR.md)**

---

## License

AGPL-3.0 - See LICENSE file.
