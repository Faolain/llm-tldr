---
name: llm-tldr-usage
description: Use when Codex needs to run llm-tldr CLI to index, search, or analyze codebases. Covers semantic search, impact/slice workflows, daemon usage, .tldrignore handling, and index isolation with --cache-root and --index (including dependency indexes).
---

# SKILL: Codebase Context Extraction (llm-tldr)

## Purpose
Use this tool to obtain minimal, high-signal context required for reasoning about an existing codebase without loading full source files. Prefer index mode (`--cache-root=git`) so all caches live under the repo root and dependency indexes stay isolated.

## When to Use
Use this tool when:
- Context size is a limiting factor
- The codebase is unfamiliar
- You need callers, data flow, or change impact
- You want to understand "what breaks if I change this"
- You need architectural guidance (where to place new code)
- You're searching for code by **concept** rather than exact text

## When NOT to Use (Use grep/Glob Instead)
**tldrf operates at function/class granularity on code files only.** Use traditional tools for:

| Task | Use This Instead |
|------|------------------|
| Find every occurrence of a string (e.g., "faiss") | `grep -rn "faiss" .` |
| Search documentation/markdown files | `grep -r "pattern" --include="*.md"` |
| Search config files (pyproject.toml, package.json) | `grep "pattern" pyproject.toml` |
| Find module-level code (not in a function) | `grep` or `Read` the file |
| Get exact line numbers for implementation | `grep -n` or `Read` |
| Find test patterns (e.g., `pytest.importorskip`) | `grep -r "importorskip" tests/` |

**Rule of thumb:** If you need to find "every place string X appears", use grep. If you need to understand "what code is affected by changing function Y", use tldrf.

## Installation and Setup

```bash
# Fork install (no 'tldr' command conflicts)
uv tool install -e /path/to/llm-tldr
uv tool update-shell

cd /path/to/project

tldrf warm --cache-root=git .        # Build structural caches

tldrf semantic index --cache-root=git .  # Build semantic embeddings
```

## Essential Commands

### Structure Overview
```bash
tldrf tree src/                       # File structure
tldrf structure src/ --lang python    # Functions/classes overview
```

### Function Context
```bash
tldrf context <function> --project .  # Function summary with dependencies
tldrf extract <file>                  # Complete file analysis
```

### Impact Analysis
```bash
tldrf impact <function> .             # Who calls this? (breaks if changed)
tldrf calls .                         # Build full call graph
tldrf arch .                          # Detect architecture layers
tldrf dead .                          # Find unreachable code
```

### Finding Code by Intent
```bash
tldrf semantic search "validate auth tokens" --path .    # Natural language search
tldrf semantic search "error retry logic" --path .       # Finds behavior, not keywords
```

### Debugging
```bash
tldrf slice <file> <func> <line>      # What affects this line?
tldrf dfg <file> <function>           # Trace data flow
tldrf cfg <file> <function>           # Control flow graph
tldrf diagnostics <file>              # Type check + lint
```

## Index Isolation and Dependency Indexes
Use `--cache-root=git` inside a repo to store all indexes under the repo-root `.tldr/` regardless of where you run the command. Add `--index <id>` to isolate a dependency or alternate corpus.

### Dependency index (stored under repo root)
```bash
# Build the dependency index
tldrf --cache-root=git --index dep:requests \
  semantic index .venv/lib/python3.12/site-packages/requests --lang python

# Query it
tldrf --cache-root=git --index dep:requests \
  semantic search "HTTPAdapter.send implementation" \
  --path .venv/lib/python3.12/site-packages/requests
```

### Repo index (same cache root, different index id)
```bash
# Build the repo index
tldrf --cache-root=git --index repo:llm-tldr \
  semantic index . --lang python

# Query it
tldrf --cache-root=git --index repo:llm-tldr \
  semantic search "daemon status handling" --path .
```

### Where they live on disk
```text
<repo>/.tldr/indexes/<hash-of-dep:requests>/
<repo>/.tldr/indexes/<hash-of-repo:llm-tldr>/
```

Can they both be searched? Yes, but one at a time in the current CLI. Switch scopes by changing `--index`.

### Dependency index helper (preferred)
Use the included dependency indexing skill to resolve the correct installed version and create or reuse a versioned index:
```bash
python skills/llm-tldr-dep-indexer/scripts/ensure_dep_index.py python requests
```
Use the returned `index_id` and `scan_root` for subsequent `tldrf` commands.

## Index Management
Use these to keep a clean cache and verify what exists.

```bash
tldrf index list --cache-root=git
tldrf index info --cache-root=git dep:requests@2.31.0:site
tldrf index rm --cache-root=git dep:requests@2.31.0:site
```

## Default Agent Workflow

1. Get structure: `tldrf tree` / `tldrf structure`
2. Narrow context: `tldrf context <function>`
3. Validate impact before changes: `tldrf impact <function>`

## Workflow Templates

### Refactoring a module
```bash
tldrf impact <target_function> .           # Find all callers
tldrf context <target_function> --project . # Understand the function
tldrf slice <file> <func> <line>           # Trace specific dependencies
tldrf dead .                               # Verify no orphaned code after refactor
```

### Understanding unfamiliar code
```bash
tldrf arch .                               # See high-level layers
tldrf structure src/ --lang python         # List functions/classes
tldrf semantic search "what does X do" --path .  # Find by intent
```

### Planning a migration (e.g., swapping a dependency)
**Use tldrf for impact analysis, grep for exhaustive touchpoint discovery:**
```bash
# Step 1: Understand impact (tldrf)
tldrf impact <old_api_function> .          # Who calls this function?
tldrf arch .                               # Where should new code live?
tldrf context <key_type> --project .       # Understand data structures

# Step 2: Find ALL references (grep) - tldrf only finds function calls
grep -rn "old_dependency" --include="*.py" --include="*.toml" --include="*.md"

# Step 3: Verify cleanup
tldrf dead .                               # Find unused code to remove
```

**Why both tools?** tldrf finds function-level dependencies (callers/callees). grep finds every string occurrence including imports, config files, documentation, and module-level code that tldrf doesn't index.

### Debugging a specific line
```bash
tldrf slice <file> <func> <line>           # What affects this line?
tldrf dfg <file> <function>                # Trace data flow through function
tldrf cfg <file> <function>                # See control flow paths
```

## tldrf vs grep: Complementary Tools

tldrf and grep serve different purposes. Using the wrong tool wastes time.

| Question | Tool | Why |
|----------|------|-----|
| "Who calls `build_index()`?" | `tldrf impact build_index .` | Call graph analysis |
| "Where is the string 'faiss' used?" | `grep -rn "faiss" .` | Text search |
| "What happens if I change this function's signature?" | `tldrf impact` | Finds all callers |
| "What imports this module?" | `grep "import faiss"` or `tldrf importers` | Either works |
| "Where should I put a new utility module?" | `tldrf arch .` | Architecture layers |
| "What does this function do?" | `tldrf context <func>` | Summarizes with deps |
| "Find code that handles authentication" | `tldrf semantic search "auth"` | Concept search |
| "Find the exact line with `IndexFlatIP`" | `grep -n "IndexFlatIP"` | Exact text match |
| "What's in pyproject.toml?" | `grep` or `Read` | tldrf doesn't index config |
| "What tests cover this function?" | `tldrf impact` shows test callers | Reverse call graph |

### Performance Comparison

| Task Type | tldrf | grep |
|-----------|-------|------|
| Find all callers of a function | 0.2s (cached index) | 30s+ (manual analysis) |
| Find every string occurrence | N/A (wrong tool) | 0.5s |
| Understand function dependencies | 0.3s | Minutes of reading |
| Search documentation files | N/A (doesn't index .md) | 0.2s |

**Best practice:** Start with `tldrf impact` to understand what's affected, then use `grep` to find every reference for implementation.

## How It Works

llm-tldr builds 5 analysis layers:

1. AST (L1): Structure - functions/classes
2. Call Graph (L2): Dependencies - who calls what
3. Control Flow (L3): Logic paths - complexity metrics
4. Data Flow (L4): Value tracking - where data goes
5. Program Dependence (L5): Line-level impact - minimal slices

The semantic layer combines all 5 layers into searchable embeddings, enabling natural language search by what code does rather than what it says.

## Configuration

### Exclude Files (.tldrignore)

Create `.tldrignore` in project root (gitignore syntax):

```gitignore
node_modules/
.venv/
__pycache__/
dist/
build/
*.egg-info/
```

In index mode, `.tldrignore` is index-scoped, so each isolated index can have its own ignore rules.

### Daemon Settings (.tldr/config.json)

```json
{
  "semantic": {
    "enabled": true,
    "auto_reindex_threshold": 20
  }
}
```

### Monorepo Support

For monorepos, create `.claude/workspace.json`:
```json
{
  "active_packages": ["packages/core", "packages/api"],
  "exclude_patterns": ["**/fixtures/**"]
}
```

## MCP Integration

**For Claude Code** (`.mcp.json` in project root, or `~/.claude.json` for user-scope):
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

**For Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tldrf": {
      "command": "tldrf-mcp",
      "args": ["--project", "/absolute/path/to/project"]
    }
  }
}
```

## Language Support

Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP, C#, Kotlin, Scala, Swift, Lua, Elixir (16 languages total). Language auto-detected or specify with `--lang`.

## Troubleshooting

**Daemon not responding?**
```bash
tldrf daemon status
tldrf daemon stop && tldrf daemon start
```

**Index out of date?**
```bash
tldrf warm --cache-root=git .  # Full rebuild
```

**Semantic search not finding relevant code?**
```bash
tldrf semantic index --cache-root=git .
```

**Not sure which indexes exist?**
```bash
tldrf index list --cache-root=git
```

## Principle

Rule of thumb: If raw code does not change your decision, do not include it in context.
