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

Avoid this tool when:
- Working on small, self-contained files
- Writing new code without dependencies

## Installation and Setup

```bash
pip install llm-tldr
cd /path/to/project

tldr warm --cache-root=git .        # Build structural caches

tldr semantic index --cache-root=git .  # Build semantic embeddings
```

## Essential Commands

### Structure Overview
```bash
tldr tree src/                       # File structure
tldr structure src/ --lang python    # Functions/classes overview
```

### Function Context
```bash
tldr context <function> --project .  # Function summary with dependencies
tldr extract <file>                  # Complete file analysis
```

### Impact Analysis
```bash
tldr impact <function> .             # Who calls this? (breaks if changed)
tldr calls .                         # Build full call graph
tldr arch .                          # Detect architecture layers
tldr dead .                          # Find unreachable code
```

### Finding Code by Intent
```bash
tldr semantic search "validate auth tokens" --path .    # Natural language search
tldr semantic search "error retry logic" --path .       # Finds behavior, not keywords
```

### Debugging
```bash
tldr slice <file> <func> <line>      # What affects this line?
tldr dfg <file> <function>           # Trace data flow
tldr cfg <file> <function>           # Control flow graph
tldr diagnostics <file>              # Type check + lint
```

## Index Isolation and Dependency Indexes
Use `--cache-root=git` inside a repo to store all indexes under the repo-root `.tldr/` regardless of where you run the command. Add `--index <id>` to isolate a dependency or alternate corpus.

### Dependency index (stored under repo root)
```bash
# Build the dependency index
tldr --cache-root=git --index dep:requests \
  semantic index .venv/lib/python3.12/site-packages/requests --lang python

# Query it
tldr --cache-root=git --index dep:requests \
  semantic search "HTTPAdapter.send implementation" \
  --path .venv/lib/python3.12/site-packages/requests
```

### Repo index (same cache root, different index id)
```bash
# Build the repo index
tldr --cache-root=git --index repo:llm-tldr \
  semantic index . --lang python

# Query it
tldr --cache-root=git --index repo:llm-tldr \
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
Use the returned `index_id` and `scan_root` for subsequent `tldr` commands.

## Index Management
Use these to keep a clean cache and verify what exists.

```bash
tldr index list --cache-root=git
tldr index info --cache-root=git dep:requests@2.31.0:site
tldr index rm --cache-root=git dep:requests@2.31.0:site
```

## Default Agent Workflow

1. Get structure: `tldr tree` / `tldr structure`
2. Narrow context: `tldr context <function>`
3. Validate impact before changes: `tldr impact <function>`

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
    "tldr": {
      "command": "tldr-mcp",
      "args": ["--project", "."]
    }
  }
}
```

**For Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tldr": {
      "command": "tldr-mcp",
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
tldr daemon status
tldr daemon stop && tldr daemon start
```

**Index out of date?**
```bash
tldr warm --cache-root=git .  # Full rebuild
```

**Semantic search not finding relevant code?**
```bash
tldr semantic index --cache-root=git .
```

**Not sure which indexes exist?**
```bash
tldr index list --cache-root=git
```

## Principle

Rule of thumb: If raw code does not change your decision, do not include it in context.
