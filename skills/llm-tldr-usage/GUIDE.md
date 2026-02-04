# tldrf Usage Guide

## Where tldrf Excels

### 1. Impact Analysis (tldrf's killer feature)
```bash
tldrf impact <function> .
```
**Use case:** "What breaks if I change this function?"
- Returns all callers instantly (0.2s from cached index)
- Shows the full caller chain (callers of callers)
- grep can't do this - it would require manual tracing

### 2. Architectural Understanding
```bash
tldrf arch .
```
**Use case:** "Where should I put this new module?"

Analyzes your codebase's call graph structure to identify architectural layers:

| Layer | Meaning | Examples |
|-------|---------|----------|
| **Entry** | Functions that call others but aren't called | CLI entry points, test functions, main() |
| **Leaf** | Functions that are called but don't call anything | Utility functions, validators, formatters |
| **Middle** | Functions that both call and are called | Business logic, service layer |
| **Circular** | Files/modules that import each other | Potential design smell |

**When it's useful:**

1. **Understanding unfamiliar codebases**
   - "Where are the entry points?" → Entry layer
   - "What are the low-level utilities?" → Leaf layer
   - "What's the core business logic?" → Middle layer

2. **Deciding where to put new code**
   - New CLI command → belongs with entry layer
   - New utility function → belongs with leaf layer
   - New service → belongs with middle layer

3. **Identifying design problems**
   - Too many circular dependencies → tangled architecture
   - Entry layer calling leaf directly (skipping middle) → possible layering violation
   - Middle layer too large → might need decomposition

4. **Refactoring guidance**
   - High middle-layer count → complex interdependencies
   - Functions in wrong layer → candidates for moving

**Example output:**
```json
{
  "summary": {
    "entry_count": 303,   // CLI commands, tests, etc.
    "leaf_count": 244,    // Utilities, helpers
    "middle_count": 245,  // Core logic
    "circular_count": 2   // Potential issues
  }
}
```

### 3. Function Context Without Reading Full Files
```bash
tldrf context <function> --project .
```
**Use case:** "What does this function do and what does it depend on?"
- Signature + docstring + complexity metrics
- Key callees (what it calls)
- Saves reading a 1300-line file for one function

### 4. Semantic Search by Concept
```bash
tldrf semantic search "validate authentication tokens" --path .
```
**Use case:** "Find code that does X" (not "find string X")
- Finds `verify_access_token()` even if it doesn't contain "validate" or "authentication"
- Searches by behavior, not keywords

### 5. Program Slicing
```bash
tldrf slice <file> <function> <line>
```
**Use case:** "What code affects line 42?"
- Returns only the lines that influence that specific line
- Cuts through noise in complex functions

### 6. Dead Code Detection
```bash
tldrf dead .
```
**Use case:** "What can I safely delete?"
- Finds unreachable functions
- Useful after refactoring

---

## How to Invoke tldrf Effectively

### Always Use `--cache-root=git` for Repos
```bash
tldrf warm --cache-root=git .                # Build index once
tldrf impact build_index . --cache-root=git  # Query instantly
```
Without this, commands may rebuild indexes or fail to find cached data.

### The Optimal Workflow Pattern

```bash
# 1. FIRST: Understand blast radius (tldrf - instant from cache)
tldrf impact <function_to_change> .
tldrf arch .

# 2. THEN: Find every string reference (grep - exhaustive)
grep -rn "string_to_find" --include="*.py" --include="*.md"

# 3. FINALLY: Verify cleanup (tldrf)
tldrf dead .
```

### When to Use Each Command

| Task | Command | Time |
|------|---------|------|
| Who calls this function? | `tldrf impact <func> .` | 0.2s |
| What does this function call? | `tldrf context <func> --project .` | 0.3s |
| Where should new code go? | `tldrf arch .` | 0.5s |
| Find code by concept | `tldrf semantic search "concept" --path .` | 2-8s |
| What affects this line? | `tldrf slice <file> <func> <line>` | 0.5s |
| What's unreachable? | `tldrf dead .` | 1-2s |

---

## When NOT to Use tldrf

| Task | Why Not | Use Instead |
|------|---------|-------------|
| Find every occurrence of "faiss" | Text search, not code analysis | `grep -rn "faiss" .` |
| Search markdown/docs | tldrf only indexes code | `grep --include="*.md"` |
| Search pyproject.toml | tldrf doesn't index config | `grep` or `Read` |
| Get exact line numbers | tldrf returns function-level | `grep -n` |
| Find module-level code | tldrf indexes functions/classes | `grep` or `Read` |

---

## Key Insight

**tldrf answers "what is connected to what" questions instantly.**

**grep answers "where does this string appear" questions exhaustively.**

Use both. Start with tldrf to understand impact, then grep for implementation details.

---

## Quick Reference

### Build Index (do this once per repo)
```bash
tldrf warm --cache-root=git .
tldrf semantic index --cache-root=git .
```

### Common Queries
```bash
# Impact analysis
tldrf impact <function> .

# Function summary
tldrf context <function> --project .

# Architecture layers
tldrf arch .

# Semantic search
tldrf semantic search "what you're looking for" --path .

# Program slice
tldrf slice <file> <function> <line>

# Dead code
tldrf dead .

# Call graph
tldrf calls .
```

### Migration/Refactoring Workflow
```bash
# Step 1: Understand impact (tldrf)
tldrf impact <old_function> .
tldrf arch .

# Step 2: Find ALL references (grep)
grep -rn "old_dependency" --include="*.py" --include="*.toml" --include="*.md"

# Step 3: Make changes...

# Step 4: Verify cleanup (tldrf)
tldrf dead .
```
