# Implementation Plan: 002 - Swap FAISS for USearch (Vector Store Backend)

Spec: `specs/002-feat-swap-faiss-to-usearch.md`

## Summary

Replace the hard-wired FAISS dependency in `tldr/semantic.py` with a pluggable vector store abstraction, defaulting to USearch. This improves:

- **Cross-platform installation** (FAISS is a common pain point)
- **Flexibility** to swap ANN backends without touching core semantic logic
- **Performance** via USearch's memory-mapped serving for daemon/MCP workloads

## Immediate Next Steps (rolling)

- [ ] Run codebase analysis to map all FAISS touchpoints (see "Codebase Analysis with tldrf" below)
- [ ] Create `tldr/vector_store/` package with `base.py` protocol
- [ ] Implement `tldr/vector_store/usearch_backend.py`
- [ ] Refactor `tldr/semantic.py` to use vector store abstraction
- [ ] Add `--vector-backend` CLI flag and `TLDR_VECTOR_BACKEND` env var
- [ ] Update `IndexPaths` to support both `.faiss` and `.usearch` files
- [ ] Extend `metadata.json` schema with `vector_store` block
- [ ] Add/update tests for USearch backend
- [ ] Update documentation to remove FAISS-specific references
- [ ] Run full test suite: `uv run pytest`
- [ ] Run lint: `uv run ruff check tldr/`

## Codebase Analysis with tldrf

`tldrf` shines at understanding dependencies and impact before you change code. Here's how I'd use it:

### 1. Impact Analysis — "What breaks if I change this?"

```bash
# Who calls the functions that use FAISS?
tldrf impact build_semantic_index .
tldrf impact semantic_search .
```

This tells you every caller that would be affected when you refactor `build_semantic_index` to use the new vector store abstraction.

### 2. Call Graph — "What does this function depend on?"

```bash
# Build the full call graph, then grep for semantic.py edges
tldrf calls . | grep "semantic.py"

# Or use context to see what a specific function calls
tldrf context build_semantic_index --project .
tldrf context semantic_search --project .
```

Shows the dependency graph — helps identify all the FAISS-touching code paths.

> **Note:** `tldrf calls` does not have a `--filter` option. Use grep on the output or `tldrf context` for specific functions.

### 3. Context — "Give me just what I need to understand this function"

```bash
# Get focused context on the key functions
tldrf context build_semantic_index --project .
tldrf context semantic_search --project .
```

Returns a minimal summary with signature, docstring, and key dependencies — without loading the entire 1300+ line file.

### 4. Slice — "What affects this specific line?"

```bash
# Line 1246 is where faiss.IndexFlatIP is created
tldrf slice tldr/semantic.py build_semantic_index 1246

# Line 1349 is where faiss.read_index is called
tldrf slice tldr/semantic.py semantic_search 1349
```

Shows exactly which variables and control flow affect those FAISS calls — useful for understanding what state needs to flow into the new abstraction.

### 5. Data Flow — "Where does this value go?"

```bash
tldrf dfg tldr/semantic.py build_semantic_index
```

Traces how the embeddings matrix flows through the function — helps ensure the new `VectorIndex.build()` interface captures all inputs.

### 6. Architecture — "What layers exist?"

```bash
tldrf arch .
```

Shows module-level dependencies — helps you decide where to place `tldr/vector_store/` in the architecture.

---

### Practical Workflow for the Spec

| Task from Spec                           | tldrf Command                                 |
|------------------------------------------|-----------------------------------------------|
| Identify all FAISS touchpoints           | `tldrf impact build_semantic_index .`         |
| Understand `IndexPaths` usage            | `tldrf context IndexPaths --project .`        |
| Find all callers of `semantic_search`    | `tldrf impact semantic_search .`              |
| Check what tests cover semantic indexing | `tldrf impact test_semantic_indexes_isolated .` |
| Validate no dead code after refactor     | `tldrf dead .`                                |

### Bugs Found and Fixed in tldrf During Analysis

While testing the tldrf workflow above, we discovered and fixed bugs in the CLI:

**Bug 1: `impact` command bypassed the cache**
- Problem: Used `analyze_impact()` which rebuilt the call graph fresh with a single language, ignoring the cached multi-language graph
- Fix: Changed to use `_get_or_build_graph()` + `impact_analysis()` like the `calls` command (commit `13168ad`)

**Bug 2: No auto-detection of index-mode cache**
- Problem: Commands failed without `--cache-root=git` even when index was built with it
- Fix: Added fallback to find most recent index in `.tldr/indexes/*/cache/call_graph.json`

**Root cause identified (not yet fixed):** `_detect_project_languages()` returns alphabetically sorted languages, so JavaScript (1 file) came before Python (63 files), causing single-language rebuilds to miss Python code.

**Verification Results (after fix):**

| Command | Before Fix | After Fix | Status |
|---------|------------|-----------|--------|
| `tldrf calls .` | 0 edges | **1150 edges** | ✅ Fixed |
| `tldrf impact build_semantic_index .` | "Function not found" | **5 callers** | ✅ Fixed |
| `tldrf impact semantic_search .` | Broken | **4 callers** | ✅ Fixed |
| `tldrf structure tldr/semantic.py` | `files: []` | `files: []` | ⚠️ Expected (use directory) |
| `tldrf context build_semantic_index --project .` | Working | **Working** | ✅ Was OK |

## Gotchas / Learnings (rolling)

- USearch returns **distance** (lower is better); TLDR expects **score** (higher is better). Conversion: `score = 1 - distance` for cosine metric.
- USearch metric must be set explicitly at build time (`metric='cos'`); don't rely on upstream defaults.
- Memory-mapped serving (`view=True`) requires atomic file replacement to avoid breaking in-flight readers.
- OpenMP environment hacks in `semantic.py` are FAISS-specific; can be removed once FAISS is dropped.

## Phases (suggested delivery)

### Phase 1: Abstraction + USearch behind a flag
- Add `tldr/vector_store/` package with protocol + USearch backend
- Wire `--vector-backend` CLI flag and `TLDR_VECTOR_BACKEND` env var
- Keep FAISS as default; USearch opt-in via flag
- Add CI lane to test with `TLDR_VECTOR_BACKEND=usearch`

### Phase 2: Make USearch default
- Switch default backend to USearch
- Move FAISS to optional extra (`llm-tldr[faiss]`) for legacy index reading
- Update all docs to recommend USearch

### Phase 3: Remove FAISS (optional)
- Remove FAISS extra + legacy reading code
- Remove FAISS-specific docs/tests
- Clean up OpenMP hacks

## Goals (Acceptance Criteria)

1. Semantic indexing/search can run using **USearch** instead of FAISS:
   - `tldrf warm …` (semantic build path)
   - `tldrf semantic index …`
   - `tldrf semantic …` (search path)

2. `semantic_search()` output schema remains stable:
   - Returns list of dicts with `name`, `qualified_name`, `file`, `line`, `unit_type`, `signature`, `score`
   - Scores remain "higher is better"

3. Persisted semantic index includes metadata to:
   - Validate embedding dimensionality and model compatibility
   - Determine which backend produced the index (FAISS vs USearch)

4. Backward compatibility:
   - Legacy `.faiss` indexes remain queryable if `faiss` is installed
   - Clear error message with rebuild instructions if `.faiss` exists but `faiss` not installed

5. Tests pass without requiring FAISS when using USearch.

6. Docs updated to remove FAISS-specific instructions.

## Non-Goals

- Hosted/remote vector DB integration (Pinecone, Weaviate, etc.)
- Changing the embedding model or 5-layer embedding text format
- Multi-index semantic search (search across multiple isolated indexes)
- GPU acceleration

## File Changes

### New files
```
tldr/vector_store/
├── __init__.py
├── base.py           # VectorIndex protocol + config dataclasses
├── usearch_backend.py # USearch implementation
└── faiss_backend.py   # (optional) FAISS wrapper for transition
```

### Modified files
| File | Changes |
|------|---------|
| `tldr/semantic.py` | Replace direct FAISS calls with vector store abstraction |
| `tldr/indexing/index.py` | Add `semantic_usearch` path to `IndexPaths` |
| `tldr/indexing/management.py` | Update artifact listing for new backend |
| `tldr/cli.py` | Add `--vector-backend` flag |
| `pyproject.toml` | Replace `faiss-cpu` with `usearch`; add `[faiss]` extra |
| `tests/test_semantic_index_isolated.py` | Parametrize for backend or switch to USearch |

### Documentation updates
| File | Changes |
|------|---------|
| `README.md` | Remove `faiss-cpu` mention, update cache filename |
| `docs/TLDR.md` | "FAISS index" → "vector index (USearch)" |
| `specs/001-feat-isolated-index.md` | Generalize `index.faiss` references |
| `implementations/001-feat-isolated-index.md` | Generalize FAISS references |

## Vector Store Protocol

```python
# tldr/vector_store/base.py

from typing import Protocol
from pathlib import Path
import numpy as np

class VectorIndex(Protocol):
    """Minimal interface for vector similarity search."""

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        *,
        ids: np.ndarray,
        dim: int,
        metric: str = "cos",
        config: dict | None = None,
    ) -> "VectorIndex": ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path, *, view: bool = False) -> "VectorIndex": ...

    def search(
        self, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ids, scores) where scores are higher-is-better."""
        ...
```

## Metadata Schema Changes

### `semantic/metadata.json`
```json
{
  "model": "...",
  "dimension": 1024,
  "count": 12345,
  "units": [ ... ],
  "vector_store": {
    "backend": "usearch",
    "metric": "cos",
    "dtype": "f32",
    "score_semantics": "cosine_similarity",
    "backend_version": "2.23.0",
    "build_params": {
      "connectivity": 16,
      "expansion_add": 128,
      "expansion_search": 64,
      "view_ok": true
    }
  }
}
```

### `indexes/<index_key>/meta.json`
```json
{
  "semantic": {
    "model": "...",
    "dim": 1024,
    "lang": null,
    "vector_backend": "usearch",
    "metric": "cos",
    "dtype": "f32"
  }
}
```

## USearch Default Configuration

Based on spec's "Target Defaults" section:

```python
USEARCH_DEFAULTS = {
    "metric": "cos",           # Cosine similarity (normalized embeddings)
    "dtype": "f32",            # Float32 for cross-platform parity
    "connectivity": 16,        # HNSW M parameter
    "expansion_add": 128,      # efConstruction
    "expansion_search": 64,    # ef for queries
}
```

## Testing Strategy

1. **Unit tests** for `tldr/vector_store/`:
   - Round-trip: build → save → load → search
   - Score directionality (descending order)
   - Dimension validation

2. **Integration tests**:
   - Parametrize existing semantic tests for both backends (or USearch-only)
   - Legacy `.faiss` index reading (if faiss installed)
   - Error message when `.faiss` exists but faiss not installed

3. **CI configuration**:
   - Default lane: USearch
   - Optional lane: `TLDR_VECTOR_BACKEND=faiss` (during transition)

## Migration UX

When a user has a legacy `.faiss` index:

```
Error: Semantic index requires rebuild.

Found:    .tldr/cache/semantic/index.faiss (legacy FAISS format)
Expected: .tldr/cache/semantic/index.usearch

To rebuild:
  tldrf semantic index --rebuild .

Or install FAISS support to read legacy index:
  pip install llm-tldr[faiss]
```
