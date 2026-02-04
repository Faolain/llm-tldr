# Implementation Plan: 002 - Swap FAISS for USearch (Vector Store Backend)

Spec: `specs/002-feat-swap-faiss-to-usearch.md`

## Summary

Replace the hard-wired FAISS dependency in `tldr/semantic.py` with a pluggable vector store abstraction, defaulting to USearch. This improves:

- **Cross-platform installation** (FAISS is a common pain point)
- **Flexibility** to swap ANN backends without touching core semantic logic
- **Robustness under concurrent rebuilds** via versioned index files + atomic metadata commits (+ rebuild lock)
- **Performance** (optional) via USearch memory-mapped serving for long-lived daemon/MCP workloads

## CLI Reality Check (important)

- The project CLI entrypoint is `tldrf` (not `tldr`).
- `tldrf warm` currently builds call graph + language cache; it does **not** build a semantic index.
- Semantic commands are subcommands: `tldrf semantic index …` and `tldrf semantic search …` (there is no `tldrf semantic <query>` today).

**Plan decision:** Update acceptance criteria + docs to match the current CLI shape. A `tldrf warm --include-semantic` flag is a reasonable follow-up, but not required for swapping the vector backend.

## Phase 0: Decisions (lock these in first)

These choices affect filenames, atomicity, daemon behavior, and cross-platform support. Make them explicit before coding.

### 0.1 File + serving policy (chosen)

**Chosen policy: Option C (versioned filenames + metadata pointer)**:

- Rebuild writes a new index file with a **unique** name, e.g.:
  - USearch: `index.<build_id>.usearch`
  - FAISS (legacy/optional): `index.<build_id>.faiss`
- `semantic/metadata.json` stores the **active** `vector_store.index_file`.
- Search always loads the index file referenced by metadata (metadata is the **commit point**).

Rationale:
- Lock-free reads + atomic swaps require the index file to be immutable while referenced.
- A stable filename (`index.usearch`) combined with “index first, metadata last” creates a window where readers can observe **new index + old units/metadata**, which is correctness-breaking.

**Serving mode**:
- Default load behavior: `view=True` on non-Windows *daemon* loads, `view=False` everywhere else.
  - Windows: force `view=False` (mmapped files block rename/delete semantics in common cases).
- Keep this policy documented and easy to override via a single knob later (e.g. env/config), but pick a default now.

**Garbage collection**:
- Keep the last **N** versions per backend (recommend `N=2`) under the rebuild lock.
- GC is best-effort:
  - On Windows: if deletion fails (mapped/open), leave the file and try next rebuild.
  - On Unix: deletion is safe even if old versions are memory-mapped (existing mappings remain valid).

### 0.2 Rebuild locking (required)

**Requirement:** serialized rebuilds per semantic index.

- Use a per-index lock file under the semantic cache dir:
  - Legacy mode: `.tldr/cache/semantic/.lock`
  - Index-mode: `<cache_root>/indexes/<key>/cache/semantic/.lock`
- Rebuild acquires an **exclusive** lock for the entire window:
  - build embeddings → write versioned index → write metadata (commit) → optional GC
- Reads remain lock-free and rely on the metadata commit point.

Implementation detail:
- Prefer a small, cross-platform lock dependency (e.g. `filelock`) over rolling our own Windows locking.
- Daemon-triggered rebuild should use a **non-blocking**/short-timeout lock acquisition and skip if already locked.

## Immediate Next Steps (rolling)

- [ ] Run codebase analysis to map all FAISS touchpoints (see "Codebase Analysis with tldrf" below)
- [ ] Create `tldr/vector_store/` package with `base.py` protocol
- [ ] Implement `tldr/vector_store/usearch_backend.py`
- [ ] Refactor `tldr/semantic.py` to use vector store abstraction
- [ ] Add `--vector-backend` CLI flag and `TLDR_VECTOR_BACKEND` env var
- [ ] Update `IndexPaths` to support both `.faiss` and `.usearch` files
- [ ] Extend `metadata.json` schema with `vector_store` block
- [ ] Implement rebuild lock + atomic “metadata commit point” behavior (Phase 1, not optional)
- [ ] Update daemon background semantic reindex to actually rebuild (see below)
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

- USearch returns **distance** (lower is better); TLDR expects **score** (higher is better). For cosine distance, conversion is `score = 1 - distance` (record this as an explicit `score_transform` in metadata).
- USearch metric must be set explicitly at build time (`metric='cos'`); don't rely on upstream defaults.
- USearch keys must be integers; generate stable IDs via `np.arange(len(units), dtype=np.uint64)` and map IDs back to `units[]`.
- USearch `search()` return shape differs for 1D vs 2D queries (Matches vs BatchMatches). Backend must normalize query shape and return only valid results (no sentinel IDs), possibly fewer than `k`.
- Keep optional dependencies optional: importing `tldr.semantic` should not immediately import `usearch` or `faiss` (use a backend factory + lazy imports).
- Atomic updates must treat `metadata.json` as the commit point. This is only safe with **versioned index filenames** referenced from metadata (see Phase 0.1).
- Windows + memory-mapped `view=True` breaks in-place replacement and file deletion semantics. Force `view=False` on Windows by default.
- OpenMP environment hacks in `semantic.py` are FAISS-specific; move them behind the FAISS backend and apply only immediately before importing FAISS.

## Phases (suggested delivery)

### Phase 1: Abstraction + USearch as default (and safe-by-default rebuilds)
- Add `tldr/vector_store/` package with protocol + USearch backend
- Add a backend factory + lazy imports (so core code can run without optional deps installed)
- Make USearch the default backend for new semantic builds
- Keep FAISS behind an optional extra for legacy index reading (and optional explicit builds)
- Wire `--vector-backend` CLI flag and `TLDR_VECTOR_BACKEND` env var for **index build selection**
- Persist backend, index filename (versioned), and score transform in `semantic/metadata.json`
- Implement serialized rebuild locking + atomic metadata commit ordering (Phase 0.2 + Phase 0.1)
- Make OpenMP env hacks FAISS-only (no global env constraints when using USearch)

### Phase 2: Harden rebuild + daemon/MCP safety
- Update daemon background rebuild to pass `--rebuild` (today it can no-op) and propagate backend selection
- Implement version GC policy (keep last N; best-effort delete on Windows)
- Add consistent errors/messages for:
  - legacy `.faiss` present but FAISS extra not installed
  - backend mismatch (CLI override vs metadata; do not silently “pick one”)
- Update docs/CI and regenerate `uv.lock` after dependency changes

### Phase 3: Remove FAISS (optional)
- Remove FAISS extra + legacy reading code
- Remove FAISS-specific docs/tests
- Clean up OpenMP hacks

## Goals (Acceptance Criteria)

1. Semantic indexing/search can run using **USearch** instead of FAISS:
   - `tldrf semantic index …`
   - `tldrf semantic search …`

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

7. Concurrent rebuild safety:
   - Serialized rebuilds via per-index lock file
   - Atomic metadata commit point with versioned index filenames (no mixed “index vs units” reads)
   - Daemon background rebuild passes `--rebuild` and does not corrupt indexes under concurrent reads

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
| `tldr/daemon/core.py` | Ensure background semantic reindex uses `--rebuild` + backend selection |
| `tldr/indexing/index.py` | Extend `IndexPaths` for `.usearch` + versioned filenames (+ semantic lock path helper) |
| `tldr/indexing/management.py` | Include `.usearch` artifacts (and versions) in listing + GC surfaces |
| `tldr/cli.py` | Add `--vector-backend` flag |
| `pyproject.toml` | Add `usearch` + `filelock`; move `faiss-cpu` to `[faiss]` extra (legacy/optional) |
| `uv.lock` | Regenerate to reflect dependency changes |
| `tests/test_semantic_index_isolated.py` | Parametrize for backend or switch to USearch |

### Documentation updates
| File | Changes |
|------|---------|
| `README.md` | Remove `faiss-cpu` mention, reconcile semantic cache paths (e.g. `.tldr/cache/semantic/index.*`), update CLI examples to `tldrf` |
| `docs/TLDR.md` | "FAISS index" → "vector index (USearch)", and update CLI examples to `tldrf semantic index/search` |
| `specs/001-feat-isolated-index.md` | Generalize `index.faiss` references |
| `implementations/001-feat-isolated-index.md` | Generalize FAISS references |

## Dependencies & Packaging

- Add `usearch` as a core dependency (semantic search should work on a base install).
  - Pin a minimum version that supports `Index.search(..., exact=...)` and `Index.restore(..., view=...)` as used in the backend (e.g. `usearch>=2.23.0`).
- Add `filelock` as a core dependency for cross-platform rebuild locking.
- Move `faiss-cpu` to an optional extra, e.g. `llm-tldr[faiss]`, for legacy `.faiss` index reading during the transition window.
- Update install/docs to reflect: base install supports semantic search (USearch); `.[faiss]` is only for legacy FAISS index compatibility.

## Backend Selection & Precedence

### Index build (`tldrf semantic index …`)

Backend selection precedence:
1. `--vector-backend`
2. `TLDR_VECTOR_BACKEND`
3. default: `usearch`

### Search (`tldrf semantic search …`)

- Always use the backend recorded in `semantic/metadata.json` when present.
- If `vector_store` is missing (legacy index):
  - Treat as legacy FAISS **only** if `index.faiss` exists (and `faiss` is installed).
  - Otherwise error with rebuild instructions.
- If a user explicitly passes `--vector-backend` during search and it conflicts with metadata: raise a clear mismatch error with rebuild instructions (prefer correctness over surprising behavior).
- Do not add a `--force-backend-mismatch` escape hatch (rebuild is the correct fix).

### Daemon background rebuild (must match CLI selection rules)

Current daemon behavior: `tldr/daemon/core.py:_trigger_background_reindex()` spawns:

```py
python -m tldr.cli semantic index <project>
```

It does **not** pass `--rebuild`, so it frequently no-ops when a semantic index already exists.

Plan changes:
- Pass `--rebuild` (or the equivalent “force rebuild” flag).
- Backend selection:
  - Simple: daemon inherits `TLDR_VECTOR_BACKEND`.
  - Better: allow daemon config `semantic.vector_backend` (in `.tldr/config.json` / `.claude/settings.json`) and pass `--vector-backend`.

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
        self, query: np.ndarray, k: int, *, exact: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (ids, scores) where scores are higher-is-better.

        Contract:
        - Implementations accept `query` shaped `(dim,)` or `(1, dim)` and normalize internally.
        - Implementations return only valid results (no sentinel IDs), possibly fewer than `k`.
        - `ids` and `scores` are the same length (`<= k`).
        """
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
    "backend_version": "2.23.0",
    "build_id": "<build_id>",
    "index_file": "index.<build_id>.usearch",
    "metric_intent": "cosine_similarity",
    "backend_metric": "cos",
    "dtype": "f32",
    "embeddings_normalized": true,
    "score_semantics": "cosine_similarity",
    "distance_semantics": "cosine_distance",
    "score_transform": "one_minus_distance",
    "build_params": {
      "connectivity": 16,
      "expansion_add": 128,
      "expansion_search": 64,
      "view_ok": true
    }
  }
}
```

Notes:
- `metric_intent` is always semantic intent (`cosine_similarity`); `backend_metric` is backend-specific (`cos` for USearch, `ip` for FAISS-on-normalized-vectors).
- `score_semantics` should remain `cosine_similarity` across backends. For FAISS legacy indexes, use `score_transform: "identity"` and omit/leave `distance_semantics` null.

### `indexes/<index_key>/meta.json`
```json
{
  "semantic": {
    "model": "...",
    "dim": 1024,
    "lang": null,
    "vector_backend": "usearch",
    "build_id": "<build_id>",
    "index_file": "index.<build_id>.usearch",
    "metric_intent": "cosine_similarity",
    "backend_metric": "cos",
    "dtype": "f32",
    "embeddings_normalized": true
  }
}
```

## Atomic Writes & `view=True`

Treat `semantic/metadata.json` as the commit point. With Phase 0’s versioned filenames, readers never observe mixed “index vs units” state.

### Rebuild algorithm (required)

1. Acquire the per-index rebuild lock (`semantic/.lock`).
2. Write the new index to a temp file in the semantic dir, then `os.replace()` it into a **new versioned filename** (never replace the active index file).
3. Write `metadata.json.tmp` pointing at the new `index_file`, then `os.replace()` → `metadata.json` (**commit point**).
4. Optional: GC old index versions (keep last N) under the same lock.

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
   - Determinism: use `exact=True` for unit tests (ANN ordering can be unstable)
   - Query-shape handling: `(dim,)` vs `(1, dim)` should both work

2. **Integration tests**:
   - Parametrize existing semantic tests for whichever backend is installed (or USearch-only after default swap)
   - For approximate mode, assert inclusion in top-k instead of strict rank ordering
   - Legacy `.faiss` index reading (if faiss installed)
   - Error message when `.faiss` exists but faiss not installed

3. **Important test fixture correction**
   - Current `tests/test_semantic_index_isolated.py` uses a DummyModel that returns all-zero embeddings.
   - Cosine distance/similarity with zero vectors is a footgun; update dummy embeddings to be **non-zero** and (optionally) normalized.
   - Update skips to `pytest.importorskip("usearch")` (or backend-parametrized skips) so tests pass without FAISS installed.

4. **CI configuration**:
   - Default lane: USearch
   - Optional lane: `TLDR_VECTOR_BACKEND=faiss` (during transition)

## Migration UX

When a user has a legacy `.faiss` index:

```
Error: Semantic index requires rebuild.

Found:    .tldr/cache/semantic/index.faiss (legacy FAISS format)
Expected: .tldr/cache/semantic/index.<build_id>.usearch + updated metadata (USearch format)

To rebuild:
  tldrf semantic index --rebuild .

Or install FAISS support to read legacy index:
  uv pip install -e ".[faiss]"        # in-repo
  uv pip install "llm-tldr[faiss]"    # installed package
```

## Optional Enhancements

- Add `tldrf semantic info` to print backend/version, dim/count, active index filename, metric + score semantics, and whether `view=True` is in use.
- Add a lightweight regression fixture (synthetic embeddings) to sanity-check top-k stability under defaults (and deterministic behavior under `exact=True`).
