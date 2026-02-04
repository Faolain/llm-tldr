# Spec: 002 - Swap FAISS for USearch (Vector Store Backend)

**Problem**
- TLDR semantic search is currently hard-wired to FAISS (`faiss-cpu`) in `tldr/semantic.py`.
- That coupling makes it difficult to:
  - Replace FAISS with a different ANN library (e.g. `usearch` / `unum-cloud/USearch`).
  - Experiment with different index types / storage modes (in-RAM vs memory-mapped).
  - Keep installation predictable across platforms (FAISS is a common pain point for Python users).

This spec describes everything required to swap the semantic vector index implementation from FAISS to USearch, while keeping TLDR’s user-facing CLI and result schema stable.

Related:
- Existing semantic cache layout + index isolation work: `specs/001-feat-isolated-index.md`

---

## Goals (Acceptance Criteria)

1. Semantic indexing/search can run using **USearch** instead of FAISS:
   - `tldr warm …` (semantic build path)
   - `tldr semantic index …`
   - `tldr semantic …` (search path)
2. `semantic_search()` output schema remains stable:
   - still returns a list of dicts with `name`, `qualified_name`, `file`, `line`, `unit_type`, `signature`, `score` (+ optional graph expansion fields).
3. Persisted semantic index includes **enough metadata** to:
   - validate embedding dimensionality and model compatibility
   - determine which backend produced the index (FAISS vs USearch)
4. Backward compatibility is explicitly handled:
   - Either (A) legacy `.faiss` indexes remain queryable during a transition window, or (B) the UX clearly instructs users to rebuild.
5. Tests are updated to pass without requiring FAISS when using USearch.
6. Docs are updated to remove FAISS-specific instructions and file names.

---

## Non-Goals

- Adding a hosted/remote vector DB integration (Pinecone, Weaviate, etc.).
- Changing the embedding model(s) or the 5-layer embedding text format.
- Implementing multi-index semantic search (search across multiple isolated indexes).
- GPU acceleration.

---

## Current State (FAISS)

### Where FAISS is used
- `tldr/semantic.py`:
  - Builds embeddings (normalized) and writes a FAISS index via:
    - `faiss.IndexFlatIP(dimension)`
    - `index.add(embeddings_matrix)`
    - `faiss.write_index(index, "index.faiss")`
  - Searches via:
    - `index = faiss.read_index("index.faiss")`
    - `scores, indices = index.search(query_embedding, k)`

### Cache layout
Legacy mode:
- `.tldr/cache/semantic/index.faiss`
- `.tldr/cache/semantic/metadata.json`

Index-mode (isolated indexes):
- `<cache_root>/.tldr/indexes/<index_key>/cache/semantic/index.faiss`
- `<cache_root>/.tldr/indexes/<index_key>/cache/semantic/metadata.json`

### Result semantics
- Embeddings are L2-normalized.
- FAISS index uses Inner Product (`IndexFlatIP`) so the returned `score` is (effectively) **cosine similarity**.
- Returned scores are “higher is better”.

---

## Proposed Approach

### High-level design

1. Introduce a small “vector store” abstraction so `tldr/semantic.py` is not tied to any one backend.
2. Implement a **USearch** backend that supports:
   - building from a `float32` matrix of shape `(n, dim)`
   - persistent serialization to disk
   - top-k ANN search
3. Store a backend identifier + backend config in semantic metadata so search can correctly load and interpret scores.
4. Provide a staged rollout plan:
   - Phase 1: Land abstraction + USearch backend behind a flag, keep FAISS default.
   - Phase 2: Make USearch the default backend; move FAISS to an optional extra.
   - Phase 3: (Optional) Remove FAISS support entirely.

This keeps risk low and avoids breaking existing users immediately.

---

## Backend API Notes: USearch

USearch’s Python API (as of `usearch` on PyPI) exposes:
- `from usearch.index import Index`
- `index = Index(ndim=D, metric='cos', dtype='f32', connectivity=..., expansion_add=..., expansion_search=...)`
- `index.add(key_or_keys, vector_or_vectors)`
- `matches = index.search(vector, k)` returning keys + distances
- `index.save("index.usearch")`
- `Index.restore("index.usearch", view=True, ...)` for memory-mapped serving

Important behavioral difference vs FAISS:
- USearch commonly returns **distance**, where “lower is better”.
- TLDR currently returns `score` where “higher is better”.

So the integration must define and persist a deterministic mapping:
- If USearch uses cosine distance `d = 1 - cos_sim`, then TLDR can return `score = 1 - d`.
- If USearch uses another metric, document + implement the conversion explicitly.

The spec requires we **set metric explicitly** when building (don’t rely on upstream defaults, which can vary by version).

---

## Vector Store Abstraction (Required Refactor)

### New module layout
Add a small package:
- `tldr/vector_store/`
  - `base.py` (protocols + shared types)
  - `usearch_backend.py`
  - `faiss_backend.py` (optional during transition)
  - `__init__.py`

### Core interface
Define a minimal interface TLDR needs:

- `build_index(vectors: np.ndarray, *, ids: np.ndarray, dim: int, metric: str, config: dict) -> VectorIndex`
- `VectorIndex.save(path: Path) -> None`
- `VectorIndex.search(query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]`
  - returns `(ids, scores)` in TLDR semantics (scores higher-is-better)
- `VectorIndex.load(path: Path, *, view: bool) -> VectorIndex`

Notes:
- “TLDR semantics” means `scores` are comparable to existing FAISS scores, at least in directionality.
- The abstraction should hide whether the backend returns distances or similarities.

### Backend selection
Support selecting a backend via:
- CLI flag (recommended): `tldr semantic index --vector-backend usearch`
- Env var fallback: `TLDR_VECTOR_BACKEND=usearch|faiss`

Persist the chosen backend in metadata to ensure search loads the same backend that built the index.

---

## Persisted Metadata (Required Changes)

### `semantic/metadata.json` schema changes
Current:
- `model`, `dimension`, `count`, `units`

Proposed additions:

```json
{
  "model": "…",
  "dimension": 1024,
  "count": 12345,
  "units": [ … ],
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

Rules:
- `backend` is required for any newly-built index.
- If `vector_store` is missing, treat the index as legacy FAISS (during transition only).
- Always record the metric + dtype explicitly.

### `indexes/<index_key>/meta.json` schema changes
`tldr/indexing/index.py:update_meta_semantic()` currently stores:

```json
"semantic": {"model": "...", "dim": 1024, "lang": null}
```

Extend to also persist the vector backend:

```json
"semantic": {
  "model": "...",
  "dim": 1024,
  "lang": null,
  "vector_backend": "usearch",
  "metric": "cos",
  "dtype": "f32"
}
```

This enables:
- `tldr index info` to surface the semantic backend
- basic staleness / mismatch checks without opening semantic metadata

---

## Cache Layout / Filenames (Required Decisions)

### Index file naming
Avoid reusing `index.faiss` for non-FAISS data.

Recommended:
- FAISS: `index.faiss`
- USearch: `index.usearch`

That implies `tldr/indexing/index.py:IndexPaths` should evolve from:
- `semantic_faiss: Path`

to something like:
- `semantic_index: Path` (the resolved “active” backend file)
- `semantic_faiss: Path` (legacy path, optional)
- `semantic_usearch: Path`

And `tldr/indexing/management.py` should report artifacts as:
- `semantic_index` (primary)
- `semantic_faiss` (optional, if present)
- `semantic_usearch` (optional, if present)

### Atomic writes
Index writes should be atomic to avoid partial/corrupt files when:
- build is interrupted
- daemon rebuild overlaps with queries

Required behavior:
- write to a temp path in the same directory
- `os.replace(tmp, final)`

---

## Behavioral Parity Requirements

### Score parity
`semantic_search()` currently returns `score` from FAISS inner product over normalized vectors.

USearch integration must ensure:
- `score` remains “higher is better”
- for cosine similarity use-cases, `score` remains in a comparable numeric range (ideally `[-1, 1]` or `[0, 1]` depending on the backend definition)

If USearch returns cosine **distance** (0 == identical), then:
- TLDR should return `score = 1 - distance` (documented + persisted in metadata)

### Ordering and determinism
FAISS `IndexFlatIP` is exact and deterministic.
USearch `Index` is typically HNSW (approximate).

This spec requires explicitly choosing one of:

Option A (recommended for parity): **Exact mode**
- Persist embeddings as `embeddings.npy` (or `embeddings.f32`) alongside metadata.
- Use `usearch.index.search(vectors, query, k, metric, exact=True)` at query time.
- Pros: deterministic and should closely match FAISS flat search.
- Cons: more disk, potentially slower at query time for very large `n`.

Option B: **Approximate mode (HNSW)**
- Persist a USearch index file (`index.usearch`).
- Use `Index.search()` for queries.
- Pros: fast queries, compact.
- Cons: results may differ vs FAISS (non-deterministic-ish depending on params), tests must allow for variation.

The repo should pick one and document it as part of “swap out” scope.
If the goal is “drop-in replacement with minimal user-visible changes”, Option A is the closer match.

---

## Code Changes (Checklist)

### 1) Introduce vector store package
- Add `tldr/vector_store/base.py` defining:
  - backend enum/strings
  - dataclasses for config + metadata
  - protocol for `VectorIndex`
- Add `tldr/vector_store/usearch_backend.py` implementing:
  - build, save, load/view, search
  - score conversion to TLDR semantics
- (Optional transition) Add `tldr/vector_store/faiss_backend.py` wrapping existing FAISS behavior.

### 2) Refactor `tldr/semantic.py`
- Replace direct FAISS calls with vector-store abstraction:
  - build path: create index via selected backend, save to backend-specific filename
  - search path: read metadata, pick backend, load the correct file, query
- Remove FAISS-specific OpenMP environment hacks once FAISS is no longer a default dependency (keep only if still needed by other deps).
- Update docstrings mentioning “FAISS” to “vector store backend”.

### 3) Update indexing paths + management surfaces
- Update `tldr/indexing/index.py:IndexPaths` to include the new index file path(s).
- Update `_invalidate_index_caches()` to delete the correct semantic artifact(s) for the active backend.
- Update `tldr/indexing/management.py:get_index_info()` artifact listing keys.

### 4) CLI / env var wiring
- Add global arg + env handling:
  - `--vector-backend` on `tldr semantic index` and `tldr warm`
  - env `TLDR_VECTOR_BACKEND`
- Ensure daemon commands that trigger semantic rebuild propagate the backend selection (in index mode and legacy mode).

### 5) Dependency + packaging changes
- Replace hard dependency `faiss-cpu` with `usearch` (or make both optional):
  - Recommended interim:
    - core deps include `usearch`
    - optional extra `faiss` for legacy index reading: `llm-tldr[faiss]`
  - Final:
    - drop `faiss-cpu` entirely
- Update `README.md` and `docs/TLDR.md` install instructions accordingly.

### 6) Tests
Update semantic tests to not require FAISS:
- Replace `pytest.importorskip("faiss")` with `pytest.importorskip("usearch")` (if fully swapped), OR
- Parametrize tests to run for whichever backend is installed.

Add/adjust tests:
- Round-trip serialization:
  - build index → save → load/view → query → stable output
- Score directionality:
  - ensure results are sorted by `score` descending (even if backend returns distances)
- Legacy index behavior (only if supported):
  - if `.faiss` exists and `faiss` is installed, it should still query successfully
  - if `.faiss` exists but `faiss` is not installed, error should recommend rebuild

### 7) Documentation updates
Update:
- `README.md` (remove `faiss-cpu` mention, correct cache filename)
- `docs/TLDR.md` (“FAISS index” → “vector index (USearch)”)
- `specs/001-feat-isolated-index.md` and `implementations/001-feat-isolated-index.md` references to `index.faiss` / “FAISS” where the exact filename/backend matters (either generalize or document the new filename).

---

## Migration & Backward Compatibility

### Migration strategy options

Option 1 (recommended): “Read old, write new”
- New indexes are written as USearch (`index.usearch`) with updated metadata.
- Search path:
  - If metadata declares `vector_store.backend == "usearch"`, load USearch.
  - Else if `index.faiss` exists, try FAISS (if installed) and warn that the index is legacy.
  - Else error with rebuild instructions.

Option 2: “Hard break”
- Immediately stop supporting `.faiss`.
- If legacy `.faiss` is detected, require rebuild.
- Simplest implementation, but breaks users on upgrade.

### User-facing UX
When rebuild is required, error message should include:
- which file was expected (`index.usearch`)
- what exists (`index.faiss`)
- the exact command to rebuild (respecting index flags):
  - `tldr semantic index --rebuild …`
  - or `tldr warm …` if warm builds semantic automatically

---

## Open Questions (Must Decide Before Implementation)

1. **Exact vs approximate**: choose Option A (exact) or Option B (HNSW) as the default implementation strategy.
2. Default USearch config:
   - `metric`: `cos` vs `ip` (recommend `cos` for clarity, store normalized vectors either way)
   - `dtype`: `f32` vs quantized (recommend `f32` for parity)
   - HNSW params if approximate is chosen
3. Memory mapping in daemon:
   - Should daemon use `Index.restore(..., view=True)` to reduce RSS?
   - If so, ensure file locks / atomic replace don’t break in-flight views.
4. Deprecation window:
   - How long to keep FAISS-reading support (if any)?

---

## Rollout Plan (Recommended)

Phase 1: Abstraction + USearch behind a flag
- Add vector backend selection and USearch implementation.
- Keep FAISS as the default to minimize immediate change.
- Add a CI lane (or local doc) to run tests with `TLDR_VECTOR_BACKEND=usearch`.

Phase 2: Make USearch default
- Switch default backend to USearch.
- Move FAISS to an optional extra for legacy index reading or parity testing.
- Update docs to recommend USearch.

Phase 3: Remove FAISS (optional)
- Remove FAISS extra + legacy reading code.
- Remove FAISS-specific docs/tests.

