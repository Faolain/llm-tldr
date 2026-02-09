"""True Incremental Updates (P4) - File-level graph patching.

This module provides O(1) per-file updates to the call graph, instead of
rebuilding the entire graph on every file edit.

Key functions:
- compute_file_hash(file_path) - SHA-1 hash for content-based deduplication
- extract_edges_from_file(file_path, lang) - Extract edges from single file
- patch_call_graph(graph, edited_file, project_root, lang) - Patch graph incrementally
- has_file_changed(file_path, cached_hash) - Check if file content changed

Usage:
    from tldr.patch import patch_call_graph, compute_file_hash, has_file_changed

    # Check if file changed
    if has_file_changed(file_path, cached_hash):
        # Patch the graph for just this file
        graph = patch_call_graph(graph, file_path, project_root, lang="python")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from tldr.cross_file_calls import (
    ProjectCallGraph,
    _extract_file_calls,
    _extract_ts_file_calls,
    _extract_go_file_calls,
    _extract_rust_file_calls,
)


@dataclass(frozen=True)
class Edge:
    """Represents a call edge from one function to another.

    Attributes:
        from_file: Source file path (relative to project root)
        from_func: Source function name
        to_file: Target file path (relative to project root)
        to_func: Target function name
    """
    from_file: str
    from_func: str
    to_file: str
    to_func: str

    def to_tuple(self) -> tuple[str, str, str, str]:
        """Convert to the tuple format used by ProjectCallGraph."""
        return (self.from_file, self.from_func, self.to_file, self.to_func)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-1 hash of file content.

    Args:
        file_path: Absolute path to the file

    Returns:
        40-character hex string representing the SHA-1 hash

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(file_path)
    content = path.read_bytes()
    return hashlib.sha1(content).hexdigest()


def has_file_changed(file_path: str, cached_hash: str) -> bool:
    """Check if file content has changed from cached hash.

    Args:
        file_path: Absolute path to the file
        cached_hash: Previously computed SHA-1 hash

    Returns:
        True if file has changed (or doesn't exist), False otherwise
    """
    try:
        current_hash = compute_file_hash(file_path)
        return current_hash != cached_hash
    except (FileNotFoundError, IOError):
        # Missing or unreadable file is considered "changed"
        return True


def extract_edges_from_file(
    file_path: str,
    lang: str = "python",
    project_root: Optional[str] = None
) -> List[Edge]:
    """Extract call edges from a single source file.

    Args:
        file_path: Absolute path to the source file
        lang: Language - "python", "typescript", "go", or "rust"
        project_root: Optional project root for computing relative paths

    Returns:
        List of Edge objects representing intra-file calls
    """
    path = Path(file_path)

    if project_root:
        root = Path(project_root)
        try:
            rel_path = path.relative_to(root)
            file_name = str(rel_path)
        except ValueError:
            file_name = path.name
    else:
        file_name = path.name

    # Get the appropriate extractor based on language
    if lang == "python":
        extractor = _extract_file_calls
    elif lang == "typescript":
        extractor = _extract_ts_file_calls
    elif lang == "go":
        extractor = _extract_go_file_calls
    elif lang == "rust":
        extractor = _extract_rust_file_calls
    else:
        raise ValueError(f"Unsupported language: {lang}")

    try:
        # Use project root or file's parent as root
        root_path = Path(project_root) if project_root else path.parent
        calls_by_func = extractor(path, root_path)
    except Exception:
        return []

    edges = []
    for caller_func, calls in calls_by_func.items():
        for call_type, call_target in calls:
            # Only include intra-file calls for now
            # Cross-file resolution requires the full function index
            if call_type == 'intra':
                edges.append(Edge(
                    from_file=file_name,
                    from_func=caller_func,
                    to_file=file_name,
                    to_func=call_target
                ))
            elif call_type == 'ref':
                # Function references (e.g., higher-order)
                edges.append(Edge(
                    from_file=file_name,
                    from_func=caller_func,
                    to_file=file_name,
                    to_func=call_target
                ))

    return edges


def patch_call_graph(
    graph: ProjectCallGraph,
    edited_file: str,
    project_root: str,
    lang: str = "python"
) -> ProjectCallGraph:
    """Incrementally update call graph for an edited file.

    This is the core incremental update algorithm:
    1. Remove all edges where from_file == edited_file
    2. Extract new edges from the edited file
    3. Add new edges to the graph
    4. Return the updated graph

    Args:
        graph: Existing ProjectCallGraph to patch
        edited_file: Absolute path to the edited file
        project_root: Project root directory
        lang: Language - "python", "typescript", "go", or "rust"

    Returns:
        Updated ProjectCallGraph (modifies in place and returns same object)
    """
    edited_path = Path(edited_file)
    root_path = Path(project_root)

    # Compute relative path for matching
    try:
        rel_path = str(edited_path.relative_to(root_path))
    except ValueError:
        rel_path = edited_path.name

    # Step 1: Remove all edges FROM the edited file
    edges_to_remove = set()
    for edge in graph.edges:
        src_file, src_func, dst_file, dst_func = edge
        if src_file == rel_path:
            edges_to_remove.add(edge)

    # Remove the edges (modify internal state)
    graph._edges -= edges_to_remove

    # Step 2: Extract new edges from the edited file
    new_edges = extract_edges_from_file(
        str(edited_file),
        lang=lang,
        project_root=project_root
    )

    # Step 3: Add new edges to the graph
    for edge in new_edges:
        graph.add_edge(edge.from_file, edge.from_func, edge.to_file, edge.to_func)

    return graph


def get_file_hash_cache(project_root: str) -> dict[str, str]:
    """Load cached file hashes from project cache directory.

    Args:
        project_root: Project root directory

    Returns:
        Dict mapping relative file paths to their SHA-1 hashes
    """
    import json
    cache_path = Path(project_root) / ".tldr" / "cache" / "file_hashes.json"

    if not cache_path.exists():
        return {}

    try:
        return json.loads(cache_path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def save_file_hash_cache(project_root: str, cache: dict[str, str]) -> None:
    """Save file hash cache to project cache directory.

    Args:
        project_root: Project root directory
        cache: Dict mapping relative file paths to their SHA-1 hashes
    """
    import json
    cache_path = Path(project_root) / ".tldr" / "cache" / "file_hashes.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))


def patch_dirty_files(
    graph: ProjectCallGraph,
    project_root: str,
    dirty_files: list[str],
    lang: str = "python"
) -> ProjectCallGraph:
    """Patch the graph for all dirty files.

    This is the main entry point for incremental updates when the dirty
    flag system reports changed files.

    Args:
        graph: Existing ProjectCallGraph to patch
        project_root: Project root directory
        dirty_files: List of relative file paths that changed
        lang: Language for all files

    Returns:
        Updated ProjectCallGraph
    """
    root_path = Path(project_root)

    for rel_file in dirty_files:
        abs_file = root_path / rel_file
        if abs_file.exists():
            graph = patch_call_graph(graph, str(abs_file), project_root, lang=lang)

    return graph


def patch_typescript_resolved_dirty_files(
    graph: ProjectCallGraph,
    project_root: str | Path,
    dirty_files: list[str],
    *,
    trace: bool = False,
    timeout_s: int = 60,
) -> dict[str, Any]:
    """Patch a TS-resolved call graph for the given dirty files.

    This recomputes *outbound* call edges (caller == dirty file) using the
    TypeScript compiler API resolver and patches the existing graph in-place.

    Notes:
    - This is intentionally conservative: it does not attempt to update edges
      *into* the dirty file from other files (those callsites didn't change).
    - If the TS resolver cannot run, callers should fall back to a full rebuild.
    """
    root = Path(project_root).resolve()

    dirty_rel: set[str] = set()
    dirty_abs: list[Path] = []

    for item in dirty_files:
        p = Path(item)
        if p.is_absolute():
            abs_path = p
            try:
                rel = abs_path.resolve().relative_to(root).as_posix()
            except Exception:
                continue
        else:
            rel = p.as_posix()
            abs_path = root / p

        if not rel.endswith((".ts", ".tsx")):
            continue
        if not abs_path.exists():
            continue
        dirty_rel.add(rel)
        dirty_abs.append(abs_path.resolve())

    if not dirty_rel:
        return {
            "dirty_files_total": len(dirty_files),
            "patched_files": [],
            "patched_file_count": 0,
            "patched_edge_count": 0,
            "skipped_reason": "no_typescript_files",
        }

    dirty_abs.sort(key=lambda p: str(p))

    # Import lazily so normal patch workflows don't require node/typescript.
    from tldr.ts.ts_callgraph import TsResolverError, build_ts_resolved_call_graph_multi_tsconfig

    try:
        edges, ts_meta = build_ts_resolved_call_graph_multi_tsconfig(
            root,
            allow_files=dirty_abs,
            trace=trace,
            timeout_s=timeout_s,
        )
    except TsResolverError:
        # Let caller decide fallback strategy.
        raise

    # Remove all edges whose caller file is dirty.
    edges_to_remove: set[tuple[str, str, str, str]] = set()
    for edge in graph.edges:
        src_file = edge[0].replace("\\", "/")
        if src_file in dirty_rel:
            edges_to_remove.add(edge)
    graph._edges -= edges_to_remove

    # Add recomputed edges.
    for e in edges:
        src_file, src_func, dst_file, dst_func = e.to_tuple()
        graph.add_edge(src_file, src_func, dst_file, dst_func)

    # Minimal incremental meta for debugging and benchmarking.
    inc_meta = {
        "patched_files": sorted(dirty_rel),
        "patched_file_count": len(dirty_rel),
        "patched_edge_count": len(edges),
        "ts_projects_ok": ts_meta.get("ok_projects"),
        "ts_projects_err": ts_meta.get("error_projects"),
    }
    graph.meta["ts_incremental"] = inc_meta
    if int(ts_meta.get("error_projects") or 0) > 0:
        graph.meta["incomplete"] = True

    return inc_meta
