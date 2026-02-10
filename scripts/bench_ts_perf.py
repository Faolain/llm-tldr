#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from bench_util import (
    bench_cache_root,
    bench_corpora_root,
    bench_runs_root,
    gather_meta,
    get_repo_root,
    make_report,
    now_utc_compact,
    percentiles,
    write_report,
)

from tldr.cross_file_calls import ProjectCallGraph, build_project_call_graph
from tldr.daemon.startup import query_daemon, start_daemon, stop_daemon
from tldr.indexing.index import IndexContext, get_index_context
from tldr.indexing.management import get_index_info, list_indexes
from tldr.patch import patch_typescript_resolved_dirty_files


@dataclass(frozen=True)
class ImpactTarget:
    file: str
    symbol: str


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                if fp.is_symlink():
                    continue
                total += fp.stat().st_size
            except OSError:
                continue
    return total


def _cacheable_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    out = dict(meta)
    out.pop("ts_trace", None)
    ts_meta = out.get("ts_meta")
    if isinstance(ts_meta, dict):
        ts_meta2 = dict(ts_meta)
        ts_meta2.pop("skipped", None)
        out["ts_meta"] = ts_meta2
    return out


def _write_graph_cache(path: Path, graph: ProjectCallGraph, *, lang: str) -> None:
    cache_data = {
        "edges": [
            {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
            for e in graph.sorted_edges()
        ],
        "meta": _cacheable_meta(getattr(graph, "meta", None)),
        "languages": [lang],
        "timestamp": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache_data, indent=2, sort_keys=True) + "\n")


def _load_curated_impact_targets(curated_path: Path, *, limit: int) -> list[ImpactTarget]:
    data = json.loads(curated_path.read_text())
    edges = data.get("edges") if isinstance(data, dict) else data
    if not isinstance(edges, list):
        raise ValueError(f"Bad curated edges file: {curated_path}")

    seen: set[tuple[str, str]] = set()
    out: list[ImpactTarget] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        callee = e.get("callee")
        if not isinstance(callee, dict):
            continue
        tf = callee.get("file")
        ts = callee.get("symbol")
        if not isinstance(tf, str) or not isinstance(ts, str):
            continue
        key = (tf, ts)
        if key in seen:
            continue
        seen.add(key)
        out.append(ImpactTarget(file=tf, symbol=ts))
        if len(out) >= limit:
            break
    return out


def _default_touch_file(corpus_id: str | None) -> str | None:
    # Deterministic, small-ish TS files that are known to exist at the pinned refs.
    if corpus_id == "peerbit":
        return "packages/clients/peerbit/src/peer.ts"
    if corpus_id == "nextjs":
        return "packages/next/src/shared/lib/router/utils/parse-path.ts"
    return None


def _build_graph(repo_root: Path) -> tuple[float, ProjectCallGraph]:
    t0 = time.monotonic()
    graph = build_project_call_graph(
        repo_root,
        language="typescript",
        use_workspace_config=True,
        ts_trace=False,
    )
    dt = time.monotonic() - t0
    return dt, graph


def _bench_patch_vs_rebuild(
    repo_root: Path,
    *,
    graph: ProjectCallGraph,
    touch_file: str,
) -> dict[str, Any]:
    abs_path = repo_root / touch_file
    if not abs_path.exists():
        return {"ok": False, "error": f"touch file does not exist: {touch_file}"}

    original = abs_path.read_text(encoding="utf-8", errors="replace")
    marker = "\n// tldr-bench-touch\n"

    patch_s: float | None = None
    patch_meta: Any = None
    patch_error: str | None = None
    rebuild_s: float | None = None
    rebuild_graph_source: str | None = None
    rebuild_incomplete: bool | None = None

    try:
        abs_path.write_text(original + marker, encoding="utf-8")

        try:
            t0 = time.monotonic()
            patch_meta = patch_typescript_resolved_dirty_files(
                graph,
                repo_root,
                [touch_file],
                trace=False,
                timeout_s=60,
            )
            patch_s = time.monotonic() - t0
        except Exception as exc:
            patch_error = str(exc)

        t0 = time.monotonic()
        rebuilt = build_project_call_graph(
            repo_root,
            language="typescript",
            use_workspace_config=True,
            ts_trace=False,
        )
        rebuild_s = time.monotonic() - t0
        rebuilt_meta = getattr(rebuilt, "meta", {}) or {}
        rebuild_graph_source = rebuilt_meta.get("graph_source")
        rebuild_incomplete = bool(rebuilt_meta.get("incomplete"))
    finally:
        abs_path.write_text(original, encoding="utf-8")

    return {
        "ok": patch_error is None and patch_s is not None and rebuild_s is not None,
        "touch_file": touch_file,
        "patch_s": round(patch_s, 4) if patch_s is not None else None,
        "patch_meta": patch_meta,
        "patch_error": patch_error,
        "full_rebuild_after_touch_s": round(rebuild_s, 4) if rebuild_s is not None else None,
        "full_rebuild_graph_source": rebuild_graph_source,
        "full_rebuild_incomplete": rebuild_incomplete,
    }


def _bench_daemon_impacts(
    repo_root: Path,
    *,
    index_ctx: IndexContext,
    targets: list[ImpactTarget],
    iterations: int,
) -> dict[str, Any]:
    # Keep output JSON-only.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        stop_daemon(repo_root, index_ctx=index_ctx)
        start_daemon(repo_root, foreground=False, index_ctx=index_ctx)

    warm_t0 = time.monotonic()
    warm_res = query_daemon(
        repo_root,
        {"cmd": "warm", "language": "typescript"},
        index_ctx=index_ctx,
    )
    warm_s = time.monotonic() - warm_t0

    per_target: list[dict[str, Any]] = []
    all_ms: list[float] = []

    for t in targets:
        cmd: dict[str, Any] = {"cmd": "impact", "func": t.symbol, "depth": 1, "file": t.file}

        # Warmup (not counted)
        query_daemon(repo_root, cmd, index_ctx=index_ctx)

        samples_ms: list[float] = []
        ok = True
        for _ in range(iterations):
            t0 = time.perf_counter()
            res = query_daemon(repo_root, cmd, index_ctx=index_ctx)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            samples_ms.append(dt_ms)
            if res.get("status") != "ok":
                ok = False
        all_ms.extend(samples_ms)
        per_target.append(
            {
                "target": f"{t.file}:{t.symbol}",
                "ok": ok,
                "samples_ms": [round(x, 4) for x in samples_ms],
                "percentiles_ms": {k: round(v * 1000.0, 4) for k, v in percentiles([x / 1000.0 for x in samples_ms]).items()},
            }
        )

    # Stop daemon to avoid side effects for other work.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        stop_daemon(repo_root, index_ctx=index_ctx)

    return {
        "warm_s": round(warm_s, 4),
        "warm_status": warm_res.get("status"),
        "targets": len(targets),
        "iterations": iterations,
        "impact_all_percentiles_ms": {k: round(v * 1000.0, 4) for k, v in percentiles([x / 1000.0 for x in all_ms]).items()},
        "per_target": per_target,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 TS perf benchmarks (build, patch, daemon impact latency).")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", default=None, help="Corpus id from benchmarks/corpora.json (e.g. peerbit, nextjs).")
    group.add_argument("--repo-root", default=None, help="Path to the corpus repo root.")
    ap.add_argument(
        "--curated",
        default=None,
        help="Curated edges JSON to derive impact targets (default: benchmarks/ts/<corpus>_curated_edges.json).",
    )
    ap.add_argument(
        "--cache-root",
        default=str(bench_cache_root(get_repo_root())),
        help="Index-mode cache root (default: benchmark/cache-root).",
    )
    ap.add_argument("--index", default=None, help="Index id (default: repo:<corpus>).")
    ap.add_argument("--touch-file", default=None, help="Repo-root-relative file to modify for patch benchmark.")
    ap.add_argument("--clear-callgraph-cache", action="store_true", help="Delete existing call_graph.json before build.")
    ap.add_argument("--daemon", action="store_true", help="Measure daemon warm + impact latencies.")
    ap.add_argument("--impact-targets", type=int, default=5, help="Max unique callee targets to measure via daemon.")
    ap.add_argument("--iterations", type=int, default=10, help="Impact iterations per target (daemon).")
    ap.add_argument("--out", default=None, help="Write JSON report to this path (default under benchmark/runs/).")
    args = ap.parse_args()

    tldr_repo_root = get_repo_root()
    corpus_id = args.corpus
    if corpus_id:
        repo_root = (bench_corpora_root(tldr_repo_root) / corpus_id).resolve()
        default_index_id = f"repo:{corpus_id}"
        default_curated = tldr_repo_root / "benchmarks" / "ts" / f"{corpus_id}_curated_edges.json"
    else:
        repo_root = Path(args.repo_root).resolve()
        default_index_id = None
        default_curated = None
        corpus_id = repo_root.name

    if not repo_root.exists():
        raise SystemExit(f"error: repo-root does not exist: {repo_root}")

    index_id = args.index or default_index_id
    index_ctx = get_index_context(
        scan_root=repo_root,
        cache_root_arg=args.cache_root,
        index_id_arg=index_id,
        allow_create=True,
    )
    index_paths = index_ctx.paths
    call_graph_cache = index_paths.call_graph if index_paths is not None else None

    if args.clear_callgraph_cache and call_graph_cache is not None:
        try:
            call_graph_cache.unlink(missing_ok=True)
        except OSError:
            pass

    build_s, graph = _build_graph(repo_root)
    meta = getattr(graph, "meta", {}) or {}
    graph_source = meta.get("graph_source")
    incomplete = bool(meta.get("incomplete"))

    if call_graph_cache is not None:
        _write_graph_cache(call_graph_cache, graph, lang="typescript")

    results: dict[str, Any] = {
        "build": {
            "build_s": round(build_s, 4),
            "graph_source": graph_source,
            "incomplete": incomplete,
            "edge_count": len(graph.edges),
        }
    }

    # Patch benchmark (only meaningful for ts-resolved graphs).
    touch_file = args.touch_file or _default_touch_file(corpus_id)
    if touch_file is None:
        results["patch_vs_rebuild"] = {"ok": False, "error": "no touch file specified (pass --touch-file)"}
    elif not str(graph_source or "").startswith("ts-resolved"):
        results["patch_vs_rebuild"] = {
            "ok": False,
            "skipped": True,
            "reason": f"graph_source={graph_source!r} (need ts-resolved)",
            "touch_file": touch_file,
        }
    else:
        results["patch_vs_rebuild"] = _bench_patch_vs_rebuild(repo_root, graph=graph, touch_file=touch_file)

    # Daemon impact latency microbench (optional).
    curated_path = None
    if args.curated:
        curated_path = Path(args.curated).resolve()
    elif default_curated and default_curated.exists():
        curated_path = default_curated

    if args.daemon:
        if curated_path is None or not curated_path.exists():
            results["daemon"] = {"ok": False, "error": "no curated file found (pass --curated)"}
        else:
            targets = _load_curated_impact_targets(curated_path, limit=int(args.impact_targets))
            results["daemon"] = _bench_daemon_impacts(
                repo_root,
                index_ctx=index_ctx,
                targets=targets,
                iterations=int(args.iterations),
            )
            results["daemon"]["curated"] = str(curated_path)
    else:
        results["daemon"] = {"skipped": True}

    # Cache sizing.
    cache_info: dict[str, Any] = {
        "cache_root": str(index_ctx.cache_root) if index_ctx.cache_root else None,
        "cache_root_bytes": _dir_size_bytes(index_ctx.cache_root) if index_ctx.cache_root else None,
        "index_id": index_ctx.index_id,
        "index_key": index_ctx.index_key,
        "index_dir": str(index_paths.index_dir) if index_paths is not None else None,
        "index_bytes": _dir_size_bytes(index_paths.index_dir) if index_paths is not None else None,
        "call_graph_cache": str(call_graph_cache) if call_graph_cache is not None else None,
        "call_graph_cache_bytes": call_graph_cache.stat().st_size if call_graph_cache is not None and call_graph_cache.exists() else None,
    }
    if index_ctx.cache_root is not None:
        try:
            index_ref = index_ctx.index_id or index_ctx.index_key
            cache_info["index_mgmt"] = {
                "list": list_indexes(index_ctx.cache_root),
                "info": get_index_info(index_ctx.cache_root, str(index_ref)),
            }
        except Exception as exc:
            cache_info["index_mgmt"] = {"error": str(exc)}
    results["cache"] = cache_info

    ok = not incomplete
    results["ok"] = ok

    report = make_report(
        phase="phase3_ts_perf",
        meta=gather_meta(
            tldr_repo_root=tldr_repo_root,
            corpus_id=corpus_id,
            corpus_root=repo_root,
        ),
        protocol={
            "cache_root": str(index_ctx.cache_root) if index_ctx.cache_root is not None else None,
            "index_id": index_ctx.index_id,
            "clear_callgraph_cache": bool(args.clear_callgraph_cache),
            "touch_file": touch_file,
            "daemon": bool(args.daemon),
            "impact_targets": int(args.impact_targets),
            "iterations": int(args.iterations),
            "curated": str(curated_path) if curated_path is not None else None,
        },
        results=results,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        ts = now_utc_compact()
        out_path = bench_runs_root(tldr_repo_root) / f"{ts}-ts-perf-{corpus_id}.json"
    write_report(out_path, report)
    print(out_path)
    return 0 if report["results"].get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
