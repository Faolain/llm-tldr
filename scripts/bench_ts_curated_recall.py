#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench_util import (
    bench_cache_root,
    bench_runs_root,
    gather_meta,
    get_repo_root,
    make_report,
    now_utc_compact,
    write_report,
)

from tldr.analysis import impact_analysis
from tldr.cross_file_calls import ProjectCallGraph, build_project_call_graph
from tldr.indexing.index import get_index_context


@dataclass(frozen=True)
class CuratedEdge:
    caller_file: str
    caller_symbol: str
    callee_file: str
    callee_symbol: str


def _load_curated_edges(path: Path) -> tuple[str | None, list[CuratedEdge]]:
    data: Any = json.loads(path.read_text())
    corpus_id = None
    edges_raw = None
    if isinstance(data, dict):
        corpus_id = data.get("repo") if isinstance(data.get("repo"), str) else None
        edges_raw = data.get("edges")
    elif isinstance(data, list):
        edges_raw = data
    else:
        raise ValueError(f"Unsupported curated format in {path}")

    if not isinstance(edges_raw, list):
        raise ValueError(f"Unsupported curated format in {path} (missing edges list)")

    out: list[CuratedEdge] = []
    for e in edges_raw:
        try:
            caller = e["caller"]
            callee = e["callee"]
            out.append(
                CuratedEdge(
                    caller_file=str(caller["file"]),
                    caller_symbol=str(caller["symbol"]),
                    callee_file=str(callee["file"]),
                    callee_symbol=str(callee["symbol"]),
                )
            )
        except Exception as exc:
            raise ValueError(f"Bad curated edge entry: {e!r}") from exc
    return corpus_id, out


def _caller_set_from_impact_tree(tree: dict[str, Any]) -> set[tuple[str, str]]:
    callers = tree.get("callers") or []
    out: set[tuple[str, str]] = set()
    if not isinstance(callers, list):
        return out
    for c in callers:
        if not isinstance(c, dict):
            continue
        file = c.get("file")
        fn = c.get("function")
        if isinstance(file, str) and isinstance(fn, str):
            out.add((file, fn))
    return out


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


def _load_graph_cache(path: Path, *, lang: str) -> ProjectCallGraph | None:
    if not path.exists():
        return None
    try:
        cache_data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    langs = cache_data.get("languages") or []
    if isinstance(langs, list) and langs and lang not in langs and "all" not in langs:
        return None

    edges = cache_data.get("edges") or []
    if not isinstance(edges, list):
        return None

    graph = ProjectCallGraph()
    meta = cache_data.get("meta")
    if isinstance(meta, dict):
        graph.meta.update(meta)
    for e in edges:
        try:
            graph.add_edge(e["from_file"], e["from_func"], e["to_file"], e["to_func"])
        except Exception:
            continue
    return graph


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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark TS callgraph quality vs curated edges (edge recall + impact recall)."
    )
    ap.add_argument("--repo-root", required=True, help="Path to the corpus repo root to analyze.")
    ap.add_argument("--curated", required=True, help="Path to curated edges JSON.")
    ap.add_argument(
        "--cache-root",
        default=str(bench_cache_root(get_repo_root())),
        help="Index-mode cache root (recommended for repeatability).",
    )
    ap.add_argument("--index", default=None, help="Index id (e.g. repo:nextjs).")
    ap.add_argument(
        "--ts-trace",
        action="store_true",
        help="Enable TS trace collection (bounded sample) to help explain misses.",
    )
    ap.add_argument(
        "--mode",
        choices=["edges", "impact", "both"],
        default="both",
        help="Scoring mode: direct edge presence, impact queries, or both.",
    )
    ap.add_argument("--rebuild", action="store_true", help="Force a fresh graph build (ignore cached graph).")
    ap.add_argument(
        "--out",
        default=None,
        help="Write a JSON report to this path (default: benchmark/runs/<ts>-ts-curated-recall-<corpus>.json).",
    )
    ap.add_argument("--fail-under-edge-recall", type=float, default=None)
    ap.add_argument("--fail-under-impact-recall", type=float, default=None)
    args = ap.parse_args()

    tldr_repo_root = get_repo_root()
    corpus_root = Path(args.repo_root).resolve()
    curated_path = Path(args.curated).resolve()

    curated_corpus_id, curated_edges = _load_curated_edges(curated_path)
    corpus_id = curated_corpus_id or corpus_root.name
    if not curated_edges:
        raise SystemExit(f"error: no curated edges found in {curated_path}")

    # Prepare index-mode paths (optional but strongly recommended).
    cache_root_arg: str | None = None
    index_id_arg: str | None = None
    if args.cache_root:
        cache_root_arg = args.cache_root
        index_id_arg = args.index
    index_ctx = get_index_context(
        scan_root=corpus_root,
        cache_root_arg=cache_root_arg,
        index_id_arg=index_id_arg,
        allow_create=True,
    )
    index_paths = index_ctx.paths
    call_graph_cache = index_paths.call_graph if index_paths is not None else None

    protocol: dict[str, Any] = {
        "language": "typescript",
        "mode": args.mode,
        "ts_trace": bool(args.ts_trace),
        "rebuild": bool(args.rebuild),
        "curated": str(curated_path),
        "cache_root": str(index_ctx.cache_root) if index_ctx.cache_root is not None else None,
        "index_id": index_ctx.index_id,
        "index_key": index_ctx.index_key,
    }

    graph: ProjectCallGraph | None = None
    loaded_from_cache = False
    if (
        not args.rebuild
        and not args.ts_trace
        and call_graph_cache is not None
        and call_graph_cache.exists()
    ):
        graph = _load_graph_cache(call_graph_cache, lang="typescript")
        loaded_from_cache = graph is not None

    build_s = None
    if graph is None:
        t0 = time.monotonic()
        graph = build_project_call_graph(
            corpus_root,
            language="typescript",
            use_workspace_config=True,
            ts_trace=bool(args.ts_trace),
        )
        build_s = round(time.monotonic() - t0, 4)
        if call_graph_cache is not None:
            _write_graph_cache(call_graph_cache, graph, lang="typescript")

    meta = getattr(graph, "meta", {}) or {}
    build_info: dict[str, Any] = {
        "repo_root": str(corpus_root),
        "loaded_from_cache": loaded_from_cache,
        "build_s": build_s,
        "graph_source": meta.get("graph_source"),
        "incomplete": bool(meta.get("incomplete")),
        "edge_count": len(graph.edges),
    }
    if isinstance(meta.get("ts_projects"), list):
        build_info["ts_projects_ok"] = sum(
            1 for p in (meta.get("ts_projects") or []) if isinstance(p, dict) and p.get("status") == "ok"
        )
        build_info["ts_projects_err"] = sum(
            1 for p in (meta.get("ts_projects") or []) if isinstance(p, dict) and p.get("status") != "ok"
        )

    results: dict[str, Any] = {"build": build_info}
    ok = True

    if args.mode in ("edges", "both"):
        present = 0
        missing: list[dict[str, Any]] = []
        for e in curated_edges:
            tup = (e.caller_file, e.caller_symbol, e.callee_file, e.callee_symbol)
            if tup in graph.edges:
                present += 1
            else:
                missing.append(
                    {
                        "caller": f"{e.caller_file}:{e.caller_symbol}",
                        "callee": f"{e.callee_file}:{e.callee_symbol}",
                    }
                )
        recall = present / len(curated_edges) if curated_edges else 1.0
        results["edges"] = {
            "present": present,
            "total": len(curated_edges),
            "recall": round(recall, 4),
            "missing_count": len(missing),
            "missing_sample": missing[:50],
        }
        if missing:
            ok = False

    if args.mode in ("impact", "both"):
        expected_by_callee: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for e in curated_edges:
            expected_by_callee[(e.callee_file, e.callee_symbol)].add((e.caller_file, e.caller_symbol))

        total_expected = 0
        total_found = 0
        per_target: list[dict[str, Any]] = []

        for (tf, tfn), expected_callers in sorted(expected_by_callee.items()):
            total_expected += len(expected_callers)

            res = impact_analysis(graph, tfn, max_depth=1, target_file=tf)
            if "error" in res:
                per_target.append(
                    {
                        "target": f"{tf}:{tfn}",
                        "expected": len(expected_callers),
                        "found": 0,
                        "recall": 0.0,
                        "error": res.get("error"),
                    }
                )
                continue

            key = f"{tf}:{tfn}"
            targets = res.get("targets") or {}
            tree = targets.get(key)
            if not isinstance(tree, dict):
                if len(targets) == 1:
                    tree = next(iter(targets.values()))
                else:
                    tree = None

            actual_callers: set[tuple[str, str]] = set()
            if isinstance(tree, dict):
                actual_callers = _caller_set_from_impact_tree(tree)

            found = len(expected_callers & actual_callers)
            total_found += found
            recall = found / len(expected_callers) if expected_callers else 1.0
            missing_callers = expected_callers - actual_callers

            per_target.append(
                {
                    "target": f"{tf}:{tfn}",
                    "expected": len(expected_callers),
                    "found": found,
                    "recall": round(recall, 4),
                    "missing_callers_count": len(missing_callers),
                    "missing_callers_sample": [f"{cf}:{cfn}" for (cf, cfn) in sorted(missing_callers)][:25],
                }
            )

        overall_recall = total_found / total_expected if total_expected else 1.0
        results["impact"] = {
            "found": total_found,
            "total": total_expected,
            "recall": round(overall_recall, 4),
            "targets": len(expected_by_callee),
            "per_target": per_target,
        }
        if total_found != total_expected:
            ok = False

    if args.ts_trace and isinstance(meta.get("ts_trace"), list):
        trace = meta.get("ts_trace") or []
        trace_count = meta.get("ts_trace_count")
        if not isinstance(trace_count, int):
            trace_count = len(trace)
        reasons = Counter()
        for item in trace:
            if not isinstance(item, dict):
                continue
            reasons[str(item.get("reason") or "")] += 1
        results["ts_trace"] = {
            "sample_len": len(trace),
            "skipped_count": trace_count,
            "top_reasons": reasons.most_common(15),
        }

    # Optional decision gates
    edge_recall = (results.get("edges") or {}).get("recall")
    impact_recall = (results.get("impact") or {}).get("recall")
    if args.fail_under_edge_recall is not None and isinstance(edge_recall, (int, float)):
        if float(edge_recall) < float(args.fail_under_edge_recall):
            ok = False
    if args.fail_under_impact_recall is not None and isinstance(impact_recall, (int, float)):
        if float(impact_recall) < float(args.fail_under_impact_recall):
            ok = False
    results["ok"] = bool(ok) and not bool(build_info.get("incomplete"))

    report = make_report(
        phase="phase1_ts_curated_recall",
        meta=gather_meta(
            tldr_repo_root=tldr_repo_root,
            corpus_id=corpus_id,
            corpus_root=corpus_root,
        ),
        protocol=protocol,
        results=results,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        ts = now_utc_compact()
        out_path = bench_runs_root(tldr_repo_root) / f"{ts}-ts-curated-recall-{corpus_id}.json"
    write_report(out_path, report)
    print(out_path)
    return 0 if report["results"].get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

