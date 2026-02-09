from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tldr.analysis import impact_analysis
from tldr.cross_file_calls import build_project_call_graph


def _load_curated_edges(path: Path) -> list[tuple[str, str, str, str]]:
    data: Any = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("edges"), list):
        edges = data["edges"]
    elif isinstance(data, list):
        edges = data
    else:
        raise ValueError(f"Unsupported curated format in {path}")

    out: list[tuple[str, str, str, str]] = []
    for e in edges:
        try:
            caller = e["caller"]
            callee = e["callee"]
            out.append(
                (
                    str(caller["file"]),
                    str(caller["symbol"]),
                    str(callee["file"]),
                    str(callee["symbol"]),
                )
            )
        except Exception as exc:
            raise ValueError(f"Bad curated edge entry: {e!r}") from exc
    return out


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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate TS-resolved callgraph recall on a local Peerbit checkout via a curated edge list.",
    )
    ap.add_argument(
        "--peerbit-root",
        default="/Users/aristotle/Documents/Projects/peerbit",
        help="Path to the Peerbit repo root (defaults to local dev path).",
    )
    ap.add_argument(
        "--curated",
        default=str(Path(__file__).with_name("peerbit_curated_edges.json")),
        help="Path to curated edges JSON (default: scripts/peerbit_curated_edges.json).",
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        help="Enable TS trace collection (bounded sample) to help explain misses.",
    )
    ap.add_argument(
        "--mode",
        choices=["edges", "impact", "both"],
        default="both",
        help="Validation mode: direct edge presence, impact queries, or both.",
    )
    ap.add_argument(
        "--show-misses",
        action="store_true",
        help="Print missing edges/callers.",
    )
    args = ap.parse_args()

    repo_root = Path(args.peerbit_root).resolve()
    curated_path = Path(args.curated).resolve()

    curated_edges = _load_curated_edges(curated_path)
    if not curated_edges:
        raise ValueError(f"No curated edges found in {curated_path}")

    t0 = time.monotonic()
    graph = build_project_call_graph(
        repo_root,
        language="typescript",
        use_workspace_config=True,
        ts_trace=bool(args.trace),
    )
    dt = time.monotonic() - t0

    meta = getattr(graph, "meta", {}) or {}
    print(
        json.dumps(
            {
                "repo_root": str(repo_root),
                "curated": str(curated_path),
                "build_s": round(dt, 2),
                "graph_source": meta.get("graph_source"),
                "incomplete": bool(meta.get("incomplete")),
                "edge_count": len(graph.edges),
                "ts_projects_ok": (
                    sum(1 for p in (meta.get("ts_projects") or []) if p.get("status") == "ok")
                    if isinstance(meta.get("ts_projects"), list)
                    else None
                ),
                "ts_projects_err": (
                    sum(1 for p in (meta.get("ts_projects") or []) if p.get("status") != "ok")
                    if isinstance(meta.get("ts_projects"), list)
                    else None
                ),
            },
            indent=2,
        )
    )

    ok = True

    if args.mode in ("edges", "both"):
        present = 0
        missing: list[tuple[str, str, str, str]] = []
        for e in curated_edges:
            if e in graph.edges:
                present += 1
            else:
                missing.append(e)

        recall = present / len(curated_edges)
        print(
            json.dumps(
                {
                    "mode": "edges",
                    "present": present,
                    "total": len(curated_edges),
                    "recall": round(recall, 4),
                },
                indent=2,
            )
        )
        if missing:
            ok = False
            if args.show_misses:
                for (cf, cfn, tf, tfn) in missing:
                    print(f"MISSING_EDGE {cf}:{cfn} -> {tf}:{tfn}")

    if args.mode in ("impact", "both"):
        expected_by_callee: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for cf, cfn, tf, tfn in curated_edges:
            expected_by_callee[(tf, tfn)].add((cf, cfn))

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
                # Best-effort fallback if the exact key changed or the analysis
                # matched multiple targets.
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
            entry: dict[str, Any] = {
                "target": f"{tf}:{tfn}",
                "expected": len(expected_callers),
                "found": found,
                "recall": round(recall, 4),
            }

            if args.show_misses and expected_callers - actual_callers:
                entry["missing_callers"] = [
                    f"{cf}:{cfn}" for (cf, cfn) in sorted(expected_callers - actual_callers)
                ]

            per_target.append(entry)

        overall_recall = total_found / total_expected if total_expected else 1.0
        print(
            json.dumps(
                {
                    "mode": "impact",
                    "found": total_found,
                    "total": total_expected,
                    "recall": round(overall_recall, 4),
                    "per_target": per_target,
                },
                indent=2,
            )
        )
        if total_found != total_expected:
            ok = False

    if args.trace and isinstance(meta.get("ts_trace"), list):
        trace = meta.get("ts_trace") or []
        trace_count = meta.get("ts_trace_count")
        if not isinstance(trace_count, int):
            trace_count = len(trace)
        reasons = Counter()
        for item in trace:
            if not isinstance(item, dict):
                continue
            reasons[str(item.get("reason") or "")] += 1
        top = reasons.most_common(10)
        print(
            json.dumps(
                {
                    "mode": "trace",
                    "sample_len": len(trace),
                    "skipped_count": trace_count,
                    "top_reasons": top,
                },
                indent=2,
            )
        )

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

