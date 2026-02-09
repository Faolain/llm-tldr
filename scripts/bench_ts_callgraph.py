from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from tldr.analysis import impact_analysis
from tldr.cross_file_calls import build_project_call_graph
from tldr.patch import patch_typescript_resolved_dirty_files


def _time_it(fn):
    t0 = time.monotonic()
    out = fn()
    return time.monotonic() - t0, out


def _bench_build(root: Path, *, ts_trace: bool = False) -> dict[str, Any]:
    dt, graph = _time_it(
        lambda: build_project_call_graph(
            root,
            language="typescript",
            use_workspace_config=True,
            ts_trace=ts_trace,
        )
    )
    meta = getattr(graph, "meta", {}) or {}
    return {
        "root": str(root),
        "build_s": round(dt, 4),
        "graph_source": meta.get("graph_source"),
        "incomplete": bool(meta.get("incomplete")),
        "edge_count": len(graph.edges),
        "graph": graph,
    }


def _bench_impacts(graph, targets: list[tuple[str, str | None]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sym, file_filter in targets:
        dt, res = _time_it(lambda: impact_analysis(graph, sym, max_depth=1, target_file=file_filter))
        out.append(
            {
                "symbol": sym,
                "file_filter": file_filter,
                "impact_s": round(dt, 6),
                "ok": "error" not in res,
            }
        )
    return out


def _bench_daemon_peerbit(
    root: Path,
    targets: list[tuple[str, str | None]],
) -> dict[str, Any]:
    from tldr.daemon import query_daemon, start_daemon

    # Ensure daemon is running, but keep benchmark output JSON-only.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        start_daemon(root, foreground=False)

    warm_dt, warm_res = _time_it(
        lambda: query_daemon(
            root,
            {
                "cmd": "warm",
                "language": "typescript",
            },
        )
    )

    impacts: list[dict[str, Any]] = []
    for sym, file_filter in targets:
        cmd: dict[str, Any] = {
            "cmd": "impact",
            "func": sym,
            "depth": 1,
        }
        if file_filter is not None:
            cmd["file"] = file_filter

        dt, res = _time_it(lambda: query_daemon(root, cmd))
        ok = bool(res.get("status") == "ok")
        result = res.get("result")
        if isinstance(result, dict) and "error" in result:
            ok = False

        impacts.append(
            {
                "symbol": sym,
                "file_filter": file_filter,
                "impact_s": round(dt, 6),
                "ok": ok,
            }
        )

    return {
        "warm_s": round(warm_dt, 2),
        "warm_status": warm_res.get("status"),
        "warm_edges": warm_res.get("edges"),
        "impacts": impacts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick TS callgraph benchmarks (manual).")
    ap.add_argument(
        "--fixture-root",
        default=str(Path("tests/fixtures/ts-monorepo")),
        help="TS fixture root to benchmark (default: tests/fixtures/ts-monorepo).",
    )
    ap.add_argument(
        "--peerbit-root",
        default=None,
        help="Optional Peerbit repo root to benchmark (builds can be slow).",
    )
    ap.add_argument(
        "--peerbit-daemon",
        action="store_true",
        help="Benchmark warm + 5 impact queries via the TLDR daemon (Peerbit only).",
    )
    ap.add_argument(
        "--ts-trace",
        action="store_true",
        help="Enable TS trace during builds (can be slower).",
    )
    args = ap.parse_args()

    fixture_src = Path(args.fixture_root).resolve()
    if not fixture_src.exists():
        raise ValueError(f"Fixture root not found: {fixture_src}")

    # Run fixture benchmarks in a temp copy so we can safely do a "touch/edit" rebuild.
    with tempfile.TemporaryDirectory(prefix="llm-tldr-bench-fixture-") as td:
        fixture_tmp = Path(td) / "ts-monorepo"
        shutil.copytree(fixture_src, fixture_tmp)

        fixture_build = _bench_build(fixture_tmp, ts_trace=bool(args.ts_trace))
        graph = fixture_build.pop("graph")

        fixture_impacts = _bench_impacts(
            graph,
            [
                ("foo", "packages/a/src/foo.ts"),
                ("createCache", "packages/a/src/arrow.ts"),
                ("typedFn", "packages/a/src/typed.ts"),
                ("C.m", "packages/a/src/dispatch.ts"),
                ("onFoo", "packages/a/src/handlers.ts"),
            ],
        )

        # True incremental patch benchmark: recompute edges just for the touched file
        # (TypeScript-resolved mode) and patch the existing graph in place.
        touched = fixture_tmp / "packages" / "a" / "src" / "foo.ts"
        touched.write_text(touched.read_text() + "\n// bench_touch\n")

        patch_dt = None
        patch_meta: dict[str, Any] | None = None
        patch_err: str | None = None
        try:
            patch_dt, patch_meta = _time_it(
                lambda: patch_typescript_resolved_dirty_files(
                    graph,
                    fixture_tmp,
                    ["packages/a/src/foo.ts"],
                    trace=False,
                    timeout_s=60,
                )
            )
        except Exception as exc:
            patch_err = str(exc)

        # "Incremental" rebuild benchmark (today: TS uses full rebuild when dirty; this
        # still gives us a consistent measurement point).
        fixture_rebuild = _bench_build(fixture_tmp, ts_trace=bool(args.ts_trace))
        fixture_rebuild.pop("graph")

        print(
            json.dumps(
                {
                    "fixture": {
                        "build": {k: v for k, v in fixture_build.items() if k != "graph"},
                        "impacts": fixture_impacts,
                        "incremental_patch_after_touch": {
                            "patch_s": round(patch_dt, 4) if isinstance(patch_dt, (int, float)) else None,
                            "meta": patch_meta,
                            "error": patch_err,
                        },
                        "rebuild_after_touch": fixture_rebuild,
                    }
                },
                indent=2,
            )
        )

    if args.peerbit_root:
        peerbit_root = Path(args.peerbit_root).resolve()
        peerbit_targets = [
            ("Peerbit.dial", "packages/clients/peerbit/src/peer.ts"),
            ("createLibp2pExtended", "packages/clients/peerbit/src/libp2p.ts"),
            ("Handler.open", "packages/programs/program/program/src/handler.ts"),
            ("getKeypairFromPrivateKey", "packages/utils/crypto/src/from.ts"),
            ("LevelStore.open", "packages/utils/any-store/any-store/src/level.ts"),
        ]

        if args.peerbit_daemon:
            res = _bench_daemon_peerbit(peerbit_root, peerbit_targets)
            print(json.dumps({"peerbit_daemon": res}, indent=2))
        else:
            peerbit_build = _bench_build(peerbit_root, ts_trace=bool(args.ts_trace))
            peerbit_graph = peerbit_build.pop("graph")
            peerbit_impacts = _bench_impacts(peerbit_graph, peerbit_targets)
            print(
                json.dumps(
                    {
                        "peerbit": {
                            "build": peerbit_build,
                            "impacts": peerbit_impacts,
                        }
                    },
                    indent=2,
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
