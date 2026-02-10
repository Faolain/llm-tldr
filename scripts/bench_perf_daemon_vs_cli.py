#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
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

from tldr.daemon.startup import query_daemon, start_daemon, stop_daemon
from tldr.indexing.index import IndexContext, get_index_context
from tldr.indexing.management import get_index_info, list_indexes


@dataclass(frozen=True)
class Target:
    file: str
    symbol: str


@dataclass(frozen=True)
class CommandSpec:
    name: str
    cli_args: list[str]
    daemon_cmd: dict[str, Any]
    iterations: int | None = None  # override global iterations


def _load_curated_edges(curated_path: Path) -> list[dict[str, Any]]:
    data = json.loads(curated_path.read_text())
    edges = data.get("edges") if isinstance(data, dict) else data
    if not isinstance(edges, list):
        raise ValueError(f"Bad curated edges file: {curated_path}")
    out: list[dict[str, Any]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        caller = e.get("caller")
        callee = e.get("callee")
        if not isinstance(caller, dict) or not isinstance(callee, dict):
            continue
        if not isinstance(caller.get("file"), str) or not isinstance(caller.get("symbol"), str):
            continue
        if not isinstance(callee.get("file"), str) or not isinstance(callee.get("symbol"), str):
            continue
        out.append(e)
    return out


def _pick_smallest_callee(edges: list[dict[str, Any]]) -> Target | None:
    # Prefer a callee with few curated callers to keep perf payloads bounded.
    counts: dict[tuple[str, str], int] = {}
    for e in edges:
        callee = e.get("callee") or {}
        if not isinstance(callee, dict):
            continue
        tf = callee.get("file")
        ts = callee.get("symbol")
        if not isinstance(tf, str) or not isinstance(ts, str):
            continue
        key = (tf, ts)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    best = min(counts.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1]))[0]
    return Target(file=best[0], symbol=best[1])


def _escape_regex_literal(text: str) -> str:
    # Basic regex escaping for ripgrep and Python's re syntax (good enough here).
    special = set("\\.^$|?*+()[]{}")
    return "".join("\\" + ch if ch in special else ch for ch in text)


def _search_pattern_for_symbol(symbol: str) -> str:
    # Keep it simple and deterministic: search for the last identifier token.
    leaf = symbol.split(".")[-1]
    return r"\b" + _escape_regex_literal(leaf) + r"\b"


def _exts_for_language(language: str) -> list[str]:
    lang = language.lower()
    if lang in {"typescript", "ts"}:
        return [".ts", ".tsx"]
    if lang in {"javascript", "js"}:
        return [".js", ".jsx", ".ts", ".tsx"]
    if lang in {"python", "py"}:
        return [".py"]
    return []


def _cli_base_cmd(repo_root: Path, *, cache_root: str | None, index_id: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "tldr.cli"]
    if cache_root:
        cmd.extend(["--cache-root", cache_root])
    if index_id:
        cmd.extend(["--index", index_id])
    cmd.extend(["--scan-root", str(repo_root)])
    return cmd


def _run_cli_once(cmd: list[str], *, cwd: Path) -> tuple[int, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return int(proc.returncode), dt_ms


def _run_daemon_once(
    repo_root: Path,
    *,
    index_ctx: IndexContext,
    command: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    try:
        res = query_daemon(repo_root, command, index_ctx=index_ctx)
    except Exception as exc:
        res = {"status": "error", "message": str(exc), "exc_type": type(exc).__name__}
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return res, dt_ms


def _summarize_samples_ms(samples_ms: list[float]) -> dict[str, Any]:
    if not samples_ms:
        return {}
    mean = statistics.mean(samples_ms)
    stdev = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    ps = percentiles(samples_ms)
    return {
        "mean": round(mean, 4),
        "stdev": round(stdev, 4),
        **{k: round(v, 4) for k, v in ps.items()},
    }


def _bench_cli(
    *,
    repo_root: Path,
    cmd: list[str],
    iterations: int,
) -> dict[str, Any]:
    # Warmup (not counted)
    _run_cli_once(cmd, cwd=repo_root)

    samples_ms: list[float] = []
    ok = True
    errors: list[str] = []
    for _ in range(iterations):
        rc, dt_ms = _run_cli_once(cmd, cwd=repo_root)
        samples_ms.append(dt_ms)
        if rc != 0:
            ok = False
            errors.append(f"returncode={rc}")
    return {
        "ok": ok,
        "iterations": iterations,
        "samples_ms": [round(x, 4) for x in samples_ms],
        "stats_ms": _summarize_samples_ms(samples_ms),
        "errors": errors[:5],
    }


def _bench_daemon(
    repo_root: Path,
    *,
    index_ctx: IndexContext,
    command: dict[str, Any],
    iterations: int,
) -> dict[str, Any]:
    # Warmup (not counted)
    _run_daemon_once(repo_root, index_ctx=index_ctx, command=command)

    samples_ms: list[float] = []
    ok = True
    errors: list[str] = []
    for _ in range(iterations):
        res, dt_ms = _run_daemon_once(repo_root, index_ctx=index_ctx, command=command)
        samples_ms.append(dt_ms)
        if res.get("status") != "ok":
            ok = False
            msg = res.get("message") or res.get("error") or "status!=ok"
            errors.append(str(msg))
    return {
        "ok": ok,
        "iterations": iterations,
        "samples_ms": [round(x, 4) for x in samples_ms],
        "stats_ms": _summarize_samples_ms(samples_ms),
        "errors": errors[:5],
    }


def _speedup(cli_stats: dict[str, Any], daemon_stats: dict[str, Any]) -> float | None:
    try:
        cli_mean = float(cli_stats["stats_ms"]["mean"])
        daemon_mean = float(daemon_stats["stats_ms"]["mean"])
    except Exception:
        return None
    if daemon_mean <= 0:
        return None
    return round(cli_mean / daemon_mean, 4)


def _maybe_warm_call_graph(
    repo_root: Path,
    *,
    index_ctx: IndexContext,
    language: str,
) -> dict[str, Any]:
    # If a call graph cache already exists on disk for this index, assume warm.
    index_paths = index_ctx.paths
    cache_file = index_paths.call_graph if index_paths is not None else None
    if cache_file is not None and cache_file.exists():
        try:
            size = cache_file.stat().st_size
        except OSError:
            size = None
        return {"skipped": True, "reason": "call graph cache already exists", "call_graph_cache_bytes": size}

    res, dt_ms = _run_daemon_once(
        repo_root,
        index_ctx=index_ctx,
        command={"cmd": "warm", "language": language},
    )
    out: dict[str, Any] = {
        "skipped": False,
        "status": res.get("status"),
        "edges": res.get("edges"),
        "files": res.get("files"),
        "wall_time_ms": round(dt_ms, 4),
    }
    if cache_file is not None and cache_file.exists():
        try:
            out["call_graph_cache_bytes"] = cache_file.stat().st_size
        except OSError:
            out["call_graph_cache_bytes"] = None
    return out


def _wait_for_daemon_ready(
    repo_root: Path,
    *,
    index_ctx: IndexContext,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_err: str | None = None
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        try:
            res = query_daemon(repo_root, {"cmd": "ping"}, index_ctx=index_ctx)
            if res.get("status") == "ok":
                return {"ok": True, "attempts": attempts}
            last_err = res.get("message") or "status!=ok"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(0.05)
    return {"ok": False, "attempts": attempts, "last_error": last_err}


def _build_command_set(
    *,
    repo_root: Path,
    target: Target | None,
    language: str,
    iterations: int,
    include_calls: bool,
    calls_iterations: int,
) -> tuple[list[CommandSpec], dict[str, Any]]:
    exts = _exts_for_language(language)
    symbol = target.symbol if target is not None else "main"
    file_filter = target.file if target is not None else ""
    extract_file = str((repo_root / target.file).resolve()) if target is not None else None

    protocol_details: dict[str, Any] = {
        "language": language,
        "target": {"file": target.file, "symbol": target.symbol} if target is not None else None,
        "extensions": exts,
    }

    search_pattern = _search_pattern_for_symbol(symbol)

    # NOTE: Keep commands comparable between CLI and daemon: only use parameters supported by both.
    if extract_file is None:
        # Best-effort fallback: avoid an expensive repo-wide walk by checking a few common paths.
        candidates = [
            repo_root / "src" / "index.ts",
            repo_root / "src" / "index.js",
            repo_root / "index.ts",
            repo_root / "index.js",
        ]
        for c in candidates:
            if c.exists():
                extract_file = str(c.resolve())
                break

    tree_cli_args = ["tree", "."]
    tree_daemon_cmd: dict[str, Any] = {"cmd": "tree", "exclude_hidden": True}
    if exts:
        tree_cli_args = ["tree", "--ext", *exts, "."]
        tree_daemon_cmd["extensions"] = exts

    cmds: list[CommandSpec] = [
        CommandSpec(
            name="search",
            cli_args=["search", "--max", "20", search_pattern, "."],
            daemon_cmd={"cmd": "search", "pattern": search_pattern, "max_results": 20},
            iterations=iterations,
        ),
        CommandSpec(
            name="extract",
            cli_args=["extract", extract_file or str(repo_root / "README.md")],
            daemon_cmd={"cmd": "extract", "file": extract_file} if extract_file else {"cmd": "extract", "file": str(repo_root / "README.md")},
            iterations=iterations,
        ),
        CommandSpec(
            name="impact",
            cli_args=[
                "impact",
                "--lang",
                language,
                "--depth",
                "1",
                "--file",
                file_filter,
                symbol,
                ".",
            ],
            daemon_cmd={"cmd": "impact", "language": language, "func": symbol, "depth": 1, "file": file_filter},
            iterations=iterations,
        ),
        CommandSpec(
            name="tree",
            cli_args=tree_cli_args,
            daemon_cmd=tree_daemon_cmd,
            iterations=iterations,
        ),
        CommandSpec(
            name="structure",
            cli_args=["structure", "--lang", language, "--max", "50", "."],
            daemon_cmd={"cmd": "structure", "language": language, "max_results": 50},
            iterations=iterations,
        ),
        CommandSpec(
            name="context",
            cli_args=["context", "--lang", language, "--depth", "1", symbol],
            daemon_cmd={"cmd": "context", "language": language, "depth": 1, "entry": symbol},
            iterations=iterations,
        ),
    ]
    if include_calls:
        # Warning: daemon-side `calls` currently rebuilds the call graph on every request.
        # Keep iterations low to avoid turning this microbench into an index build benchmark.
        cmds.append(
            CommandSpec(
                name="calls",
                cli_args=["calls", "--lang", language, "."],
                daemon_cmd={"cmd": "calls", "language": language},
                iterations=calls_iterations,
            )
        )
    return cmds, protocol_details


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 perf microbench: daemon vs CLI latency for key commands.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", default=None, help="Corpus id from benchmarks/corpora.json (e.g. peerbit, nextjs).")
    group.add_argument("--repo-root", default=None, help="Path to the corpus repo root.")
    ap.add_argument("--curated", default=None, help="Curated edges JSON to derive a stable benchmark target.")
    ap.add_argument(
        "--cache-root",
        default=str(bench_cache_root(get_repo_root())),
        help="Index-mode cache root (default: benchmark/cache-root).",
    )
    ap.add_argument("--index", default=None, help="Index id (default: repo:<corpus>).")
    ap.add_argument("--lang", default="typescript", help="Language for commands (default: typescript).")
    ap.add_argument("--iterations", type=int, default=10, help="Iterations per command (default: 10).")
    ap.add_argument("--include-calls", action="store_true", help="Also benchmark `calls` (expensive; default off).")
    ap.add_argument(
        "--calls-iterations",
        type=int,
        default=1,
        help="Iterations for `calls` when enabled (default: 1).",
    )
    ap.add_argument(
        "--include-semantic",
        action="store_true",
        help="Also benchmark `semantic search` (only runs if semantic index artifacts exist).",
    )
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

    curated_path = Path(args.curated).resolve() if args.curated else None
    if curated_path is None and default_curated and default_curated.exists():
        curated_path = default_curated

    target: Target | None = None
    if curated_path is not None and curated_path.exists():
        try:
            edges = _load_curated_edges(curated_path)
            target = _pick_smallest_callee(edges)
        except Exception:
            target = None

    index_id = args.index or default_index_id
    index_ctx = get_index_context(
        scan_root=repo_root,
        cache_root_arg=args.cache_root,
        index_id_arg=index_id,
        allow_create=True,
    )

    # Start daemon and make sure a call graph cache exists so impact/context are "warm".
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        stop_daemon(repo_root, index_ctx=index_ctx)
        start_daemon(repo_root, foreground=False, index_ctx=index_ctx)

    daemon_ready = _wait_for_daemon_ready(repo_root, index_ctx=index_ctx)
    warm_info = _maybe_warm_call_graph(repo_root, index_ctx=index_ctx, language=str(args.lang))

    cmds, protocol_details = _build_command_set(
        repo_root=repo_root,
        target=target,
        language=str(args.lang),
        iterations=int(args.iterations),
        include_calls=bool(args.include_calls),
        calls_iterations=int(args.calls_iterations),
    )

    cli_base = _cli_base_cmd(repo_root, cache_root=str(index_ctx.cache_root) if index_ctx.cache_root else None, index_id=index_ctx.index_id)

    semantic_status: dict[str, Any] = {"enabled": bool(args.include_semantic)}
    if args.include_semantic:
        index_paths = index_ctx.paths
        semantic_ok = (
            index_paths is not None
            and index_paths.semantic_faiss.exists()
            and index_paths.semantic_metadata.exists()
        )
        if semantic_ok:
            leaf = (target.symbol if target else "entry").split(".")[-1]
            query = f"{leaf} implementation"
            cmds.append(
                CommandSpec(
                    name="semantic_search",
                    cli_args=["semantic", "search", "--k", "10", "--lang", str(args.lang), query],
                    daemon_cmd={"cmd": "semantic", "action": "search", "language": str(args.lang), "query": query, "k": 10},
                    iterations=int(args.iterations),
                )
            )
            semantic_status["ok"] = True
            semantic_status["query"] = query
        else:
            semantic_status["ok"] = False
            semantic_status["skipped"] = True
            semantic_status["reason"] = "semantic index artifacts missing (run `tldrf semantic index` first)"

    results: list[dict[str, Any]] = []
    for c in cmds:
        cli_cmd = cli_base + c.cli_args
        daemon_cmd = dict(c.daemon_cmd)

        daemon_res = _bench_daemon(
            repo_root,
            index_ctx=index_ctx,
            command=daemon_cmd,
            iterations=int(c.iterations or args.iterations),
        )
        cli_res = _bench_cli(
            repo_root=repo_root,
            cmd=cli_cmd,
            iterations=int(c.iterations or args.iterations),
        )
        results.append(
            {
                "cmd": c.name,
                "daemon": daemon_res,
                "cli": cli_res,
                "speedup": _speedup(cli_res, daemon_res),
                "daemon_cmd": daemon_cmd,
                "cli_cmd": cli_cmd,
            }
        )

    # Stop daemon to avoid side effects for other work.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        stop_daemon(repo_root, index_ctx=index_ctx)

    # Cache sizing / index artifacts (via the same JSON contract as `tldrf index list/info`).
    cache_root_path = index_ctx.cache_root
    index_mgmt: dict[str, Any] | None = None
    if cache_root_path is not None:
        try:
            index_ref = index_ctx.index_id or index_ctx.index_key
            if not index_ref:
                raise ValueError("index context missing index_id/index_key")
            index_mgmt = {
                "list": list_indexes(cache_root_path),
                "info": get_index_info(cache_root_path, str(index_ref)),
            }
        except Exception as exc:
            index_mgmt = {"error": str(exc)}

    report = make_report(
        phase="phase3_perf_daemon_vs_cli",
        meta=gather_meta(
            tldr_repo_root=tldr_repo_root,
            corpus_id=corpus_id,
            corpus_root=repo_root,
        ),
        protocol={
            "cache_root": str(index_ctx.cache_root) if index_ctx.cache_root is not None else None,
            "index_id": index_ctx.index_id,
            "iterations": int(args.iterations),
            "include_calls": bool(args.include_calls),
            "calls_iterations": int(args.calls_iterations),
            "semantic": semantic_status,
            "curated": str(curated_path) if curated_path is not None else None,
            "daemon_ready": daemon_ready,
            "warm_call_graph": warm_info,
            **protocol_details,
        },
        results={
            "results": results,
            "index_mgmt": index_mgmt,
        },
    )

    if args.out:
        out_path = Path(args.out)
    else:
        ts = now_utc_compact()
        out_path = bench_runs_root(tldr_repo_root) / f"{ts}-perf-daemon-vs-cli-{corpus_id}.json"
    write_report(out_path, report)
    print(out_path)

    ok = all(r.get("daemon", {}).get("ok") and r.get("cli", {}).get("ok") for r in report["results"]["results"])
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
