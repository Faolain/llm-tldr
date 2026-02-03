#!/usr/bin/env python3
"""Benchmark legacy vs index-mode semantic indexing for dependency sources.

Runs three benchmarks:
1) Parity: same corpus indexed legacy vs index mode (recall@k, MRR).
2) Workflow: main-repo index vs dependency index (dependency queries).
3) Storage + time: index time and disk usage for each index.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class RunResult:
    stdout: str
    stderr: str
    duration_s: float


@dataclass
class SearchMetrics:
    recall_at_k: float
    mrr: float
    hits: int
    total: int
    avg_time_s: float | None
    per_query: list[dict[str, Any]]


def _run(cmd: list[str], env: dict[str, str]) -> RunResult:
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    duration = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return RunResult(stdout=proc.stdout, stderr=proc.stderr, duration_s=duration)


def _run_tldr(args: list[str], env: dict[str, str]) -> RunResult:
    cmd = [sys.executable, "-m", "tldr.cli", *args]
    return _run(cmd, env)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                if file_path.is_symlink():
                    continue
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _load_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("queries must be a JSON list")
    return data


def _resolve_package_root(dep: str) -> Path:
    mod = importlib.import_module(dep)
    mod_path = Path(mod.__file__).resolve()
    if mod_path.name == "__init__.py":
        return mod_path.parent
    if (mod_path.parent / "__init__.py").exists():
        return mod_path.parent
    return mod_path


def _copy_source(src: Path, dest: Path) -> Path:
    if src.is_dir():
        shutil.copytree(src, dest)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest / src.name)
    return dest


def _parse_indexed_count(output: str) -> int | None:
    match = re.search(r"Indexed (\d+) code units", output)
    if not match:
        return None
    return int(match.group(1))


def _evaluate_queries(
    queries: Iterable[dict[str, Any]],
    search_fn,
    k: int,
) -> SearchMetrics:
    hits = 0
    total = 0
    rr_sum = 0.0
    time_hits: list[float] = []
    per_query: list[dict[str, Any]] = []

    for entry in queries:
        query = entry["query"]
        expected = entry["expect_path_contains"]
        total += 1
        results, duration_s = search_fn(query, k)

        hit_rank = None
        expected_norm = expected.replace("\\", "/")
        expected_name = Path(expected_norm).name
        for idx, result in enumerate(results, start=1):
            path = result.get("file") or ""
            path_norm = path.replace("\\", "/")
            path_name = Path(path_norm).name
            if expected_norm in path_norm or expected_name == path_name:
                hit_rank = idx
                break

        if hit_rank is not None:
            hits += 1
            rr_sum += 1.0 / hit_rank
            time_hits.append(duration_s)

        per_query.append(
            {
                "query": query,
                "expected": expected,
                "hit_rank": hit_rank,
                "duration_s": duration_s,
                "top_path": results[0]["file"] if results else None,
            }
        )

    recall = hits / total if total else 0.0
    mrr = rr_sum / total if total else 0.0
    avg_time = sum(time_hits) / len(time_hits) if time_hits else None

    return SearchMetrics(
        recall_at_k=recall,
        mrr=mrr,
        hits=hits,
        total=total,
        avg_time_s=avg_time,
        per_query=per_query,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark isolated indexes")
    parser.add_argument("--dep", default="requests", help="Dependency import name")
    parser.add_argument(
        "--queries",
        default="scripts/bench_queries.json",
        help="Path to queries JSON",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k for recall")
    parser.add_argument("--lang", default="python", help="Language")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model",
    )
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep temp directories (print paths)",
    )
    args = parser.parse_args()

    queries = _load_queries(Path(args.queries))
    dep_root = _resolve_package_root(args.dep)

    env = os.environ.copy()
    env.setdefault("TLDR_AUTO_DOWNLOAD", "1")

    with tempfile.TemporaryDirectory(prefix="tldr-bench-") as tmp:
        tmp_path = Path(tmp)
        legacy_src = tmp_path / f"{args.dep}-legacy"
        index_src = tmp_path / f"{args.dep}-index"
        cache_root = tmp_path / "cache-root"

        _copy_source(dep_root, legacy_src)
        _copy_source(dep_root, index_src)

        # Parity: legacy index
        legacy_index_start = time.perf_counter()
        legacy_index_res = _run_tldr(
            [
                "semantic",
                "index",
                str(legacy_src),
                "--lang",
                args.lang,
                "--model",
                args.model,
                "--device",
                args.device,
                "--no-ignore",
            ],
            env,
        )
        legacy_index_time = time.perf_counter() - legacy_index_start
        legacy_units = _parse_indexed_count(legacy_index_res.stdout)

        # Parity: index mode
        dep_index_id = f"dep:{args.dep}"
        index_index_start = time.perf_counter()
        index_index_res = _run_tldr(
            [
                "--cache-root",
                str(cache_root),
                "--index",
                dep_index_id,
                "semantic",
                "index",
                str(index_src),
                "--lang",
                args.lang,
                "--model",
                args.model,
                "--device",
                args.device,
                "--no-ignore",
            ],
            env,
        )
        index_index_time = time.perf_counter() - index_index_start
        index_units = _parse_indexed_count(index_index_res.stdout)

        def legacy_search(query: str, k: int):
            res = _run_tldr(
                [
                    "semantic",
                    "search",
                    query,
                    "--path",
                    str(legacy_src),
                    "--k",
                    str(k),
                    "--device",
                    args.device,
                ],
                env,
            )
            results = json.loads(res.stdout)
            return results, res.duration_s

        def index_search(query: str, k: int):
            res = _run_tldr(
                [
                    "--cache-root",
                    str(cache_root),
                    "--index",
                    dep_index_id,
                    "semantic",
                    "search",
                    query,
                    "--path",
                    str(index_src),
                    "--k",
                    str(k),
                    "--device",
                    args.device,
                ],
                env,
            )
            results = json.loads(res.stdout)
            return results, res.duration_s

        legacy_metrics = _evaluate_queries(queries, legacy_search, args.k)
        index_metrics = _evaluate_queries(queries, index_search, args.k)

        legacy_cache = legacy_src / ".tldr"
        legacy_size = _dir_size_bytes(legacy_cache) if legacy_cache.exists() else 0

        # Workflow benchmark: main repo index vs dependency index
        repo_root = Path.cwd()
        repo_scan = repo_root / "tldr"
        main_index_id = "main:llm-tldr"
        main_index_start = time.perf_counter()
        _run_tldr(
            [
                "--cache-root",
                str(cache_root),
                "--index",
                main_index_id,
                "semantic",
                "index",
                str(repo_scan),
                "--lang",
                args.lang,
                "--model",
                args.model,
                "--device",
                args.device,
                "--no-ignore",
            ],
            env,
        )
        main_index_time = time.perf_counter() - main_index_start

        def main_search(query: str, k: int):
            res = _run_tldr(
                [
                    "--cache-root",
                    str(cache_root),
                    "--index",
                    main_index_id,
                    "semantic",
                    "search",
                    query,
                    "--path",
                    str(repo_scan),
                    "--k",
                    str(k),
                    "--device",
                    args.device,
                ],
                env,
            )
            results = json.loads(res.stdout)
            return results, res.duration_s

        main_metrics = _evaluate_queries(queries, main_search, args.k)

        index_list = _run_tldr(
            ["--cache-root", str(cache_root), "index", "list"], env
        )
        index_list_data = json.loads(index_list.stdout)
        index_sizes = {
            entry.get("index_id"): entry.get("size_bytes")
            for entry in index_list_data.get("indexes", [])
        }

        summary = {
            "dependency": args.dep,
            "dep_source": str(dep_root),
            "model": args.model,
            "device": args.device,
            "k": args.k,
            "paths": {
                "legacy_src": str(legacy_src),
                "index_src": str(index_src),
                "cache_root": str(cache_root),
                "repo_scan": str(repo_scan),
            },
            "parity": {
                "legacy": {
                    "index_time_s": legacy_index_time,
                    "indexed_units": legacy_units,
                    "disk_usage_bytes": legacy_size,
                    "metrics": legacy_metrics.__dict__,
                },
                "index_mode": {
                    "index_time_s": index_index_time,
                    "indexed_units": index_units,
                    "disk_usage_bytes": index_sizes.get(dep_index_id),
                    "metrics": index_metrics.__dict__,
                },
            },
            "workflow": {
                "main_repo": {
                    "index_time_s": main_index_time,
                    "metrics": main_metrics.__dict__,
                },
                "dependency": {
                    "metrics": index_metrics.__dict__,
                },
            },
            "index_list": index_list_data,
        }

        print(json.dumps(summary, indent=2))

        if args.keep:
            print(f"\nKept temp data at: {tmp_path}")
        else:
            # TemporaryDirectory will clean up
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
