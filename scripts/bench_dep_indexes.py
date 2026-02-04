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
    negative_total: int
    total_queries: int
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
    negative_total = 0
    rr_sum = 0.0
    time_hits: list[float] = []
    per_query: list[dict[str, Any]] = []

    for entry in queries:
        query = entry["query"]
        expected = entry.get("expect_path_contains")
        is_negative = bool(entry.get("expect_none") or expected is None)
        if is_negative:
            negative_total += 1
        else:
            total += 1
        results, duration_s = search_fn(query, k)

        hit_rank = None
        if not is_negative and expected is not None:
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
                "negative": is_negative,
                "hit_rank": hit_rank,
                "duration_s": duration_s,
                "top_path": results[0]["file"] if results else None,
                "top_paths": [res.get("file") for res in results[:k]],
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
        negative_total=negative_total,
        total_queries=total + negative_total,
        avg_time_s=avg_time,
        per_query=per_query,
    )


def _path_within_root(
    path_str: str | None,
    *,
    result_root: Path,
    scope_root: Path,
) -> bool:
    if not path_str:
        return False
    try:
        path = Path(path_str)
        if not path.is_absolute():
            path = (result_root / path).resolve()
        else:
            path = path.resolve()
        scope_resolved = scope_root.resolve()
        return path == scope_resolved or str(path).startswith(
            f"{scope_resolved}{os.sep}"
        )
    except OSError:
        return False


def _scope_metrics(
    per_query: list[dict[str, Any]],
    *,
    result_root: Path,
    scope_root: Path,
) -> dict[str, Any]:
    total = len(per_query)
    in_scope = 0
    off_scope = 0
    in_scope_total = 0
    result_total = 0
    any_in_scope = 0
    negative_total = 0
    negative_any_in_scope = 0
    for entry in per_query:
        top_paths = entry.get("top_paths") or []
        if not top_paths:
            continue
        top_path = top_paths[0]
        if _path_within_root(
            top_path, result_root=result_root, scope_root=scope_root
        ):
            in_scope += 1
        else:
            off_scope += 1
        in_scope_k = 0
        for path in top_paths:
            result_total += 1
            if _path_within_root(
                path, result_root=result_root, scope_root=scope_root
            ):
                in_scope_total += 1
                in_scope_k += 1
        if in_scope_k > 0:
            any_in_scope += 1
        if entry.get("negative"):
            negative_total += 1
            if in_scope_k > 0:
                negative_any_in_scope += 1
    return {
        "result_root": str(result_root),
        "scope_root": str(scope_root),
        "scope_hit_rate": in_scope / total if total else 0.0,
        "off_scope_rate": off_scope / total if total else 0.0,
        "topk_in_scope_rate": in_scope_total / result_total if result_total else 0.0,
        "any_in_scope_rate": any_in_scope / total if total else 0.0,
        "negative_queries": negative_total,
        "negative_any_in_scope_rate": (
            negative_any_in_scope / negative_total if negative_total else None
        ),
        "off_scope_hits": off_scope,
        "total": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark isolated indexes")
    parser.add_argument("--dep", default="requests", help="Dependency import name (comma-separated for multiple)")
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
    dep_list = [d.strip() for d in args.dep.split(",") if d.strip()]
    if any("dep" not in entry for entry in queries) and len(dep_list) > 1:
        raise ValueError(
            "queries missing 'dep' while multiple deps requested; add dep fields"
        )
    deps_from_queries = sorted({entry.get("dep") for entry in queries if entry.get("dep")})
    deps = dep_list if dep_list else (deps_from_queries or [args.dep])
    missing_query_deps = sorted(set(deps_from_queries) - set(deps))
    if missing_query_deps:
        print(
            f"Skipping queries for deps not requested: {', '.join(missing_query_deps)}",
            file=sys.stderr,
        )

    env = os.environ.copy()
    env.setdefault("TLDR_AUTO_DOWNLOAD", "1")

    with tempfile.TemporaryDirectory(prefix="tldr-bench-") as tmp:
        tmp_path = Path(tmp)
        cache_root = tmp_path / "cache-root"

        dep_info: dict[str, dict[str, Any]] = {}

        for dep in deps:
            try:
                dep_root = _resolve_package_root(dep)
            except Exception as exc:
                print(f"Skipping dep {dep}: {exc}", file=sys.stderr)
                continue

            dep_queries = [
                entry for entry in queries if (entry.get("dep") or dep) == dep
            ]
            if not dep_queries:
                print(f"Skipping dep {dep}: no queries found", file=sys.stderr)
                continue

            legacy_src = tmp_path / f"{dep}-legacy"
            index_src = tmp_path / f"{dep}-index"

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
            dep_index_id = f"dep:{dep}"
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

            legacy_metrics = _evaluate_queries(dep_queries, legacy_search, args.k)
            index_metrics = _evaluate_queries(dep_queries, index_search, args.k)

            legacy_cache = legacy_src / ".tldr"
            legacy_size = _dir_size_bytes(legacy_cache) if legacy_cache.exists() else 0

            dep_info[dep] = {
                "dep_source": str(dep_root),
                "paths": {
                    "legacy_src": str(legacy_src),
                    "index_src": str(index_src),
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
                        "disk_usage_bytes": None,
                        "metrics": index_metrics.__dict__,
                    },
                },
                "index_id": dep_index_id,
            }

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

        index_list = _run_tldr(
            ["--cache-root", str(cache_root), "index", "list"], env
        )
        index_list_data = json.loads(index_list.stdout)
        index_sizes = {
            entry.get("index_id"): entry.get("size_bytes")
            for entry in index_list_data.get("indexes", [])
        }

        for dep, info in dep_info.items():
            dep_index_id = info["index_id"]
            info["parity"]["index_mode"]["disk_usage_bytes"] = index_sizes.get(
                dep_index_id
            )

        primary_dep = next(iter(dep_info.keys()), args.dep)
        primary_info = dep_info.get(primary_dep)

        main_metrics = None
        scope_dep = None
        scope_main = None
        if primary_info is not None:
            dep_queries = [
                entry
                for entry in queries
                if (entry.get("dep") or primary_dep) == primary_dep
            ]
            main_metrics = _evaluate_queries(dep_queries, main_search, args.k)
            scope_dep = _scope_metrics(
                primary_info["parity"]["index_mode"]["metrics"]["per_query"],
                result_root=Path(primary_info["paths"]["index_src"]),
                scope_root=Path(primary_info["paths"]["index_src"]),
            )
            scope_main = _scope_metrics(
                main_metrics.per_query,
                result_root=repo_scan,
                scope_root=Path(primary_info["paths"]["index_src"]),
            )

        cross_dependency = []
        dep_names = list(dep_info.keys())
        for i, dep_a in enumerate(dep_names):
            for dep_b in dep_names[i + 1 :]:
                info_a = dep_info[dep_a]
                info_b = dep_info[dep_b]
                dep_a_queries = [
                    entry
                    for entry in queries
                    if (entry.get("dep") or dep_a) == dep_a
                ]

                def dep_b_search(query: str, k: int):
                    res = _run_tldr(
                        [
                            "--cache-root",
                            str(cache_root),
                            "--index",
                            info_b["index_id"],
                            "semantic",
                            "search",
                            query,
                            "--path",
                            info_b["paths"]["index_src"],
                            "--k",
                            str(k),
                            "--device",
                            args.device,
                        ],
                        env,
                    )
                    results = json.loads(res.stdout)
                    return results, res.duration_s

                cross_metrics = _evaluate_queries(dep_a_queries, dep_b_search, args.k)
                cross_scope = _scope_metrics(
                    cross_metrics.per_query,
                    result_root=Path(info_b["paths"]["index_src"]),
                    scope_root=Path(info_a["paths"]["index_src"]),
                )
                cross_dependency.append(
                    {
                        "from_dep": dep_a,
                        "to_dep": dep_b,
                        "metrics": cross_metrics.__dict__,
                        "scope_precision": cross_scope,
                    }
                )

        summary = {
            "dependency": primary_dep,
            "dep_source": primary_info["dep_source"] if primary_info else None,
            "model": args.model,
            "device": args.device,
            "k": args.k,
            "paths": {
                "legacy_src": primary_info["paths"]["legacy_src"] if primary_info else None,
                "index_src": primary_info["paths"]["index_src"] if primary_info else None,
                "cache_root": str(cache_root),
                "repo_scan": str(repo_scan),
            },
            "parity": primary_info["parity"] if primary_info else None,
            "workflow": {
                "main_repo": {
                    "index_time_s": main_index_time,
                    "metrics": main_metrics.__dict__ if main_metrics else None,
                },
                "dependency": {
                    "metrics": primary_info["parity"]["index_mode"]["metrics"]
                    if primary_info
                    else None,
                },
            },
            "scope_precision": {
                "dependency_index": scope_dep,
                "main_repo_index": scope_main,
            },
            "multi_dep": {
                "deps": dep_info,
                "cross_dependency": cross_dependency,
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
