#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
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
    query_daemon_ok,
    restart_benchmark_daemon,
    stop_benchmark_daemon,
    write_report,
)
from tldr.indexing.index import get_index_context
from tldr.semantic import (
    _lane4_extract_callers,
    _lane4_impact_targets,
    _lane4_languages_from_semantic_rows,
    compound_semantic_impact_search,
    semantic_search,
)

try:
    from tldr.stats import count_tokens as _count_tokens
except Exception:  # pragma: no cover

    def _count_tokens(text: str) -> int:
        return len(str(text or "").split())


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query: str
    rg_pattern: str | None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_queries(path: Path, *, limit: int | None = None) -> list[QuerySpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError(f"bad query file: {path}")

    out: list[QuerySpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        qid = item.get("id")
        query = item.get("query")
        rg_pattern = item.get("rg_pattern")
        if isinstance(qid, str) and isinstance(query, str) and query.strip():
            out.append(
                QuerySpec(
                    query_id=qid,
                    query=query.strip(),
                    rg_pattern=rg_pattern if isinstance(rg_pattern, str) and rg_pattern.strip() else None,
                )
            )

    if limit is not None and limit > 0:
        return out[: int(limit)]
    return out


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    ordered = sorted(float(v) for v in values)
    p50 = float(statistics.median(ordered))
    idx_95 = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    p95 = float(ordered[idx_95])
    return {"p50": p50, "p95": p95}


def _extract_caller_set(result_row: dict[str, Any]) -> set[tuple[str, str]]:
    impact = result_row.get("impact") if isinstance(result_row, dict) else None
    callers = impact.get("callers") if isinstance(impact, dict) else None
    out: set[tuple[str, str]] = set()
    if not isinstance(callers, list):
        return out
    for item in callers:
        if not isinstance(item, dict):
            continue
        file_path = item.get("file")
        function = item.get("function")
        if isinstance(file_path, str) and isinstance(function, str):
            out.add((file_path, function))
    return out


def _jaccard(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _impact_payload_from_daemon_response(response: dict[str, Any]) -> dict[str, Any] | None:
    payload = response.get("result")
    if isinstance(payload, dict):
        return payload

    callers = response.get("callers")
    if not isinstance(callers, list):
        return None

    normalized_callers: list[dict[str, Any]] = []
    for item in callers:
        if not isinstance(item, dict):
            continue
        file_path = item.get("file")
        function = item.get("caller")
        if not isinstance(file_path, str) or not isinstance(function, str):
            continue
        normalized_callers.append(
            {
                "file": file_path,
                "function": function,
                "line": item.get("line") if isinstance(item.get("line"), int) else None,
            }
        )
    return {"targets": {"legacy": {"callers": normalized_callers}}}


def _sequential_baseline(
    repo_root: Path,
    *,
    index_ctx: Any,
    project_path: str,
    query: str,
    k: int,
    model: str | None,
    device: str | None,
    index_paths,
    index_config,
    retrieval_mode: str,
    no_result_guard: str,
    rg_pattern: str | None,
    rg_glob: str | None,
    rrf_k: int,
    abstain_threshold: float | None,
    abstain_empty: bool,
    rerank: bool,
    rerank_top_n: int,
    max_latency_ms_p50_ratio: float | None,
    max_payload_tokens_median_ratio: float | None,
    budget_tokens: int | None,
    impact_depth: int,
    impact_limit: int,
    impact_language: str,
    ignore_spec,
    workspace_root,
    use_daemon: bool = False,
) -> dict[str, Any]:
    from tldr.analysis import impact_analysis
    from tldr.cross_file_calls import ProjectCallGraph, build_project_call_graph

    total_start = time.perf_counter()
    semantic_start = time.perf_counter()
    if use_daemon:
        semantic_res = query_daemon_ok(
            repo_root,
            index_ctx=index_ctx,
            command={
                "cmd": "semantic",
                "action": "search",
                "query": query,
                "k": int(k),
                "model": model,
                "retrieval_mode": retrieval_mode,
                "no_result_guard": no_result_guard,
                "rg_pattern": rg_pattern,
                "rg_glob": rg_glob,
                "rrf_k": int(rrf_k),
                "abstain_threshold": abstain_threshold,
                "abstain_empty": bool(abstain_empty),
                "rerank": bool(rerank),
                "rerank_top_n": int(rerank_top_n),
                "max_latency_ms_p50_ratio": max_latency_ms_p50_ratio,
                "max_payload_tokens_median_ratio": max_payload_tokens_median_ratio,
                "budget_tokens": budget_tokens,
            },
        )
        semantic_rows = semantic_res.get("results") or semantic_res.get("result") or []
    else:
        semantic_rows = semantic_search(
            project_path,
            query,
            k=k,
            model=model,
            device=device,
            index_paths=index_paths,
            index_config=index_config,
            retrieval_mode=retrieval_mode,
            no_result_guard=no_result_guard,
            rg_pattern=rg_pattern,
            rg_glob=rg_glob,
            rrf_k=rrf_k,
            abstain_threshold=abstain_threshold,
            abstain_empty=abstain_empty,
            rerank=rerank,
            rerank_top_n=rerank_top_n,
            max_latency_ms_p50_ratio=max_latency_ms_p50_ratio,
            max_payload_tokens_median_ratio=max_payload_tokens_median_ratio,
            budget_tokens=budget_tokens,
        )
    semantic_ms = (time.perf_counter() - semantic_start) * 1000.0

    normalized_rows = [dict(row) for row in semantic_rows if isinstance(row, dict)]
    targets, selection_failures = _lane4_impact_targets(
        normalized_rows,
        impact_limit=int(impact_limit),
    )
    if not targets:
        total_ms = (time.perf_counter() - total_start) * 1000.0
        payload_tokens = _count_tokens(
            json.dumps({"semantic": normalized_rows, "impact": []}, sort_keys=True)
        )
        status = "ok" if not selection_failures else "partial"
        return {
            "status": status,
            "time_to_evidence_ms": float(total_ms),
            "semantic_ms": float(semantic_ms),
            "impact_ms_total": 0.0,
            "impact_ms_p50": 0.0,
            "payload_tokens": int(payload_tokens),
            "partial_failures_count": int(len(selection_failures)),
            "retrieval_ranked_files": [
                row.get("file")
                for row in normalized_rows
                if isinstance(row.get("file"), str)
            ],
            "impact_rows": [],
            "impact_ok": 0,
            "impact_attempted": 0,
        }

    combined_graph = ProjectCallGraph()
    partial_failures = len(selection_failures)
    if not use_daemon:
        languages = _lane4_languages_from_semantic_rows(
            normalized_rows,
            impact_language=impact_language,
        )
        if languages:
            for language in languages:
                try:
                    graph = build_project_call_graph(
                        project_path,
                        language=language,
                        ignore_spec=ignore_spec,
                        workspace_root=workspace_root,
                    )
                except Exception:
                    partial_failures += 1
                    continue
                for edge in graph.sorted_edges():
                    combined_graph.add_edge(*edge)
        else:
            partial_failures += len(targets)

    impact_rows: list[dict[str, Any]] = []
    impact_latencies: list[float] = []
    impact_ok = 0
    for target in targets:
        aliases = [alias for alias in target.get("aliases", []) if isinstance(alias, str)]
        file_path = target.get("file")
        start = time.perf_counter()
        payload = None
        for alias in aliases:
            if use_daemon:
                try:
                    response = query_daemon_ok(
                        repo_root,
                        index_ctx=index_ctx,
                        command={
                            "cmd": "impact",
                            "func": alias,
                            "depth": int(impact_depth),
                            **({"file": file_path} if isinstance(file_path, str) and file_path else {}),
                        },
                    )
                except Exception:
                    continue
                candidate = _impact_payload_from_daemon_response(response)
                if isinstance(candidate, dict) and candidate.get("error"):
                    continue
                payload = candidate
                break

            if not combined_graph.edges:
                continue
            candidate = impact_analysis(
                combined_graph,
                alias,
                max_depth=int(impact_depth),
                target_file=file_path,
            )
            if isinstance(candidate, dict) and candidate.get("error"):
                continue
            payload = candidate
            break
        latency_ms = (time.perf_counter() - start) * 1000.0
        impact_latencies.append(float(latency_ms))
        if isinstance(payload, dict):
            callers, truncated = _lane4_extract_callers(payload)
            impact_rows.append(
                {
                    "rank": int(target["row_index"]),
                    "status": "ok",
                    "callers": callers,
                    "truncated": bool(truncated),
                }
            )
            impact_ok += 1
        else:
            partial_failures += 1
            impact_rows.append(
                {
                    "rank": int(target["row_index"]),
                    "status": "error",
                    "callers": [],
                    "truncated": None,
                }
            )

    total_ms = (time.perf_counter() - total_start) * 1000.0
    impact_total_ms = float(sum(impact_latencies))
    impact_p50_ms = float(statistics.median(impact_latencies)) if impact_latencies else 0.0

    payload_tokens = _count_tokens(
        json.dumps({"semantic": normalized_rows, "impact": impact_rows}, sort_keys=True)
    )

    status = "ok" if partial_failures == 0 else "partial"
    return {
        "status": status,
        "time_to_evidence_ms": float(total_ms),
        "semantic_ms": float(semantic_ms),
        "impact_ms_total": float(impact_total_ms),
        "impact_ms_p50": float(impact_p50_ms),
        "payload_tokens": int(payload_tokens),
        "partial_failures_count": int(partial_failures),
        "retrieval_ranked_files": [
            row.get("file")
            for row in normalized_rows
            if isinstance(row.get("file"), str)
        ],
        "impact_rows": impact_rows,
        "impact_ok": int(impact_ok),
        "impact_attempted": int(len(targets)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Deterministic lane4 compound semantic+impact benchmark (compound vs sequential)."
    )
    corpus_group = ap.add_mutually_exclusive_group(required=True)
    corpus_group.add_argument("--corpus", default=None)
    corpus_group.add_argument("--repo-root", default=None)
    ap.add_argument(
        "--queries",
        default=str(get_repo_root() / "benchmarks" / "retrieval" / "django_queries.json"),
    )
    ap.add_argument("--cache-root", default=str(bench_cache_root(get_repo_root())))
    ap.add_argument("--index", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget-tokens", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--query-limit", type=int, default=12)
    ap.add_argument("--retrieval-mode", choices=["semantic", "hybrid"], default="hybrid")
    ap.add_argument("--no-result-guard", choices=["none", "rg_empty"], default="rg_empty")
    ap.add_argument("--impact-depth", type=int, default=3)
    ap.add_argument("--impact-limit", type=int, default=3)
    ap.add_argument("--impact-language", default="python")
    ap.add_argument("--lane2-abstain-threshold", type=float, default=0.35)
    ap.add_argument("--lane2-abstain-empty", action="store_true")
    ap.add_argument("--lane2-rerank", action="store_true")
    ap.add_argument("--lane2-rerank-top-n", type=int, default=8)
    ap.add_argument("--lane2-max-latency-ms-p50-ratio", type=float, default=1.10)
    ap.add_argument("--lane2-max-payload-tokens-median-ratio", type=float, default=0.90)
    ap.add_argument(
        "--use-daemon",
        action="store_true",
        help="Route semantic and impact queries through the daemon instead of in-process API calls.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo_root = get_repo_root()
    if args.corpus:
        corpus_root = (bench_corpora_root(repo_root) / str(args.corpus)).resolve()
        corpus_id = str(args.corpus)
        default_index = f"repo:{corpus_id}"
    else:
        corpus_root = Path(str(args.repo_root)).resolve()
        corpus_id = corpus_root.name
        default_index = None

    if not corpus_root.exists():
        raise SystemExit(f"error: repo root does not exist: {corpus_root}")

    queries = _load_queries(Path(args.queries).resolve(), limit=args.query_limit)
    if not queries:
        raise SystemExit("error: no valid queries found")

    index_ctx = get_index_context(
        scan_root=corpus_root,
        cache_root_arg=args.cache_root,
        index_id_arg=args.index or default_index,
        allow_create=False,
    )

    if index_ctx.paths is None or index_ctx.config is None:
        raise SystemExit("error: index context missing paths/config")

    if not index_ctx.paths.semantic_faiss.exists() or not index_ctx.paths.semantic_metadata.exists():
        raise SystemExit("error: semantic index artifacts missing")

    semantic_meta: dict[str, Any] = {}
    try:
        semantic_meta = json.loads(index_ctx.paths.semantic_metadata.read_text())
    except Exception:
        semantic_meta = {}

    daemon_ready: dict[str, Any] | None = None
    if args.use_daemon:
        daemon_ready = restart_benchmark_daemon(corpus_root, index_ctx=index_ctx)
        if not daemon_ready.get("ok"):
            raise SystemExit(f"error: daemon did not become ready: {daemon_ready}")

    compound_tte_ms: list[float] = []
    sequential_tte_ms: list[float] = []
    compound_payload: list[float] = []
    sequential_payload: list[float] = []
    overlap_scores: list[float] = []
    jaccard_scores: list[float] = []
    compound_win_count = 0
    compound_partial = 0
    sequential_partial = 0

    per_query: list[dict[str, Any]] = []

    try:
        for trial in range(1, int(args.trials) + 1):
            for query_spec in queries:
                compound_start = time.perf_counter()
                if args.use_daemon:
                    compound_res = query_daemon_ok(
                        corpus_root,
                        index_ctx=index_ctx,
                        command={
                            "cmd": "semantic",
                            "action": "search",
                            "query": query_spec.query,
                            "k": int(args.k),
                            "retrieval_mode": str(args.retrieval_mode),
                            "no_result_guard": str(args.no_result_guard),
                            "rg_pattern": query_spec.rg_pattern,
                            "rg_glob": "*.py",
                            "compound_impact": True,
                            "impact_depth": int(args.impact_depth),
                            "impact_limit": int(args.impact_limit),
                            "impact_language": str(args.impact_language),
                            "abstain_threshold": args.lane2_abstain_threshold,
                            "abstain_empty": bool(args.lane2_abstain_empty),
                            "rerank": bool(args.lane2_rerank),
                            "rerank_top_n": int(args.lane2_rerank_top_n),
                            "max_latency_ms_p50_ratio": args.lane2_max_latency_ms_p50_ratio,
                            "max_payload_tokens_median_ratio": args.lane2_max_payload_tokens_median_ratio,
                            "budget_tokens": args.budget_tokens,
                        },
                    )
                    compound = compound_res.get("result") or compound_res.get("results") or {}
                else:
                    compound = compound_semantic_impact_search(
                        str(corpus_root),
                        query_spec.query,
                        k=int(args.k),
                        index_paths=index_ctx.paths,
                        index_config=index_ctx.config,
                        retrieval_mode=str(args.retrieval_mode),
                        no_result_guard=str(args.no_result_guard),
                        rg_pattern=query_spec.rg_pattern,
                        rg_glob="*.py",
                        abstain_threshold=args.lane2_abstain_threshold,
                        abstain_empty=bool(args.lane2_abstain_empty),
                        rerank=bool(args.lane2_rerank),
                        rerank_top_n=int(args.lane2_rerank_top_n),
                        max_latency_ms_p50_ratio=args.lane2_max_latency_ms_p50_ratio,
                        max_payload_tokens_median_ratio=args.lane2_max_payload_tokens_median_ratio,
                        budget_tokens=args.budget_tokens,
                        impact_depth=int(args.impact_depth),
                        impact_limit=int(args.impact_limit),
                        impact_language=str(args.impact_language),
                        ignore_spec=None,
                        workspace_root=None,
                    )
                compound_elapsed_ms = (time.perf_counter() - compound_start) * 1000.0

                sequential = _sequential_baseline(
                    corpus_root,
                    index_ctx=index_ctx,
                    project_path=str(corpus_root),
                    query=query_spec.query,
                    k=int(args.k),
                    model=None,
                    device=None,
                    index_paths=index_ctx.paths,
                    index_config=index_ctx.config,
                    retrieval_mode=str(args.retrieval_mode),
                    no_result_guard=str(args.no_result_guard),
                    rg_pattern=query_spec.rg_pattern,
                    rg_glob="*.py",
                    rrf_k=60,
                    abstain_threshold=args.lane2_abstain_threshold,
                    abstain_empty=bool(args.lane2_abstain_empty),
                    rerank=bool(args.lane2_rerank),
                    rerank_top_n=int(args.lane2_rerank_top_n),
                    max_latency_ms_p50_ratio=args.lane2_max_latency_ms_p50_ratio,
                    max_payload_tokens_median_ratio=args.lane2_max_payload_tokens_median_ratio,
                    budget_tokens=args.budget_tokens,
                    impact_depth=int(args.impact_depth),
                    impact_limit=int(args.impact_limit),
                    impact_language=str(args.impact_language),
                    ignore_spec=None,
                    workspace_root=None,
                    use_daemon=bool(args.use_daemon),
                )

                compound_retrieval = [
                    row.get("file")
                    for row in compound.get("results", [])
                    if isinstance(row, dict) and isinstance(row.get("file"), str)
                ]
                sequential_retrieval = [
                    path for path in sequential.get("retrieval_ranked_files", []) if isinstance(path, str)
                ]

                top_k = int(args.k)
                set_compound = set(compound_retrieval[:top_k])
                set_sequential = set(sequential_retrieval[:top_k])
                overlap = 1.0 if not set_compound and not set_sequential else (
                    len(set_compound & set_sequential) / max(1, len(set_compound | set_sequential))
                )
                overlap_scores.append(float(overlap))

                comp_rows = {
                    int(row.get("rank")): row
                    for row in compound.get("results", [])
                    if isinstance(row, dict) and isinstance(row.get("rank"), int)
                }
                seq_rows = {
                    int(row.get("rank")): row
                    for row in sequential.get("impact_rows", [])
                    if isinstance(row, dict) and isinstance(row.get("rank"), int)
                }
                jaccards: list[float] = []
                for rank in sorted(set(comp_rows) & set(seq_rows)):
                    comp_callers = _extract_caller_set(comp_rows[rank])
                    seq_callers = _extract_caller_set(seq_rows[rank])
                    jaccards.append(_jaccard(comp_callers, seq_callers))
                impact_jaccard = float(statistics.mean(jaccards)) if jaccards else 1.0
                jaccard_scores.append(impact_jaccard)

                compound_tte = _safe_float(compound.get("timing_ms", {}).get("total"))
                if compound_tte is None:
                    compound_tte = float(compound_elapsed_ms)
                sequential_tte = _safe_float(sequential.get("time_to_evidence_ms")) or 0.0
                compound_tte_ms.append(float(compound_tte))
                sequential_tte_ms.append(float(sequential_tte))
                if compound_tte <= sequential_tte:
                    compound_win_count += 1

                compound_payload_tokens = _safe_float(
                    compound.get("regression_metadata", {}).get("payload_tokens_median")
                ) or 0.0
                sequential_payload_tokens = _safe_float(sequential.get("payload_tokens")) or 0.0
                compound_payload.append(float(compound_payload_tokens))
                sequential_payload.append(float(sequential_payload_tokens))

                if str(compound.get("status")) != "ok":
                    compound_partial += 1
                if str(sequential.get("status")) != "ok":
                    sequential_partial += 1

                per_query.append(
                    {
                        "query_id": query_spec.query_id,
                        "trial": int(trial),
                        "budget_tokens": int(args.budget_tokens),
                        "compound": {
                            "status": str(compound.get("status")),
                            "time_to_evidence_ms": float(compound_tte),
                            "semantic_ms": float(compound.get("timing_ms", {}).get("semantic", 0.0)),
                            "impact_ms_total": float(compound.get("timing_ms", {}).get("impact_total", 0.0)),
                            "payload_tokens": float(compound_payload_tokens),
                            "partial_failures_count": int(len(compound.get("partial_failures", []))),
                        },
                        "sequential": {
                            "status": str(sequential.get("status")),
                            "time_to_evidence_ms": float(sequential_tte),
                            "semantic_ms": float(sequential.get("semantic_ms", 0.0)),
                            "impact_ms_total": float(sequential.get("impact_ms_total", 0.0)),
                            "payload_tokens": float(sequential_payload_tokens),
                            "partial_failures_count": int(sequential.get("partial_failures_count", 0)),
                        },
                        "delta": {
                            "tte_ms": float(compound_tte - sequential_tte),
                            "payload_tokens": float(compound_payload_tokens - sequential_payload_tokens),
                        },
                        "quality": {
                            "retrieval_overlap_at_k": float(overlap),
                            "impact_callers_jaccard": float(impact_jaccard),
                        },
                    }
                )
    finally:
        if args.use_daemon:
            stop_benchmark_daemon(corpus_root, index_ctx=index_ctx)

    compound_stats = _summary(compound_tte_ms)
    sequential_stats = _summary(sequential_tte_ms)
    compound_payload_stats = _summary(compound_payload)
    sequential_payload_stats = _summary(sequential_payload)
    total_items = len(per_query)

    report = make_report(
        phase="phase8_compound_semantic_impact",
        meta=gather_meta(tldr_repo_root=repo_root, corpus_id=corpus_id, corpus_root=corpus_root),
        protocol={
            "schema_version": SCHEMA_VERSION,
            "feature_set_id": "feature.compound-semantic-impact.v1",
            "cache_root": str(index_ctx.cache_root) if index_ctx.cache_root is not None else None,
            "index_id": index_ctx.index_id,
            "semantic_model": semantic_meta.get("model") if isinstance(semantic_meta, dict) else None,
            "semantic_dimension": semantic_meta.get("dimension") if isinstance(semantic_meta, dict) else None,
            "queries": str(Path(args.queries).resolve()),
            "query_limit": int(args.query_limit),
            "budget_tokens": int(args.budget_tokens),
            "k": int(args.k),
            "trials": int(args.trials),
            "retrieval_mode": str(args.retrieval_mode),
            "impact_depth": int(args.impact_depth),
            "impact_limit": int(args.impact_limit),
            "impact_language": str(args.impact_language),
            "use_daemon": bool(args.use_daemon),
            "daemon_ready": daemon_ready,
            "sequential_baseline_definition": "semantic_search + bounded per-row impact_analysis",
            "compound_command_template": (
                "uv run tldrf semantic search \"{query}\" --path {repo_root} --k {k} --lang python "
                "--hybrid --no-result-guard rg_empty --compound-impact --impact-depth {impact_depth} "
                "--impact-limit {impact_limit} --impact-language {impact_language}"
            ),
        },
        results={
            "summary": {
                "compound": {
                    "n": int(total_items),
                    "time_to_evidence_ms_p50": float(compound_stats["p50"]),
                    "time_to_evidence_ms_p95": float(compound_stats["p95"]),
                    "payload_tokens_median": float(compound_payload_stats["p50"]),
                    "error_rate": 0.0,
                    "timeout_rate": 0.0,
                    "partial_rate": float(compound_partial / max(1, total_items)),
                },
                "sequential": {
                    "n": int(total_items),
                    "time_to_evidence_ms_p50": float(sequential_stats["p50"]),
                    "time_to_evidence_ms_p95": float(sequential_stats["p95"]),
                    "payload_tokens_median": float(sequential_payload_stats["p50"]),
                    "error_rate": 0.0,
                    "timeout_rate": 0.0,
                    "partial_rate": float(sequential_partial / max(1, total_items)),
                },
                "delta": {
                    "tte_ms_p50_delta": float(compound_stats["p50"] - sequential_stats["p50"]),
                    "tte_ms_p50_ratio": float(compound_stats["p50"] / max(1e-9, sequential_stats["p50"])),
                    "payload_tokens_median_delta": float(compound_payload_stats["p50"] - sequential_payload_stats["p50"]),
                    "error_rate_delta": 0.0,
                    "timeout_rate_delta": 0.0,
                    "partial_rate_delta": float(
                        (compound_partial - sequential_partial) / max(1, total_items)
                    ),
                    "retrieval_overlap_at_k_mean": float(statistics.mean(overlap_scores)) if overlap_scores else 0.0,
                    "impact_callers_jaccard_mean": float(statistics.mean(jaccard_scores)) if jaccard_scores else 1.0,
                    "compound_tte_win_rate": float(compound_win_count / max(1, total_items)),
                },
            },
            "per_query": per_query,
        },
    )

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        ts = now_utc_compact()
        out_path = (
            bench_runs_root(repo_root)
            / f"{ts}-compound-semantic-impact-{corpus_id}-lane4-b{int(args.budget_tokens)}.json"
        )
    write_report(out_path, report)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
