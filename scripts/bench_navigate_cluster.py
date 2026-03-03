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
    write_report,
)
from tldr.indexing.index import get_index_context
from tldr.semantic import semantic_navigation_cluster_search

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    query: str
    relevant_files: tuple[str, ...]
    rg_pattern: str | None
    is_negative: bool


def _normalize_file(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _load_queries(path: Path, *, limit: int | None = None) -> list[RetrievalQuery]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError(f"bad query file: {path}")

    out: list[RetrievalQuery] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        qid = item.get("id")
        query = item.get("query")
        relevant = item.get("relevant_files")
        if not isinstance(qid, str) or not isinstance(query, str):
            continue
        if not isinstance(relevant, list):
            relevant = []
        rel_files = tuple(
            fp for fp in (_normalize_file(v) for v in relevant) if isinstance(fp, str)
        )
        out.append(
            RetrievalQuery(
                query_id=qid,
                query=query.strip(),
                relevant_files=rel_files,
                rg_pattern=(
                    item.get("rg_pattern")
                    if isinstance(item.get("rg_pattern"), str) and item.get("rg_pattern").strip()
                    else None
                ),
                is_negative=bool(item.get("is_negative")),
            )
        )

    if limit is not None and limit > 0:
        return out[: int(limit)]
    return out


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return float(ordered[idx])


def _cluster_member_files(cluster: dict[str, Any]) -> set[str]:
    files: set[str] = set()
    members = cluster.get("members")
    if not isinstance(members, list):
        return files
    for member in members:
        if not isinstance(member, dict):
            continue
        fp = _normalize_file(member.get("file"))
        if isinstance(fp, str):
            files.add(fp)
    return files


def _row_payload_bytes(row: dict[str, Any]) -> int:
    return len(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _determinism_match_rate(per_query: list[dict[str, Any]]) -> float:
    digest_by_query: dict[str, set[str]] = {}
    for row in per_query:
        qid = row.get("query_id")
        digest = row.get("assignment_digest")
        if not isinstance(qid, str):
            continue
        digest_by_query.setdefault(qid, set()).add(str(digest or ""))
    if not digest_by_query:
        return 0.0
    stable = sum(1 for digests in digest_by_query.values() if len(digests) == 1)
    return float(stable) / float(len(digest_by_query))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Deterministic lane5 benchmark for semantic navigation/clustering."
    )
    corpus_group = ap.add_mutually_exclusive_group(required=True)
    corpus_group.add_argument("--corpus", default=None)
    corpus_group.add_argument("--repo-root", default=None)
    ap.add_argument(
        "--queries",
        default=str(get_repo_root() / "benchmarks" / "retrieval" / "django_queries.json"),
    )
    ap.add_argument(
        "--cache-root",
        default=str(bench_cache_root(get_repo_root())),
    )
    ap.add_argument("--index", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cluster-count", type=int, default=8)
    ap.add_argument("--cluster-min-size", type=int, default=1)
    ap.add_argument("--cluster-max-members", type=int, default=12)
    ap.add_argument(
        "--cluster-label-mode",
        choices=["auto", "file", "symbol"],
        default="auto",
    )
    ap.add_argument("--budget-tokens", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--query-limit", type=int, default=None)
    ap.add_argument(
        "--retrieval-mode",
        choices=["semantic", "hybrid"],
        default="hybrid",
    )
    ap.add_argument(
        "--no-result-guard",
        choices=["none", "rg_empty"],
        default="rg_empty",
    )
    ap.add_argument("--abstain-threshold", type=float, default=0.35)
    ap.add_argument("--abstain-empty", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank-top-n", type=int, default=8)
    ap.add_argument("--max-latency-ms-p50-ratio", type=float, default=1.10)
    ap.add_argument("--max-payload-tokens-median-ratio", type=float, default=0.90)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo_root = get_repo_root()
    corpus_id = str(args.corpus) if args.corpus else None
    if corpus_id:
        project_root = (bench_corpora_root(repo_root) / corpus_id).resolve()
        index_id = str(args.index or f"repo:{corpus_id}")
    else:
        project_root = Path(str(args.repo_root)).resolve()
        corpus_id = project_root.name
        index_id = str(args.index) if args.index else None

    if not project_root.exists():
        raise SystemExit(f"repo root not found: {project_root}")

    queries_path = Path(str(args.queries)).resolve()
    queries = _load_queries(queries_path, limit=args.query_limit)
    if not queries:
        raise SystemExit(f"no queries loaded from {queries_path}")

    index_ctx = get_index_context(
        scan_root=project_root,
        cache_root_arg=args.cache_root,
        index_id_arg=index_id,
        allow_create=False,
    )

    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    payload_tokens: list[float] = []
    payload_bytes: list[float] = []
    coverage_rates: list[float] = []
    recall1_hits: list[float] = []
    recall3_hits: list[float] = []
    errors = 0
    partials = 0

    for trial in range(1, int(args.trials) + 1):
        for q in queries:
            t0 = time.perf_counter()
            out = semantic_navigation_cluster_search(
                str(project_root),
                q.query,
                k=int(args.k),
                index_paths=index_ctx.paths,
                index_config=index_ctx.config,
                retrieval_mode=str(args.retrieval_mode),
                no_result_guard=str(args.no_result_guard),
                rg_pattern=q.rg_pattern,
                abstain_threshold=args.abstain_threshold,
                abstain_empty=bool(args.abstain_empty),
                rerank=bool(args.rerank),
                rerank_top_n=int(args.rerank_top_n),
                max_latency_ms_p50_ratio=args.max_latency_ms_p50_ratio,
                max_payload_tokens_median_ratio=args.max_payload_tokens_median_ratio,
                budget_tokens=int(args.budget_tokens),
                cluster_count=int(args.cluster_count),
                cluster_min_size=int(args.cluster_min_size),
                cluster_max_members=int(args.cluster_max_members),
                cluster_label_mode=str(args.cluster_label_mode),
            )
            elapsed_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)

            status = str(out.get("status") or "error")
            if status == "error":
                errors += 1
            elif status == "partial":
                partials += 1

            regression_metadata = out.get("regression_metadata")
            if not isinstance(regression_metadata, dict):
                regression_metadata = {}
            timing = out.get("timing_ms")
            if not isinstance(timing, dict):
                timing = {}

            latency = _safe_float(regression_metadata.get("latency_ms_p50"))
            if latency is None:
                latency = _safe_float(timing.get("total"))
            if latency is None:
                latency = float(elapsed_ms)
            latencies.append(float(latency))

            payload_tok = _safe_float(regression_metadata.get("payload_tokens_median"))
            if payload_tok is None:
                payload_tok = 0.0
            payload_tokens.append(float(payload_tok))

            row_bytes = float(_row_payload_bytes(out))
            payload_bytes.append(row_bytes)

            results = out.get("results")
            clusters = out.get("clusters")
            if not isinstance(results, list):
                results = []
            if not isinstance(clusters, list):
                clusters = []

            result_files = {
                fp
                for fp in (_normalize_file(row.get("file")) for row in results if isinstance(row, dict))
                if isinstance(fp, str)
            }
            cluster_files: set[str] = set()
            for cluster in clusters:
                if isinstance(cluster, dict):
                    cluster_files.update(_cluster_member_files(cluster))

            coverage_rate = 1.0
            if result_files:
                coverage_rate = float(len(result_files & cluster_files)) / float(len(result_files))
            coverage_rates.append(float(coverage_rate))

            relevant_set = {fp for fp in q.relevant_files if isinstance(fp, str)}
            positive = bool(relevant_set) and not bool(q.is_negative)
            top1_files: set[str] = set()
            top3_files: set[str] = set()
            for idx, cluster in enumerate(clusters):
                if not isinstance(cluster, dict):
                    continue
                member_files = _cluster_member_files(cluster)
                if idx == 0:
                    top1_files.update(member_files)
                if idx < 3:
                    top3_files.update(member_files)
            recall1_hit = None
            recall3_hit = None
            if positive:
                recall1_hit = 1.0 if (top1_files & relevant_set) else 0.0
                recall3_hit = 1.0 if (top3_files & relevant_set) else 0.0
                recall1_hits.append(float(recall1_hit))
                recall3_hits.append(float(recall3_hit))

            per_query.append(
                {
                    "trial": int(trial),
                    "query_id": q.query_id,
                    "query": q.query,
                    "status": status,
                    "is_positive": bool(positive),
                    "relevant_files": sorted(relevant_set),
                    "cluster_count": int(len(clusters)),
                    "retrieval_result_count": int(len(results)),
                    "cluster_coverage_rate": float(coverage_rate),
                    "query_cluster_recall@1_hit": recall1_hit,
                    "query_cluster_recall@3_hit": recall3_hit,
                    "assignment_digest": str(regression_metadata.get("assignment_digest") or ""),
                    "latency_ms_p50": float(latency),
                    "payload_tokens_median": float(payload_tok),
                    "payload_bytes": float(row_bytes),
                    "artifact": out,
                }
            )

    per_query = sorted(per_query, key=lambda row: (int(row["trial"]), str(row["query_id"])))

    summary = {
        "navigate": {
            "n": int(len(per_query)),
            "cluster_coverage_rate_mean": float(_mean(coverage_rates)),
            "determinism_assignment_digest_match_rate": float(_determinism_match_rate(per_query)),
            "query_cluster_recall@1_mean": float(_mean(recall1_hits)),
            "query_cluster_recall@3_mean": float(_mean(recall3_hits)),
            "latency_ms_p50": float(_p(latencies, 0.50)),
            "latency_ms_p95": float(_p(latencies, 0.95)),
            "payload_tokens_median": float(_p(payload_tokens, 0.50)),
            "payload_bytes_median": float(_p(payload_bytes, 0.50)),
            "partial_rate": float(float(partials) / float(len(per_query) or 1)),
            "error_rate": float(float(errors) / float(len(per_query) or 1)),
            "positive_queries_count": int(len(recall1_hits)),
        }
    }

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "feature_set_id": "feature.navigate-cluster.v1",
        "queries": str(queries_path),
        "query_count": int(len(queries)),
        "trials": int(args.trials),
        "budget_tokens": int(args.budget_tokens),
        "k": int(args.k),
        "cluster_count": int(args.cluster_count),
        "cluster_min_size": int(args.cluster_min_size),
        "cluster_max_members": int(args.cluster_max_members),
        "cluster_label_mode": str(args.cluster_label_mode),
        "retrieval_mode": str(args.retrieval_mode),
        "no_result_guard": str(args.no_result_guard),
        "abstain_threshold": args.abstain_threshold,
        "abstain_empty": bool(args.abstain_empty),
        "rerank": bool(args.rerank),
        "rerank_top_n": int(args.rerank_top_n),
        "max_latency_ms_p50_ratio": args.max_latency_ms_p50_ratio,
        "max_payload_tokens_median_ratio": args.max_payload_tokens_median_ratio,
        "cache_root": str(args.cache_root),
        "index": index_id,
        "repo_root": str(project_root),
    }

    report = make_report(
        phase="phase9_navigate_cluster",
        meta=gather_meta(
            tldr_repo_root=repo_root,
            corpus_id=corpus_id,
            corpus_root=project_root,
        ),
        protocol=protocol,
        results={
            "summary": summary,
            "per_query": per_query,
        },
        schema_version=SCHEMA_VERSION,
    )

    out_path = Path(str(args.out)).resolve() if args.out else (
        bench_runs_root(repo_root)
        / f"{now_utc_compact()}-navigate-cluster-{corpus_id}-lane5-b{int(args.budget_tokens)}.json"
    )
    write_report(out_path, report)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
