#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
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

from tldr.indexing.index import IndexContext, get_index_context


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    query: str
    relevant_files: tuple[str, ...]
    rg_pattern: str | None


def _load_queries(path: Path) -> list[RetrievalQuery]:
    data = json.loads(path.read_text())
    queries = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(queries, list):
        raise ValueError(f"Bad query file: {path}")

    out: list[RetrievalQuery] = []
    for q in queries:
        if not isinstance(q, dict):
            continue
        qid = q.get("id")
        query = q.get("query")
        relevant = q.get("relevant_files", [])
        rg_pattern = q.get("rg_pattern")
        if not isinstance(qid, str) or not isinstance(query, str) or not isinstance(relevant, list):
            continue
        rel_files = tuple(x for x in relevant if isinstance(x, str))
        out.append(
            RetrievalQuery(
                id=qid,
                query=query,
                relevant_files=rel_files,
                rg_pattern=rg_pattern if isinstance(rg_pattern, str) else None,
            )
        )
    return out


def _rg_rank_files(repo_root: Path, *, pattern: str, glob: str | None) -> list[str]:
    cmd = ["rg", "-n", "--no-messages"]
    if glob:
        cmd.extend(["--glob", glob])
    cmd.append(pattern)
    cmd.append(".")
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1):  # 1 = no matches
        raise RuntimeError(f"rg failed (rc={proc.returncode}): {proc.stderr.strip()}")

    hits_by_file: dict[str, dict[str, int]] = {}
    for line in (proc.stdout or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        fp, line_s, _ = parts
        if fp.startswith("./"):
            fp = fp[2:]
        try:
            ln = int(line_s)
        except ValueError:
            continue
        info = hits_by_file.setdefault(fp, {"hits": 0, "min_line": ln})
        info["hits"] += 1
        if ln < info["min_line"]:
            info["min_line"] = ln

    ranked = sorted(
        hits_by_file.items(),
        key=lambda kv: (-kv[1]["hits"], kv[1]["min_line"], kv[0]),
    )
    return [fp for fp, _ in ranked]


def _semantic_rank_files(
    repo_root: Path,
    *,
    index_ctx: IndexContext,
    query: str,
    k: int,
) -> list[str] | None:
    paths = index_ctx.paths
    cfg = index_ctx.config
    if paths is None or cfg is None:
        return None
    if not (paths.semantic_faiss.exists() and paths.semantic_metadata.exists()):
        return None

    from tldr.semantic import semantic_search

    results = semantic_search(
        str(repo_root),
        query,
        k=int(k),
        index_paths=paths,
        index_config=cfg,
    )

    ranked: list[str] = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        fp = r.get("file") or r.get("path")
        if isinstance(fp, str):
            if fp.startswith("./"):
                fp = fp[2:]
            ranked.append(fp)
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for fp in ranked:
        if fp in seen:
            continue
        seen.add(fp)
        out.append(fp)
    return out


def _rrf_fuse(rankings: list[list[str]], *, k: int = 60) -> list[str]:
    # Reciprocal Rank Fusion (RRF). Score = sum(1 / (k + rank)).
    scores: dict[str, float] = {}
    for ranking in rankings:
        for i, fp in enumerate(ranking, start=1):
            scores[fp] = scores.get(fp, 0.0) + 1.0 / (k + i)
    return [fp for fp, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _mrr(ranking: list[str], relevant: set[str]) -> float:
    for i, fp in enumerate(ranking, start=1):
        if fp in relevant:
            return 1.0 / i
    return 0.0


def _recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def _precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranking[:k]) & relevant) / k


def _fpr_at_k(ranking: list[str], *, k: int) -> float:
    # For negative queries (no relevant files), any returned file is a false positive.
    return 1.0 if ranking[:k] else 0.0


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 retrieval-quality benchmarks (rg vs semantic vs hybrid).")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", default=None, help="Corpus id from benchmarks/corpora.json (e.g. django).")
    group.add_argument("--repo-root", default=None, help="Path to corpus repo root.")
    ap.add_argument(
        "--queries",
        default=str(get_repo_root() / "benchmarks" / "retrieval" / "django_queries.json"),
        help="Query set JSON (default: benchmarks/retrieval/django_queries.json).",
    )
    ap.add_argument(
        "--cache-root",
        default=str(bench_cache_root(get_repo_root())),
        help="Index-mode cache root (default: benchmark/cache-root).",
    )
    ap.add_argument("--index", default=None, help="Index id (default: repo:<corpus>).")
    ap.add_argument("--ks", default="5,10", help="Comma-separated K values (default: 5,10).")
    ap.add_argument(
        "--rg-glob",
        default="*.py",
        help="ripgrep --glob filter (default: *.py). Use empty string to disable.",
    )
    ap.add_argument(
        "--no-result-guard",
        choices=["none", "rg_empty"],
        default="none",
        help=(
            "Bench-only gate to allow semantic/hybrid to return 'no results'. "
            "'rg_empty' suppresses semantic/hybrid when rg finds no matches for the query's rg_pattern."
        ),
    )
    ap.add_argument("--out", default=None, help="Write JSON report to this path (default under benchmark/runs/).")
    args = ap.parse_args()

    tldr_repo_root = get_repo_root()
    corpus_id = args.corpus
    if corpus_id:
        repo_root = (bench_corpora_root(tldr_repo_root) / corpus_id).resolve()
        default_index_id = f"repo:{corpus_id}"
    else:
        repo_root = Path(args.repo_root).resolve()
        default_index_id = None
        corpus_id = repo_root.name

    if not repo_root.exists():
        raise SystemExit(f"error: repo-root does not exist: {repo_root}")

    queries_path = Path(args.queries).resolve()
    queries = _load_queries(queries_path)

    ks = []
    for part in str(args.ks).split(","):
        part = part.strip()
        if not part:
            continue
        ks.append(int(part))
    if not ks:
        ks = [5, 10]
    max_k = max(ks)

    index_id = args.index or default_index_id
    index_ctx = get_index_context(
        scan_root=repo_root,
        cache_root_arg=args.cache_root,
        index_id_arg=index_id,
        allow_create=True,
    )

    glob = str(args.rg_glob)
    glob_arg = glob if glob.strip() else None
    no_result_guard = str(args.no_result_guard)

    per_query: list[dict[str, Any]] = []

    # Aggregate metrics (positive queries only).
    agg: dict[str, dict[str, list[float]]] = {
        "rg": {"mrr": [], **{f"recall@{k}": [] for k in ks}, **{f"precision@{k}": [] for k in ks}},
        "semantic": {"mrr": [], **{f"recall@{k}": [] for k in ks}, **{f"precision@{k}": [] for k in ks}},
        "hybrid_rrf": {"mrr": [], **{f"recall@{k}": [] for k in ks}, **{f"precision@{k}": [] for k in ks}},
    }
    neg_fpr: dict[str, dict[str, list[float]]] = {
        "rg": {f"fpr@{k}": [] for k in ks},
        "semantic": {f"fpr@{k}": [] for k in ks},
        "hybrid_rrf": {f"fpr@{k}": [] for k in ks},
    }

    semantic_available = (
        index_ctx.paths is not None
        and index_ctx.paths.semantic_faiss.exists()
        and index_ctx.paths.semantic_metadata.exists()
    )
    semantic_meta: dict[str, Any] | None = None
    if semantic_available and index_ctx.paths is not None:
        try:
            semantic_meta = json.loads(index_ctx.paths.semantic_metadata.read_text())
        except (OSError, json.JSONDecodeError):
            semantic_meta = None

    for q in queries:
        relevant = set(q.relevant_files)
        is_negative = not relevant

        rg_pattern = q.rg_pattern or re.escape(q.query)
        t0 = time.monotonic()
        rg_rank = _rg_rank_files(repo_root, pattern=rg_pattern, glob=glob_arg)
        rg_time_s = time.monotonic() - t0

        sem_rank: list[str] | None = None
        sem_time_s: float | None = None
        if semantic_available:
            if no_result_guard == "rg_empty" and not rg_rank:
                sem_rank = []
                sem_time_s = 0.0
            else:
                t0 = time.monotonic()
                sem_rank = _semantic_rank_files(repo_root, index_ctx=index_ctx, query=q.query, k=max_k)
                sem_time_s = time.monotonic() - t0

        hybrid_rank: list[str] | None = None
        if sem_rank is not None:
            hybrid_rank = _rrf_fuse([rg_rank[:max_k], sem_rank[:max_k]])

        def record(method: str, ranking: list[str]) -> dict[str, Any]:
            out = {
                "mrr": _mrr(ranking, relevant) if relevant else None,
                "top_files": ranking[:max_k],
            }
            for k in ks:
                if relevant:
                    out[f"recall@{k}"] = _recall_at_k(ranking, relevant, k)
                    out[f"precision@{k}"] = _precision_at_k(ranking, relevant, k)
                else:
                    out[f"fpr@{k}"] = _fpr_at_k(ranking, k=k)
            return out

        q_res: dict[str, Any] = {
            "id": q.id,
            "query": q.query,
            "relevant_files": list(q.relevant_files),
            "no_result_guard_triggered": bool(no_result_guard == "rg_empty" and not rg_rank),
            "rg": {
                "pattern": rg_pattern,
                "time_s": round(rg_time_s, 6),
                **record("rg", rg_rank),
            },
            "semantic": None,
            "hybrid_rrf": None,
        }

        if sem_rank is not None:
            q_res["semantic"] = {
                "time_s": round(float(sem_time_s or 0.0), 6),
                **record("semantic", sem_rank),
            }
        else:
            q_res["semantic"] = {"skipped": True, "reason": "semantic index artifacts missing"}

        if hybrid_rank is not None:
            q_res["hybrid_rrf"] = record("hybrid_rrf", hybrid_rank)
        else:
            q_res["hybrid_rrf"] = {"skipped": True, "reason": "semantic ranking unavailable"}

        per_query.append(q_res)

        # Aggregate metrics.
        if is_negative:
            for k in ks:
                neg_fpr["rg"][f"fpr@{k}"].append(_fpr_at_k(rg_rank, k=k))
            if sem_rank is not None:
                for k in ks:
                    neg_fpr["semantic"][f"fpr@{k}"].append(_fpr_at_k(sem_rank, k=k))
            if hybrid_rank is not None:
                for k in ks:
                    neg_fpr["hybrid_rrf"][f"fpr@{k}"].append(_fpr_at_k(hybrid_rank, k=k))
        else:
            agg["rg"]["mrr"].append(_mrr(rg_rank, relevant))
            for k in ks:
                agg["rg"][f"recall@{k}"].append(_recall_at_k(rg_rank, relevant, k))
                agg["rg"][f"precision@{k}"].append(_precision_at_k(rg_rank, relevant, k))

            if sem_rank is not None:
                agg["semantic"]["mrr"].append(_mrr(sem_rank, relevant))
                for k in ks:
                    agg["semantic"][f"recall@{k}"].append(_recall_at_k(sem_rank, relevant, k))
                    agg["semantic"][f"precision@{k}"].append(_precision_at_k(sem_rank, relevant, k))

            if hybrid_rank is not None:
                agg["hybrid_rrf"]["mrr"].append(_mrr(hybrid_rank, relevant))
                for k in ks:
                    agg["hybrid_rrf"][f"recall@{k}"].append(_recall_at_k(hybrid_rank, relevant, k))
                    agg["hybrid_rrf"][f"precision@{k}"].append(_precision_at_k(hybrid_rank, relevant, k))

    def summarize(xs: dict[str, list[float]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, vs in xs.items():
            out[k] = _mean(vs)
        return out

    report = make_report(
        phase="phase5_retrieval_quality",
        meta=gather_meta(tldr_repo_root=tldr_repo_root, corpus_id=corpus_id, corpus_root=repo_root),
        protocol={
            "schema_version": SCHEMA_VERSION,
            "queries": str(queries_path),
            "cache_root": str(index_ctx.cache_root) if index_ctx.cache_root is not None else None,
            "index_id": index_ctx.index_id,
            "ks": ks,
            "rg_glob": glob_arg,
            "no_result_guard": no_result_guard,
            "semantic_available": bool(semantic_available),
            "semantic_model": semantic_meta.get("model") if isinstance(semantic_meta, dict) else None,
            "semantic_dimension": semantic_meta.get("dimension") if isinstance(semantic_meta, dict) else None,
            "semantic_count": semantic_meta.get("count") if isinstance(semantic_meta, dict) else None,
            "semantic_faiss_bytes": (
                int(index_ctx.paths.semantic_faiss.stat().st_size)
                if semantic_available and index_ctx.paths is not None
                else None
            ),
            "semantic_metadata_bytes": (
                int(index_ctx.paths.semantic_metadata.stat().st_size)
                if semantic_available and index_ctx.paths is not None
                else None
            ),
        },
        results={
            "agg_positive": {
                "rg": summarize(agg["rg"]),
                "semantic": summarize(agg["semantic"]),
                "hybrid_rrf": summarize(agg["hybrid_rrf"]),
            },
            "agg_negative": {
                "rg": summarize(neg_fpr["rg"]),
                "semantic": summarize(neg_fpr["semantic"]),
                "hybrid_rrf": summarize(neg_fpr["hybrid_rrf"]),
            },
            "per_query": per_query,
        },
    )

    if args.out:
        out_path = Path(args.out)
    else:
        ts = now_utc_compact()
        out_path = bench_runs_root(tldr_repo_root) / f"{ts}-retrieval-{corpus_id}.json"
    write_report(out_path, report)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
