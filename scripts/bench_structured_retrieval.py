#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    percentiles,
    query_daemon_ok,
    restart_benchmark_daemon,
    stop_benchmark_daemon,
    write_report,
)

from tldr.indexing.index import IndexContext, get_index_context


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StructuredTarget:
    file: str
    qualified_symbol: str | None
    symbol_kind: str | None
    start_line: int | None
    end_line: int | None


@dataclass(frozen=True)
class StructuredQuery:
    id: str
    query: str
    rg_pattern: str
    targets: tuple[StructuredTarget, ...]


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    kind: str
    display_name: str
    index_id: str | None = None
    model: str | None = None
    dimension: int | None = None
    index_ctx: IndexContext | None = None
    retrieval_mode: str = "semantic"
    no_result_guard: str = "none"
    rg_glob: str | None = None
    rerank: bool = False
    rerank_top_n: int = 5
    projection_unit_k: int | None = None


@dataclass(frozen=True)
class Score:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        return (2.0 * precision * recall) / (precision + recall) if (precision + recall) else 0.0


def _normalize_rel_path(value: str) -> str:
    path = str(value or "").strip().replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    return path


def _normalize_symbol_kind(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _target_to_dict(target: StructuredTarget, *, rank: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "file": target.file,
        "qualified_symbol": target.qualified_symbol,
        "symbol_kind": target.symbol_kind,
        "start_line": target.start_line,
        "end_line": target.end_line,
    }
    if rank is not None:
        out["rank"] = int(rank)
    return out


def _load_queries(path: Path) -> list[StructuredQuery]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_queries = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(raw_queries, list):
        raise ValueError(f"bad structured retrieval query file: {path}")

    queries: list[StructuredQuery] = []
    for raw in raw_queries:
        if not isinstance(raw, dict):
            continue
        query_id = raw.get("id")
        query_text = raw.get("query")
        rg_pattern = raw.get("rg_pattern")
        raw_targets = raw.get("targets", [])
        if not isinstance(query_id, str) or not isinstance(query_text, str) or not isinstance(rg_pattern, str):
            continue
        if not isinstance(raw_targets, list):
            raise ValueError(f"query {query_id} has non-list targets")

        targets: list[StructuredTarget] = []
        for raw_target in raw_targets:
            if not isinstance(raw_target, dict):
                continue
            file_path = _normalize_rel_path(str(raw_target.get("file") or ""))
            if not file_path:
                continue
            qualified_symbol = raw_target.get("qualified_symbol")
            start_line = _safe_int(raw_target.get("start_line"))
            end_line = _safe_int(raw_target.get("end_line"))
            if start_line is not None and end_line is None:
                end_line = start_line
            target = StructuredTarget(
                file=file_path,
                qualified_symbol=qualified_symbol if isinstance(qualified_symbol, str) and qualified_symbol.strip() else None,
                symbol_kind=_normalize_symbol_kind(raw_target.get("symbol_kind")),
                start_line=start_line,
                end_line=end_line,
            )
            if _dedupe_identity_key(_target_to_dict(target)) is None:
                raise ValueError(
                    f"query {query_id} target must define a scoreable identity: {raw_target}"
                )
            targets.append(target)

        queries.append(
            StructuredQuery(
                id=query_id,
                query=query_text.strip(),
                rg_pattern=rg_pattern,
                targets=tuple(targets),
            )
        )
    return queries


def _symbol_key(item: dict[str, Any]) -> tuple[str, str] | None:
    file_path = item.get("file")
    symbol = item.get("qualified_symbol")
    if isinstance(file_path, str) and file_path and isinstance(symbol, str) and symbol:
        return (file_path, symbol)
    return None


def _span_key(item: dict[str, Any]) -> tuple[str, int, int] | None:
    file_path = item.get("file")
    start_line = _safe_int(item.get("start_line"))
    end_line = _safe_int(item.get("end_line"))
    if isinstance(file_path, str) and file_path and start_line is not None and end_line is not None:
        return (file_path, start_line, end_line)
    return None


def _dedupe_identity_key(item: dict[str, Any]) -> tuple[Any, ...] | None:
    symbol_key = _symbol_key(item)
    if symbol_key is not None:
        return ("symbol",) + symbol_key
    span_key = _span_key(item)
    if span_key is not None:
        return ("span",) + span_key
    return None


def _dedupe_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    unscorable = 0
    for item in items:
        key = _dedupe_identity_key(item)
        if key is None:
            unscorable += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
    return deduped, unscorable


def _score_structured_predictions(
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    gold, unscorable_gold = _dedupe_items(targets)
    got, unscorable_predictions = _dedupe_items(predictions)

    unmatched_gold = set(range(len(gold)))
    unmatched_predictions = set(range(len(got)))
    matched_pairs: list[dict[str, Any]] = []

    gold_symbol_map: dict[tuple[str, str], list[int]] = {}
    for idx, item in enumerate(gold):
        key = _symbol_key(item)
        if key is not None:
            gold_symbol_map.setdefault(key, []).append(idx)

    for pred_idx, pred in enumerate(got):
        pred_symbol = _symbol_key(pred)
        if pred_symbol is None:
            continue
        candidates = gold_symbol_map.get(pred_symbol, [])
        match_idx = next((idx for idx in candidates if idx in unmatched_gold), None)
        if match_idx is None:
            continue
        unmatched_gold.remove(match_idx)
        unmatched_predictions.remove(pred_idx)
        matched_pairs.append(
            {
                "mode": "symbol",
                "prediction": dict(pred),
                "target": dict(gold[match_idx]),
            }
        )

    gold_span_map: dict[tuple[str, int, int], list[int]] = {}
    for idx in unmatched_gold:
        key = _span_key(gold[idx])
        if key is not None:
            gold_span_map.setdefault(key, []).append(idx)

    for pred_idx, pred in enumerate(got):
        if pred_idx not in unmatched_predictions:
            continue
        pred_span = _span_key(pred)
        if pred_span is None:
            continue
        candidates = gold_span_map.get(pred_span, [])
        match_idx = None
        for candidate_idx in candidates:
            if candidate_idx not in unmatched_gold:
                continue
            gold_has_symbol = _symbol_key(gold[candidate_idx]) is not None
            pred_has_symbol = _symbol_key(pred) is not None
            if gold_has_symbol and pred_has_symbol:
                continue
            match_idx = candidate_idx
            break
        if match_idx is None:
            continue
        unmatched_gold.remove(match_idx)
        unmatched_predictions.remove(pred_idx)
        matched_pairs.append(
            {
                "mode": "span",
                "prediction": dict(pred),
                "target": dict(gold[match_idx]),
            }
        )

    score = Score(
        tp=len(matched_pairs),
        fp=len(unmatched_predictions),
        fn=len(unmatched_gold),
    )
    return {
        "tp": int(score.tp),
        "fp": int(score.fp),
        "fn": int(score.fn),
        "precision": float(score.precision),
        "recall": float(score.recall),
        "f1": float(score.f1),
        "gold_targets": gold,
        "predictions": got,
        "matched_pairs": matched_pairs,
        "unmatched_gold": [gold[idx] for idx in sorted(unmatched_gold)],
        "unmatched_predictions": [got[idx] for idx in sorted(unmatched_predictions)],
        "unscorable_gold_targets": int(unscorable_gold),
        "unscorable_predictions": int(unscorable_predictions),
    }


def _load_semantic_metadata(index_ctx: IndexContext) -> dict[str, Any] | None:
    paths = getattr(index_ctx, "paths", None)
    if paths is None or not paths.semantic_metadata.exists():
        return None
    try:
        return json.loads(paths.semantic_metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _default_semantic_backends(corpus_id: str | None) -> list[str]:
    if corpus_id == "django":
        return ["bge=repo:django-bge", "jina05b=repo:django-jina05b"]
    return []


def _parse_semantic_backend_arg(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text or "=" not in text:
        raise ValueError(f"bad --semantic-backend value: {value!r}")
    label, index_id = text.split("=", 1)
    label = label.strip()
    index_id = index_id.strip()
    if not label or not index_id:
        raise ValueError(f"bad --semantic-backend value: {value!r}")
    return (label, index_id)


def _rg_structured_predictions(
    repo_root: Path,
    *,
    pattern: str,
    top_k: int,
    glob: str | None,
) -> list[dict[str, Any]]:
    argv = ["rg", "--json", "--line-number", "--no-messages"]
    if isinstance(glob, str) and glob.strip():
        argv.extend(["--glob", glob.strip()])
    argv.extend(["--", pattern, "."])
    proc = subprocess.run(
        argv,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        stderr = str(proc.stderr or "").strip()
        raise RuntimeError(f"rg failed with exit={proc.returncode}: {stderr}")

    raw_matches: list[dict[str, Any]] = []
    file_hit_counts: dict[str, int] = {}
    for raw_line in str(proc.stdout or "").splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "match":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        path_obj = data.get("path")
        if not isinstance(path_obj, dict):
            continue
        path_text = path_obj.get("text")
        if not isinstance(path_text, str):
            continue
        file_path = _normalize_rel_path(path_text)
        line_number = _safe_int(data.get("line_number"))
        if not file_path or line_number is None:
            continue
        raw_matches.append(
            {
                "file": file_path,
                "qualified_symbol": None,
                "symbol_kind": "unknown",
                "start_line": int(line_number),
                "end_line": int(line_number),
            }
        )
        file_hit_counts[file_path] = file_hit_counts.get(file_path, 0) + 1

    raw_matches.sort(
        key=lambda item: (
            -file_hit_counts.get(str(item.get("file") or ""), 0),
            int(item.get("start_line") or 0),
            str(item.get("file") or ""),
        )
    )

    ranked: list[dict[str, Any]] = []
    for rank, item in enumerate(raw_matches, start=1):
        entry = dict(item)
        entry["rank"] = int(rank)
        ranked.append(entry)

    deduped, _ = _dedupe_items(ranked)
    limited = deduped[: max(0, int(top_k))]
    for rank, item in enumerate(limited, start=1):
        item["rank"] = int(rank)
    return limited


def _semantic_structured_predictions(
    repo_root: Path,
    *,
    backend: BackendSpec,
    query: str,
    top_k: int,
    use_daemon: bool,
    rg_pattern: str | None = None,
) -> list[dict[str, Any]]:
    rows = _semantic_search_rows(
        repo_root,
        backend=backend,
        query=query,
        top_k=int(top_k),
        use_daemon=use_daemon,
        retrieval_mode="semantic",
        rg_pattern=rg_pattern,
    )
    predictions = _rows_to_structured_predictions(rows)
    deduped, _ = _dedupe_items(predictions)
    limited = deduped[: max(0, int(top_k))]
    for rank, item in enumerate(limited, start=1):
        item["rank"] = int(rank)
    return limited


def _semantic_search_rows(
    repo_root: Path,
    *,
    backend: BackendSpec,
    query: str,
    top_k: int,
    use_daemon: bool,
    retrieval_mode: str = "semantic",
    rg_pattern: str | None = None,
) -> list[dict[str, Any]]:
    if backend.index_ctx is None:
        return []
    glob_arg = backend.rg_glob if isinstance(backend.rg_glob, str) and backend.rg_glob.strip() else None
    if use_daemon:
        command: dict[str, Any] = {
            "cmd": "semantic",
            "action": "search",
            "query": query,
            "k": int(top_k),
            "model": backend.model,
            "retrieval_mode": retrieval_mode,
            "no_result_guard": backend.no_result_guard,
            "rerank": bool(backend.rerank),
            "rerank_top_n": int(backend.rerank_top_n),
        }
        if isinstance(rg_pattern, str) and rg_pattern.strip():
            command["rg_pattern"] = rg_pattern
        if glob_arg is not None:
            command["rg_glob"] = glob_arg
        response = query_daemon_ok(
            repo_root,
            index_ctx=backend.index_ctx,
            command=command,
        )
        rows = response.get("results") or response.get("result") or []
    else:
        from tldr.semantic import semantic_search

        rows = semantic_search(
            str(repo_root),
            query,
            k=int(top_k),
            model=backend.model,
            index_paths=getattr(backend.index_ctx, "paths", None),
            index_config=getattr(backend.index_ctx, "config", None),
            retrieval_mode=retrieval_mode,
            no_result_guard=backend.no_result_guard,
            rg_pattern=rg_pattern,
            rg_glob=glob_arg,
            rerank=bool(backend.rerank),
            rerank_top_n=int(backend.rerank_top_n),
        )
    return [dict(row) for row in rows or [] if isinstance(row, dict)]


def _row_to_structured_prediction(row: dict[str, Any], *, rank: int) -> dict[str, Any] | None:
    file_path = row.get("file") or row.get("path")
    if not isinstance(file_path, str):
        return None
    qualified_symbol = None
    if isinstance(row.get("qualified_name"), str):
        qualified_symbol = row.get("qualified_name")
    else:
        symbol_obj = row.get("symbol")
        if isinstance(symbol_obj, dict) and isinstance(symbol_obj.get("qualified_name"), str):
            qualified_symbol = symbol_obj.get("qualified_name")
    start_line = _safe_int(row.get("line"))
    if start_line is None:
        start_line = _safe_int(row.get("start_line"))
    end_line = _safe_int(row.get("end_line"))
    if start_line is not None and end_line is None:
        end_line = start_line
    return {
        "file": _normalize_rel_path(file_path),
        "qualified_symbol": qualified_symbol if isinstance(qualified_symbol, str) and qualified_symbol.strip() else None,
        "symbol_kind": _normalize_symbol_kind(row.get("unit_type") or row.get("symbol_kind")),
        "start_line": start_line,
        "end_line": end_line,
        "rank": int(rank),
    }


def _rows_to_structured_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for rank, row in enumerate(rows or [], start=1):
        prediction = _row_to_structured_prediction(row, rank=rank)
        if prediction is not None:
            predictions.append(prediction)
    return predictions


def _hybrid_structured_predictions(
    repo_root: Path,
    *,
    backend: BackendSpec,
    query: str,
    rg_pattern: str,
    top_k: int,
    use_daemon: bool,
) -> list[dict[str, Any]]:
    file_rows = _semantic_search_rows(
        repo_root,
        backend=backend,
        query=query,
        top_k=int(top_k),
        use_daemon=use_daemon,
        retrieval_mode="hybrid",
        rg_pattern=rg_pattern,
    )
    if not file_rows:
        return []

    file_rank_by_path: dict[str, int] = {}
    for rank, row in enumerate(file_rows, start=1):
        file_path = row.get("file") or row.get("path")
        if not isinstance(file_path, str):
            continue
        normalized = _normalize_rel_path(file_path)
        if normalized and normalized not in file_rank_by_path:
            file_rank_by_path[normalized] = int(rank)
    if not file_rank_by_path:
        return []

    projection_unit_k = backend.projection_unit_k
    if projection_unit_k is None:
        projection_unit_k = max(int(top_k) * 20, 100)
    unit_rows = _semantic_search_rows(
        repo_root,
        backend=backend,
        query=query,
        top_k=max(int(top_k), int(projection_unit_k)),
        use_daemon=use_daemon,
        retrieval_mode="semantic",
    )

    candidates: list[dict[str, Any]] = []
    for semantic_rank, row in enumerate(unit_rows, start=1):
        prediction = _row_to_structured_prediction(row, rank=semantic_rank)
        if prediction is None:
            continue
        file_rank = file_rank_by_path.get(str(prediction.get("file") or ""))
        if file_rank is None:
            continue
        prediction["hybrid_file_rank"] = int(file_rank)
        prediction["semantic_rank"] = int(semantic_rank)
        candidates.append(prediction)

    candidates.sort(
        key=lambda item: (
            int(item.get("hybrid_file_rank") or 0),
            int(item.get("semantic_rank") or 0),
            str(item.get("file") or ""),
            str(item.get("qualified_symbol") or ""),
            int(item.get("start_line") or 0),
        )
    )
    deduped, _ = _dedupe_items(candidates)
    limited = deduped[: max(0, int(top_k))]
    for rank, item in enumerate(limited, start=1):
        item["rank"] = int(rank)
    return limited


def _finalize_backend_summary(acc: dict[str, Any]) -> dict[str, Any]:
    micro = Score(tp=int(acc["tp"]), fp=int(acc["fp"]), fn=int(acc["fn"]))
    latency_values = [float(value) for value in acc["latency_ms"]]
    latency_percentiles = percentiles(latency_values, ps=[0.5, 0.95]) if latency_values else {}
    macro_precision = statistics.mean(acc["precision"]) if acc["precision"] else None
    macro_recall = statistics.mean(acc["recall"]) if acc["recall"] else None
    macro_f1 = statistics.mean(acc["f1"]) if acc["f1"] else None
    return {
        "n_queries": int(acc["n_queries"]),
        "micro": {
            "tp": int(micro.tp),
            "fp": int(micro.fp),
            "fn": int(micro.fn),
            "precision": float(micro.precision),
            "recall": float(micro.recall),
            "f1": float(micro.f1),
        },
        "macro": {
            "precision_mean": float(macro_precision) if macro_precision is not None else None,
            "recall_mean": float(macro_recall) if macro_recall is not None else None,
            "f1_mean": float(macro_f1) if macro_f1 is not None else None,
        },
        "latency_ms": {
            "p50": float(latency_percentiles.get("p50", 0.0)),
            "p95": float(latency_percentiles.get("p95", 0.0)),
            "mean": float(statistics.mean(latency_values)) if latency_values else 0.0,
        },
        "unscorable_predictions": int(acc["unscorable_predictions"]),
        "unscorable_gold_targets": int(acc["unscorable_gold_targets"]),
    }


def _compare_backend_summaries(
    lhs: str,
    rhs: str,
    *,
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    lhs_summary = summaries.get(lhs)
    rhs_summary = summaries.get(rhs)
    if not isinstance(lhs_summary, dict) or not isinstance(rhs_summary, dict):
        return None

    lhs_f1 = float(lhs_summary["micro"]["f1"])
    rhs_f1 = float(rhs_summary["micro"]["f1"])
    lhs_p50 = float(lhs_summary["latency_ms"]["p50"])
    rhs_p50 = float(rhs_summary["latency_ms"]["p50"])

    winner = "tie"
    if lhs_f1 > rhs_f1:
        winner = lhs
    elif rhs_f1 > lhs_f1:
        winner = rhs

    return {
        "lhs": lhs,
        "rhs": rhs,
        "micro_f1_delta": float(lhs_f1 - rhs_f1),
        "micro_precision_delta": float(lhs_summary["micro"]["precision"] - rhs_summary["micro"]["precision"]),
        "micro_recall_delta": float(lhs_summary["micro"]["recall"] - rhs_summary["micro"]["recall"]),
        "latency_ms_p50_delta": float(lhs_p50 - rhs_p50),
        "winner_by_micro_f1": winner,
    }


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Deterministic structured retrieval benchmark "
            "(rg-native vs TLDR semantic and optional hybrid backends)."
        )
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", default=None, help="Corpus id under benchmark/corpora/ (e.g. django).")
    group.add_argument("--repo-root", default=None, help="Corpus repo root.")
    ap.add_argument(
        "--queries",
        default=str(get_repo_root() / "benchmarks" / "retrieval" / "django_structured_queries.json"),
        help="Structured retrieval query manifest.",
    )
    ap.add_argument(
        "--cache-root",
        default=str(bench_cache_root(get_repo_root())),
        help="Index-mode cache root.",
    )
    ap.add_argument(
        "--semantic-backend",
        action="append",
        default=[],
        help="Repeated LABEL=INDEX_ID mapping. Default for django: bge=repo:django-bge and jina05b=repo:django-jina05b.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of structured rows to score per backend/query.",
    )
    ap.add_argument(
        "--rg-glob",
        default="*.py",
        help="Optional rg --glob filter for the native lexical baseline.",
    )
    ap.add_argument(
        "--use-daemon",
        action="store_true",
        help="Route TLDR semantic queries through the daemon for repeated-query benchmarking.",
    )
    ap.add_argument(
        "--include-hybrid",
        action="store_true",
        help=(
            "Also benchmark hybrid file retrieval projected back to structured unit predictions "
            "via semantic-unit rows from the same top files."
        ),
    )
    ap.add_argument(
        "--hybrid-no-result-guard",
        choices=["none", "rg_empty"],
        default="none",
        help="Guard mode for hybrid retrieval (default: none).",
    )
    ap.add_argument(
        "--hybrid-rerank",
        action="store_true",
        help="Enable deterministic reranking on hybrid file rows before unit projection.",
    )
    ap.add_argument(
        "--hybrid-rerank-top-n",
        type=int,
        default=5,
        help="Top-N candidate files considered by --hybrid-rerank (default: 5).",
    )
    ap.add_argument(
        "--hybrid-projection-unit-k",
        type=int,
        default=100,
        help=(
            "Semantic-unit candidate count used to project hybrid file hits back to structured "
            "predictions (default: 100)."
        ),
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Write JSON report here (default: benchmark/runs/<timestamp>-structured-retrieval-<corpus>.json).",
    )
    return ap


def main() -> int:
    args = _build_parser().parse_args()

    tldr_repo_root = get_repo_root()
    corpus_id = str(args.corpus) if args.corpus else None
    if corpus_id:
        repo_root = (bench_corpora_root(tldr_repo_root) / corpus_id).resolve()
    else:
        repo_root = Path(args.repo_root).resolve()
        corpus_id = repo_root.name
    if not repo_root.exists():
        raise SystemExit(f"error: repo-root does not exist: {repo_root}")

    queries_path = Path(args.queries).resolve()
    queries = _load_queries(queries_path)

    semantic_backend_args = list(args.semantic_backend or ())
    if not semantic_backend_args:
        semantic_backend_args = _default_semantic_backends(corpus_id)

    semantic_backends: list[BackendSpec] = []
    for raw_backend in semantic_backend_args:
        label, index_id = _parse_semantic_backend_arg(raw_backend)
        index_ctx = get_index_context(
            scan_root=repo_root,
            cache_root_arg=args.cache_root,
            index_id_arg=index_id,
            allow_create=True,
        )
        meta = _load_semantic_metadata(index_ctx)
        semantic_backends.append(
            BackendSpec(
                backend_id=label,
                kind="semantic",
                display_name=f"{label} ({meta.get('model') if isinstance(meta, dict) else index_id})",
                index_id=index_id,
                model=meta.get("model") if isinstance(meta, dict) and isinstance(meta.get("model"), str) else None,
                dimension=_safe_int(meta.get("dimension")) if isinstance(meta, dict) else None,
                index_ctx=index_ctx,
            )
        )

    backends: list[BackendSpec] = [
        BackendSpec(
            backend_id="rg_native",
            kind="rg_native",
            display_name="rg-native (lexical)",
        )
    ]
    backends.extend(semantic_backends)
    if args.include_hybrid:
        backends.extend(
            BackendSpec(
                backend_id=f"{backend.backend_id}_hybrid",
                kind="semantic_hybrid",
                display_name=f"{backend.backend_id} hybrid ({backend.model or backend.index_id})",
                index_id=backend.index_id,
                model=backend.model,
                dimension=backend.dimension,
                index_ctx=backend.index_ctx,
                retrieval_mode="hybrid",
                no_result_guard=str(args.hybrid_no_result_guard),
                rg_glob=str(args.rg_glob) if str(args.rg_glob).strip() else None,
                rerank=bool(args.hybrid_rerank),
                rerank_top_n=int(args.hybrid_rerank_top_n),
                projection_unit_k=int(args.hybrid_projection_unit_k),
            )
            for backend in semantic_backends
        )

    daemon_ready: dict[str, Any] = {}
    if args.use_daemon:
        for backend in semantic_backends:
            if backend.index_ctx is None:
                continue
            ready = restart_benchmark_daemon(repo_root, index_ctx=backend.index_ctx)
            if not ready.get("ok"):
                raise SystemExit(
                    f"error: daemon did not become ready for {backend.backend_id}: {ready}"
                )
            daemon_ready[backend.backend_id] = ready

    accumulators: dict[str, dict[str, Any]] = {
        backend.backend_id: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_queries": 0,
            "precision": [],
            "recall": [],
            "f1": [],
            "latency_ms": [],
            "unscorable_predictions": 0,
            "unscorable_gold_targets": 0,
        }
        for backend in backends
    }

    per_query: list[dict[str, Any]] = []
    try:
        for query in queries:
            gold_targets = [_target_to_dict(target) for target in query.targets]
            row: dict[str, Any] = {
                "id": query.id,
                "query": query.query,
                "rg_pattern": query.rg_pattern,
                "targets": gold_targets,
                "backends": {},
            }

            for backend in backends:
                started = time.monotonic()
                if backend.kind == "rg_native":
                    predictions = _rg_structured_predictions(
                        repo_root,
                        pattern=query.rg_pattern,
                        top_k=int(args.top_k),
                        glob=args.rg_glob,
                    )
                elif backend.kind == "semantic_hybrid":
                    predictions = _hybrid_structured_predictions(
                        repo_root,
                        backend=backend,
                        query=query.query,
                        rg_pattern=query.rg_pattern,
                        top_k=int(args.top_k),
                        use_daemon=bool(args.use_daemon),
                    )
                else:
                    predictions = _semantic_structured_predictions(
                        repo_root,
                        backend=backend,
                        query=query.query,
                        top_k=int(args.top_k),
                        use_daemon=bool(args.use_daemon),
                        rg_pattern=query.rg_pattern,
                    )
                elapsed_ms = (time.monotonic() - started) * 1000.0

                scored = _score_structured_predictions(gold_targets, predictions)
                row["backends"][backend.backend_id] = {
                    "backend_id": backend.backend_id,
                    "display_name": backend.display_name,
                    "latency_ms": float(elapsed_ms),
                    "prediction_count": int(len(scored["predictions"])),
                    "tp": int(scored["tp"]),
                    "fp": int(scored["fp"]),
                    "fn": int(scored["fn"]),
                    "precision": float(scored["precision"]),
                    "recall": float(scored["recall"]),
                    "f1": float(scored["f1"]),
                    "predictions": scored["predictions"],
                    "matched_pairs": scored["matched_pairs"],
                    "unmatched_gold": scored["unmatched_gold"],
                    "unmatched_predictions": scored["unmatched_predictions"],
                    "unscorable_predictions": int(scored["unscorable_predictions"]),
                    "unscorable_gold_targets": int(scored["unscorable_gold_targets"]),
                }

                acc = accumulators[backend.backend_id]
                acc["tp"] += int(scored["tp"])
                acc["fp"] += int(scored["fp"])
                acc["fn"] += int(scored["fn"])
                acc["n_queries"] += 1
                acc["precision"].append(float(scored["precision"]))
                acc["recall"].append(float(scored["recall"]))
                acc["f1"].append(float(scored["f1"]))
                acc["latency_ms"].append(float(elapsed_ms))
                acc["unscorable_predictions"] += int(scored["unscorable_predictions"])
                acc["unscorable_gold_targets"] += int(scored["unscorable_gold_targets"])

            per_query.append(row)
    finally:
        if args.use_daemon:
            for backend in reversed(semantic_backends):
                if backend.index_ctx is None:
                    continue
                stop_benchmark_daemon(repo_root, index_ctx=backend.index_ctx)

    summaries = {
        backend_id: _finalize_backend_summary(acc)
        for backend_id, acc in accumulators.items()
    }

    comparisons: list[dict[str, Any]] = []
    comparison_pairs: list[tuple[str, str]] = []
    for backend in backends:
        if backend.backend_id != "rg_native":
            comparison_pairs.append((backend.backend_id, "rg_native"))
    available_ids = {backend.backend_id for backend in backends}
    for lhs, rhs in (
        ("jina05b", "bge"),
        ("jina05b_hybrid", "bge_hybrid"),
        ("bge_hybrid", "bge"),
        ("jina05b_hybrid", "jina05b"),
    ):
        if lhs in available_ids and rhs in available_ids:
            comparison_pairs.append((lhs, rhs))
    seen_pairs: set[tuple[str, str]] = set()
    for lhs, rhs in comparison_pairs:
        pair = (lhs, rhs)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        comparison = _compare_backend_summaries(lhs, rhs, summaries=summaries)
        if comparison is not None:
            comparisons.append(comparison)

    protocol_backends: list[dict[str, Any]] = []
    for backend in backends:
        backend_entry = {
            "backend_id": backend.backend_id,
            "kind": backend.kind,
            "display_name": backend.display_name,
        }
        if backend.index_id is not None:
            backend_entry["index_id"] = backend.index_id
        if backend.model is not None:
            backend_entry["semantic_model"] = backend.model
        if backend.dimension is not None:
            backend_entry["semantic_dimension"] = backend.dimension
        if backend.retrieval_mode != "semantic":
            backend_entry["retrieval_mode"] = backend.retrieval_mode
            backend_entry["no_result_guard"] = backend.no_result_guard
            backend_entry["rerank"] = bool(backend.rerank)
            backend_entry["rerank_top_n"] = int(backend.rerank_top_n)
            if backend.projection_unit_k is not None:
                backend_entry["projection_unit_k"] = int(backend.projection_unit_k)
            if backend.rg_glob is not None:
                backend_entry["rg_glob"] = backend.rg_glob
        if backend.backend_id in daemon_ready:
            backend_entry["daemon_ready"] = daemon_ready[backend.backend_id]
        protocol_backends.append(backend_entry)

    report = make_report(
        phase="phase5_structured_retrieval",
        meta=gather_meta(tldr_repo_root=tldr_repo_root, corpus_id=corpus_id, corpus_root=repo_root),
        protocol={
            "schema_version": SCHEMA_VERSION,
            "feature_set_id": "feature.structured-retrieval.v1",
            "queries": str(queries_path),
            "cache_root": str(Path(args.cache_root).resolve()),
            "top_k": int(args.top_k),
            "use_daemon": bool(args.use_daemon),
            "include_hybrid": bool(args.include_hybrid),
            "hybrid_no_result_guard": str(args.hybrid_no_result_guard),
            "hybrid_rerank": bool(args.hybrid_rerank),
            "hybrid_rerank_top_n": int(args.hybrid_rerank_top_n),
            "hybrid_projection_unit_k": int(args.hybrid_projection_unit_k),
            "backends": protocol_backends,
            "rg_glob": str(args.rg_glob),
        },
        results={
            "by_backend": summaries,
            "comparisons": comparisons,
            "per_query": per_query,
        },
        schema_version=SCHEMA_VERSION,
    )

    out_path = Path(args.out).resolve() if args.out else (
        bench_runs_root(tldr_repo_root) / f"{now_utc_compact()}-structured-retrieval-{corpus_id}.json"
    )
    write_report(out_path, report)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
