#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_util import bench_runs_root, gather_meta, get_repo_root, now_utc_compact, write_report

SCHEMA_VERSION = 1
ALLOWED_STATUS = {"ok", "unsupported", "timeout", "error", "pending"}
CATEGORY_KEYS = ("retrieval", "impact", "slice", "complexity", "data_flow")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_out(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def _git_sha(repo_root: Path) -> str | None:
    return _run_out(["git", "rev-parse", "HEAD"], cwd=repo_root)


def _normalize_rel_path(path: str) -> str:
    return path.replace("\\", "/")


def _coerce_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return None
    return None


def _safe_mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _safe_median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    prec = None if tp + fp == 0 else tp / (tp + fp)
    rec = None if tp + fn == 0 else tp / (tp + fn)
    if prec is None or rec is None or prec + rec == 0:
        f1 = None
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


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
    return 1.0 if ranking[:k] else 0.0


def _kendall_tau_b(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    denom = (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    if denom <= 0:
        return None
    return (concordant - discordant) / (denom**0.5)


def _validate_suite(suite: dict[str, Any], repo_root: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    info: dict[str, Any] = {}

    if int(suite.get("schema_version", 0)) != SCHEMA_VERSION:
        errors.append("suite.schema_version must be 1")
    suite_id = suite.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id:
        errors.append("suite.suite_id must be a non-empty string")

    dataset = suite.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("suite.dataset must be an object")
        dataset = {}

    corpus_manifest_rel = dataset.get("corpus_manifest")
    corpus_id = dataset.get("corpus_id")
    required_sha = dataset.get("required_git_sha")
    required_ref = dataset.get("required_ref")

    if not isinstance(corpus_manifest_rel, str) or not corpus_manifest_rel:
        errors.append("dataset.corpus_manifest must be a non-empty string")
    if not isinstance(corpus_id, str) or not corpus_id:
        errors.append("dataset.corpus_id must be a non-empty string")
    if not isinstance(required_sha, str) or len(required_sha) < 7:
        errors.append("dataset.required_git_sha must be a git SHA string")
    if not isinstance(required_ref, str) or not required_ref:
        errors.append("dataset.required_ref must be a non-empty string")

    manifest_path = repo_root / str(corpus_manifest_rel)
    info["corpus_manifest"] = str(manifest_path)
    if not manifest_path.exists():
        errors.append(f"corpus manifest not found: {manifest_path}")
    else:
        data = _read_json(manifest_path)
        corpora = data.get("corpora") if isinstance(data, dict) else None
        if not isinstance(corpora, list):
            errors.append("corpus manifest has invalid 'corpora' list")
        else:
            found = None
            for c in corpora:
                if isinstance(c, dict) and c.get("id") == corpus_id:
                    found = c
                    break
            if found is None:
                errors.append(f"corpus '{corpus_id}' not found in {manifest_path}")
            else:
                pinned = found.get("pinned_sha")
                pinned_ref = found.get("pinned_ref")
                info["manifest_pinned_sha"] = pinned
                info["manifest_pinned_ref"] = pinned_ref
                if isinstance(required_sha, str) and pinned != required_sha:
                    errors.append(
                        "suite required_git_sha does not match manifest pinned_sha "
                        f"({required_sha} != {pinned})"
                    )
                if isinstance(required_ref, str) and pinned_ref != required_ref:
                    errors.append(
                        "suite required_ref does not match manifest pinned_ref "
                        f"({required_ref} != {pinned_ref})"
                    )

    sources = suite.get("sources")
    if not isinstance(sources, dict):
        errors.append("suite.sources must be an object")
        sources = {}

    for key in ("retrieval_queries", "structural_queries"):
        rel = sources.get(key)
        if not isinstance(rel, str) or not rel:
            errors.append(f"sources.{key} must be a non-empty string")
            continue
        p = repo_root / rel
        info[key] = str(p)
        if not p.exists():
            errors.append(f"query file not found: {p}")

    budgets = suite.get("budgets")
    if not isinstance(budgets, dict):
        errors.append("suite.budgets must be an object")
        budgets = {}
    token_budgets = budgets.get("token_budgets")
    if not isinstance(token_budgets, list) or not token_budgets:
        errors.append("budgets.token_budgets must be a non-empty list")
    else:
        if not all(isinstance(x, int) and x > 0 for x in token_budgets):
            errors.append("budgets.token_budgets must contain positive integers")

    protocol = suite.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("suite.protocol must be an object")
    else:
        trials = protocol.get("trials")
        seeds = protocol.get("seeds")
        if not isinstance(trials, int) or trials <= 0:
            errors.append("protocol.trials must be a positive integer")
        if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
            errors.append("protocol.seeds must be a list of integers")

    lanes = suite.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        errors.append("suite.lanes must be a non-empty list")
    else:
        lane_ids: set[str] = set()
        categories_seen: set[str] = set()
        for lane in lanes:
            if not isinstance(lane, dict):
                errors.append("lane must be an object")
                continue
            lid = lane.get("id")
            cats = lane.get("categories")
            if not isinstance(lid, str) or not lid:
                errors.append("lane.id must be a non-empty string")
            elif lid in lane_ids:
                errors.append(f"duplicate lane.id: {lid}")
            else:
                lane_ids.add(lid)
            if not isinstance(cats, list) or not cats:
                errors.append(f"lane.categories must be non-empty for lane {lid}")
                continue
            for cat in cats:
                if cat not in CATEGORY_KEYS:
                    errors.append(f"unknown category in lane {lid}: {cat}")
                categories_seen.add(str(cat))
        if "retrieval" not in categories_seen:
            errors.append("suite must include retrieval in at least one lane")

    return errors, info


def _validate_tool_profile(profile: dict[str, Any], suite_id: str) -> list[str]:
    errors: list[str] = []

    if int(profile.get("schema_version", 0)) != SCHEMA_VERSION:
        errors.append("tool profile schema_version must be 1")
    if profile.get("suite_id") != suite_id:
        errors.append("tool profile suite_id mismatch")

    tool_id = profile.get("tool_id")
    if not isinstance(tool_id, str) or not tool_id:
        errors.append("tool profile tool_id must be non-empty")

    caps = profile.get("capabilities")
    if not isinstance(caps, dict):
        errors.append("tool profile capabilities must be an object")
        caps = {}

    commands = profile.get("commands")
    if not isinstance(commands, dict):
        errors.append("tool profile commands must be an object")
        commands = {}

    feature_set_id = profile.get("feature_set_id")
    if feature_set_id is not None and (not isinstance(feature_set_id, str) or not feature_set_id.strip()):
        errors.append("tool profile feature_set_id must be a non-empty string when provided")

    for cat in CATEGORY_KEYS:
        v = caps.get(cat)
        if not isinstance(v, bool):
            errors.append(f"capabilities.{cat} must be boolean")
            continue
        if v:
            cmd = commands.get(cat)
            if not isinstance(cmd, dict):
                errors.append(f"commands.{cat} must be present when capability is true")
                continue
            tmpl = cmd.get("template")
            if not isinstance(tmpl, str) or not tmpl:
                errors.append(f"commands.{cat}.template must be a non-empty string")

    return errors


def _find_python_span(tree: ast.AST, function: str) -> tuple[int, int] | None:
    class_name = None
    func_name = function
    if "." in function:
        class_name, func_name = function.split(".", 1)

    def _match(node: ast.AST, *, want: str) -> bool:
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == want

    if class_name:
        for n in getattr(tree, "body", []):
            if isinstance(n, ast.ClassDef) and n.name == class_name:
                for m in n.body:
                    if _match(m, want=func_name) and hasattr(m, "lineno") and hasattr(m, "end_lineno"):
                        return int(m.lineno), int(m.end_lineno)  # type: ignore[attr-defined]
        return None

    for n in getattr(tree, "body", []):
        if _match(n, want=func_name) and hasattr(n, "lineno") and hasattr(n, "end_lineno"):
            return int(n.lineno), int(n.end_lineno)  # type: ignore[attr-defined]

    for n in ast.walk(tree):
        if _match(n, want=func_name) and hasattr(n, "lineno") and hasattr(n, "end_lineno"):
            return int(n.lineno), int(n.end_lineno)  # type: ignore[attr-defined]
    return None


def _radon_cc(source: str, function: str) -> int | None:
    try:
        from radon.complexity import cc_visit  # type: ignore[import-not-found]
    except Exception:
        return None

    class_name = None
    func_name = function
    if "." in function:
        class_name, func_name = function.split(".", 1)

    blocks = cc_visit(source)
    for b in blocks:
        name = getattr(b, "name", None)
        if class_name and name == class_name:
            for m in getattr(b, "methods", []) or []:
                if getattr(m, "name", None) == func_name:
                    return int(getattr(m, "complexity", 0))
        if not class_name and name == func_name:
            return int(getattr(b, "complexity", 0))
    return None


def _materialize_tasks(
    *,
    suite: dict[str, Any],
    repo_root: Path,
    corpus_root: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    warnings: list[str] = []
    sources = suite["sources"]

    retrieval_path = (repo_root / str(sources["retrieval_queries"])).resolve()
    structural_path = (repo_root / str(sources["structural_queries"])).resolve()
    retrieval_data = _read_json(retrieval_path)
    structural_data = _read_json(structural_path)

    retrieval_queries = retrieval_data.get("queries") if isinstance(retrieval_data, dict) else None
    structural_queries = structural_data.get("queries") if isinstance(structural_data, dict) else None

    if not isinstance(retrieval_queries, list):
        raise SystemExit(f"error: bad retrieval query file: {retrieval_path}")
    if not isinstance(structural_queries, list):
        raise SystemExit(f"error: bad structural query file: {structural_path}")

    source_hashes = {
        str(Path(sources["retrieval_queries"])): _sha256_file(retrieval_path),
        str(Path(sources["structural_queries"])): _sha256_file(structural_path),
    }

    tasks: list[dict[str, Any]] = []
    source_cache: dict[str, tuple[str, list[str], ast.AST | None]] = {}

    def load_source(file_rel: str) -> tuple[str, list[str], ast.AST | None]:
        key = _normalize_rel_path(file_rel)
        if key in source_cache:
            return source_cache[key]
        abs_path = corpus_root / key
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            source = ""
        lines = source.splitlines()
        tree = None
        if source:
            try:
                tree = ast.parse(source)
            except SyntaxError:
                tree = None
        source_cache[key] = (source, lines, tree)
        return source_cache[key]

    for q in retrieval_queries:
        if not isinstance(q, dict):
            continue
        qid = q.get("id")
        query = q.get("query")
        relevant_files = q.get("relevant_files")
        if not isinstance(qid, str) or not isinstance(query, str) or not isinstance(relevant_files, list):
            warnings.append(f"skipped malformed retrieval query: {q}")
            continue
        relevant = [
            _normalize_rel_path(x)
            for x in relevant_files
            if isinstance(x, str)
        ]
        tasks.append(
            {
                "task_id": f"retrieval:{qid}",
                "query_id": qid,
                "category": "retrieval",
                "input": {
                    "query": query,
                    "rg_pattern": q.get("rg_pattern") if isinstance(q.get("rg_pattern"), str) else None,
                },
                "ground_truth": {
                    "relevant_files": relevant,
                    "is_negative": len(relevant) == 0,
                },
                "source": str(Path(sources["retrieval_queries"])),
            }
        )

    for q in structural_queries:
        if not isinstance(q, dict):
            continue
        qid = q.get("id")
        category = q.get("category")
        if not isinstance(qid, str) or not isinstance(category, str):
            warnings.append(f"skipped malformed structural query: {q}")
            continue
        if category not in {"impact", "slice", "complexity", "data_flow"}:
            continue

        base: dict[str, Any] = {
            "task_id": f"{category}:{qid}",
            "query_id": qid,
            "category": category,
            "difficulty": q.get("difficulty") if isinstance(q.get("difficulty"), str) else None,
            "source": str(Path(sources["structural_queries"])),
        }

        if category == "impact":
            function = q.get("function")
            file_rel = q.get("file")
            expected_callers = q.get("expected_callers")
            if not isinstance(function, str) or not isinstance(file_rel, str) or not isinstance(expected_callers, list):
                warnings.append(f"skipped malformed impact query {qid}")
                continue
            callers: list[dict[str, str]] = []
            for c in expected_callers:
                if not isinstance(c, dict):
                    continue
                fp = c.get("file")
                fn = c.get("function")
                if isinstance(fp, str) and isinstance(fn, str):
                    callers.append({"file": _normalize_rel_path(fp), "function": fn})
            base["input"] = {
                "function": function,
                "file": _normalize_rel_path(file_rel),
            }
            base["ground_truth"] = {"callers": callers}
            tasks.append(base)
            continue

        if category == "slice":
            file_rel = q.get("file")
            function = q.get("function")
            target_line = q.get("target_line")
            expected_lines = q.get("expected_slice_lines")
            if (
                not isinstance(file_rel, str)
                or not isinstance(function, str)
                or not isinstance(target_line, int)
                or not isinstance(expected_lines, list)
            ):
                warnings.append(f"skipped malformed slice query {qid}")
                continue

            _, _, tree = load_source(file_rel)
            total_lines = None
            if tree is not None:
                span = _find_python_span(tree, function)
                if span is not None:
                    total_lines = max(1, int(span[1] - span[0] + 1))

            exp = sorted({int(x) for x in expected_lines if isinstance(x, int)})
            base["input"] = {
                "file": _normalize_rel_path(file_rel),
                "function": function,
                "target_line": int(target_line),
            }
            base["ground_truth"] = {
                "lines": exp,
                "total_function_lines": total_lines,
            }
            tasks.append(base)
            continue

        if category == "complexity":
            file_rel = q.get("file")
            function = q.get("function")
            if not isinstance(file_rel, str) or not isinstance(function, str):
                warnings.append(f"skipped malformed complexity query {qid}")
                continue

            source, _, _ = load_source(file_rel)
            expected_cc = _radon_cc(source, function=function) if source else None
            if expected_cc is None:
                warnings.append(f"complexity ground truth missing for {qid} ({file_rel}:{function})")

            base["input"] = {
                "file": _normalize_rel_path(file_rel),
                "function": function,
            }
            base["ground_truth"] = {
                "cyclomatic_complexity": expected_cc,
            }
            tasks.append(base)
            continue

        if category == "data_flow":
            file_rel = q.get("file")
            function = q.get("function")
            variable = q.get("variable")
            expected_flow = q.get("expected_flow")
            if (
                not isinstance(file_rel, str)
                or not isinstance(function, str)
                or not isinstance(variable, str)
                or not isinstance(expected_flow, list)
            ):
                warnings.append(f"skipped malformed data_flow query {qid}")
                continue

            lines: list[int] = []
            origin_line = None
            for ev in expected_flow:
                if not isinstance(ev, dict):
                    continue
                ln = ev.get("line")
                if isinstance(ln, int):
                    lines.append(int(ln))
                    if origin_line is None and ev.get("event") == "defined":
                        origin_line = int(ln)

            base["input"] = {
                "file": _normalize_rel_path(file_rel),
                "function": function,
                "variable": variable,
            }
            base["ground_truth"] = {
                "flow_lines": sorted(set(lines)),
                "origin_line": origin_line,
            }
            tasks.append(base)
            continue

    tasks.sort(key=lambda t: t["task_id"])
    return tasks, warnings, source_hashes


def _required_categories(suite: dict[str, Any], capabilities: dict[str, bool]) -> set[str]:
    required: set[str] = set()
    lanes = suite.get("lanes") if isinstance(suite, dict) else None
    if not isinstance(lanes, list):
        return set(CATEGORY_KEYS)
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        categories = lane.get("categories")
        if not isinstance(categories, list):
            continue
        req_for_all = bool(lane.get("required_for_all_tools"))
        for c in categories:
            if c not in CATEGORY_KEYS:
                continue
            if req_for_all:
                required.add(c)
            elif bool(capabilities.get(c, False)):
                required.add(c)
    return required


def _empty_acc(ks: list[int]) -> dict[str, dict[str, list[float]]]:
    return {
        "retrieval": {
            "mrr": [],
            **{f"recall@{k}": [] for k in ks},
            **{f"precision@{k}": [] for k in ks},
            **{f"fpr@{k}": [] for k in ks},
            "latency_ms": [],
            "payload_tokens": [],
            "payload_bytes": [],
        },
        "impact": {
            "precision": [],
            "recall": [],
            "f1": [],
            "latency_ms": [],
            "payload_tokens": [],
            "payload_bytes": [],
        },
        "slice": {
            "precision": [],
            "recall": [],
            "f1": [],
            "noise_reduction": [],
            "latency_ms": [],
            "payload_tokens": [],
            "payload_bytes": [],
        },
        "complexity": {
            "accuracy": [],
            "abs_error": [],
            "predicted_scores": [],
            "expected_scores": [],
            "latency_ms": [],
            "payload_tokens": [],
            "payload_bytes": [],
        },
        "data_flow": {
            "origin_accuracy": [],
            "flow_completeness": [],
            "latency_ms": [],
            "payload_tokens": [],
            "payload_bytes": [],
        },
    }


def _summarize_acc(acc: dict[str, dict[str, list[float]]], ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    r = acc["retrieval"]
    out["retrieval"] = {
        "mrr_mean": _safe_mean(r["mrr"]),
        **{f"recall@{k}_mean": _safe_mean(r[f"recall@{k}"]) for k in ks},
        **{f"precision@{k}_mean": _safe_mean(r[f"precision@{k}"]) for k in ks},
        **{f"fpr@{k}_mean": _safe_mean(r[f"fpr@{k}"]) for k in ks},
        "latency_ms_p50": _safe_median(r["latency_ms"]),
        "payload_tokens_median": _safe_median(r["payload_tokens"]),
        "payload_bytes_median": _safe_median(r["payload_bytes"]),
    }

    i = acc["impact"]
    out["impact"] = {
        "precision_mean": _safe_mean(i["precision"]),
        "recall_mean": _safe_mean(i["recall"]),
        "f1_mean": _safe_mean(i["f1"]),
        "latency_ms_p50": _safe_median(i["latency_ms"]),
        "payload_tokens_median": _safe_median(i["payload_tokens"]),
    }

    s = acc["slice"]
    out["slice"] = {
        "precision_mean": _safe_mean(s["precision"]),
        "recall_mean": _safe_mean(s["recall"]),
        "f1_mean": _safe_mean(s["f1"]),
        "noise_reduction_mean": _safe_mean(s["noise_reduction"]),
        "latency_ms_p50": _safe_median(s["latency_ms"]),
        "payload_tokens_median": _safe_median(s["payload_tokens"]),
    }

    c = acc["complexity"]
    out["complexity"] = {
        "accuracy_mean": _safe_mean(c["accuracy"]),
        "mae": _safe_mean(c["abs_error"]),
        "kendall_tau_b": _kendall_tau_b(c["predicted_scores"], c["expected_scores"]),
        "latency_ms_p50": _safe_median(c["latency_ms"]),
        "payload_tokens_median": _safe_median(c["payload_tokens"]),
    }

    d = acc["data_flow"]
    out["data_flow"] = {
        "origin_accuracy_mean": _safe_mean(d["origin_accuracy"]),
        "flow_completeness_mean": _safe_mean(d["flow_completeness"]),
        "latency_ms_p50": _safe_median(d["latency_ms"]),
        "payload_tokens_median": _safe_median(d["payload_tokens"]),
    }

    return out


def _score_retrieval_entry(gt: dict[str, Any], result: dict[str, Any], ks: list[int]) -> dict[str, float]:
    ranked_raw = result.get("ranked_files")
    ranking = []
    if isinstance(ranked_raw, list):
        for x in ranked_raw:
            if isinstance(x, str):
                ranking.append(_normalize_rel_path(x))

    relevant = {
        _normalize_rel_path(x)
        for x in gt.get("relevant_files", [])
        if isinstance(x, str)
    }

    out: dict[str, float] = {}
    if relevant:
        out["mrr"] = _mrr(ranking, relevant)
        for k in ks:
            out[f"recall@{k}"] = _recall_at_k(ranking, relevant, k)
            out[f"precision@{k}"] = _precision_at_k(ranking, relevant, k)
    else:
        for k in ks:
            out[f"fpr@{k}"] = _fpr_at_k(ranking, k=k)
    return out


def _score_impact_entry(gt: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    expected_callers = set()
    for c in gt.get("callers", []):
        if not isinstance(c, dict):
            continue
        fp = c.get("file")
        fn = c.get("function")
        if isinstance(fp, str) and isinstance(fn, str):
            expected_callers.add((_normalize_rel_path(fp), fn))

    predicted = set()
    for c in result.get("callers", []) if isinstance(result.get("callers"), list) else []:
        if not isinstance(c, dict):
            continue
        fp = c.get("file")
        fn = c.get("function")
        if isinstance(fp, str) and isinstance(fn, str):
            predicted.add((_normalize_rel_path(fp), fn))

    tp = len(predicted & expected_callers)
    fp = len(predicted - expected_callers)
    fn = len(expected_callers - predicted)
    prec, rec, f1 = _prf(tp, fp, fn)

    out: dict[str, float] = {}
    if prec is not None:
        out["precision"] = prec
    if rec is not None:
        out["recall"] = rec
    if f1 is not None:
        out["f1"] = f1
    return out


def _score_slice_entry(gt: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    expected = {int(x) for x in gt.get("lines", []) if isinstance(x, int)}
    predicted = {int(x) for x in result.get("lines", []) if isinstance(x, int)}

    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    prec, rec, f1 = _prf(tp, fp, fn)

    out: dict[str, float] = {}
    if prec is not None:
        out["precision"] = prec
    if rec is not None:
        out["recall"] = rec
    if f1 is not None:
        out["f1"] = f1

    total_lines = _coerce_int(gt.get("total_function_lines"))
    if total_lines and total_lines > 0:
        out["noise_reduction"] = 1.0 - (len(predicted) / total_lines)
    return out


def _score_complexity_entry(gt: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    expected = _coerce_int(gt.get("cyclomatic_complexity"))
    predicted = _coerce_int(result.get("cyclomatic_complexity"))
    if expected is None or predicted is None:
        return {}
    return {
        "accuracy": 1.0 if expected == predicted else 0.0,
        "abs_error": float(abs(expected - predicted)),
        "predicted_score": float(predicted),
        "expected_score": float(expected),
    }


def _score_data_flow_entry(gt: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    expected_lines = {int(x) for x in gt.get("flow_lines", []) if isinstance(x, int)}
    predicted_lines = {int(x) for x in result.get("flow_lines", []) if isinstance(x, int)}

    out: dict[str, float] = {}
    if expected_lines:
        out["flow_completeness"] = len(expected_lines & predicted_lines) / len(expected_lines)

    exp_origin = _coerce_int(gt.get("origin_line"))
    pred_origin = _coerce_int(result.get("origin_line"))
    if exp_origin is not None and pred_origin is not None:
        out["origin_accuracy"] = 1.0 if exp_origin == pred_origin else 0.0
    return out


def _result_shape_matches_category(category: str, result: dict[str, Any]) -> bool:
    if category == "retrieval":
        return isinstance(result.get("ranked_files"), list)
    if category == "impact":
        return isinstance(result.get("callers"), list)
    if category == "slice":
        return isinstance(result.get("lines"), list)
    if category == "complexity":
        return _coerce_int(result.get("cyclomatic_complexity")) is not None
    if category == "data_flow":
        has_flow_lines = isinstance(result.get("flow_lines"), list)
        has_origin_line = _coerce_int(result.get("origin_line")) is not None
        return has_flow_lines or has_origin_line
    return True


def _append_latency_and_payload(cat_acc: dict[str, list[float]], pred: dict[str, Any]) -> None:
    latency_ms = pred.get("latency_ms")
    payload_tokens = pred.get("payload_tokens")
    payload_bytes = pred.get("payload_bytes")

    if isinstance(latency_ms, (int, float)):
        cat_acc["latency_ms"].append(float(latency_ms))
    if isinstance(payload_tokens, int):
        cat_acc["payload_tokens"].append(float(payload_tokens))
    if isinstance(payload_bytes, int):
        cat_acc["payload_bytes"].append(float(payload_bytes))


def _add_gate(
    gates: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    actual: Any,
    expected: Any,
    skipped: bool = False,
    reason: str | None = None,
) -> None:
    entry = {
        "name": name,
        "pass": bool(passed),
        "actual": actual,
        "expected": expected,
        "skipped": bool(skipped),
    }
    if reason:
        entry["reason"] = reason
    gates.append(entry)


def cmd_validate_suite(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    suite_path = Path(args.suite).resolve()
    if not suite_path.exists():
        raise SystemExit(f"error: suite file not found: {suite_path}")

    suite = _read_json(suite_path)
    if not isinstance(suite, dict):
        raise SystemExit("error: suite must be a JSON object")

    errors, info = _validate_suite(suite, repo_root)

    tool_results: list[dict[str, Any]] = []
    for p in args.tool_profile or []:
        tp = Path(p).resolve()
        if not tp.exists():
            errors.append(f"tool profile not found: {tp}")
            continue
        profile = _read_json(tp)
        if not isinstance(profile, dict):
            errors.append(f"tool profile must be object: {tp}")
            continue
        perr = _validate_tool_profile(profile, str(suite.get("suite_id")))
        tool_results.append(
            {
                "path": str(tp),
                "tool_id": profile.get("tool_id"),
                "errors": perr,
            }
        )
        errors.extend([f"{tp}: {e}" for e in perr])

    out = {
        "schema_version": SCHEMA_VERSION,
        "suite": str(suite_path),
        "suite_sha256": _sha256_file(suite_path),
        "ok": not errors,
        "errors": errors,
        "suite_info": info,
        "tool_profiles": tool_results,
        "validated_at_utc": _utc_now(),
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = bench_runs_root(repo_root) / f"{now_utc_compact()}-h2h-validate.json"
    write_report(out_path, out)
    print(out_path)
    return 0 if not errors else 2


def cmd_materialize_tasks(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    suite_path = Path(args.suite).resolve()
    suite = _read_json(suite_path)
    if not isinstance(suite, dict):
        raise SystemExit("error: suite must be a JSON object")

    errors, _ = _validate_suite(suite, repo_root)
    if errors:
        raise SystemExit("error: suite validation failed:\n- " + "\n- ".join(errors))

    dataset = suite["dataset"]
    corpus_id = str(dataset["corpus_id"])
    if args.corpus_root:
        corpus_root = Path(args.corpus_root).resolve()
    else:
        corpus_root = (repo_root / "benchmark" / "corpora" / corpus_id).resolve()

    if not corpus_root.exists():
        raise SystemExit(f"error: corpus root does not exist: {corpus_root}")

    actual_sha = _git_sha(corpus_root)
    required_sha = str(dataset["required_git_sha"])
    # Allow manifests that pin an annotated tag object SHA by resolving it to
    # the underlying commit before comparing with HEAD.
    resolved_required_sha = (
        _run_out(["git", "rev-parse", f"{required_sha}^{{}}"], cwd=corpus_root) or required_sha
    )
    if actual_sha != resolved_required_sha:
        raise SystemExit(
            "error: corpus SHA mismatch: "
            f"expected {resolved_required_sha}, got {actual_sha or 'unknown'}"
        )

    tasks, warnings, source_hashes = _materialize_tasks(
        suite=suite,
        repo_root=repo_root,
        corpus_root=corpus_root,
    )

    by_category: dict[str, int] = {}
    for t in tasks:
        cat = t.get("category")
        if isinstance(cat, str):
            by_category[cat] = by_category.get(cat, 0) + 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite.get("suite_id"),
        "generated_at_utc": _utc_now(),
        "suite_path": str(suite_path),
        "suite_sha256": _sha256_file(suite_path),
        "dataset": {
            "corpus_id": corpus_id,
            "corpus_root": str(corpus_root),
            "required_git_sha": required_sha,
            "required_git_sha_resolved": resolved_required_sha,
            "actual_git_sha": actual_sha,
        },
        "source_hashes": source_hashes,
        "task_counts": {
            "total": len(tasks),
            "by_category": by_category,
        },
        "tasks": tasks,
        "warnings": warnings,
    }
    manifest["task_manifest_sha256"] = _sha256_json(manifest["tasks"])

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = bench_runs_root(repo_root) / f"{now_utc_compact()}-h2h-task-manifest.json"
    write_report(out_path, manifest)
    print(out_path)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    suite_path = Path(args.suite).resolve()
    tasks_path = Path(args.tasks).resolve()
    preds_path = Path(args.predictions).resolve()

    suite = _read_json(suite_path)
    tasks_doc = _read_json(tasks_path)
    preds_doc = _read_json(preds_path)

    if not isinstance(suite, dict) or not isinstance(tasks_doc, dict) or not isinstance(preds_doc, dict):
        raise SystemExit("error: suite/tasks/predictions must all be JSON objects")

    suite_id = str(suite.get("suite_id"))
    if tasks_doc.get("suite_id") != suite_id:
        raise SystemExit("error: task manifest suite_id mismatch")
    if preds_doc.get("suite_id") != suite_id:
        raise SystemExit("error: predictions suite_id mismatch")

    tasks = tasks_doc.get("tasks")
    if not isinstance(tasks, list):
        raise SystemExit("error: task manifest missing 'tasks' list")

    manifest_hash = tasks_doc.get("task_manifest_sha256")
    actual_manifest_hash = _sha256_json(tasks)
    if manifest_hash != actual_manifest_hash:
        raise SystemExit("error: task manifest hash mismatch (file may have been edited)")

    preds_manifest_hash = preds_doc.get("task_manifest_sha256")
    if isinstance(preds_manifest_hash, str) and preds_manifest_hash != actual_manifest_hash:
        raise SystemExit("error: predictions task_manifest_sha256 mismatch")

    tokenizer = preds_doc.get("tokenizer")
    expected_tokenizer = suite.get("budgets", {}).get("tokenizer")

    profile_path = Path(args.tool_profile).resolve() if args.tool_profile else None
    profile = None
    profile_errors: list[str] = []
    capabilities: dict[str, bool] = {k: True for k in CATEGORY_KEYS}
    tool_id = preds_doc.get("tool_id")
    feature_set_id = preds_doc.get("feature_set_id")

    if profile_path is not None:
        profile = _read_json(profile_path)
        if not isinstance(profile, dict):
            raise SystemExit("error: tool profile must be a JSON object")
        profile_errors = _validate_tool_profile(profile, suite_id)
        if profile_errors:
            raise SystemExit("error: invalid tool profile:\n- " + "\n- ".join(profile_errors))
        caps = profile.get("capabilities")
        if isinstance(caps, dict):
            capabilities = {k: bool(caps.get(k, False)) for k in CATEGORY_KEYS}
        if not isinstance(tool_id, str) or not tool_id:
            tool_id = profile.get("tool_id")
        if not isinstance(feature_set_id, str) or not feature_set_id:
            feature_set_id = profile.get("feature_set_id")

    if not isinstance(tool_id, str) or not tool_id:
        tool_id = "unknown-tool"
    if not isinstance(feature_set_id, str) or not feature_set_id:
        feature_set_id = "unspecified"

    required_categories = _required_categories(suite, capabilities)

    budgets_raw = suite.get("budgets", {}).get("token_budgets", [2000])
    budgets = [int(x) for x in budgets_raw if isinstance(x, int)]
    if not budgets:
        budgets = [2000]

    ks_raw = suite.get("budgets", {}).get("retrieval_ks", [1, 5, 10])
    ks = sorted({int(x) for x in ks_raw if isinstance(x, int) and x > 0})
    if not ks:
        ks = [1, 5, 10]

    trials = int(suite.get("protocol", {}).get("trials", 1))
    if trials <= 0:
        trials = 1

    tasks_by_id: dict[str, dict[str, Any]] = {}
    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = t.get("task_id")
        cat = t.get("category")
        if isinstance(tid, str) and isinstance(cat, str):
            tasks_by_id[tid] = t

    expected_keys: set[tuple[str, int, int]] = set()
    expected_by_category: dict[str, int] = {k: 0 for k in CATEGORY_KEYS}
    for tid, task in tasks_by_id.items():
        cat = str(task.get("category"))
        if cat not in required_categories:
            continue
        for budget in budgets:
            for trial in range(1, trials + 1):
                expected_keys.add((tid, budget, trial))
                expected_by_category[cat] = expected_by_category.get(cat, 0) + 1

    preds = preds_doc.get("predictions")
    if not isinstance(preds, list):
        raise SystemExit("error: predictions must contain a 'predictions' list")

    pred_map: dict[tuple[str, int, int], dict[str, Any]] = {}
    duplicate_keys = 0
    for p in preds:
        if not isinstance(p, dict):
            continue
        tid = p.get("task_id")
        budget = _coerce_int(p.get("budget_tokens"))
        trial = _coerce_int(p.get("trial"))
        if not isinstance(tid, str) or budget is None or trial is None:
            continue
        key = (tid, budget, trial)
        if key in pred_map:
            duplicate_keys += 1
            continue
        pred_map[key] = p

    acc_by_budget: dict[int, dict[str, dict[str, list[float]]]] = {
        b: _empty_acc(ks) for b in budgets
    }
    overall_acc = _empty_acc(ks)

    status_counts = {
        "expected_total": len(expected_keys),
        "ok": 0,
        "unsupported": 0,
        "timeout": 0,
        "error": 0,
        "pending": 0,
        "missing": 0,
        "extra": 0,
        "duplicate": duplicate_keys,
        "budget_violations": 0,
    }
    ok_by_category: dict[str, int] = {k: 0 for k in CATEGORY_KEYS}
    shape_mismatch_by_category: dict[str, int] = {k: 0 for k in CATEGORY_KEYS}
    result_shape_counters: dict[str, Any] = {
        "non_object_result": 0,
        "empty_result_object": 0,
        "category_shape_mismatch": 0,
        "category_shape_mismatch_by_category": shape_mismatch_by_category,
    }

    parse_errors: list[str] = []

    def merge_entry_metrics(
        *,
        budget: int,
        category: str,
        metrics: dict[str, float],
    ) -> None:
        bacc = acc_by_budget[budget][category]
        oacc = overall_acc[category]
        for k, v in metrics.items():
            if k not in bacc:
                continue
            bacc[k].append(float(v))
            oacc[k].append(float(v))

    for key in sorted(expected_keys):
        tid, budget, trial = key
        task = tasks_by_id.get(tid)
        if task is None:
            status_counts["missing"] += 1
            parse_errors.append(f"missing task definition for task_id={tid}")
            continue

        category = str(task.get("category"))
        pred = pred_map.get(key)
        if pred is None:
            status_counts["missing"] += 1
            continue

        status = str(pred.get("status") or "error").lower()
        if status not in ALLOWED_STATUS:
            status = "error"

        if status != "ok":
            status_counts[status] += 1
            continue

        status_counts["ok"] += 1
        ok_by_category[category] = ok_by_category.get(category, 0) + 1

        payload_tokens = _coerce_int(pred.get("payload_tokens"))
        if payload_tokens is not None and payload_tokens > budget:
            status_counts["budget_violations"] += 1

        result = pred.get("result")
        if not isinstance(result, dict):
            result_shape_counters["non_object_result"] += 1
            status_counts["error"] += 1
            parse_errors.append(f"{tid}@{budget}/trial{trial}: result must be object")
            continue
        if not result:
            result_shape_counters["empty_result_object"] += 1
        elif not _result_shape_matches_category(category, result):
            result_shape_counters["category_shape_mismatch"] += 1
            if category in shape_mismatch_by_category:
                shape_mismatch_by_category[category] += 1

        try:
            metrics: dict[str, float]
            gt = task.get("ground_truth")
            if not isinstance(gt, dict):
                gt = {}

            if category == "retrieval":
                metrics = _score_retrieval_entry(gt, result, ks)
            elif category == "impact":
                metrics = _score_impact_entry(gt, result)
            elif category == "slice":
                metrics = _score_slice_entry(gt, result)
            elif category == "complexity":
                metrics = _score_complexity_entry(gt, result)
            elif category == "data_flow":
                metrics = _score_data_flow_entry(gt, result)
            else:
                metrics = {}

            merge_entry_metrics(budget=budget, category=category, metrics=metrics)
            _append_latency_and_payload(acc_by_budget[budget][category], pred)
            _append_latency_and_payload(overall_acc[category], pred)

        except Exception as exc:  # defensive: malformed external predictions
            status_counts["error"] += 1
            parse_errors.append(f"{tid}@{budget}/trial{trial}: {exc}")

    for key in pred_map:
        if key not in expected_keys:
            status_counts["extra"] += 1

    result_shape_counters["total"] = (
        int(result_shape_counters["non_object_result"])
        + int(result_shape_counters["empty_result_object"])
        + int(result_shape_counters["category_shape_mismatch"])
    )

    summary_by_budget: dict[str, Any] = {}
    for budget in budgets:
        summary_by_budget[str(budget)] = _summarize_acc(acc_by_budget[budget], ks)
    summary_overall = _summarize_acc(overall_acc, ks)

    expected_total = max(1, status_counts["expected_total"])
    retrieval_expected = expected_by_category.get("retrieval", 0)
    retrieval_ok = ok_by_category.get("retrieval", 0)

    rates = {
        "timeout_rate": status_counts["timeout"] / expected_total,
        "error_rate": (status_counts["error"] + status_counts["missing"]) / expected_total,
        "unsupported_rate": status_counts["unsupported"] / expected_total,
        "pending_rate": status_counts["pending"] / expected_total,
        "budget_violation_rate": status_counts["budget_violations"] / expected_total,
        "common_lane_coverage": (retrieval_ok / retrieval_expected) if retrieval_expected else None,
        "capability_coverage": status_counts["ok"] / expected_total,
    }

    gates: list[dict[str, Any]] = []
    gate_cfg = suite.get("gates", {}) if isinstance(suite.get("gates"), dict) else {}

    run_validity = gate_cfg.get("run_validity") if isinstance(gate_cfg.get("run_validity"), dict) else {}
    max_timeout_rate = float(run_validity.get("max_timeout_rate", 1.0))
    max_error_rate = float(run_validity.get("max_error_rate", 1.0))
    max_budget_violation_rate = float(run_validity.get("max_budget_violation_rate", 1.0))

    _add_gate(
        gates,
        name="run_validity.max_timeout_rate",
        passed=(rates["timeout_rate"] <= max_timeout_rate),
        actual=rates["timeout_rate"],
        expected=f"<= {max_timeout_rate}",
    )
    _add_gate(
        gates,
        name="run_validity.max_error_rate",
        passed=(rates["error_rate"] <= max_error_rate),
        actual=rates["error_rate"],
        expected=f"<= {max_error_rate}",
    )
    _add_gate(
        gates,
        name="run_validity.max_budget_violation_rate",
        passed=(rates["budget_violation_rate"] <= max_budget_violation_rate),
        actual=rates["budget_violation_rate"],
        expected=f"<= {max_budget_violation_rate}",
    )

    fairness = gate_cfg.get("fairness") if isinstance(gate_cfg.get("fairness"), dict) else {}
    expected_tok = fairness.get("require_same_tokenizer")
    _add_gate(
        gates,
        name="fairness.tokenizer_match",
        passed=(expected_tok is None or tokenizer == expected_tok),
        actual=tokenizer,
        expected=expected_tok,
    )

    budgets_in_preds = sorted(
        {
            int(_coerce_int(p.get("budget_tokens")) or -1)
            for p in preds
            if isinstance(p, dict) and _coerce_int(p.get("budget_tokens")) is not None
        }
    )
    _add_gate(
        gates,
        name="fairness.budget_set_match",
        passed=(budgets_in_preds == sorted(budgets)),
        actual=budgets_in_preds,
        expected=sorted(budgets),
    )

    if bool(fairness.get("require_identical_task_manifest_hash", False)):
        _add_gate(
            gates,
            name="fairness.task_manifest_hash_match",
            passed=(preds_manifest_hash == actual_manifest_hash),
            actual=preds_manifest_hash,
            expected=actual_manifest_hash,
        )

    quality_cfg = gate_cfg.get("tool_quality") if isinstance(gate_cfg.get("tool_quality"), dict) else {}
    primary_budget = 2000
    h2h_cfg = gate_cfg.get("head_to_head") if isinstance(gate_cfg.get("head_to_head"), dict) else {}
    if isinstance(h2h_cfg.get("primary_budget"), int):
        primary_budget = int(h2h_cfg["primary_budget"])

    pb = summary_by_budget.get(str(primary_budget), {})
    pb_retrieval = pb.get("retrieval", {}) if isinstance(pb, dict) else {}

    min_common_cov = float(quality_cfg.get("common_lane_min_coverage", 0.0))
    common_cov = rates.get("common_lane_coverage")
    _add_gate(
        gates,
        name="tool_quality.common_lane_min_coverage",
        passed=(common_cov is not None and common_cov >= min_common_cov),
        actual=common_cov,
        expected=f">= {min_common_cov}",
    )

    min_mrr = quality_cfg.get("retrieval_min_mrr_at_budget_2000")
    if isinstance(min_mrr, (int, float)):
        mrr_v = pb_retrieval.get("mrr_mean") if isinstance(pb_retrieval, dict) else None
        _add_gate(
            gates,
            name="tool_quality.retrieval_min_mrr_at_budget_2000",
            passed=(isinstance(mrr_v, (int, float)) and float(mrr_v) >= float(min_mrr)),
            actual=mrr_v,
            expected=f">= {float(min_mrr)}",
        )

    max_fpr5 = quality_cfg.get("retrieval_max_fpr5_at_budget_2000")
    if isinstance(max_fpr5, (int, float)):
        fpr_v = pb_retrieval.get("fpr@5_mean") if isinstance(pb_retrieval, dict) else None
        _add_gate(
            gates,
            name="tool_quality.retrieval_max_fpr5_at_budget_2000",
            passed=(fpr_v is None or (isinstance(fpr_v, (int, float)) and float(fpr_v) <= float(max_fpr5))),
            actual=fpr_v,
            expected=f"<= {float(max_fpr5)}",
        )

    def _optional_category_gate(
        *,
        category: str,
        metric_key: str,
        gate_name: str,
        threshold: float,
        lower_is_better: bool,
    ) -> None:
        supported = bool(capabilities.get(category, False))
        cat_obj = pb.get(category, {}) if isinstance(pb, dict) else {}
        metric = cat_obj.get(metric_key) if isinstance(cat_obj, dict) else None
        if not supported:
            _add_gate(
                gates,
                name=gate_name,
                passed=True,
                actual=None,
                expected=f"skipped (unsupported {category})",
                skipped=True,
                reason="tool capability disabled",
            )
            return

        if not isinstance(metric, (int, float)):
            _add_gate(
                gates,
                name=gate_name,
                passed=False,
                actual=metric,
                expected="numeric metric",
            )
            return

        if lower_is_better:
            passed = float(metric) <= threshold
            expected_txt = f"<= {threshold}"
        else:
            passed = float(metric) >= threshold
            expected_txt = f">= {threshold}"

        _add_gate(
            gates,
            name=gate_name,
            passed=passed,
            actual=float(metric),
            expected=expected_txt,
        )

    v = quality_cfg.get("impact_min_f1_at_budget_2000_if_supported")
    if isinstance(v, (int, float)):
        _optional_category_gate(
            category="impact",
            metric_key="f1_mean",
            gate_name="tool_quality.impact_min_f1_at_budget_2000_if_supported",
            threshold=float(v),
            lower_is_better=False,
        )

    v = quality_cfg.get("slice_min_recall_at_budget_2000_if_supported")
    if isinstance(v, (int, float)):
        _optional_category_gate(
            category="slice",
            metric_key="recall_mean",
            gate_name="tool_quality.slice_min_recall_at_budget_2000_if_supported",
            threshold=float(v),
            lower_is_better=False,
        )

    v = quality_cfg.get("data_flow_min_origin_accuracy_at_budget_2000_if_supported")
    if isinstance(v, (int, float)):
        _optional_category_gate(
            category="data_flow",
            metric_key="origin_accuracy_mean",
            gate_name="tool_quality.data_flow_min_origin_accuracy_at_budget_2000_if_supported",
            threshold=float(v),
            lower_is_better=False,
        )

    v = quality_cfg.get("complexity_max_mae_at_budget_2000_if_supported")
    if isinstance(v, (int, float)):
        _optional_category_gate(
            category="complexity",
            metric_key="mae",
            gate_name="tool_quality.complexity_max_mae_at_budget_2000_if_supported",
            threshold=float(v),
            lower_is_better=True,
        )

    gates_passed = all(g["pass"] for g in gates if not g.get("skipped"))

    report = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "tool_id": tool_id,
        "feature_set_id": feature_set_id,
        "scored_at_utc": _utc_now(),
        "meta": gather_meta(tldr_repo_root=repo_root),
        "inputs": {
            "suite": str(suite_path),
            "suite_sha256": _sha256_file(suite_path),
            "tasks": str(tasks_path),
            "task_manifest_sha256": actual_manifest_hash,
            "predictions": str(preds_path),
            "predictions_sha256": _sha256_file(preds_path),
            "predictions_feature_set_id": feature_set_id,
            "tool_profile": str(profile_path) if profile_path else None,
            "tool_profile_sha256": _sha256_file(profile_path) if profile_path else None,
            "tokenizer": tokenizer,
            "expected_tokenizer": expected_tokenizer,
        },
        "capabilities": capabilities,
        "required_categories": sorted(required_categories),
        "status_counts": status_counts,
        "expected_by_category": expected_by_category,
        "ok_by_category": ok_by_category,
        "rates": rates,
        "metrics": {
            "by_budget": summary_by_budget,
            "overall": summary_overall,
        },
        "diagnostics": {
            "result_shape_counters": result_shape_counters,
        },
        "parse_errors": parse_errors[:200],
        "gate_checks": gates,
        "gates_passed": gates_passed,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = bench_runs_root(repo_root) / f"{now_utc_compact()}-h2h-score-{tool_id}.json"
    write_report(out_path, report)
    print(out_path)
    return 0 if gates_passed else 2


def cmd_compare(args: argparse.Namespace) -> int:
    repo_root = get_repo_root()
    suite_path = Path(args.suite).resolve()
    score_a_path = Path(args.score_a).resolve()
    score_b_path = Path(args.score_b).resolve()

    suite = _read_json(suite_path)
    score_a = _read_json(score_a_path)
    score_b = _read_json(score_b_path)

    if not isinstance(suite, dict) or not isinstance(score_a, dict) or not isinstance(score_b, dict):
        raise SystemExit("error: suite and score files must all be JSON objects")

    suite_id = suite.get("suite_id")
    if score_a.get("suite_id") != suite_id or score_b.get("suite_id") != suite_id:
        raise SystemExit("error: suite_id mismatch between compare inputs")

    label_a = args.label_a or str(score_a.get("tool_id") or "tool-a")
    label_b = args.label_b or str(score_b.get("tool_id") or "tool-b")

    gate_cfg = suite.get("gates", {}) if isinstance(suite.get("gates"), dict) else {}
    h2h_cfg = gate_cfg.get("head_to_head") if isinstance(gate_cfg.get("head_to_head"), dict) else {}
    primary_budget = int(h2h_cfg.get("primary_budget", 2000))

    metrics_a = (
        score_a.get("metrics", {})
        .get("by_budget", {})
        .get(str(primary_budget), {})
        .get("retrieval", {})
    )
    metrics_b = (
        score_b.get("metrics", {})
        .get("by_budget", {})
        .get(str(primary_budget), {})
        .get("retrieval", {})
    )

    def metric(name: str, obj: dict[str, Any]) -> float | None:
        v = obj.get(name)
        if isinstance(v, (int, float)):
            return float(v)
        return None

    comparisons: list[dict[str, Any]] = []
    wins = {label_a: 0, label_b: 0}

    candidates = [
        ("mrr_mean", "higher"),
        ("recall@5_mean", "higher"),
        ("precision@5_mean", "higher"),
        ("latency_ms_p50", "lower"),
        ("payload_tokens_median", "lower"),
    ]

    for key, direction in candidates:
        va = metric(key, metrics_a) if isinstance(metrics_a, dict) else None
        vb = metric(key, metrics_b) if isinstance(metrics_b, dict) else None
        winner = "tie"
        if va is not None and vb is not None:
            if direction == "higher":
                if va > vb:
                    winner = label_a
                elif vb > va:
                    winner = label_b
            else:
                if va < vb:
                    winner = label_a
                elif vb < va:
                    winner = label_b
            if winner in wins:
                wins[winner] += 1
        comparisons.append(
            {
                "metric": key,
                "direction": direction,
                "a": va,
                "b": vb,
                "winner": winner,
            }
        )

    rule = str(h2h_cfg.get("winner_rule") or "win_at_least_three_primary_metrics_in_common_lane")
    winner = "tie"
    if wins[label_a] >= 3 and wins[label_a] > wins[label_b]:
        winner = label_a
    elif wins[label_b] >= 3 and wins[label_b] > wins[label_a]:
        winner = label_b

    tie_breakers = h2h_cfg.get("tie_breakers") if isinstance(h2h_cfg.get("tie_breakers"), list) else []
    feature_set_a = score_a.get("feature_set_id") if isinstance(score_a.get("feature_set_id"), str) else "unspecified"
    feature_set_b = score_b.get("feature_set_id") if isinstance(score_b.get("feature_set_id"), str) else "unspecified"

    if winner == "tie":
        for tb in tie_breakers:
            if tb == "lower_timeout_rate":
                ra = score_a.get("rates", {}).get("timeout_rate")
                rb = score_b.get("rates", {}).get("timeout_rate")
                if isinstance(ra, (int, float)) and isinstance(rb, (int, float)) and ra != rb:
                    winner = label_a if ra < rb else label_b
                    break
            if tb == "lower_payload_tokens_median":
                pa = metric("payload_tokens_median", metrics_a) if isinstance(metrics_a, dict) else None
                pb = metric("payload_tokens_median", metrics_b) if isinstance(metrics_b, dict) else None
                if pa is not None and pb is not None and pa != pb:
                    winner = label_a if pa < pb else label_b
                    break
            if tb == "lower_latency_ms_p50":
                la = metric("latency_ms_p50", metrics_a) if isinstance(metrics_a, dict) else None
                lb = metric("latency_ms_p50", metrics_b) if isinstance(metrics_b, dict) else None
                if la is not None and lb is not None and la != lb:
                    winner = label_a if la < lb else label_b
                    break

    report = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "compared_at_utc": _utc_now(),
        "meta": gather_meta(tldr_repo_root=repo_root),
        "inputs": {
            "suite": str(suite_path),
            "suite_sha256": _sha256_file(suite_path),
            "score_a": str(score_a_path),
            "score_b": str(score_b_path),
            "score_a_sha256": _sha256_file(score_a_path),
            "score_b_sha256": _sha256_file(score_b_path),
            "primary_budget": primary_budget,
            "feature_set_a": feature_set_a,
            "feature_set_b": feature_set_b,
        },
        "labels": {
            "a": label_a,
            "b": label_b,
        },
        "feature_sets": {"a": feature_set_a, "b": feature_set_b},
        "winner_rule": rule,
        "metric_comparisons": comparisons,
        "wins": wins,
        "tie_breakers": tie_breakers,
        "winner": winner,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = bench_runs_root(repo_root) / f"{now_utc_compact()}-h2h-compare.json"
    write_report(out_path, report)
    print(out_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Head-to-head benchmark harness (llm-tldr vs contextplus).")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("validate-suite", help="Validate suite and optional tool profiles.")
    p.add_argument("--suite", default="benchmarks/head_to_head/suite.v1.json")
    p.add_argument("--tool-profile", action="append", default=[])
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_validate_suite)

    p = sp.add_parser("materialize-tasks", help="Build canonical task manifest from query sets.")
    p.add_argument("--suite", default="benchmarks/head_to_head/suite.v1.json")
    p.add_argument("--corpus-root", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_materialize_tasks)

    p = sp.add_parser("score", help="Score one tool predictions file against task manifest.")
    p.add_argument("--suite", default="benchmarks/head_to_head/suite.v1.json")
    p.add_argument("--tasks", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--tool-profile", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_score)

    p = sp.add_parser("compare", help="Compare two score reports using suite winner rule.")
    p.add_argument("--suite", default="benchmarks/head_to_head/suite.v1.json")
    p.add_argument("--score-a", required=True)
    p.add_argument("--score-b", required=True)
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_compare)

    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
