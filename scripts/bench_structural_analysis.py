#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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
    percentiles,
    write_report,
)

from tldr.analysis import impact_analysis
from tldr.api import get_cfg_context, get_dfg_context, get_slice
from tldr.cross_file_calls import ProjectCallGraph
from tldr.indexing.index import IndexContext, get_index_context
from tldr.stats import count_tokens
from tldr.tldrignore import IgnoreSpec


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Caller:
    file: str
    function: str


def _load_query_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    queries = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(queries, list):
        raise ValueError(f"Bad query file: {path}")
    out: list[dict[str, Any]] = []
    for q in queries:
        if isinstance(q, dict):
            out.append(q)
    return out


def _load_call_graph_cache(path: Path) -> ProjectCallGraph:
    data = json.loads(path.read_text())
    edges = data.get("edges", [])
    g = ProjectCallGraph()
    for e in edges:
        if not isinstance(e, dict):
            continue
        ff = e.get("from_file")
        ffunc = e.get("from_func")
        tf = e.get("to_file")
        tfunc = e.get("to_func")
        if not all(isinstance(x, str) for x in (ff, ffunc, tf, tfunc)):
            continue
        g.add_edge(ff, ffunc, tf, tfunc)
    meta = data.get("meta")
    if isinstance(meta, dict):
        g.meta = meta
    return g


def _write_call_graph_cache(path: Path, graph: ProjectCallGraph, *, language: str) -> None:
    cache_data = {
        "edges": [
            {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
            for e in graph.sorted_edges()
        ],
        "meta": getattr(graph, "meta", {}) or {},
        "languages": [language],
        "timestamp": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache_data, indent=2, sort_keys=True) + "\n")


def _payload_stats(payload: str) -> dict[str, Any]:
    b = payload.encode("utf-8")
    return {"payload_bytes": len(b), "payload_tokens": int(count_tokens(payload))}


def _safe_mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return statistics.mean(xs)


def _safe_stdev(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return 0.0 if xs else None
    return statistics.stdev(xs)


def _summarize_times_s(times_s: list[float]) -> dict[str, Any]:
    if not times_s:
        return {}
    return {
        "mean_s": round(_safe_mean(times_s) or 0.0, 6),
        "stdev_s": round(_safe_stdev(times_s) or 0.0, 6),
        "p50_s": round(percentiles(times_s).get("p50", 0.0), 6),
        "p95_s": round(percentiles(times_s).get("p95", 0.0), 6),
    }


def _flatten_impact_targets(result: dict[str, Any]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    targets = result.get("targets")
    if not isinstance(targets, dict):
        return out
    for _, tree in targets.items():
        if not isinstance(tree, dict):
            continue
        callers = tree.get("callers")
        if not isinstance(callers, list):
            continue
        for c in callers:
            if not isinstance(c, dict):
                continue
            fn = c.get("function")
            fp = c.get("file")
            if isinstance(fn, str) and isinstance(fp, str):
                out.add((fp, fn))
    return out


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    prec = None if tp + fp == 0 else tp / (tp + fp)
    rec = None if tp + fn == 0 else tp / (tp + fn)
    if prec is None or rec is None or prec + rec == 0:
        f1 = None
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


_DEF_RE = re.compile(
    r"^(?P<indent>\s*)(async\s+def|def)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_CLASS_RE = re.compile(r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")


def _guess_enclosing_python_symbol(lines: list[str], *, line_1based: int) -> str | None:
    idx = max(0, min(len(lines) - 1, line_1based - 1))
    def_name: str | None = None
    def_indent: int | None = None
    for j in range(idx, -1, -1):
        m = _DEF_RE.match(lines[j])
        if m:
            def_name = m.group("name")
            def_indent = len(m.group("indent").expandtabs(4))
            break
    if def_name is None or def_indent is None:
        return None

    class_name: str | None = None
    for j in range(j - 1, -1, -1):
        m = _CLASS_RE.match(lines[j])
        if not m:
            continue
        class_indent = len(m.group("indent").expandtabs(4))
        if class_indent < def_indent:
            class_name = m.group("name")
            break
    return f"{class_name}.{def_name}" if class_name else def_name


def _rg_hits(repo_root: Path, *, pattern: str, file_glob: str = "*.py") -> list[tuple[str, int, str]]:
    cmd = ["rg", "-n", "--no-messages", "--glob", file_glob, pattern, str(repo_root)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), check=False)
    if proc.returncode not in (0, 1):  # 1 = no matches
        raise RuntimeError(f"rg failed (rc={proc.returncode}): {proc.stderr.strip()}")
    hits: list[tuple[str, int, str]] = []
    for line in (proc.stdout or "").splitlines():
        # path:line:content
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        path_s, line_s, content = parts
        try:
            n = int(line_s)
        except ValueError:
            continue
        hits.append((path_s, n, content))
    hits.sort(key=lambda x: (x[0], x[1]))
    return hits


def _python_function_span(source: str, *, function: str) -> tuple[int, int] | None:
    """Return (start_line, end_line) for a function or method (best-effort).

    Supports names like "func" or "Class.method".
    """
    class_name = None
    func_name = function
    if "." in function:
        class_name, func_name = function.split(".", 1)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    def match_fn(node: ast.AST, *, want: str) -> bool:
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == want

    if class_name:
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and n.name == class_name:
                for m in n.body:
                    if match_fn(m, want=func_name) and hasattr(m, "lineno") and hasattr(m, "end_lineno"):
                        return int(m.lineno), int(m.end_lineno)  # type: ignore[attr-defined]
        return None

    # Prefer module-level functions when unqualified.
    for n in tree.body:
        if match_fn(n, want=func_name) and hasattr(n, "lineno") and hasattr(n, "end_lineno"):
            return int(n.lineno), int(n.end_lineno)  # type: ignore[attr-defined]
    # Fall back to any matching function (e.g., class methods), which is
    # sufficient for benchmark query sets that avoid duplicate method names.
    for n in ast.walk(tree):
        if match_fn(n, want=func_name) and hasattr(n, "lineno") and hasattr(n, "end_lineno"):
            return int(n.lineno), int(n.end_lineno)  # type: ignore[attr-defined]
    return None


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


def _radon_version() -> str | None:
    try:
        import radon  # type: ignore[import-not-found]
    except Exception:
        return None
    return getattr(radon, "__version__", None)


def _radon_cc_for_function(source: str, *, function: str) -> int | None:
    """Return radon CC score for a function/method in the given source, if available."""
    try:
        from radon.complexity import cc_visit  # type: ignore[import-not-found]
    except Exception:
        return None

    class_name = None
    func_name = function
    if "." in function:
        class_name, func_name = function.split(".", 1)

    blocks = cc_visit(source)
    # cc_visit returns blocks for functions/classes; class blocks have `methods`.
    for b in blocks:
        name = getattr(b, "name", None)
        if class_name and name == class_name:
            for m in getattr(b, "methods", []) or []:
                if getattr(m, "name", None) == func_name:
                    return int(getattr(m, "complexity", 0))
        if not class_name and name == func_name:
            return int(getattr(b, "complexity", 0))
    return None


def _complexity_heuristic(source_lines: list[str], *, span: tuple[int, int]) -> int:
    start, end = span
    decision = re.compile(r"^\\s*(if|elif|else:|for|while|except|with)\\b")
    count = 0
    for line in source_lines[start - 1 : end]:
        if decision.search(line):
            count += 1
    return 1 + count


def _data_flow_eval(
    dfg: dict[str, Any],
    *,
    variable: str,
    expected_lines: set[int],
) -> dict[str, Any]:
    refs = dfg.get("refs")
    edges = dfg.get("edges")
    var_refs: set[int] = set()
    if isinstance(refs, list):
        for r in refs:
            if not isinstance(r, dict):
                continue
            if r.get("name") != variable:
                continue
            line = r.get("line")
            if isinstance(line, int):
                var_refs.add(line)
    var_edge_lines: set[int] = set()
    origin_line: int | None = None
    if isinstance(edges, list):
        for e in edges:
            if not isinstance(e, dict):
                continue
            if e.get("var") != variable:
                continue
            dl = e.get("def_line")
            ul = e.get("use_line")
            if isinstance(dl, int):
                var_edge_lines.add(dl)
                if origin_line is None or dl < origin_line:
                    origin_line = dl
            if isinstance(ul, int):
                var_edge_lines.add(ul)
    seen = var_refs | var_edge_lines
    if not expected_lines:
        completeness = None
    else:
        completeness = len(seen & expected_lines) / len(expected_lines)
    return {
        "origin_line": origin_line,
        "flow_completeness": completeness,
        "seen_lines": sorted(seen),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4: Python structural analysis quality (Django).")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", default=None, help="Corpus id from benchmarks/corpora.json (e.g. django).")
    group.add_argument("--repo-root", default=None, help="Path to the corpus repo root.")
    ap.add_argument(
        "--queries",
        default=str(get_repo_root() / "benchmarks" / "python" / "django_structural_queries.json"),
        help="Path to structural query set JSON.",
    )
    ap.add_argument(
        "--cache-root",
        default=str(bench_cache_root(get_repo_root())),
        help="Index-mode cache root (default: benchmark/cache-root).",
    )
    ap.add_argument("--index", default=None, help="Index id (default: repo:<corpus>).")
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
    queries = _load_query_file(queries_path)

    index_id = args.index or default_index_id
    index_ctx: IndexContext = get_index_context(
        scan_root=repo_root,
        cache_root_arg=args.cache_root,
        index_id_arg=index_id,
        allow_create=True,
    )
    index_paths = index_ctx.paths

    ignore_spec = IgnoreSpec(
        project_dir=repo_root,
        use_gitignore=bool(index_ctx.config.use_gitignore) if index_ctx.config else True,
        cli_patterns=list(index_ctx.config.cli_patterns or ()) if index_ctx.config else None,
        ignore_file=index_ctx.config.ignore_file if index_ctx.config else None,
        gitignore_root=index_ctx.config.gitignore_root if index_ctx.config else None,
    )

    # Call graph: load cache if present, else build and write cache for repeatability.
    call_graph: ProjectCallGraph | None = None
    call_graph_build_s: float | None = None
    if index_paths is not None and index_paths.call_graph.exists():
        call_graph = _load_call_graph_cache(index_paths.call_graph)
    else:
        from tldr.api import build_project_call_graph

        t0 = time.monotonic()
        call_graph = build_project_call_graph(
            repo_root,
            language="python",
            ignore_spec=ignore_spec,
        )
        call_graph_build_s = time.monotonic() - t0
        if index_paths is not None:
            _write_call_graph_cache(index_paths.call_graph, call_graph, language="python")

    assert call_graph is not None

    radon_version = _radon_version()

    per_query: list[dict[str, Any]] = []

    # Aggregates.
    impact_tp = impact_fp = impact_fn = 0
    grep_impact_tp = grep_impact_fp = grep_impact_fn = 0

    slice_precisions: list[float] = []
    slice_recalls: list[float] = []
    slice_noise_reductions: list[float] = []

    cfg_tldr_scores: list[int] = []
    cfg_radon_scores: list[int] = []
    cfg_heur_scores: list[int] = []

    dfg_origin_hits = 0
    dfg_total = 0
    dfg_completeness: list[float] = []
    grep_noise_ratios: list[float] = []

    times_tldr_s: list[float] = []
    times_grep_s: list[float] = []

    for q in queries:
        qid = q.get("id") or q.get("name") or "unknown"
        category = q.get("category")

        entry: dict[str, Any] = {"id": qid, "category": category}

        if category == "impact":
            func = q.get("function")
            file_filter = q.get("file")
            expected_callers_raw = q.get("expected_callers", [])
            if not isinstance(func, str) or not isinstance(file_filter, str) or not isinstance(expected_callers_raw, list):
                per_query.append({**entry, "error": "bad impact query schema"})
                continue
            expected: set[tuple[str, str]] = set()
            for c in expected_callers_raw:
                if not isinstance(c, dict):
                    continue
                cf = c.get("file")
                cn = c.get("function")
                if isinstance(cf, str) and isinstance(cn, str):
                    expected.add((cf, cn))

            t0 = time.monotonic()
            tldr_res = impact_analysis(call_graph, func, max_depth=1, target_file=file_filter)
            tldr_payload = json.dumps(tldr_res, sort_keys=True)
            tldr_time_s = time.monotonic() - t0

            predicted = _flatten_impact_targets(tldr_res)
            tp = len(predicted & expected)
            fp = len(predicted - expected)
            fn = len(expected - predicted)
            prec, rec, f1 = _prf(tp, fp, fn)
            impact_tp += tp
            impact_fp += fp
            impact_fn += fn

            # Grep baseline for impact: rg for the leaf call expression + enclosing symbol heuristic.
            leaf = func.split(".")[-1]
            pattern = r"\." + re.escape(leaf) + r"\s*\(" if "." in func else r"\b" + re.escape(leaf) + r"\s*\("
            t0 = time.monotonic()
            hits = _rg_hits(repo_root, pattern=pattern)
            grep_callers: set[tuple[str, str]] = set()
            by_file: dict[str, list[tuple[int, str]]] = {}
            for fp_s, ln, _ in hits:
                by_file.setdefault(fp_s, []).append((ln, ""))
            for fp_s, items in by_file.items():
                abs_fp = Path(fp_s)
                try:
                    lines = abs_fp.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                rel = str(abs_fp.relative_to(repo_root))
                for ln, _ in items:
                    sym = _guess_enclosing_python_symbol(lines, line_1based=ln)
                    if sym:
                        grep_callers.add((rel, sym))
            grep_payload = json.dumps(
                [{"file": f, "function": fn} for f, fn in sorted(grep_callers)],
                sort_keys=True,
            )
            grep_time_s = time.monotonic() - t0

            gtp = len(grep_callers & expected)
            gfp = len(grep_callers - expected)
            gfn = len(expected - grep_callers)
            gprec, grec, gf1 = _prf(gtp, gfp, gfn)
            grep_impact_tp += gtp
            grep_impact_fp += gfp
            grep_impact_fn += gfn

            times_tldr_s.append(tldr_time_s)
            times_grep_s.append(grep_time_s)

            entry.update(
                {
                    "tldr": {
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "time_s": round(tldr_time_s, 6),
                        **_payload_stats(tldr_payload),
                    },
                    "grep": {
                        "tp": gtp,
                        "fp": gfp,
                        "fn": gfn,
                        "precision": gprec,
                        "recall": grec,
                        "f1": gf1,
                        "time_s": round(grep_time_s, 6),
                        **_payload_stats(grep_payload),
                    },
                    "expected_callers": len(expected),
                    "predicted_callers": len(predicted),
                    "grep_callers": len(grep_callers),
                    "function": func,
                    "file": file_filter,
                    "rg_pattern": pattern,
                }
            )
            per_query.append(entry)
            continue

        if category == "slice":
            file_rel = q.get("file")
            function = q.get("function")
            target_line = q.get("target_line")
            expected_lines = q.get("expected_slice_lines", [])
            if not isinstance(file_rel, str) or not isinstance(function, str) or not isinstance(target_line, int) or not isinstance(expected_lines, list):
                per_query.append({**entry, "error": "bad slice query schema"})
                continue
            expected = {int(x) for x in expected_lines if isinstance(x, int)}
            abs_path = repo_root / file_rel
            source = abs_path.read_text(encoding="utf-8", errors="replace")
            source_lines = source.splitlines()
            span = _python_function_span(source, function=function)
            if span is None:
                per_query.append({**entry, "error": f"function not found: {function}"})
                continue
            start, end = span
            total_lines = max(1, end - start + 1)

            t0 = time.monotonic()
            tldr_lines = set(get_slice(str(abs_path), function, int(target_line), direction="backward", variable=None, language="python"))
            tldr_payload = json.dumps({"lines": sorted(tldr_lines), "count": len(tldr_lines)}, sort_keys=True)
            tldr_time_s = time.monotonic() - t0

            tp = len(tldr_lines & expected)
            fp = len(tldr_lines - expected)
            fn = len(expected - tldr_lines)
            prec, rec, _ = _prf(tp, fp, fn)
            if prec is not None:
                slice_precisions.append(prec)
            if rec is not None:
                slice_recalls.append(rec)
            noise_reduction = None if not tldr_lines else 1.0 - (len(tldr_lines) / total_lines)
            if noise_reduction is not None:
                slice_noise_reductions.append(noise_reduction)

            baseline_lines = set(range(start, end + 1))
            baseline_payload = json.dumps(
                {"file": file_rel, "function": function, "start": start, "end": end, "lines": total_lines},
                sort_keys=True,
            )
            # Baseline "read it all" is in-process; treat as ~0 tool time.
            baseline_time_s = 0.0

            times_tldr_s.append(tldr_time_s)
            times_grep_s.append(baseline_time_s)

            entry.update(
                {
                    "tldr": {
                        "precision": prec,
                        "recall": rec,
                        "slice_size": len(tldr_lines),
                        "total_function_lines": total_lines,
                        "noise_reduction": noise_reduction,
                        "time_s": round(tldr_time_s, 6),
                        **_payload_stats(tldr_payload),
                    },
                    "grep": {
                        "slice_size": len(baseline_lines),
                        "total_function_lines": total_lines,
                        "noise_reduction": 0.0,
                        "time_s": round(baseline_time_s, 6),
                        **_payload_stats(baseline_payload),
                    },
                    "file": file_rel,
                    "function": function,
                    "target_line": int(target_line),
                }
            )
            per_query.append(entry)
            continue

        if category == "complexity":
            file_rel = q.get("file")
            function = q.get("function")
            if not isinstance(file_rel, str) or not isinstance(function, str):
                per_query.append({**entry, "error": "bad complexity query schema"})
                continue
            abs_path = repo_root / file_rel
            source = abs_path.read_text(encoding="utf-8", errors="replace")
            source_lines = source.splitlines()
            span = _python_function_span(source, function=function)
            if span is None:
                per_query.append({**entry, "error": f"function not found: {function}"})
                continue

            t0 = time.monotonic()
            cfg = get_cfg_context(str(abs_path), function, language="python")
            tldr_payload = json.dumps(cfg, sort_keys=True)
            tldr_time_s = time.monotonic() - t0

            tldr_cc = cfg.get("cyclomatic_complexity")
            tldr_cc_int = int(tldr_cc) if isinstance(tldr_cc, int) else None

            radon_cc = _radon_cc_for_function(source, function=function)
            heur_cc = _complexity_heuristic(source_lines, span=span)

            if tldr_cc_int is not None and radon_cc is not None:
                cfg_tldr_scores.append(tldr_cc_int)
                cfg_radon_scores.append(radon_cc)
                cfg_heur_scores.append(heur_cc)

            times_tldr_s.append(tldr_time_s)

            entry.update(
                {
                    "tldr": {
                        "cyclomatic_complexity": tldr_cc_int,
                        "blocks": len(cfg.get("blocks", []) or []) if isinstance(cfg.get("blocks"), list) else None,
                        "time_s": round(tldr_time_s, 6),
                        **_payload_stats(tldr_payload),
                    },
                    "radon": {"cyclomatic_complexity": radon_cc},
                    "grep_heuristic": {"cyclomatic_complexity": heur_cc},
                    "file": file_rel,
                    "function": function,
                }
            )
            per_query.append(entry)
            continue

        if category == "data_flow":
            file_rel = q.get("file")
            function = q.get("function")
            variable = q.get("variable")
            expected_flow = q.get("expected_flow", [])
            if not isinstance(file_rel, str) or not isinstance(function, str) or not isinstance(variable, str) or not isinstance(expected_flow, list):
                per_query.append({**entry, "error": "bad data_flow query schema"})
                continue
            expected_lines: set[int] = set()
            origin_expected: int | None = None
            for ev in expected_flow:
                if not isinstance(ev, dict):
                    continue
                ln = ev.get("line")
                if isinstance(ln, int):
                    expected_lines.add(ln)
                if origin_expected is None and ev.get("event") == "defined" and isinstance(ln, int):
                    origin_expected = ln

            abs_path = repo_root / file_rel
            source = abs_path.read_text(encoding="utf-8", errors="replace")
            span = _python_function_span(source, function=function)

            t0 = time.monotonic()
            dfg = get_dfg_context(str(abs_path), function, language="python")
            tldr_payload = json.dumps(dfg, sort_keys=True)
            tldr_time_s = time.monotonic() - t0

            eval_res = _data_flow_eval(dfg, variable=variable, expected_lines=expected_lines)
            origin_line = eval_res.get("origin_line")
            origin_ok = bool(origin_expected is not None and origin_line == origin_expected)
            if origin_expected is not None:
                dfg_total += 1
                if origin_ok:
                    dfg_origin_hits += 1
            if isinstance(eval_res.get("flow_completeness"), float):
                dfg_completeness.append(float(eval_res["flow_completeness"]))

            # Grep baseline: count matching lines inside the function span (if available).
            hits = []
            if span is not None:
                start, end = span
                lines = source.splitlines()
                for ln in range(start, end + 1):
                    if variable in lines[ln - 1]:
                        hits.append(ln)
            denom = max(1, len(expected_lines))
            grep_noise_ratios.append(len(hits) / denom)
            grep_payload = json.dumps({"hits": hits, "count": len(hits)}, sort_keys=True)

            times_tldr_s.append(tldr_time_s)

            entry.update(
                {
                    "tldr": {
                        "origin_line": origin_line,
                        "origin_expected": origin_expected,
                        "origin_ok": origin_ok,
                        "flow_completeness": eval_res.get("flow_completeness"),
                        "time_s": round(tldr_time_s, 6),
                        **_payload_stats(tldr_payload),
                    },
                    "grep": {
                        "hits": len(hits),
                        "expected_flow_lines": len(expected_lines),
                        "noise_ratio": len(hits) / denom,
                        **_payload_stats(grep_payload),
                    },
                    "file": file_rel,
                    "function": function,
                    "variable": variable,
                }
            )
            per_query.append(entry)
            continue

        per_query.append({**entry, "skipped": True, "reason": "unknown category"})

    impact_prec, impact_rec, impact_f1 = _prf(impact_tp, impact_fp, impact_fn)
    grep_impact_prec, grep_impact_rec, grep_impact_f1 = _prf(
        grep_impact_tp, grep_impact_fp, grep_impact_fn
    )

    cfg_accuracy = None
    cfg_mae = None
    heur_mae = None
    tau_tldr = None
    tau_heur = None
    if cfg_radon_scores:
        cfg_accuracy = sum(1 for a, b in zip(cfg_tldr_scores, cfg_radon_scores) if a == b) / len(cfg_radon_scores)
        cfg_mae = statistics.mean(abs(a - b) for a, b in zip(cfg_tldr_scores, cfg_radon_scores))
        heur_mae = statistics.mean(abs(a - b) for a, b in zip(cfg_heur_scores, cfg_radon_scores))
        tau_tldr = _kendall_tau_b([float(x) for x in cfg_tldr_scores], [float(x) for x in cfg_radon_scores])
        tau_heur = _kendall_tau_b([float(x) for x in cfg_heur_scores], [float(x) for x in cfg_radon_scores])

    report = make_report(
        phase="phase4_python_structural",
        meta=gather_meta(tldr_repo_root=tldr_repo_root, corpus_id=corpus_id, corpus_root=repo_root),
        protocol={
            "schema_version": SCHEMA_VERSION,
            "queries": str(queries_path),
            "cache_root": str(index_ctx.cache_root) if index_ctx.cache_root is not None else None,
            "index_id": index_ctx.index_id,
            "radon_version": radon_version,
            "call_graph_cache": str(index_paths.call_graph) if index_paths is not None else None,
            "call_graph_build_s": round(call_graph_build_s, 6) if call_graph_build_s is not None else None,
        },
        results={
            "impact": {
                "tldr": {"precision": impact_prec, "recall": impact_rec, "f1": impact_f1},
                "grep": {"precision": grep_impact_prec, "recall": grep_impact_rec, "f1": grep_impact_f1},
                "tp": impact_tp,
                "fp": impact_fp,
                "fn": impact_fn,
            },
            "slice": {
                "tldr": {
                    "precision_mean": _safe_mean(slice_precisions),
                    "recall_mean": _safe_mean(slice_recalls),
                    "noise_reduction_mean": _safe_mean(slice_noise_reductions),
                }
            },
            "complexity": {
                "radon_version": radon_version,
                "tldr": {"accuracy": cfg_accuracy, "mae": cfg_mae, "kendall_tau_b": tau_tldr},
                "grep_heuristic": {"mae": heur_mae, "kendall_tau_b": tau_heur},
            },
            "data_flow": {
                "tldr": {
                    "origin_accuracy": (dfg_origin_hits / dfg_total) if dfg_total else None,
                    "flow_completeness_mean": _safe_mean(dfg_completeness),
                },
                "grep": {"noise_ratio_mean": _safe_mean(grep_noise_ratios)},
            },
            "timing": {
                "tldr": _summarize_times_s(times_tldr_s),
                "grep": _summarize_times_s(times_grep_s),
            },
            "per_query": per_query,
        },
    )

    if args.out:
        out_path = Path(args.out)
    else:
        ts = now_utc_compact()
        out_path = bench_runs_root(tldr_repo_root) / f"{ts}-python-structural-{corpus_id}.json"
    write_report(out_path, report)
    print(out_path)

    # Consider the run "ok" if we processed at least one query per category present.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
