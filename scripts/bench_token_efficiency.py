#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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

from tldr.analysis import impact_analysis
from tldr.api import get_cfg_context, get_dfg_context, get_slice
from tldr.cross_file_calls import ProjectCallGraph
from tldr.indexing.index import IndexContext, get_index_context
from tldr.stats import count_tokens
from tldr.tldrignore import IgnoreSpec


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    query: str
    relevant_files: tuple[str, ...]
    rg_pattern: str | None


@dataclass(frozen=True)
class RgHit:
    file: str
    line: int
    text: str


def _load_structural_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    queries = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(queries, list):
        raise ValueError(f"Bad query file: {path}")
    out: list[dict[str, Any]] = []
    for q in queries:
        if isinstance(q, dict):
            out.append(q)
    return out


def _load_retrieval_queries(path: Path) -> list[RetrievalQuery]:
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


def _parse_budgets(spec: str) -> list[int]:
    budgets: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        budgets.append(int(part))
    budgets = sorted(set(budgets))
    if not budgets:
        raise SystemExit("error: no budgets provided")
    return budgets


def _payload_stats(payload: str) -> dict[str, int]:
    payload_bytes = len(payload.encode("utf-8"))
    payload_tokens = int(count_tokens(payload))
    return {"payload_tokens": payload_tokens, "payload_bytes": payload_bytes}


def _apply_budget(pieces: list[str], budget_tokens: int) -> tuple[str, int, int, int]:
    """Greedy deterministic prefix selection by tokens. Returns payload + counts + pieces_used."""
    payload_parts: list[str] = []
    used = 0
    for piece in pieces:
        candidate = "\n\n".join([*payload_parts, piece]) if payload_parts else piece
        toks = int(count_tokens(candidate))
        if toks > budget_tokens:
            break
        payload_parts.append(piece)
        used += 1
    payload = "\n\n".join(payload_parts)
    payload_bytes = len(payload.encode("utf-8"))
    payload_tokens = int(count_tokens(payload))
    return payload, payload_tokens, payload_bytes, used


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
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


def _rg_hits(repo_root: Path, *, pattern: str, file_glob: str = "*.py") -> list[RgHit]:
    cmd = ["rg", "-n", "--no-messages", "--glob", file_glob, pattern, "."]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root), check=False)
    if proc.returncode not in (0, 1):  # 1 = no matches
        raise RuntimeError(f"rg failed (rc={proc.returncode}): {proc.stderr.strip()}")
    hits: list[RgHit] = []
    for line in (proc.stdout or "").splitlines():
        # path:line:content
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        path_s, line_s, content = parts
        if path_s.startswith("./"):
            path_s = path_s[2:]
        try:
            n = int(line_s)
        except ValueError:
            continue
        hits.append(RgHit(file=path_s.replace("\\", "/"), line=n, text=content))
    hits.sort(key=lambda h: (h.file, h.line, h.text))
    return hits


def _filter_py_def_hits(hits: list[RgHit], *, callee_leaf: str) -> list[RgHit]:
    def_re = re.compile(rf"^\s*(?:async\s+def|def)\s+{re.escape(callee_leaf)}\s*\(")
    out: list[RgHit] = []
    for h in hits:
        if def_re.match(h.text):
            continue
        out.append(h)
    return out


def _read_lines(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


def _indent_width(s: str) -> int:
    return len(s[: len(s) - len(s.lstrip())].expandtabs(4))


_DEF_RE = re.compile(r"^(?P<indent>\s*)(?:async\s+def|def)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CLASS_RE = re.compile(r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")


def _python_find_def_span(lines: list[str], *, function_name: str) -> tuple[int, int] | None:
    """Best-effort (start_line, end_line) for a top-level def by name using indentation heuristics.

    This intentionally avoids AST parsing so it can be used as a "grep workflow" proxy.
    """
    start_def: int | None = None
    def_indent: int | None = None

    for i, line in enumerate(lines, start=1):
        m = _DEF_RE.match(line)
        if not m:
            continue
        if m.group("name") != function_name:
            continue
        start_def = i
        def_indent = _indent_width(m.group("indent"))
        break

    if start_def is None or def_indent is None:
        return None

    # Include decorators immediately above the def.
    start = start_def
    for j in range(start_def - 1, 0, -1):
        prev = lines[j - 1]
        if prev.lstrip().startswith("@") and _indent_width(prev) == def_indent:
            start = j
            continue
        break

    end = len(lines)
    for k in range(start_def + 1, len(lines) + 1):
        ln = lines[k - 1]
        if not ln.strip():
            continue
        m_def = _DEF_RE.match(ln)
        m_cls = _CLASS_RE.match(ln)
        if m_def or m_cls:
            if _indent_width(ln) <= def_indent:
                end = k - 1
                break
    # Trim trailing blank lines for a tighter span.
    while end > start_def and not lines[end - 1].strip():
        end -= 1
    end = max(end, start_def)
    return start, end


def _python_enclosing_def_span(lines: list[str], *, line_1based: int) -> tuple[int, int, str] | None:
    """Return (start, end, function_name) for the def enclosing line_1based."""
    idx = max(0, min(len(lines) - 1, line_1based - 1))
    # Determine indentation of the current (non-empty) line; if it's at module
    # indent, treat it as having no enclosing def.
    cur_indent = 0
    for t in range(idx, -1, -1):
        if lines[t].strip():
            cur_indent = _indent_width(lines[t])
            break
    if cur_indent <= 0:
        return None

    def_line: int | None = None
    def_indent: int | None = None
    def_name: str | None = None
    fallback: tuple[int, int, str] | None = None
    for j in range(idx, -1, -1):
        m = _DEF_RE.match(lines[j])
        if not m:
            continue
        dl = j + 1
        di = _indent_width(m.group("indent"))
        dn = m.group("name")
        if fallback is None:
            fallback = (dl, di, dn)
        if di < cur_indent:
            def_line, def_indent, def_name = dl, di, dn
            break
    if def_line is None or def_indent is None or def_name is None:
        if fallback is None:
            return None
        def_line, def_indent, def_name = fallback

    # Include decorators.
    start = def_line
    for j in range(def_line - 1, 0, -1):
        prev = lines[j - 1]
        if prev.lstrip().startswith("@") and _indent_width(prev) == def_indent:
            start = j
            continue
        break

    end = len(lines)
    for k in range(def_line + 1, len(lines) + 1):
        ln = lines[k - 1]
        if not ln.strip():
            continue
        m_def = _DEF_RE.match(ln)
        m_cls = _CLASS_RE.match(ln)
        if m_def or m_cls:
            if _indent_width(ln) <= def_indent:
                end = k - 1
                break
    while end > def_line and not lines[end - 1].strip():
        end -= 1
    end = max(end, def_line)
    return start, end, def_name


def _merge_windows(windows: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if not out:
            out.append((start, end))
            continue
        prev_s, prev_e = out[-1]
        if start <= prev_e + 1:
            out[-1] = (prev_s, max(prev_e, end))
        else:
            out.append((start, end))
    return out


def _render_code_block(file_rel: str, *, start: int, end: int, lines: list[str]) -> str:
    end = min(end, len(lines))
    start = max(1, start)
    header = f"# {file_rel}:{start}-{end}"
    body = "\n".join(f"{ln}: {lines[ln - 1].rstrip()}" for ln in range(start, end + 1))
    return f"{header}\n{body}"


def _extract_lines_for_numbers(file_rel: str, *, lines: list[str], line_nos: list[int]) -> tuple[str, set[int]]:
    # Merge contiguous runs for readability + fewer headers.
    sorted_lines = sorted({ln for ln in line_nos if 1 <= ln <= len(lines)})
    if not sorted_lines:
        return "", set()

    ranges: list[tuple[int, int]] = []
    s = e = sorted_lines[0]
    for ln in sorted_lines[1:]:
        if ln == e + 1:
            e = ln
            continue
        ranges.append((s, e))
        s = e = ln
    ranges.append((s, e))

    pieces = [_render_code_block(file_rel, start=s, end=e, lines=lines) for s, e in ranges]
    return "\n\n".join(pieces), set(sorted_lines)


def _python_function_span_ast(source: str, *, function: str) -> tuple[int, int] | None:
    """Return (start_line, end_line) for a function/method (best-effort) using AST.

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
    for n in ast.walk(tree):
        if match_fn(n, want=func_name) and hasattr(n, "lineno") and hasattr(n, "end_lineno"):
            return int(n.lineno), int(n.end_lineno)  # type: ignore[attr-defined]
    return None


def _radon_version() -> str | None:
    try:
        import radon  # type: ignore[import-not-found]
    except Exception:
        return None
    return getattr(radon, "__version__", None)


def _radon_cc_for_function(source: str, *, function: str) -> int | None:
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


def _complexity_heuristic(lines: list[str], *, span: tuple[int, int]) -> int:
    start, end = span
    decision = re.compile(r"^\s*(if|elif|else:|for|while|except|with)\b")
    count = 0
    for line in lines[start - 1 : end]:
        if decision.search(line):
            count += 1
    return 1 + count


def _rg_rank_files(repo_root: Path, *, pattern: str, glob: str | None) -> list[str]:
    cmd = ["rg", "-n", "--no-messages"]
    if glob:
        cmd.extend(["--glob", glob])
    cmd.append(pattern)
    cmd.append(".")
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1):
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

    ranked = sorted(hits_by_file.items(), key=lambda kv: (-kv[1]["hits"], kv[1]["min_line"], kv[0]))
    return [fp.replace("\\", "/") for fp, _ in ranked]


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
            ranked.append(fp.replace("\\", "/"))

    seen = set()
    out: list[str] = []
    for fp in ranked:
        if fp in seen:
            continue
        seen.add(fp)
        out.append(fp)
    return out


def _rrf_fuse(rankings: list[list[str]], *, k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for i, fp in enumerate(ranking, start=1):
            scores[fp] = scores.get(fp, 0.0) + 1.0 / (k + i)
    return [fp for fp, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _first_regex_line(lines: list[str], *, pattern: re.Pattern[str]) -> int | None:
    for i, line in enumerate(lines, start=1):
        if pattern.search(line):
            return i
    return None


def _render_snippet_for_file(
    repo_root: Path,
    *,
    file_rel: str,
    rg_pattern: str,
    before: int,
    after: int,
) -> tuple[str, set[int]]:
    path = repo_root / file_rel
    lines = _read_lines(path) or []
    if not lines:
        return f"# {file_rel} (unreadable)", set()
    try:
        pat = re.compile(rg_pattern)
    except re.error:
        pat = re.compile(re.escape(rg_pattern))
    ln = _first_regex_line(lines, pattern=pat)
    if ln is None:
        return f"# {file_rel} (no rg_pattern match)", set()
    start = max(1, ln - before)
    end = min(len(lines), ln + after)
    return _render_code_block(file_rel, start=start, end=end, lines=lines), set(range(start, end + 1))


def _select_window_within_span(
    *,
    file_rel: str,
    lines: list[str],
    span: tuple[int, int] | None,
    target_line: int,
    budget_tokens: int,
) -> tuple[str, set[int], dict[str, int]]:
    """Deterministically choose the largest contiguous window around target_line that fits budget."""
    lo = 1
    hi = len(lines)
    if span is not None:
        lo, hi = span
    lo = max(1, lo)
    hi = min(hi, len(lines))
    target_line = max(lo, min(hi, target_line))

    best_start = best_end = target_line
    while True:
        cur = _render_code_block(file_rel, start=best_start, end=best_end, lines=lines)
        if int(count_tokens(cur)) > budget_tokens:
            # Can't even fit a single line. Return empty.
            return "", set(), {"start": best_start, "end": best_end}

        expanded = False
        cand_start = best_start
        cand_end = best_end
        if cand_start > lo:
            cand_start -= 1
        if cand_end < hi:
            cand_end += 1
        if cand_start == best_start and cand_end == best_end:
            break

        cand = _render_code_block(file_rel, start=cand_start, end=cand_end, lines=lines)
        if int(count_tokens(cand)) <= budget_tokens:
            best_start, best_end = cand_start, cand_end
            expanded = True

        if not expanded:
            # Try expanding one side at a time.
            tried = False
            if best_start > lo:
                tried = True
                cand2 = _render_code_block(file_rel, start=best_start - 1, end=best_end, lines=lines)
                if int(count_tokens(cand2)) <= budget_tokens:
                    best_start -= 1
                    continue
            if best_end < hi:
                tried = True
                cand2 = _render_code_block(file_rel, start=best_start, end=best_end + 1, lines=lines)
                if int(count_tokens(cand2)) <= budget_tokens:
                    best_end += 1
                    continue
            if not tried:
                break
            break

    payload = _render_code_block(file_rel, start=best_start, end=best_end, lines=lines)
    included = set(range(best_start, best_end + 1))
    return payload, included, {"start": best_start, "end": best_end}


def _load_or_build_call_graph(
    *,
    repo_root: Path,
    index_ctx: IndexContext,
    language: str,
    ignore_spec: IgnoreSpec,
) -> tuple[ProjectCallGraph, float | None, str | None]:
    index_paths = index_ctx.paths
    if index_paths is not None and index_paths.call_graph.exists():
        try:
            cache_data = json.loads(index_paths.call_graph.read_text())
        except (OSError, json.JSONDecodeError):
            cache_data = None
        if isinstance(cache_data, dict):
            edges = cache_data.get("edges", [])
            g = ProjectCallGraph()
            for e in edges if isinstance(edges, list) else []:
                if not isinstance(e, dict):
                    continue
                ff = e.get("from_file")
                ffunc = e.get("from_func")
                tf = e.get("to_file")
                tfunc = e.get("to_func")
                if not all(isinstance(x, str) for x in (ff, ffunc, tf, tfunc)):
                    continue
                g.add_edge(ff, ffunc, tf, tfunc)
            meta = cache_data.get("meta")
            if isinstance(meta, dict):
                g.meta = meta
            return g, None, str(index_paths.call_graph)

    from tldr.api import build_project_call_graph

    import time

    t0 = time.monotonic()
    g = build_project_call_graph(repo_root, language=language, ignore_spec=ignore_spec)
    build_s = time.monotonic() - t0

    if index_paths is not None:
        cache_data2 = {
            "edges": [
                {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
                for e in g.sorted_edges()
            ],
            "meta": getattr(g, "meta", {}) or {},
            "languages": [language],
            "timestamp": time.time(),
        }
        index_paths.call_graph.parent.mkdir(parents=True, exist_ok=True)
        index_paths.call_graph.write_text(json.dumps(cache_data2, indent=2, sort_keys=True) + "\n")

    return g, build_s, str(index_paths.call_graph) if index_paths is not None else None


def _impact_grep_pieces(
    repo_root: Path,
    *,
    hits: list[RgHit],
    before: int,
    after: int,
) -> dict[str, Any]:
    """Return pieces for grep strategies plus aux data for caller extraction."""
    # match-only pieces are per hit.
    pieces_match_only = [f"{h.file}:{h.line}:{h.text}" for h in hits]

    # match+context pieces are per merged window (per file).
    file_cache: dict[str, list[str] | None] = {}
    windows_by_file: dict[str, list[tuple[int, int, int]]] = {}
    # windows_by_file[file] = list of (start, end, hit_line)
    for h in hits:
        start = max(1, h.line - before)
        end = h.line + after
        windows_by_file.setdefault(h.file, []).append((start, end, h.line))

    pieces_context: list[str] = []
    callers_by_piece: list[set[tuple[str, str]]] = []
    files_by_piece: list[set[str]] = []

    for file in sorted(windows_by_file.keys()):
        lines = file_cache.get(file)
        if lines is None and file not in file_cache:
            lines = _read_lines(repo_root / file)
            file_cache[file] = lines
        if not lines:
            continue
        merged = _merge_windows([(s, e) for (s, e, _) in windows_by_file[file]])

        # For each merged window, include the snippet and try to infer caller names from
        # hits inside the window by scanning backwards within the snippet for `def`.
        for start, end in merged:
            end2 = min(end, len(lines))
            text = _render_code_block(file, start=start, end=end2, lines=lines)
            pieces_context.append(text)
            files_by_piece.append({file})

            callers: set[tuple[str, str]] = set()
            for _, _, hit_line in windows_by_file[file]:
                if not (start <= hit_line <= end2):
                    continue
                rel_idx = hit_line - start
                # Scan within the snippet.
                for j in range(rel_idx, -1, -1):
                    m = _DEF_RE.match(lines[start - 1 + j])
                    if m:
                        callers.add((file, m.group("name")))
                        break
            callers_by_piece.append(callers)

    # function/window extraction: full enclosing def per hit.
    pieces_fn: list[str] = []
    callers_by_fn_piece: list[set[tuple[str, str]]] = []
    files_by_fn_piece: list[set[str]] = []
    seen_spans: set[tuple[str, int, int]] = set()
    for h in hits:
        lines = file_cache.get(h.file)
        if lines is None and h.file not in file_cache:
            lines = _read_lines(repo_root / h.file)
            file_cache[h.file] = lines
        if not lines:
            continue
        span = _python_enclosing_def_span(lines, line_1based=h.line)
        if span is None:
            continue
        start, end, fn = span
        key = (h.file, start, end)
        if key in seen_spans:
            continue
        seen_spans.add(key)
        pieces_fn.append(_render_code_block(h.file, start=start, end=end, lines=lines))
        callers_by_fn_piece.append({(h.file, fn)})
        files_by_fn_piece.append({h.file})

    return {
        "match_only": {
            "pieces": pieces_match_only,
            "found_callers_by_piece": [set() for _ in pieces_match_only],
            "found_files_by_piece": [{h.file} for h in hits],
        },
        "match_plus_context": {
            "pieces": pieces_context,
            "found_callers_by_piece": callers_by_piece,
            "found_files_by_piece": files_by_piece,
        },
        "window_function": {
            "pieces": pieces_fn,
            "found_callers_by_piece": callers_by_fn_piece,
            "found_files_by_piece": files_by_fn_piece,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6 token-efficiency benchmarks (fixed budgets).")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", default=None, help="Corpus id from benchmarks/corpora.json (e.g. django).")
    group.add_argument("--repo-root", default=None, help="Path to corpus repo root.")
    ap.add_argument(
        "--mode",
        choices=["structural", "retrieval", "both"],
        default="both",
        help="Which query sets to run (default: both).",
    )
    ap.add_argument(
        "--structural-queries",
        default=None,
        help="Structural query set JSON (default: benchmarks/python/django_structural_queries.json when corpus=django).",
    )
    ap.add_argument(
        "--retrieval-queries",
        default=None,
        help="Retrieval query set JSON (default: benchmarks/retrieval/django_queries.json when corpus=django).",
    )
    ap.add_argument(
        "--budgets",
        default="500,1000,2000,5000,10000",
        help="Comma-separated token budgets for payload materialization.",
    )
    ap.add_argument(
        "--cache-root",
        default=str(bench_cache_root(get_repo_root())),
        help="Index-mode cache root (default: benchmark/cache-root).",
    )
    ap.add_argument("--index", default=None, help="Index id (default: repo:<corpus>).")
    ap.add_argument("--rg-glob", default="*.py", help="ripgrep --glob filter for retrieval ranking (default: *.py).")
    ap.add_argument("--max-files", type=int, default=50, help="Max files to consider per retrieval query (default: 50).")
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

    budgets = _parse_budgets(args.budgets)

    index_id = args.index or default_index_id
    index_ctx = get_index_context(
        scan_root=repo_root,
        cache_root_arg=args.cache_root,
        index_id_arg=index_id,
        allow_create=True,
    )

    glob = str(args.rg_glob)
    glob_arg = glob if glob.strip() else None

    results: dict[str, Any] = {}

    # Structural token-efficiency (Django suite).
    if args.mode in ("structural", "both"):
        structural_path: Path
        if args.structural_queries:
            structural_path = Path(args.structural_queries).resolve()
        else:
            if corpus_id != "django":
                raise SystemExit("error: --structural-queries is required unless --corpus django")
            structural_path = (tldr_repo_root / "benchmarks" / "python" / "django_structural_queries.json").resolve()
        structural_queries = _load_structural_queries(structural_path)

        ignore_spec = IgnoreSpec(
            project_dir=repo_root,
            use_gitignore=bool(index_ctx.config.use_gitignore) if index_ctx.config else True,
            cli_patterns=list(index_ctx.config.cli_patterns or ()) if index_ctx.config else None,
            ignore_file=index_ctx.config.ignore_file if index_ctx.config else None,
            gitignore_root=index_ctx.config.gitignore_root if index_ctx.config else None,
        )

        call_graph: ProjectCallGraph | None = None
        call_graph_build_s: float | None = None
        call_graph_cache: str | None = None
        if any(q.get("category") == "impact" for q in structural_queries):
            call_graph, call_graph_build_s, call_graph_cache = _load_or_build_call_graph(
                repo_root=repo_root,
                index_ctx=index_ctx,
                language="python",
                ignore_spec=ignore_spec,
            )

        radon_ver = _radon_version()

        per_query: list[dict[str, Any]] = []

        # Aggregates by category/strategy/budget.
        impact_totals: dict[str, dict[int, dict[str, float]]] = {}
        slice_totals: dict[str, dict[int, dict[str, float]]] = {}
        dfg_totals: dict[str, dict[int, dict[str, float]]] = {}
        cfg_totals: dict[str, dict[int, dict[str, float]]] = {}

        def ensure_totals(bucketed: dict[str, dict[int, dict[str, float]]], *, strategy: str, budget: int) -> dict[str, float]:
            strat = bucketed.setdefault(strategy, {})
            return strat.setdefault(
                budget,
                {
                    "tp": 0.0,
                    "fp": 0.0,
                    "fn": 0.0,
                    "payload_tokens_sum": 0.0,
                    "payload_bytes_sum": 0.0,
                    "queries": 0.0,
                },
            )

        for q in structural_queries:
            qid = q.get("id") or q.get("name") or "unknown"
            category = q.get("category")
            entry: dict[str, Any] = {"id": qid, "category": category, "budgets": {}}

            if category == "impact":
                if call_graph is None:
                    per_query.append({**entry, "error": "call_graph unavailable"})
                    continue
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

                tldr_res = impact_analysis(call_graph, func, max_depth=1, target_file=file_filter)
                predicted = sorted(_flatten_impact_targets(tldr_res))
                target_key = f"{file_filter}:{func}"

                # Grep baselines: rg for leaf call expression.
                leaf = func.split(".")[-1]
                pattern = r"\." + re.escape(leaf) + r"\s*\(" if "." in func else r"\b" + re.escape(leaf) + r"\s*\("
                hits = _rg_hits(repo_root, pattern=pattern)
                hits = _filter_py_def_hits(hits, callee_leaf=leaf)
                grep_parts = _impact_grep_pieces(repo_root, hits=hits, before=25, after=5)

                per_budget: dict[int, Any] = {}
                for budget in budgets:
                    # Strategy: TLDR structured-only (budgeted caller list).
                    tldr_callers: list[dict[str, str]] = []
                    tldr_found: set[tuple[str, str]] = set()
                    payload = json.dumps({"target": target_key, "callers": []}, sort_keys=True)
                    for fp, fn in predicted:
                        cand_callers = [*tldr_callers, {"file": fp, "function": fn}]
                        cand_payload = json.dumps({"target": target_key, "callers": cand_callers}, sort_keys=True)
                        if int(count_tokens(cand_payload)) > budget:
                            break
                        tldr_callers = cand_callers
                        tldr_found.add((fp, fn))
                        payload = cand_payload
                    stats = _payload_stats(payload)
                    tp = len(tldr_found & expected)
                    fp = len(tldr_found - expected)
                    fn = len(expected - tldr_found)
                    tot = ensure_totals(impact_totals, strategy="tldr_structured", budget=budget)
                    tot["tp"] += tp
                    tot["fp"] += fp
                    tot["fn"] += fn
                    tot["payload_tokens_sum"] += stats["payload_tokens"]
                    tot["payload_bytes_sum"] += stats["payload_bytes"]
                    tot["queries"] += 1

                    out = {
                        "tldr_structured": {
                            "payload_tokens": stats["payload_tokens"],
                            "payload_bytes": stats["payload_bytes"],
                            "callers_included": len(tldr_found),
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "metrics": _prf(tp, fp, fn),
                        }
                    }

                    # Strategy: TLDR structured + materialized code for included callers.
                    selected: list[tuple[str, str]] = []
                    code_pieces: list[str] = []
                    for fp, fn in predicted:
                        cand_selected = [*selected, (fp, fn)]
                        cand_json = json.dumps(
                            {
                                "target": target_key,
                                "callers": [{"file": f, "function": name} for f, name in cand_selected],
                            },
                            sort_keys=True,
                        )
                        cand_code_pieces = list(code_pieces)
                        # Extract code for the new caller.
                        abs_fp = repo_root / fp
                        src = abs_fp.read_text(encoding="utf-8", errors="replace")
                        span = _python_function_span_ast(src, function=fn)
                        if span is not None:
                            lines = src.splitlines()
                            cand_code_pieces.append(_render_code_block(fp, start=span[0], end=span[1], lines=lines))
                        cand_payload = cand_json
                        if cand_code_pieces:
                            cand_payload += "\n\n" + "\n\n".join(cand_code_pieces)
                        if int(count_tokens(cand_payload)) > budget:
                            break
                        selected = cand_selected
                        code_pieces = cand_code_pieces
                    payload2 = json.dumps(
                        {"target": target_key, "callers": [{"file": f, "function": name} for f, name in selected]},
                        sort_keys=True,
                    )
                    if code_pieces:
                        payload2 += "\n\n" + "\n\n".join(code_pieces)
                    stats2 = _payload_stats(payload2)
                    found2 = set(selected)
                    tp2 = len(found2 & expected)
                    fp2 = len(found2 - expected)
                    fn2 = len(expected - found2)
                    tot2 = ensure_totals(impact_totals, strategy="tldr_structured_plus_code", budget=budget)
                    tot2["tp"] += tp2
                    tot2["fp"] += fp2
                    tot2["fn"] += fn2
                    tot2["payload_tokens_sum"] += stats2["payload_tokens"]
                    tot2["payload_bytes_sum"] += stats2["payload_bytes"]
                    tot2["queries"] += 1
                    out["tldr_structured_plus_code"] = {
                        "payload_tokens": stats2["payload_tokens"],
                        "payload_bytes": stats2["payload_bytes"],
                        "callers_included": len(found2),
                        "tp": tp2,
                        "fp": fp2,
                        "fn": fn2,
                        "metrics": _prf(tp2, fp2, fn2),
                    }

                    # Grep strategies.
                    for strat_key, label in [
                        ("match_only", "rg_match_only"),
                        ("match_plus_context", "rg_match_plus_context"),
                        ("window_function", "rg_window_function"),
                    ]:
                        pieces = grep_parts[strat_key]["pieces"]
                        payload3, ptok, pbytes, used = _apply_budget(pieces, budget)
                        found_callers: set[tuple[str, str]] = set()
                        found_files: set[str] = set()
                        for s in grep_parts[strat_key]["found_callers_by_piece"][:used]:
                            found_callers |= set(s)
                        for s in grep_parts[strat_key]["found_files_by_piece"][:used]:
                            found_files |= set(s)

                        # Caller-level scoring when we have callers. For match-only, callers are empty.
                        tp3 = len(found_callers & expected)
                        fp3 = len(found_callers - expected)
                        fn3 = len(expected - found_callers)
                        tot3 = ensure_totals(impact_totals, strategy=label, budget=budget)
                        tot3["tp"] += tp3
                        tot3["fp"] += fp3
                        tot3["fn"] += fn3
                        tot3["payload_tokens_sum"] += ptok
                        tot3["payload_bytes_sum"] += pbytes
                        tot3["queries"] += 1
                        out[label] = {
                            "payload_tokens": ptok,
                            "payload_bytes": pbytes,
                            "pieces_used": used,
                            "hits_total": len(hits),
                            "found_callers": len(found_callers),
                            "found_files": len(found_files),
                            "tp": tp3,
                            "fp": fp3,
                            "fn": fn3,
                            "metrics": _prf(tp3, fp3, fn3) if found_callers else None,
                        }
                        _ = payload3  # payload not stored in report

                    per_budget[budget] = out

                entry.update(
                    {
                        "function": func,
                        "file": file_filter,
                        "expected_callers": len(expected),
                        "predicted_callers_total": len(predicted),
                        "rg_pattern": pattern,
                        "budgets": per_budget,
                    }
                )
                per_query.append(entry)
                continue

            if category == "slice":
                file_rel = q.get("file")
                function = q.get("function")
                target_line = q.get("target_line")
                expected_lines_raw = q.get("expected_slice_lines", [])
                if not isinstance(file_rel, str) or not isinstance(function, str) or not isinstance(target_line, int) or not isinstance(expected_lines_raw, list):
                    per_query.append({**entry, "error": "bad slice query schema"})
                    continue
                expected = {int(x) for x in expected_lines_raw if isinstance(x, int)}

                abs_path = repo_root / file_rel
                src = abs_path.read_text(encoding="utf-8", errors="replace")
                lines = src.splitlines()
                span = _python_find_def_span(lines, function_name=function)

                tldr_lines = sorted(
                    {
                        int(x)
                        for x in get_slice(str(abs_path), function, int(target_line), direction="backward", variable=None, language="python")
                    }
                )

                per_budget: dict[int, Any] = {}
                for budget in budgets:
                    # Grep window read around target.
                    payload_w, included_w, win = _select_window_within_span(
                        file_rel=file_rel,
                        lines=lines,
                        span=span,
                        target_line=int(target_line),
                        budget_tokens=budget,
                    )
                    stats_w = _payload_stats(payload_w) if payload_w else {"payload_tokens": 0, "payload_bytes": 0}
                    tpw = len(included_w & expected)
                    fpw = len(included_w - expected)
                    fnw = len(expected - included_w)
                    totw = ensure_totals(slice_totals, strategy="grep_window", budget=budget)
                    totw["tp"] += tpw
                    totw["fp"] += fpw
                    totw["fn"] += fnw
                    totw["payload_tokens_sum"] += stats_w["payload_tokens"]
                    totw["payload_bytes_sum"] += stats_w["payload_bytes"]
                    totw["queries"] += 1

                    out: dict[str, Any] = {
                        "grep_window": {
                            "payload_tokens": stats_w["payload_tokens"],
                            "payload_bytes": stats_w["payload_bytes"],
                            "window": win,
                            "lines_included": len(included_w),
                            "tp": tpw,
                            "fp": fpw,
                            "fn": fnw,
                            "metrics": _prf(tpw, fpw, fnw),
                            "noise_ratio": (len(included_w) / max(1, len(expected))) if expected else None,
                        }
                    }

                    # TLDR structured-only (line numbers).
                    selected_lines: list[int] = []
                    payload_s = json.dumps(
                        {"file": file_rel, "function": function, "target_line": int(target_line), "lines": []},
                        sort_keys=True,
                    )
                    for ln in tldr_lines:
                        cand = [*selected_lines, ln]
                        cand_payload = json.dumps(
                            {"file": file_rel, "function": function, "target_line": int(target_line), "lines": cand},
                            sort_keys=True,
                        )
                        if int(count_tokens(cand_payload)) > budget:
                            break
                        selected_lines = cand
                        payload_s = cand_payload
                    stats_s = _payload_stats(payload_s)
                    included_s = set(selected_lines)
                    tps = len(included_s & expected)
                    fps = len(included_s - expected)
                    fns = len(expected - included_s)
                    tots = ensure_totals(slice_totals, strategy="tldr_structured", budget=budget)
                    tots["tp"] += tps
                    tots["fp"] += fps
                    tots["fn"] += fns
                    tots["payload_tokens_sum"] += stats_s["payload_tokens"]
                    tots["payload_bytes_sum"] += stats_s["payload_bytes"]
                    tots["queries"] += 1
                    out["tldr_structured"] = {
                        "payload_tokens": stats_s["payload_tokens"],
                        "payload_bytes": stats_s["payload_bytes"],
                        "lines_included": len(included_s),
                        "tp": tps,
                        "fp": fps,
                        "fn": fns,
                        "metrics": _prf(tps, fps, fns),
                        "noise_ratio": (len(included_s) / max(1, len(expected))) if expected else None,
                    }

                    # TLDR structured + code for included slice lines.
                    selected2: list[int] = []
                    code_text = ""
                    # Prefer selecting lines closest to target first under tight budgets.
                    tldr_by_distance = sorted(tldr_lines, key=lambda x: (abs(int(x) - int(target_line)), x))
                    for ln in tldr_by_distance:
                        cand = [*selected2, int(ln)]
                        code2, included2 = _extract_lines_for_numbers(file_rel, lines=lines, line_nos=cand)
                        payload2 = json.dumps(
                            {
                                "file": file_rel,
                                "function": function,
                                "target_line": int(target_line),
                                "lines": sorted(included2),
                            },
                            sort_keys=True,
                        )
                        if code2:
                            payload2 += "\n\n" + code2
                        if int(count_tokens(payload2)) > budget:
                            continue
                        selected2 = sorted(included2)
                        code_text = code2
                    payload2 = json.dumps(
                        {"file": file_rel, "function": function, "target_line": int(target_line), "lines": selected2},
                        sort_keys=True,
                    )
                    if code_text:
                        payload2 += "\n\n" + code_text
                    stats2 = _payload_stats(payload2)
                    included2 = set(selected2)
                    tp2 = len(included2 & expected)
                    fp2 = len(included2 - expected)
                    fn2 = len(expected - included2)
                    tot2 = ensure_totals(slice_totals, strategy="tldr_structured_plus_code", budget=budget)
                    tot2["tp"] += tp2
                    tot2["fp"] += fp2
                    tot2["fn"] += fn2
                    tot2["payload_tokens_sum"] += stats2["payload_tokens"]
                    tot2["payload_bytes_sum"] += stats2["payload_bytes"]
                    tot2["queries"] += 1
                    out["tldr_structured_plus_code"] = {
                        "payload_tokens": stats2["payload_tokens"],
                        "payload_bytes": stats2["payload_bytes"],
                        "lines_included": len(included2),
                        "tp": tp2,
                        "fp": fp2,
                        "fn": fn2,
                        "metrics": _prf(tp2, fp2, fn2),
                        "noise_ratio": (len(included2) / max(1, len(expected))) if expected else None,
                    }

                    per_budget[budget] = out

                entry.update(
                    {
                        "file": file_rel,
                        "function": function,
                        "target_line": int(target_line),
                        "expected_slice_lines": len(expected),
                        "tldr_slice_lines_total": len(tldr_lines),
                        "budgets": per_budget,
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
                src = abs_path.read_text(encoding="utf-8", errors="replace")
                lines = src.splitlines()
                span = _python_find_def_span(lines, function_name=function)

                dfg = get_dfg_context(str(abs_path), function, language="python")
                dfg_lines: list[int] = []
                refs = dfg.get("refs")
                if isinstance(refs, list):
                    for r in refs:
                        if not isinstance(r, dict):
                            continue
                        if r.get("name") != variable:
                            continue
                        ln = r.get("line")
                        if isinstance(ln, int):
                            dfg_lines.append(ln)
                edges = dfg.get("edges")
                if isinstance(edges, list):
                    for e in edges:
                        if not isinstance(e, dict):
                            continue
                        if e.get("var") != variable:
                            continue
                        dl = e.get("def_line")
                        ul = e.get("use_line")
                        if isinstance(dl, int):
                            dfg_lines.append(dl)
                        if isinstance(ul, int):
                            dfg_lines.append(ul)
                dfg_lines = sorted({int(x) for x in dfg_lines})

                var_re = re.compile(rf"\b{re.escape(variable)}\b")
                grep_hit_lines: list[int] = []
                if span is not None:
                    start, end = span
                    for ln in range(start, end + 1):
                        if var_re.search(lines[ln - 1]):
                            grep_hit_lines.append(ln)
                else:
                    for ln, text in enumerate(lines, start=1):
                        if var_re.search(text):
                            grep_hit_lines.append(ln)

                per_budget: dict[int, Any] = {}
                for budget in budgets:
                    out: dict[str, Any] = {}

                    # Grep match-only: line numbers for occurrences.
                    selected_hits: list[int] = []
                    payload_m = json.dumps(
                        {"file": file_rel, "function": function, "variable": variable, "hits": []},
                        sort_keys=True,
                    )
                    for ln in grep_hit_lines:
                        cand = [*selected_hits, int(ln)]
                        cand_payload = json.dumps(
                            {"file": file_rel, "function": function, "variable": variable, "hits": cand},
                            sort_keys=True,
                        )
                        if int(count_tokens(cand_payload)) > budget:
                            break
                        selected_hits = cand
                        payload_m = cand_payload
                    stats_m = _payload_stats(payload_m)
                    included_m = set(selected_hits)
                    tp_m = len(included_m & expected_lines)
                    fp_m = len(included_m - expected_lines)
                    fn_m = len(expected_lines - included_m)
                    totm = ensure_totals(dfg_totals, strategy="grep_match_only", budget=budget)
                    totm["tp"] += tp_m
                    totm["fp"] += fp_m
                    totm["fn"] += fn_m
                    totm["payload_tokens_sum"] += stats_m["payload_tokens"]
                    totm["payload_bytes_sum"] += stats_m["payload_bytes"]
                    totm["queries"] += 1
                    out["grep_match_only"] = {
                        "payload_tokens": stats_m["payload_tokens"],
                        "payload_bytes": stats_m["payload_bytes"],
                        "lines_included": len(included_m),
                        "origin_present": bool(origin_expected is not None and origin_expected in included_m),
                        "flow_completeness": (tp_m / len(expected_lines)) if expected_lines else None,
                        "noise_ratio": (len(included_m) / max(1, len(expected_lines))) if expected_lines else None,
                    }

                    # Grep match+context: windows around each occurrence within the function.
                    windows = []
                    if span is not None:
                        lo, hi = span
                    else:
                        lo, hi = (1, len(lines))
                    for ln in selected_hits:
                        start = max(lo, ln - 3)
                        end = min(hi, ln + 3)
                        windows.append((start, end))
                    merged = _merge_windows(windows)
                    pieces_ctx = [_render_code_block(file_rel, start=s, end=e, lines=lines) for s, e in merged]
                    payload_c, ptok_c, pbytes_c, used_c = _apply_budget(pieces_ctx, budget)
                    included_c: set[int] = set()
                    for s, e in merged[:used_c]:
                        included_c |= set(range(s, e + 1))
                    tp_c = len(included_c & expected_lines)
                    fp_c = len(included_c - expected_lines)
                    fn_c = len(expected_lines - included_c)
                    totc = ensure_totals(dfg_totals, strategy="grep_match_plus_context", budget=budget)
                    totc["tp"] += tp_c
                    totc["fp"] += fp_c
                    totc["fn"] += fn_c
                    totc["payload_tokens_sum"] += ptok_c
                    totc["payload_bytes_sum"] += pbytes_c
                    totc["queries"] += 1
                    out["grep_match_plus_context"] = {
                        "payload_tokens": ptok_c,
                        "payload_bytes": pbytes_c,
                        "pieces_used": used_c,
                        "lines_included": len(included_c),
                        "origin_present": bool(origin_expected is not None and origin_expected in included_c),
                        "flow_completeness": (tp_c / len(expected_lines)) if expected_lines else None,
                        "noise_ratio": (len(included_c) / max(1, len(expected_lines))) if expected_lines else None,
                    }
                    _ = payload_c

                    # Grep window read: full function (or file) prefix under budget.
                    w_payload, w_included, win = _select_window_within_span(
                        file_rel=file_rel,
                        lines=lines,
                        span=span,
                        target_line=(span[0] if span is not None else 1),
                        budget_tokens=budget,
                    )
                    stats_w = _payload_stats(w_payload) if w_payload else {"payload_tokens": 0, "payload_bytes": 0}
                    tp_w = len(w_included & expected_lines)
                    fp_w = len(w_included - expected_lines)
                    fn_w = len(expected_lines - w_included)
                    totw = ensure_totals(dfg_totals, strategy="grep_window_function", budget=budget)
                    totw["tp"] += tp_w
                    totw["fp"] += fp_w
                    totw["fn"] += fn_w
                    totw["payload_tokens_sum"] += stats_w["payload_tokens"]
                    totw["payload_bytes_sum"] += stats_w["payload_bytes"]
                    totw["queries"] += 1
                    out["grep_window_function"] = {
                        "payload_tokens": stats_w["payload_tokens"],
                        "payload_bytes": stats_w["payload_bytes"],
                        "window": win,
                        "lines_included": len(w_included),
                        "origin_present": bool(origin_expected is not None and origin_expected in w_included),
                        "flow_completeness": (tp_w / len(expected_lines)) if expected_lines else None,
                        "noise_ratio": (len(w_included) / max(1, len(expected_lines))) if expected_lines else None,
                    }

                    # TLDR structured-only: DFG-derived line set.
                    selected_dfg: list[int] = []
                    payload_t = json.dumps(
                        {"file": file_rel, "function": function, "variable": variable, "lines": []},
                        sort_keys=True,
                    )
                    for ln in dfg_lines:
                        cand = [*selected_dfg, int(ln)]
                        cand_payload = json.dumps(
                            {"file": file_rel, "function": function, "variable": variable, "lines": cand},
                            sort_keys=True,
                        )
                        if int(count_tokens(cand_payload)) > budget:
                            break
                        selected_dfg = cand
                        payload_t = cand_payload
                    stats_t = _payload_stats(payload_t)
                    included_t = set(selected_dfg)
                    tp_t = len(included_t & expected_lines)
                    fp_t = len(included_t - expected_lines)
                    fn_t = len(expected_lines - included_t)
                    tott = ensure_totals(dfg_totals, strategy="tldr_structured", budget=budget)
                    tott["tp"] += tp_t
                    tott["fp"] += fp_t
                    tott["fn"] += fn_t
                    tott["payload_tokens_sum"] += stats_t["payload_tokens"]
                    tott["payload_bytes_sum"] += stats_t["payload_bytes"]
                    tott["queries"] += 1
                    out["tldr_structured"] = {
                        "payload_tokens": stats_t["payload_tokens"],
                        "payload_bytes": stats_t["payload_bytes"],
                        "lines_included": len(included_t),
                        "origin_present": bool(origin_expected is not None and origin_expected in included_t),
                        "flow_completeness": (tp_t / len(expected_lines)) if expected_lines else None,
                        "noise_ratio": (len(included_t) / max(1, len(expected_lines))) if expected_lines else None,
                    }

                    # TLDR structured + code for included lines.
                    selected_lines2: list[int] = []
                    code_text = ""
                    for ln in dfg_lines:
                        cand = [*selected_lines2, int(ln)]
                        code2, included2 = _extract_lines_for_numbers(file_rel, lines=lines, line_nos=cand)
                        payload2 = json.dumps(
                            {"file": file_rel, "function": function, "variable": variable, "lines": sorted(included2)},
                            sort_keys=True,
                        )
                        if code2:
                            payload2 += "\n\n" + code2
                        if int(count_tokens(payload2)) > budget:
                            break
                        selected_lines2 = sorted(included2)
                        code_text = code2
                    payload2 = json.dumps(
                        {"file": file_rel, "function": function, "variable": variable, "lines": selected_lines2},
                        sort_keys=True,
                    )
                    if code_text:
                        payload2 += "\n\n" + code_text
                    stats2 = _payload_stats(payload2)
                    included2 = set(selected_lines2)
                    tp2 = len(included2 & expected_lines)
                    fp2 = len(included2 - expected_lines)
                    fn2 = len(expected_lines - included2)
                    tot2 = ensure_totals(dfg_totals, strategy="tldr_structured_plus_code", budget=budget)
                    tot2["tp"] += tp2
                    tot2["fp"] += fp2
                    tot2["fn"] += fn2
                    tot2["payload_tokens_sum"] += stats2["payload_tokens"]
                    tot2["payload_bytes_sum"] += stats2["payload_bytes"]
                    tot2["queries"] += 1
                    out["tldr_structured_plus_code"] = {
                        "payload_tokens": stats2["payload_tokens"],
                        "payload_bytes": stats2["payload_bytes"],
                        "lines_included": len(included2),
                        "origin_present": bool(origin_expected is not None and origin_expected in included2),
                        "flow_completeness": (tp2 / len(expected_lines)) if expected_lines else None,
                        "noise_ratio": (len(included2) / max(1, len(expected_lines))) if expected_lines else None,
                    }

                    per_budget[budget] = out

                entry.update(
                    {
                        "file": file_rel,
                        "function": function,
                        "variable": variable,
                        "expected_lines": len(expected_lines),
                        "tldr_lines_total": len(dfg_lines),
                        "grep_hits_total": len(grep_hit_lines),
                        "origin_expected": origin_expected,
                        "budgets": per_budget,
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
                src = abs_path.read_text(encoding="utf-8", errors="replace")
                lines = src.splitlines()
                span = _python_find_def_span(lines, function_name=function)
                if span is None:
                    per_query.append({**entry, "error": f"function not found (heuristic): {function}"})
                    continue

                cfg = get_cfg_context(str(abs_path), function, language="python")
                tldr_cc = cfg.get("cyclomatic_complexity")
                tldr_cc_int = int(tldr_cc) if isinstance(tldr_cc, int) else None
                radon_cc = _radon_cc_for_function(src, function=function)
                heur_cc = _complexity_heuristic(lines, span=span)

                per_budget: dict[int, Any] = {}
                for budget in budgets:
                    payload_t = json.dumps(
                        {"file": file_rel, "function": function, "cyclomatic_complexity": tldr_cc_int},
                        sort_keys=True,
                    )
                    stats_t = _payload_stats(payload_t)
                    # Complexity is scalar; treat tp/fp/fn as exact-match scoring vs radon when available.
                    if radon_cc is None or tldr_cc_int is None:
                        tp = fp = fn = 0
                    else:
                        tp = 1 if tldr_cc_int == radon_cc else 0
                        fp = 0 if tp else 1
                        fn = 0 if tp else 1
                    tot = ensure_totals(cfg_totals, strategy="tldr_structured", budget=budget)
                    tot["tp"] += tp
                    tot["fp"] += fp
                    tot["fn"] += fn
                    tot["payload_tokens_sum"] += stats_t["payload_tokens"]
                    tot["payload_bytes_sum"] += stats_t["payload_bytes"]
                    tot["queries"] += 1

                    payload_h = json.dumps(
                        {"file": file_rel, "function": function, "cyclomatic_complexity": heur_cc},
                        sort_keys=True,
                    )
                    stats_h = _payload_stats(payload_h)
                    if radon_cc is None:
                        tp2 = fp2 = fn2 = 0
                    else:
                        tp2 = 1 if heur_cc == radon_cc else 0
                        fp2 = 0 if tp2 else 1
                        fn2 = 0 if tp2 else 1
                    tot2 = ensure_totals(cfg_totals, strategy="grep_heuristic", budget=budget)
                    tot2["tp"] += tp2
                    tot2["fp"] += fp2
                    tot2["fn"] += fn2
                    tot2["payload_tokens_sum"] += stats_h["payload_tokens"]
                    tot2["payload_bytes_sum"] += stats_h["payload_bytes"]
                    tot2["queries"] += 1

                    per_budget[budget] = {
                        "tldr_structured": {
                            "payload_tokens": stats_t["payload_tokens"],
                            "payload_bytes": stats_t["payload_bytes"],
                            "cyclomatic_complexity": tldr_cc_int,
                            "radon": radon_cc,
                        },
                        "grep_heuristic": {
                            "payload_tokens": stats_h["payload_tokens"],
                            "payload_bytes": stats_h["payload_bytes"],
                            "cyclomatic_complexity": heur_cc,
                            "radon": radon_cc,
                        },
                    }

                entry.update(
                    {
                        "file": file_rel,
                        "function": function,
                        "radon": radon_cc,
                        "tldr": tldr_cc_int,
                        "grep_heuristic": heur_cc,
                        "budgets": per_budget,
                    }
                )
                per_query.append(entry)
                continue

            per_query.append({**entry, "skipped": True, "reason": "unknown category"})

        def summarize_bucketed(totals: dict[str, dict[int, dict[str, float]]]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for strategy, by_budget in totals.items():
                out[strategy] = []
                for budget in budgets:
                    agg = by_budget.get(budget)
                    if not agg:
                        continue
                    tp = int(agg["tp"])
                    fp = int(agg["fp"])
                    fn = int(agg["fn"])
                    queries = int(agg["queries"])
                    payload_tokens_sum = float(agg["payload_tokens_sum"])
                    out[strategy].append(
                        {
                            "budget_tokens": budget,
                            "queries": queries,
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "metrics": _prf(tp, fp, fn),
                            "payload_tokens_sum": int(payload_tokens_sum),
                            "payload_bytes_sum": int(agg["payload_bytes_sum"]),
                            "payload_tokens_mean": round((payload_tokens_sum / queries) if queries else 0.0, 2),
                        }
                    )
            return out

        def summarize_slice(per_query_rows: list[dict[str, Any]]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            strategies = ["grep_window", "tldr_structured", "tldr_structured_plus_code"]
            for strat in strategies:
                rows: list[dict[str, Any]] = []
                for budget in budgets:
                    precs: list[float] = []
                    recs: list[float] = []
                    noise: list[float] = []
                    tok_sum = 0
                    byte_sum = 0
                    n = 0
                    for qrow in per_query_rows:
                        if qrow.get("category") != "slice":
                            continue
                        b = (qrow.get("budgets") or {}).get(budget) or {}
                        s = b.get(strat) or {}
                        if not isinstance(s, dict):
                            continue
                        m = s.get("metrics")
                        if isinstance(m, dict):
                            if isinstance(m.get("precision"), (int, float)):
                                precs.append(float(m["precision"]))
                            if isinstance(m.get("recall"), (int, float)):
                                recs.append(float(m["recall"]))
                        if isinstance(s.get("noise_ratio"), (int, float)):
                            noise.append(float(s["noise_ratio"]))
                        tok_sum += int(s.get("payload_tokens") or 0)
                        byte_sum += int(s.get("payload_bytes") or 0)
                        n += 1
                    rows.append(
                        {
                            "budget_tokens": budget,
                            "queries": n,
                            "precision_mean": _mean(precs),
                            "recall_mean": _mean(recs),
                            "noise_ratio_mean": _mean(noise),
                            "payload_tokens_sum": tok_sum,
                            "payload_bytes_sum": byte_sum,
                            "payload_tokens_mean": round((tok_sum / n) if n else 0.0, 2),
                        }
                    )
                out[strat] = rows
            return out

        def summarize_data_flow(per_query_rows: list[dict[str, Any]]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            strategies = [
                "grep_match_only",
                "grep_match_plus_context",
                "grep_window_function",
                "tldr_structured",
                "tldr_structured_plus_code",
            ]
            for strat in strategies:
                rows: list[dict[str, Any]] = []
                for budget in budgets:
                    origins: list[float] = []
                    comps: list[float] = []
                    noise: list[float] = []
                    tok_sum = 0
                    byte_sum = 0
                    n = 0
                    for qrow in per_query_rows:
                        if qrow.get("category") != "data_flow":
                            continue
                        b = (qrow.get("budgets") or {}).get(budget) or {}
                        s = b.get(strat) or {}
                        if not isinstance(s, dict):
                            continue
                        if isinstance(s.get("origin_present"), bool):
                            origins.append(1.0 if s["origin_present"] else 0.0)
                        if isinstance(s.get("flow_completeness"), (int, float)):
                            comps.append(float(s["flow_completeness"]))
                        if isinstance(s.get("noise_ratio"), (int, float)):
                            noise.append(float(s["noise_ratio"]))
                        tok_sum += int(s.get("payload_tokens") or 0)
                        byte_sum += int(s.get("payload_bytes") or 0)
                        n += 1
                    rows.append(
                        {
                            "budget_tokens": budget,
                            "queries": n,
                            "origin_accuracy": _mean(origins),
                            "flow_completeness_mean": _mean(comps),
                            "noise_ratio_mean": _mean(noise),
                            "payload_tokens_sum": tok_sum,
                            "payload_bytes_sum": byte_sum,
                            "payload_tokens_mean": round((tok_sum / n) if n else 0.0, 2),
                        }
                    )
                out[strat] = rows
            return out

        def summarize_complexity(per_query_rows: list[dict[str, Any]]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            strategies = ["tldr_structured", "grep_heuristic"]
            for strat in strategies:
                rows: list[dict[str, Any]] = []
                for budget in budgets:
                    diffs: list[float] = []
                    exact = 0
                    count = 0
                    tok_sum = 0
                    byte_sum = 0
                    n = 0
                    for qrow in per_query_rows:
                        if qrow.get("category") != "complexity":
                            continue
                        b = (qrow.get("budgets") or {}).get(budget) or {}
                        s = b.get(strat) or {}
                        if not isinstance(s, dict):
                            continue
                        radon_cc = s.get("radon")
                        cc = s.get("cyclomatic_complexity")
                        if isinstance(radon_cc, int) and isinstance(cc, int):
                            count += 1
                            if cc == radon_cc:
                                exact += 1
                            diffs.append(float(abs(cc - radon_cc)))
                        tok_sum += int(s.get("payload_tokens") or 0)
                        byte_sum += int(s.get("payload_bytes") or 0)
                        n += 1
                    rows.append(
                        {
                            "budget_tokens": budget,
                            "queries": n,
                            "scored": count,
                            "accuracy": (exact / count) if count else None,
                            "mae": _mean(diffs),
                            "payload_tokens_sum": tok_sum,
                            "payload_bytes_sum": byte_sum,
                            "payload_tokens_mean": round((tok_sum / n) if n else 0.0, 2),
                        }
                    )
                out[strat] = rows
            return out

        # Tokens-per-correct metrics for impact (caller-level).
        impact_summary = summarize_bucketed(impact_totals)
        for strategy, rows in impact_summary.items():
            for row in rows:
                budget = int(row["budget_tokens"])
                agg = impact_totals.get(strategy, {}).get(budget, {})
                expected_total = sum(
                    int(q.get("expected_callers") or 0)
                    for q in per_query
                    if q.get("category") == "impact" and isinstance(q.get("expected_callers"), int)
                )
                tp = int(row.get("tp") or 0)
                payload_tokens_sum = int(row.get("payload_tokens_sum") or 0)
                row["tokens_per_true_caller"] = (
                    round(payload_tokens_sum / expected_total, 2) if expected_total else None
                )
                row["tokens_per_correct_caller"] = round(payload_tokens_sum / tp, 2) if tp else None

        results["structural"] = {
            "queries": str(structural_path),
            "budgets": budgets,
            "call_graph_cache": call_graph_cache,
            "call_graph_build_s": round(call_graph_build_s, 6) if call_graph_build_s is not None else None,
            "radon_version": radon_ver,
            "impact": {"summary": impact_summary},
            "slice": {"summary": summarize_slice(per_query)},
            "data_flow": {"summary": summarize_data_flow(per_query)},
            "complexity": {"summary": summarize_complexity(per_query)},
            "per_query": per_query,
        }

    # Retrieval token-efficiency (Django retrieval suite).
    if args.mode in ("retrieval", "both"):
        retrieval_path: Path
        if args.retrieval_queries:
            retrieval_path = Path(args.retrieval_queries).resolve()
        else:
            if corpus_id != "django":
                raise SystemExit("error: --retrieval-queries is required unless --corpus django")
            retrieval_path = (tldr_repo_root / "benchmarks" / "retrieval" / "django_queries.json").resolve()
        retrieval_queries = _load_retrieval_queries(retrieval_path)

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

        per_query: list[dict[str, Any]] = []
        totals: dict[str, dict[int, dict[str, float]]] = {}

        def ensure(strategy: str, budget: int) -> dict[str, float]:
            strat = totals.setdefault(strategy, {})
            return strat.setdefault(
                budget,
                {
                    "mrr_sum": 0.0,
                    "queries_pos": 0.0,
                    "fpr_sum": 0.0,
                    "queries_neg": 0.0,
                    "payload_tokens_sum": 0.0,
                    "payload_bytes_sum": 0.0,
                },
            )

        max_files = max(1, int(args.max_files))

        for q in retrieval_queries:
            relevant = set(q.relevant_files)
            is_negative = not relevant
            rg_pattern = q.rg_pattern or re.escape(q.query)

            rg_rank = _rg_rank_files(repo_root, pattern=rg_pattern, glob=glob_arg)[:max_files]
            sem_rank = None
            if semantic_available:
                sem_rank = _semantic_rank_files(repo_root, index_ctx=index_ctx, query=q.query, k=max_files) or []
                sem_rank = sem_rank[:max_files]

            hybrid_rank = _rrf_fuse([rg_rank, sem_rank])[:max_files] if sem_rank is not None else None

            rankings: dict[str, list[str] | None] = {
                "rg": rg_rank,
                "semantic": sem_rank,
                "hybrid_rrf": hybrid_rank,
            }

            q_entry: dict[str, Any] = {
                "id": q.id,
                "query": q.query,
                "relevant_files": list(q.relevant_files),
                "rg_pattern": rg_pattern,
                "budgets": {},
            }

            for method, ranking in rankings.items():
                if ranking is None:
                    continue
                pieces: list[str] = []
                files_for_piece: list[str] = []
                for fp in ranking:
                    snippet, _ = _render_snippet_for_file(
                        repo_root,
                        file_rel=fp,
                        rg_pattern=rg_pattern,
                        before=2,
                        after=2,
                    )
                    pieces.append(snippet)
                    files_for_piece.append(fp)

                for budget in budgets:
                    payload, ptok, pbytes, used = _apply_budget(pieces, budget)
                    included_files = set(files_for_piece[:used])

                    rr = 0.0
                    if relevant:
                        for i, fp in enumerate(ranking, start=1):
                            if fp in relevant:
                                if fp in included_files:
                                    rr = 1.0 / i
                                break
                        agg = ensure(method, budget)
                        agg["mrr_sum"] += rr
                        agg["queries_pos"] += 1
                    else:
                        fpr = 1.0 if included_files else 0.0
                        agg = ensure(method, budget)
                        agg["fpr_sum"] += fpr
                        agg["queries_neg"] += 1

                    agg["payload_tokens_sum"] += ptok
                    agg["payload_bytes_sum"] += pbytes

                    q_entry["budgets"].setdefault(budget, {})[method] = {
                        "payload_tokens": ptok,
                        "payload_bytes": pbytes,
                        "pieces_used": used,
                        "mrr": rr if relevant else None,
                        "fpr": (1.0 if included_files else 0.0) if is_negative else None,
                    }
                    _ = payload

            per_query.append(q_entry)

        summary: dict[str, Any] = {}
        for strategy, by_budget in totals.items():
            rows: list[dict[str, Any]] = []
            for budget in budgets:
                agg = by_budget.get(budget)
                if not agg:
                    continue
                qpos = int(agg["queries_pos"])
                qneg = int(agg["queries_neg"])
                rows.append(
                    {
                        "budget_tokens": budget,
                        "mrr_mean": (agg["mrr_sum"] / qpos) if qpos else None,
                        "fpr_mean": (agg["fpr_sum"] / qneg) if qneg else None,
                        "queries_pos": qpos,
                        "queries_neg": qneg,
                        "payload_tokens_mean": round((agg["payload_tokens_sum"] / (qpos + qneg)) if (qpos + qneg) else 0.0, 2),
                    }
                )
            summary[strategy] = rows

        results["retrieval"] = {
            "queries": str(retrieval_path),
            "budgets": budgets,
            "rg_glob": glob_arg,
            "semantic_available": bool(semantic_available),
            "semantic_model": semantic_meta.get("model") if isinstance(semantic_meta, dict) else None,
            "semantic_dimension": semantic_meta.get("dimension") if isinstance(semantic_meta, dict) else None,
            "summary": summary,
            "per_query": per_query,
        }

    report = make_report(
        phase="phase6_token_efficiency",
        meta=gather_meta(tldr_repo_root=tldr_repo_root, corpus_id=corpus_id, corpus_root=repo_root),
        protocol={
            "schema_version": SCHEMA_VERSION,
            "mode": args.mode,
            "budgets": budgets,
            "cache_root": str(index_ctx.cache_root) if index_ctx.cache_root is not None else None,
            "index_id": index_ctx.index_id,
            "rg_glob": glob_arg,
        },
        results=results,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        ts = now_utc_compact()
        out_path = bench_runs_root(tldr_repo_root) / f"{ts}-token-efficiency-{corpus_id}.json"
    write_report(out_path, report)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
