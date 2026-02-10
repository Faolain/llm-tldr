#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from bench_util import (
    bench_cache_root,
    bench_corpora_root,
    bench_root,
    bench_runs_root,
    gather_meta,
    get_repo_root,
    make_report,
    now_utc_compact,
    write_report,
)

import bench_token_efficiency as bte

from tldr.analysis import impact_analysis
from tldr.api import get_dfg_context, get_slice
from tldr.indexing.index import get_index_context
from tldr.stats import count_tokens
from tldr.tldrignore import IgnoreSpec


SCHEMA_VERSION = 1


def _stable_shuffle(seed: int, task_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    h = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
    s = int.from_bytes(h[:8], "big", signed=False)
    rnd = random.Random(s)
    xs = list(items)
    rnd.shuffle(xs)
    return xs


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        raise ValueError(f"Bad tasks file: {path}")
    out: list[dict[str, Any]] = []
    for t in tasks:
        if isinstance(t, dict):
            out.append(t)
    return out


def _index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        qid = r.get("id")
        if isinstance(qid, str):
            out[qid] = r
    return out


def _prompt_for_task(*, question: str, context: str, schema_hint: str) -> str:
    return (
        "You are given TOOL OUTPUT context. Answer the question using ONLY the context.\n\n"
        f"Question:\n{question}\n\n"
        f"Output format:\n{schema_hint}\n\n"
        "Context:\n<BEGIN_CONTEXT>\n"
        f"{context}\n"
        "<END_CONTEXT>\n\n"
        "Return ONLY the JSON output (no prose)."
    )


def _prompt_for_open_ended_task(*, question: str, context: str) -> str:
    return (
        "You are given TOOL OUTPUT context. Answer the question using ONLY the context.\n\n"
        f"Question:\n{question}\n\n"
        "Context:\n<BEGIN_CONTEXT>\n"
        f"{context}\n"
        "<END_CONTEXT>\n\n"
        "Answer in plain text. Do not guess; if the context is insufficient, say what is missing."
    )


def _label_variants(variants: list[dict[str, Any]]) -> None:
    labels = [chr(ord("A") + i) for i in range(26)]
    for i, v in enumerate(variants):
        v["label"] = labels[i] if i < len(labels) else f"V{i+1}"


def _impact_context_tldr(
    *,
    call_graph: Any,
    callee_func: str,
    callee_file: str,
    budget_tokens: int,
) -> tuple[str, int, int]:
    res = impact_analysis(call_graph, callee_func, max_depth=1, target_file=callee_file)
    predicted = sorted(bte._flatten_impact_targets(res))

    callers: list[dict[str, str]] = []
    payload = json.dumps({"callers": []}, sort_keys=True)
    for fp, fn in predicted:
        cand = [*callers, {"file": fp, "function": fn}]
        cand_payload = json.dumps({"callers": cand}, sort_keys=True)
        if int(count_tokens(cand_payload)) > budget_tokens:
            break
        callers = cand
        payload = cand_payload
    return payload, int(count_tokens(payload)), len(payload.encode("utf-8"))


def _impact_context_tldr_plus_code(
    *,
    repo_root: Path,
    call_graph: Any,
    callee_func: str,
    callee_file: str,
    budget_tokens: int,
) -> tuple[str, int, int]:
    res = impact_analysis(call_graph, callee_func, max_depth=1, target_file=callee_file)
    predicted = sorted(bte._flatten_impact_targets(res))

    callers: list[dict[str, str]] = []
    code_pieces: list[str] = []
    payload = json.dumps({"callers": []}, sort_keys=True)

    for fp, fn in predicted:
        cand_callers = [*callers, {"file": fp, "function": fn}]
        cand_json = json.dumps({"callers": cand_callers}, sort_keys=True)
        if int(count_tokens(cand_json)) > budget_tokens:
            break

        # Try to include code for this caller if it fits; otherwise include the caller list only.
        cand_code_pieces = list(code_pieces)
        try:
            abs_fp = repo_root / fp
            src = abs_fp.read_text(encoding="utf-8", errors="replace")
            span = bte._python_function_span_ast(src, function=fn)
            if span is not None:
                lines = src.splitlines()
                cand_code_pieces.append(bte._render_code_block(fp, start=span[0], end=span[1], lines=lines))
        except OSError:
            pass

        cand_payload = cand_json
        if cand_code_pieces:
            cand_payload += "\n\n" + "\n\n".join(cand_code_pieces)
        if int(count_tokens(cand_payload)) <= budget_tokens:
            callers = cand_callers
            code_pieces = cand_code_pieces
            payload = cand_payload
            continue

        # Fall back to callers-only (still useful for open-ended tasks).
        callers = cand_callers
        payload = cand_json

    return payload, int(count_tokens(payload)), len(payload.encode("utf-8"))


def _impact_context_rg(
    *,
    repo_root: Path,
    callee_func: str,
    budget_tokens: int,
) -> tuple[str, int, int]:
    leaf = callee_func.split(".")[-1]
    pattern = r"\." + re.escape(leaf) + r"\s*\(" if "." in callee_func else r"\b" + re.escape(leaf) + r"\s*\("
    hits = bte._rg_hits(repo_root, pattern=pattern)
    hits = bte._filter_py_def_hits(hits, callee_leaf=leaf)

    def _apply_budget_skip(pieces: list[str], budget: int) -> tuple[str, int, int]:
        payload_parts: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            candidate = "\n\n".join([*payload_parts, piece]) if payload_parts else piece
            if int(count_tokens(candidate)) > int(budget):
                continue
            payload_parts.append(piece)
        payload = "\n\n".join(payload_parts)
        return payload, int(count_tokens(payload)), len(payload.encode("utf-8"))

    # Prefer a "caller summary" piece per enclosing def, so the model can name callers
    # without having to see an entire (possibly huge) function body.
    group_pieces: list[str] = []
    file_cache: dict[str, list[str] | None] = {}
    grouped: dict[tuple[str, int, int, str], list[int]] = {}
    for h in hits:
        lines = file_cache.get(h.file)
        if lines is None and h.file not in file_cache:
            try:
                lines = (repo_root / h.file).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = None
            file_cache[h.file] = lines
        if not lines:
            continue
        span = bte._python_enclosing_def_span(lines, line_1based=int(h.line))
        if span is None:
            continue
        start, end, fn = span
        grouped.setdefault((h.file, int(start), int(end), str(fn)), []).append(int(h.line))

    for file_rel, start, end, fn in sorted(grouped.keys(), key=lambda x: (x[0], x[1], x[3])):
        lines = file_cache.get(file_rel)
        if not lines:
            continue
        hit_lines = sorted(set(grouped[(file_rel, start, end, fn)]))
        # Include a small header region + tight windows around each call site.
        line_nos: list[int] = []
        # Header region (decorators + def line + a couple lines).
        for ln in range(int(start), min(int(end), int(start) + 3) + 1):
            line_nos.append(int(ln))
        for hl in hit_lines:
            for ln in range(max(int(start), int(hl) - 2), min(int(end), int(hl) + 2) + 1):
                line_nos.append(int(ln))
        code_text, _included = bte._extract_lines_for_numbers(file_rel, lines=lines, line_nos=line_nos)
        if not code_text:
            continue
        group_pieces.append(f"# caller: {file_rel}:{fn}\n{code_text}")

    payload, ptok, pbytes = _apply_budget_skip(group_pieces, int(budget_tokens))
    if payload.strip():
        return payload, int(ptok), int(pbytes)

    # Fallback: windows around matches (smaller than full enclosing def).
    parts = bte._impact_grep_pieces(repo_root, hits=hits, before=60, after=10)
    pieces2 = parts["match_plus_context"]["pieces"]
    payload2, ptok2, pbytes2 = _apply_budget_skip(list(pieces2), int(budget_tokens))
    if payload2.strip():
        return payload2, int(ptok2), int(pbytes2)

    # Last resort: raw match lines.
    pieces3 = parts["match_only"]["pieces"]
    payload3, ptok3, pbytes3 = _apply_budget_skip(list(pieces3), int(budget_tokens))
    return payload3, int(ptok3), int(pbytes3)


def _slice_context_tldr(
    *,
    repo_root: Path,
    file_rel: str,
    function: str,
    target_line: int,
    budget_tokens: int,
) -> tuple[str, int, int]:
    abs_path = repo_root / file_rel
    tldr_lines = sorted(
        {
            int(x)
            for x in get_slice(
                str(abs_path),
                function,
                int(target_line),
                direction="backward",
                variable=None,
                language="python",
            )
        }
    )

    # Keep this purely structured (no code), otherwise models often confuse
    # code line numbers with slice line numbers.
    selected: list[int] = []
    for ln in tldr_lines:
        cand = [*selected, int(ln)]
        payload2 = json.dumps({"lines": cand}, sort_keys=True)
        if int(count_tokens(payload2)) > budget_tokens:
            break
        selected = cand

    payload = json.dumps({"lines": selected}, sort_keys=True)
    return payload, int(count_tokens(payload)), len(payload.encode("utf-8"))


def _slice_context_tldr_plus_code(
    *,
    repo_root: Path,
    file_rel: str,
    function: str,
    target_line: int,
    budget_tokens: int,
) -> tuple[str, int, int]:
    abs_path = repo_root / file_rel
    try:
        src = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        payload0 = json.dumps(
            {
                "file": file_rel,
                "function": function,
                "target_line": int(target_line),
                "error": "unreadable",
            },
            sort_keys=True,
        )
        return payload0, int(count_tokens(payload0)), len(payload0.encode("utf-8"))

    lines = src.splitlines()
    span = bte._python_find_def_span(lines, function_name=function)
    if span is not None:
        lo, hi = span
    else:
        lo, hi = (1, len(lines))
    lo = max(1, int(lo))
    hi = min(int(hi), len(lines))
    target_line = max(int(lo), min(int(hi), int(target_line)))

    tldr_lines = sorted(
        {
            int(x)
            for x in get_slice(
                str(abs_path),
                function,
                int(target_line),
                direction="backward",
                variable=None,
                language="python",
            )
        }
    )

    # Open-ended slice tasks often need control-flow continuity. Pack context as:
    # - one large contiguous "target window" around target_line (rg-style)
    # - plus extra, budgeted small windows around slice-selected lines outside that window
    # This tends to keep explanations grounded while still surfacing remote dependencies.
    meta, code_text = _pack_open_ended_slice_context(
        file_rel=file_rel,
        lines=lines,
        function=function,
        span=(lo, hi),
        target_line=int(target_line),
        slice_lines=tldr_lines,
        budget_tokens=int(budget_tokens),
    )

    payload = json.dumps(meta, sort_keys=True)
    if code_text:
        payload += "\n\n" + code_text
    return payload, int(count_tokens(payload)), len(payload.encode("utf-8"))


def _line_in_windows(windows: list[tuple[int, int]], line: int) -> bool:
    for s, e in windows:
        if int(s) <= int(line) <= int(e):
            return True
    return False


_CALL_NAME_RE = re.compile(r"(?<![A-Za-z0-9_.])(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CLASS_DEF_RE = re.compile(r"^(?P<indent>\s*)class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b")


def _indent_width(s: str) -> int:
    return len(s[: len(s) - len(s.lstrip())].expandtabs(4))


def _python_find_class_span(lines: list[str], *, class_name: str) -> tuple[int, int] | None:
    """Best-effort (start_line, end_line) for a top-level class by name using indentation heuristics."""
    start_cls: int | None = None
    cls_indent: int | None = None
    for i, line in enumerate(lines, start=1):
        m = _CLASS_DEF_RE.match(line)
        if not m:
            continue
        if m.group("name") != class_name:
            continue
        start_cls = i
        cls_indent = _indent_width(m.group("indent"))
        break

    if start_cls is None or cls_indent is None:
        return None

    start = start_cls
    end = len(lines)
    for k in range(start_cls + 1, len(lines) + 1):
        ln = lines[k - 1]
        if not ln.strip():
            continue
        if bte._DEF_RE.match(ln) or _CLASS_DEF_RE.match(ln):
            if _indent_width(ln) <= cls_indent:
                end = k - 1
                break
    while end > start_cls and not lines[end - 1].strip():
        end -= 1
    end = max(end, start_cls)
    return start, end


def _extract_call_names_from_windows(
    *, lines: list[str], windows: list[tuple[int, int]], target_line: int, line_filter: set[int] | None = None
) -> dict[str, int]:
    """Return name -> min abs(line - target_line) for unqualified call names in windows."""
    out: dict[str, int] = {}
    for s, e in windows:
        for ln in range(int(s), int(e) + 1):
            if not (1 <= int(ln) <= len(lines)):
                continue
            if line_filter is not None and int(ln) not in line_filter:
                continue
            text = lines[int(ln) - 1]
            if text.lstrip().startswith("#"):
                continue
            # Drop inline comments (best-effort; doesn't attempt to handle "#" inside strings).
            if "#" in text:
                text = text.split("#", 1)[0]
            for m in _CALL_NAME_RE.finditer(text):
                name = m.group("name")
                dist = abs(int(ln) - int(target_line))
                prev = out.get(name)
                if prev is None or dist < prev:
                    out[name] = dist
    return out


def _add_related_definitions(
    *,
    file_rel: str,
    lines: list[str],
    function_span: tuple[int, int],
    windows: list[tuple[int, int]],
    target_line: int,
    budget_tokens: int,
    meta: dict[str, Any],
    code_text: str,
) -> tuple[dict[str, Any], str]:
    """Append same-file def/class snippets for call targets referenced in windows, if budget allows."""
    base_payload = json.dumps(meta, sort_keys=True)
    if code_text:
        base_payload += "\n\n" + code_text
    base_tokens = int(count_tokens(base_payload))
    if base_tokens >= int(budget_tokens):
        return meta, code_text

    line_filter: set[int] | None = None
    slice_lines = meta.get("slice_lines")
    if isinstance(slice_lines, list) and all(isinstance(x, int) for x in slice_lines):
        line_filter = {int(x) for x in slice_lines}
    call_dists = _extract_call_names_from_windows(
        lines=lines,
        windows=windows,
        target_line=int(target_line),
        line_filter=line_filter,
    )
    # Stable priority: closest reference to target_line first, then name.
    candidates = sorted(call_dists.items(), key=lambda kv: (int(kv[1]), str(kv[0])))

    related: list[dict[str, Any]] = []
    extra_blocks: list[str] = []

    lo, hi = function_span
    max_related = 4
    for name, _dist in candidates:
        if len(related) >= max_related:
            break

        # Prefer top-level defs/classes only, and avoid the current function span.
        span = bte._python_find_def_span(lines, function_name=str(name))
        kind = "def"
        if span is None:
            span = _python_find_class_span(lines, class_name=str(name))
            kind = "class"
        if span is None:
            continue
        s, e = span
        if int(lo) <= int(s) <= int(hi):
            continue
        if _line_in_windows(windows, int(s)) or _line_in_windows(windows, int(e)):
            continue

        # Cap very large defs/classes; include start + end slices.
        max_lines = 80
        spans: list[tuple[int, int]] = [(int(s), int(e))]
        if int(e) - int(s) + 1 > int(max_lines):
            head = (int(s), min(int(e), int(s) + 39))
            tail = (max(int(s), int(e) - 19), int(e))
            spans = bte._merge_windows([head, tail])

        blocks = [bte._render_code_block(file_rel, start=ss, end=ee, lines=lines) for ss, ee in spans]
        piece = "\n\n".join([b for b in blocks if b.strip()])
        if not piece:
            continue

        # Greedy budget add: update meta + append blocks if the whole payload still fits.
        cand_related = [*related, {"name": str(name), "kind": kind, "start": int(s), "end": int(e)}]
        meta2 = dict(meta)
        meta2["related_definitions"] = cand_related
        payload2 = json.dumps(meta2, sort_keys=True)
        if code_text:
            payload2 += "\n\n" + code_text
        if extra_blocks:
            payload2 += "\n\n" + "\n\n".join(extra_blocks)
        payload2 += "\n\n" + piece
        if int(count_tokens(payload2)) > int(budget_tokens):
            continue

        related = cand_related
        extra_blocks.append(piece)

    if related:
        meta3 = dict(meta)
        meta3["related_definitions"] = related
        if extra_blocks:
            code_text = (code_text + "\n\n" if code_text else "") + "\n\n".join(extra_blocks)
        return meta3, code_text

    return meta, code_text


def _pack_open_ended_slice_context(
    *,
    file_rel: str,
    lines: list[str],
    function: str,
    span: tuple[int, int] | None,
    target_line: int,
    slice_lines: list[int],
    budget_tokens: int,
) -> tuple[dict[str, Any], str]:
    window_radius = 3
    lo = 1
    hi = len(lines)
    if span is not None:
        lo, hi = span
    lo = max(1, int(lo))
    hi = min(int(hi), len(lines))
    target_line = max(int(lo), min(int(hi), int(target_line)))

    # Ensure target_line always appears in the slice line set.
    slice_all = sorted({int(target_line), *[int(x) for x in slice_lines]})
    slice_all = [int(x) for x in slice_all if int(lo) <= int(x) <= int(hi)]
    by_distance = sorted(slice_all, key=lambda x: (abs(int(x) - int(target_line)), int(x)))

    # A tiny header helps the model keep track of the function signature and any decorators/docstring.
    header_window = (int(lo), min(int(hi), int(lo) + 3))

    # Prefer the largest contiguous target window that still leaves room for at least one extra slice window
    # (when there are slice lines outside the target window).
    best_fallback_meta: dict[str, Any] | None = None
    best_fallback_code = ""
    fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    for frac in fractions:
        base_budget = max(0, int(int(budget_tokens) * float(frac)))
        _, _, base_win = bte._select_window_within_span(
            file_rel=file_rel,
            lines=lines,
            span=(int(lo), int(hi)),
            target_line=int(target_line),
            budget_tokens=int(base_budget),
        )
        # Even the target window didn't fit; try a smaller one.
        if not isinstance(base_win, dict) or "start" not in base_win or "end" not in base_win:
            continue

        fixed_windows = bte._merge_windows(
            [header_window, (int(base_win["start"]), int(base_win["end"]))]  # type: ignore[arg-type]
        )
        extra_windows: list[tuple[int, int]] = []

        def build(extra: list[tuple[int, int]]) -> tuple[dict[str, Any], str, list[tuple[int, int]]]:
            all_windows = bte._merge_windows([*fixed_windows, *extra])
            code_pieces = [bte._render_code_block(file_rel, start=s, end=e, lines=lines) for s, e in all_windows]
            code = "\n\n".join([p for p in code_pieces if p.strip()])
            included_slice = [int(x) for x in slice_all if _line_in_windows(all_windows, int(x))]
            meta = {
                "file": file_rel,
                "function": function,
                "target_line": int(target_line),
                "slice_lines": included_slice,
                "target_window": {"start": int(base_win["start"]), "end": int(base_win["end"])},
                "slice_window_radius": int(window_radius),
                "extra_windows": [{"start": int(s), "end": int(e)} for s, e in extra],
                "strategy": "target_window_plus_slice_windows",
            }
            return meta, code, all_windows

        meta0, code0, _all0 = build(extra_windows)
        payload0 = json.dumps(meta0, sort_keys=True)
        if code0:
            payload0 += "\n\n" + code0
        if int(count_tokens(payload0)) > int(budget_tokens):
            # Shrink the target window (try a smaller fraction).
            continue

        # Greedily add windows around slice-selected lines that are outside the current windows.
        for ln in by_distance:
            _, _, all_cur = build(extra_windows)
            if _line_in_windows(all_cur, int(ln)):
                continue
            w = (max(int(lo), int(ln) - int(window_radius)), min(int(hi), int(ln) + int(window_radius)))
            cand_extra = bte._merge_windows([*extra_windows, w])
            meta2, code2, _all2 = build(cand_extra)
            payload2 = json.dumps(meta2, sort_keys=True)
            if code2:
                payload2 += "\n\n" + code2
            if int(count_tokens(payload2)) > int(budget_tokens):
                continue
            extra_windows = cand_extra

        meta_final, code_final, all_final = build(extra_windows)

        meta_final, code_final = _add_related_definitions(
            file_rel=file_rel,
            lines=lines,
            function_span=(int(lo), int(hi)),
            windows=all_final,
            target_line=int(target_line),
            budget_tokens=int(budget_tokens),
            meta=meta_final,
            code_text=code_final,
        )

        # Record a fallback even if we couldn't fit any extra windows.
        if best_fallback_meta is None:
            best_fallback_meta = meta_final
            best_fallback_code = code_final

        # If there are slice lines outside the fixed windows, prefer a fraction that can include at least
        # one extra window. Otherwise, keep the largest target window possible (earliest fraction).
        needs_extras = any(not _line_in_windows(fixed_windows, int(ln)) for ln in slice_all)
        has_extras = bool(meta_final.get("extra_windows"))
        if not needs_extras or has_extras:
            return meta_final, code_final

    # Worst case: emit a minimal, budget-respecting payload.
    if best_fallback_meta is not None:
        return best_fallback_meta, best_fallback_code

    meta_min = {
        "file": file_rel,
        "function": function,
        "target_line": int(target_line),
        "slice_lines": [int(target_line)],
        "strategy": "target_window_plus_slice_windows",
    }
    return meta_min, ""


def _slice_context_rg(
    *,
    repo_root: Path,
    file_rel: str,
    function: str,
    target_line: int,
    budget_tokens: int,
) -> tuple[str, int, int]:
    abs_path = repo_root / file_rel
    src = abs_path.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    span = bte._python_find_def_span(lines, function_name=function)
    payload, _included, _win = bte._select_window_within_span(
        file_rel=file_rel,
        lines=lines,
        span=span,
        target_line=int(target_line),
        budget_tokens=int(budget_tokens),
    )
    return payload, int(count_tokens(payload)), len(payload.encode("utf-8"))


def _data_flow_context_tldr(
    *,
    repo_root: Path,
    file_rel: str,
    function: str,
    variable: str,
    budget_tokens: int,
) -> tuple[str, int, int]:
    abs_path = repo_root / file_rel
    dfg = get_dfg_context(str(abs_path), function, language="python")

    # Keep variable-specific events compact.
    events: list[dict[str, Any]] = []
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
                events.append({"line": dl, "event": "defined"})
            if isinstance(ul, int):
                events.append({"line": ul, "event": "used"})
    # Dedup (line,event) deterministically.
    seen = set()
    events2: list[dict[str, Any]] = []
    for ev in sorted(events, key=lambda x: (int(x.get("line") or 0), str(x.get("event") or ""))):
        key = (ev.get("line"), ev.get("event"))
        if key in seen:
            continue
        seen.add(key)
        events2.append(ev)

    selected: list[dict[str, Any]] = []
    for ev in events2:
        cand = [*selected, ev]
        payload2 = json.dumps({"flow": cand}, sort_keys=True)
        if int(count_tokens(payload2)) > budget_tokens:
            break
        selected = cand

    payload = json.dumps({"flow": selected}, sort_keys=True)
    return payload, int(count_tokens(payload)), len(payload.encode("utf-8"))


def _data_flow_context_tldr_plus_code(
    *,
    repo_root: Path,
    file_rel: str,
    function: str,
    variable: str,
    budget_tokens: int,
) -> tuple[str, int, int]:
    abs_path = repo_root / file_rel
    src = abs_path.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    span = bte._python_find_def_span(lines, function_name=function)
    if span is not None:
        lo, hi = span
    else:
        lo, hi = (1, len(lines))
    lo = max(1, int(lo))
    hi = min(int(hi), len(lines))

    dfg = get_dfg_context(str(abs_path), function, language="python")

    events: list[dict[str, Any]] = []
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
                events.append({"line": dl, "event": "defined"})
            if isinstance(ul, int):
                events.append({"line": ul, "event": "used"})

    # Dedup (line,event) deterministically.
    seen = set()
    events2: list[dict[str, Any]] = []
    for ev in sorted(events, key=lambda x: (int(x.get("line") or 0), str(x.get("event") or ""))):
        key = (ev.get("line"), ev.get("event"))
        if key in seen:
            continue
        seen.add(key)
        events2.append(ev)

    selected: list[dict[str, Any]] = []
    code_text = ""
    window_radius = 3
    for ev in events2:
        cand = [*selected, ev]
        line_nos = [
            int(x.get("line") or 0)
            for x in cand
            if isinstance(x.get("line"), int) and lo <= int(x.get("line") or 0) <= hi
        ]
        windows = [(int(lo), min(int(hi), int(lo) + 3))]
        windows.extend(
            [
                (max(int(lo), int(x) - int(window_radius)), min(int(hi), int(x) + int(window_radius)))
                for x in line_nos
            ]
        )
        merged = bte._merge_windows(windows)
        code_pieces = [bte._render_code_block(file_rel, start=s, end=e, lines=lines) for s, e in merged]
        code2 = "\n\n".join([p for p in code_pieces if p.strip()])
        payload2 = json.dumps(
            {
                "file": file_rel,
                "function": function,
                "variable": variable,
                "flow": cand,
                "code_window_radius": int(window_radius),
            },
            sort_keys=True,
        )
        if code2:
            payload2 += "\n\n" + code2
        if int(count_tokens(payload2)) > budget_tokens:
            continue
        selected = cand
        code_text = code2

    payload = json.dumps(
        {
            "file": file_rel,
            "function": function,
            "variable": variable,
            "flow": selected,
            "code_window_radius": int(window_radius),
        },
        sort_keys=True,
    )
    if code_text:
        payload += "\n\n" + code_text
    return payload, int(count_tokens(payload)), len(payload.encode("utf-8"))


def _data_flow_context_rg(
    *,
    repo_root: Path,
    file_rel: str,
    function: str,
    variable: str,
    budget_tokens: int,
) -> tuple[str, int, int]:
    abs_path = repo_root / file_rel
    src = abs_path.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    span = bte._python_find_def_span(lines, function_name=function)
    if span is not None:
        lo, hi = span
    else:
        lo, hi = (1, len(lines))

    var_re = re.compile(rf"\b{re.escape(variable)}\b")
    hit_lines = [ln for ln in range(lo, hi + 1) if var_re.search(lines[ln - 1])]

    windows = [(max(lo, ln - 3), min(hi, ln + 3)) for ln in hit_lines]
    merged = bte._merge_windows(windows)
    pieces = [bte._render_code_block(file_rel, start=s, end=e, lines=lines) for s, e in merged]
    payload, ptok, pbytes, _used = bte._apply_budget(pieces, int(budget_tokens))
    return payload, int(ptok), int(pbytes)


def _retrieval_context_from_ranked_files(
    *,
    repo_root: Path,
    ranked_files: list[str],
    rg_pattern: str,
    budget_tokens: int,
    before: int,
    after: int,
) -> tuple[str, int, int]:
    pieces: list[str] = []
    for fp in ranked_files:
        snippet, _ = bte._render_snippet_for_file(
            repo_root,
            file_rel=fp,
            rg_pattern=rg_pattern,
            before=int(before),
            after=int(after),
        )
        pieces.append(snippet)
    payload, ptok, pbytes, _used = bte._apply_budget(pieces, int(budget_tokens))
    return payload, int(ptok), int(pbytes)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 7: generate randomized LLM prompt packets with multiple context variants (rg/TLDR/retrieval)."
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", default=None, help="Corpus id from benchmarks/corpora.json (e.g. django).")
    group.add_argument("--repo-root", default=None, help="Path to corpus repo root.")
    ap.add_argument(
        "--tasks",
        default=str(get_repo_root() / "benchmarks" / "llm" / "tasks.json"),
        help="Tasks JSON (default: benchmarks/llm/tasks.json).",
    )
    ap.add_argument(
        "--structural-queries",
        default=str(get_repo_root() / "benchmarks" / "python" / "django_structural_queries.json"),
        help="Structural query set JSON (default: benchmarks/python/django_structural_queries.json).",
    )
    ap.add_argument(
        "--retrieval-queries",
        default=str(get_repo_root() / "benchmarks" / "retrieval" / "django_queries.json"),
        help="Retrieval query set JSON (default: benchmarks/retrieval/django_queries.json).",
    )
    ap.add_argument(
        "--budget-tokens",
        type=int,
        default=2000,
        help="Context token budget per variant (default: 2000).",
    )
    ap.add_argument(
        "--retrieval-rg-glob",
        default="*.py",
        help="ripgrep --glob filter for retrieval ranking (default: *.py).",
    )
    ap.add_argument("--retrieval-max-files", type=int, default=50, help="Max files per retrieval variant (default: 50).")
    ap.add_argument("--seed", type=int, default=0, help="Deterministic seed for per-task A/B label shuffling.")
    ap.add_argument(
        "--cache-root",
        default=str(bench_cache_root(get_repo_root())),
        help="Index-mode cache root (default: benchmark/cache-root).",
    )
    ap.add_argument("--index", default=None, help="Index id (default: repo:<corpus>).")
    ap.add_argument("--out", default=None, help="Write JSON report to this path (default under benchmark/runs/).")
    ap.add_argument(
        "--prompts-out",
        default=None,
        help="Write a JSONL prompts file to this path (default: benchmark/llm/<ts>-llm-ab-<corpus>.jsonl).",
    )
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

    tasks_path = Path(args.tasks).resolve()
    structural_path = Path(args.structural_queries).resolve()
    tasks = _load_tasks(tasks_path)
    structural_queries = bte._load_structural_queries(structural_path)
    by_id = _index_by_id(structural_queries)

    retrieval_by_id: dict[str, bte.RetrievalQuery] = {}
    retrieval_path = Path(args.retrieval_queries).resolve()
    if any(t.get("category") == "retrieval" for t in tasks):
        retrieval_queries = bte._load_retrieval_queries(retrieval_path)
        retrieval_by_id = {q.id: q for q in retrieval_queries}

    index_id = args.index or default_index_id
    index_ctx = get_index_context(
        scan_root=repo_root,
        cache_root_arg=args.cache_root,
        index_id_arg=index_id,
        allow_create=True,
    )

    ignore_spec = IgnoreSpec(
        project_dir=repo_root,
        use_gitignore=bool(index_ctx.config.use_gitignore) if index_ctx.config else True,
        cli_patterns=list(index_ctx.config.cli_patterns or ()) if index_ctx.config else None,
        ignore_file=index_ctx.config.ignore_file if index_ctx.config else None,
        gitignore_root=index_ctx.config.gitignore_root if index_ctx.config else None,
    )

    call_graph = None
    if any(t.get("category") == "impact" for t in tasks):
        call_graph, _build_s, _cache = bte._load_or_build_call_graph(
            repo_root=repo_root,
            index_ctx=index_ctx,
            language="python",
            ignore_spec=ignore_spec,
        )

    prompt_dir = bench_root(tldr_repo_root) / "llm"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    ts = now_utc_compact()
    if args.prompts_out:
        prompts_path = Path(args.prompts_out).resolve()
    else:
        prompts_path = (prompt_dir / f"{ts}-llm-ab-{corpus_id}.jsonl").resolve()

    records: list[dict[str, Any]] = []
    tok_by_source: dict[str, list[int]] = {}

    semantic_available = (
        index_ctx.paths is not None
        and index_ctx.paths.semantic_faiss.exists()
        and index_ctx.paths.semantic_metadata.exists()
    )

    for t in tasks:
        tid = t.get("id")
        task_type = t.get("task_type") or "structured"
        category = t.get("category")
        qid = t.get("query_id")
        question = t.get("question")
        rubric = t.get("rubric") if task_type == "open_ended" else None
        if not all(isinstance(x, str) for x in (tid, task_type, category, qid, question)):
            continue
        if task_type == "open_ended" and not (isinstance(rubric, str) and rubric.strip()):
            continue
        q = by_id.get(qid)
        if category != "retrieval" and q is None:
            continue

        budget = int(args.budget_tokens)
        variants: list[dict[str, Any]] = []
        record_expected: dict[str, Any] | None = None

        if category == "impact":
            if call_graph is None:
                continue
            func = q.get("function")
            file_filter = q.get("file")
            expected_callers = q.get("expected_callers", [])
            if not isinstance(func, str) or not isinstance(file_filter, str) or not isinstance(expected_callers, list):
                continue
            expected = sorted(
                [
                    {"file": str(c.get("file")), "function": str(c.get("function"))}
                    for c in expected_callers
                    if isinstance(c, dict) and isinstance(c.get("file"), str) and isinstance(c.get("function"), str)
                ],
                key=lambda x: (x["file"], x["function"]),
            )
            schema_hint = "{\"callers\": [{\"file\": \"path\", \"function\": \"name\"}, ...]}"

            if task_type == "open_ended":
                ctx_tldr, t_tok, t_bytes = _impact_context_tldr_plus_code(
                    repo_root=repo_root,
                    call_graph=call_graph,
                    callee_func=func,
                    callee_file=file_filter,
                    budget_tokens=budget,
                )
            else:
                ctx_tldr, t_tok, t_bytes = _impact_context_tldr(
                    call_graph=call_graph,
                    callee_func=func,
                    callee_file=file_filter,
                    budget_tokens=budget,
                )
            ctx_rg, r_tok, r_bytes = _impact_context_rg(
                repo_root=repo_root,
                callee_func=func,
                budget_tokens=budget,
            )
            variants = [
                {
                    "source": "tldr",
                    "context": ctx_tldr,
                    "context_tokens": t_tok,
                    "context_bytes": t_bytes,
                    "prompt": _prompt_for_open_ended_task(question=question, context=ctx_tldr)
                    if task_type == "open_ended"
                    else _prompt_for_task(question=question, context=ctx_tldr, schema_hint=schema_hint),
                },
                {
                    "source": "rg",
                    "context": ctx_rg,
                    "context_tokens": r_tok,
                    "context_bytes": r_bytes,
                    "prompt": _prompt_for_open_ended_task(question=question, context=ctx_rg)
                    if task_type == "open_ended"
                    else _prompt_for_task(question=question, context=ctx_rg, schema_hint=schema_hint),
                },
            ]

            record_expected = {"callers": expected} if task_type != "open_ended" else None

        elif category == "slice":
            file_rel = q.get("file")
            function = q.get("function")
            target_line = q.get("target_line")
            expected_lines = q.get("expected_slice_lines", [])
            if not isinstance(file_rel, str) or not isinstance(function, str) or not isinstance(target_line, int) or not isinstance(expected_lines, list):
                continue
            expected = sorted([int(x) for x in expected_lines if isinstance(x, int)])
            schema_hint = "{\"lines\": [int, ...]}"

            if task_type == "open_ended":
                ctx_tldr, t_tok, t_bytes = _slice_context_tldr_plus_code(
                    repo_root=repo_root,
                    file_rel=file_rel,
                    function=function,
                    target_line=int(target_line),
                    budget_tokens=budget,
                )
            else:
                ctx_tldr, t_tok, t_bytes = _slice_context_tldr(
                    repo_root=repo_root,
                    file_rel=file_rel,
                    function=function,
                    target_line=int(target_line),
                    budget_tokens=budget,
                )
            ctx_rg, r_tok, r_bytes = _slice_context_rg(
                repo_root=repo_root,
                file_rel=file_rel,
                function=function,
                target_line=int(target_line),
                budget_tokens=budget,
            )
            variants = [
                {
                    "source": "tldr",
                    "context": ctx_tldr,
                    "context_tokens": t_tok,
                    "context_bytes": t_bytes,
                    "prompt": _prompt_for_open_ended_task(question=question, context=ctx_tldr)
                    if task_type == "open_ended"
                    else _prompt_for_task(question=question, context=ctx_tldr, schema_hint=schema_hint),
                },
                {
                    "source": "rg",
                    "context": ctx_rg,
                    "context_tokens": r_tok,
                    "context_bytes": r_bytes,
                    "prompt": _prompt_for_open_ended_task(question=question, context=ctx_rg)
                    if task_type == "open_ended"
                    else _prompt_for_task(question=question, context=ctx_rg, schema_hint=schema_hint),
                },
            ]
            record_expected = {"lines": expected} if task_type != "open_ended" else None

        elif category == "data_flow":
            file_rel = q.get("file")
            function = q.get("function")
            variable = q.get("variable")
            expected_flow = q.get("expected_flow", [])
            if not isinstance(file_rel, str) or not isinstance(function, str) or not isinstance(variable, str) or not isinstance(expected_flow, list):
                continue
            expected = sorted(
                [
                    {"line": int(ev.get("line")), "event": str(ev.get("event"))}
                    for ev in expected_flow
                    if isinstance(ev, dict) and isinstance(ev.get("line"), int) and isinstance(ev.get("event"), str)
                ],
                key=lambda x: (x["line"], x["event"]),
            )
            schema_hint = "{\"flow\": [{\"line\": int, \"event\": \"defined\"|\"used\"}, ...]}"

            if task_type == "open_ended":
                ctx_tldr, t_tok, t_bytes = _data_flow_context_tldr_plus_code(
                    repo_root=repo_root,
                    file_rel=file_rel,
                    function=function,
                    variable=variable,
                    budget_tokens=budget,
                )
            else:
                ctx_tldr, t_tok, t_bytes = _data_flow_context_tldr(
                    repo_root=repo_root,
                    file_rel=file_rel,
                    function=function,
                    variable=variable,
                    budget_tokens=budget,
                )
            ctx_rg, r_tok, r_bytes = _data_flow_context_rg(
                repo_root=repo_root,
                file_rel=file_rel,
                function=function,
                variable=variable,
                budget_tokens=budget,
            )
            variants = [
                {
                    "source": "tldr",
                    "context": ctx_tldr,
                    "context_tokens": t_tok,
                    "context_bytes": t_bytes,
                    "prompt": _prompt_for_open_ended_task(question=question, context=ctx_tldr)
                    if task_type == "open_ended"
                    else _prompt_for_task(question=question, context=ctx_tldr, schema_hint=schema_hint),
                },
                {
                    "source": "rg",
                    "context": ctx_rg,
                    "context_tokens": r_tok,
                    "context_bytes": r_bytes,
                    "prompt": _prompt_for_open_ended_task(question=question, context=ctx_rg)
                    if task_type == "open_ended"
                    else _prompt_for_task(question=question, context=ctx_rg, schema_hint=schema_hint),
                },
            ]
            record_expected = {"flow": expected} if task_type != "open_ended" else None

        elif category == "retrieval":
            if task_type == "open_ended":
                # Judge-mode currently assumes 2 variants (A/B). Keep retrieval tasks deterministic for now.
                continue
            rq = retrieval_by_id.get(qid)
            if rq is None:
                continue

            rg_pattern = str(rq.rg_pattern or re.escape(rq.query))
            expected_paths = sorted({p.replace("\\", "/").lstrip("./") for p in rq.relevant_files})
            schema_hint = "{\"paths\": [\"path\", ...]}"

            max_files = max(1, int(args.retrieval_max_files))
            glob_arg = str(args.retrieval_rg_glob) if args.retrieval_rg_glob else None

            rg_rank = bte._rg_rank_files(repo_root, pattern=rg_pattern, glob=glob_arg)[:max_files]
            ctx_rg, r_tok, r_bytes = _retrieval_context_from_ranked_files(
                repo_root=repo_root,
                ranked_files=rg_rank,
                rg_pattern=rg_pattern,
                budget_tokens=budget,
                before=2,
                after=2,
            )
            variants = [
                {
                    "source": "rg",
                    "context": ctx_rg,
                    "context_tokens": r_tok,
                    "context_bytes": r_bytes,
                    "prompt": _prompt_for_task(question=question, context=ctx_rg, schema_hint=schema_hint),
                }
            ]

            if semantic_available:
                sem_rank = bte._semantic_rank_files(repo_root, index_ctx=index_ctx, query=rq.query, k=max_files) or []
                sem_rank = sem_rank[:max_files]
                ctx_sem, s_tok, s_bytes = _retrieval_context_from_ranked_files(
                    repo_root=repo_root,
                    ranked_files=sem_rank,
                    rg_pattern=rg_pattern,
                    budget_tokens=budget,
                    before=2,
                    after=2,
                )
                variants.append(
                    {
                        "source": "semantic",
                        "context": ctx_sem,
                        "context_tokens": s_tok,
                        "context_bytes": s_bytes,
                        "prompt": _prompt_for_task(question=question, context=ctx_sem, schema_hint=schema_hint),
                    }
                )

                hybrid_rank = bte._rrf_fuse([rg_rank, sem_rank])[:max_files]
                ctx_h, h_tok, h_bytes = _retrieval_context_from_ranked_files(
                    repo_root=repo_root,
                    ranked_files=hybrid_rank,
                    rg_pattern=rg_pattern,
                    budget_tokens=budget,
                    before=2,
                    after=2,
                )
                variants.append(
                    {
                        "source": "hybrid_rrf",
                        "context": ctx_h,
                        "context_tokens": h_tok,
                        "context_bytes": h_bytes,
                        "prompt": _prompt_for_task(question=question, context=ctx_h, schema_hint=schema_hint),
                    }
                )

            record_expected = {"paths": expected_paths}

        else:
            continue

        variants = _stable_shuffle(int(args.seed), str(tid), variants)
        _label_variants(variants)

        for v in variants:
            src = str(v.get("source"))
            tok_by_source.setdefault(src, []).append(int(v.get("context_tokens") or 0))

        rec: dict[str, Any] = {
            "task_id": tid,
            "task_type": task_type,
            "category": category,
            "query_id": qid,
            "question": question,
            "budget_tokens": budget,
            "variants": variants,
        }
        if rubric is not None:
            rec["rubric"] = rubric
        if record_expected is not None:
            rec["expected"] = record_expected
        records.append(rec)

    # Write JSONL prompts (untracked).
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    with prompts_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    report = make_report(
        phase="phase7_llm_ab_prompts",
        meta=gather_meta(tldr_repo_root=tldr_repo_root, corpus_id=corpus_id, corpus_root=repo_root),
        protocol={
            "schema_version": SCHEMA_VERSION,
            "tasks": str(tasks_path),
            "structural_queries": str(structural_path),
            "retrieval_queries": str(retrieval_path),
            "budget_tokens": int(args.budget_tokens),
            "seed": int(args.seed),
            "retrieval_rg_glob": str(args.retrieval_rg_glob) if args.retrieval_rg_glob else None,
            "retrieval_max_files": int(args.retrieval_max_files),
            "semantic_available": bool(semantic_available),
            "cache_root": str(index_ctx.cache_root) if index_ctx.cache_root is not None else None,
            "index_id": index_ctx.index_id,
            "prompts_path": str(prompts_path),
        },
        results={
            "tasks_total": len(records),
            "tokens_context_mean": {k: (sum(v) / len(v) if v else None) for k, v in tok_by_source.items()},
        },
    )

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = bench_runs_root(tldr_repo_root) / f"{ts}-llm-ab-prompts-{corpus_id}.json"
    write_report(out_path, report)
    print(out_path)
    print(prompts_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
