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


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 7: generate randomized A/B LLM prompts (rg vs TLDR).")
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
        "--budget-tokens",
        type=int,
        default=2000,
        help="Context token budget per variant (default: 2000).",
    )
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
    tok_by_source: dict[str, list[int]] = {"rg": [], "tldr": []}

    for t in tasks:
        tid = t.get("id")
        category = t.get("category")
        qid = t.get("query_id")
        question = t.get("question")
        if not all(isinstance(x, str) for x in (tid, category, qid, question)):
            continue
        q = by_id.get(qid)
        if q is None:
            continue

        budget = int(args.budget_tokens)
        variants: list[dict[str, Any]] = []

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
                    "prompt": _prompt_for_task(question=question, context=ctx_tldr, schema_hint=schema_hint),
                },
                {
                    "source": "rg",
                    "context": ctx_rg,
                    "context_tokens": r_tok,
                    "context_bytes": r_bytes,
                    "prompt": _prompt_for_task(question=question, context=ctx_rg, schema_hint=schema_hint),
                },
            ]

            record_expected = {"callers": expected}

        elif category == "slice":
            file_rel = q.get("file")
            function = q.get("function")
            target_line = q.get("target_line")
            expected_lines = q.get("expected_slice_lines", [])
            if not isinstance(file_rel, str) or not isinstance(function, str) or not isinstance(target_line, int) or not isinstance(expected_lines, list):
                continue
            expected = sorted([int(x) for x in expected_lines if isinstance(x, int)])
            schema_hint = "{\"lines\": [int, ...]}"

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
                    "prompt": _prompt_for_task(question=question, context=ctx_tldr, schema_hint=schema_hint),
                },
                {
                    "source": "rg",
                    "context": ctx_rg,
                    "context_tokens": r_tok,
                    "context_bytes": r_bytes,
                    "prompt": _prompt_for_task(question=question, context=ctx_rg, schema_hint=schema_hint),
                },
            ]
            record_expected = {"lines": expected}

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
                    "prompt": _prompt_for_task(question=question, context=ctx_tldr, schema_hint=schema_hint),
                },
                {
                    "source": "rg",
                    "context": ctx_rg,
                    "context_tokens": r_tok,
                    "context_bytes": r_bytes,
                    "prompt": _prompt_for_task(question=question, context=ctx_rg, schema_hint=schema_hint),
                },
            ]
            record_expected = {"flow": expected}

        else:
            continue

        variants = _stable_shuffle(int(args.seed), str(tid), variants)
        variants[0]["label"] = "A"
        variants[1]["label"] = "B"

        for v in variants:
            tok_by_source[v["source"]].append(int(v["context_tokens"]))

        records.append(
            {
                "task_id": tid,
                "category": category,
                "query_id": qid,
                "question": question,
                "budget_tokens": budget,
                "expected": record_expected,
                "variants": variants,
            }
        )

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
            "budget_tokens": int(args.budget_tokens),
            "seed": int(args.seed),
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
