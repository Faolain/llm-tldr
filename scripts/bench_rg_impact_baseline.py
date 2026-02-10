#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from bench_util import (
    bench_runs_root,
    gather_meta,
    get_repo_root,
    make_report,
    now_utc_compact,
    write_report,
)

from tldr.stats import count_tokens


@dataclass(frozen=True)
class CuratedEdge:
    caller_file: str
    caller_symbol: str
    callee_file: str
    callee_symbol: str
    rg_pattern: str | None = None


@dataclass(frozen=True)
class RgMatch:
    file: str
    line: int
    text: str


def _load_curated_edges(path: Path) -> tuple[str | None, list[CuratedEdge]]:
    data: Any = json.loads(path.read_text())
    corpus_id = None
    edges_raw = None
    if isinstance(data, dict):
        corpus_id = data.get("repo") if isinstance(data.get("repo"), str) else None
        edges_raw = data.get("edges")
    elif isinstance(data, list):
        edges_raw = data
    else:
        raise ValueError(f"Unsupported curated format in {path}")

    if not isinstance(edges_raw, list):
        raise ValueError(f"Unsupported curated format in {path} (missing edges list)")

    out: list[CuratedEdge] = []
    for e in edges_raw:
        try:
            caller = e["caller"]
            callee = e["callee"]
            rg_pattern = None
            if isinstance(e.get("rg_pattern"), str):
                rg_pattern = e["rg_pattern"]
            elif isinstance(callee, dict) and isinstance(callee.get("rg_pattern"), str):
                rg_pattern = callee["rg_pattern"]
            out.append(
                CuratedEdge(
                    caller_file=str(caller["file"]),
                    caller_symbol=str(caller["symbol"]),
                    callee_file=str(callee["file"]),
                    callee_symbol=str(callee["symbol"]),
                    rg_pattern=rg_pattern,
                )
            )
        except Exception as exc:
            raise ValueError(f"Bad curated edge entry: {e!r}") from exc
    return corpus_id, out


def _derive_rg_pattern(symbol: str) -> str:
    # For Class.method, match `.method(`.
    if "." in symbol:
        cls, member = symbol.rsplit(".", 1)
        if member == "constructor":
            # Best-effort TS constructor call: `new Class<...>(` or `new Class(`.
            cls_esc = re.escape(cls)
            return rf"\bnew\s+{cls_esc}(?:\s*<[^>\n]+>)?\s*\("
        member_esc = re.escape(member)
        return rf"\.{member_esc}\s*\("
    # For functionName, match `functionName(`.
    sym_esc = re.escape(symbol)
    return rf"\b{sym_esc}\s*\("


def _run_rg(repo_root: Path, *, pattern: str) -> list[RgMatch]:
    cmd = [
        "rg",
        "--no-heading",
        "--color",
        "never",
        "--line-number",
        "--sort",
        "path",
        "--glob",
        "*.ts",
        "--glob",
        "*.tsx",
        "--regexp",
        pattern,
        ".",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    # rg exit codes:
    # - 0: matches found
    # - 1: no matches
    # - 2: error
    if proc.returncode == 2:
        raise RuntimeError(f"rg failed: {' '.join(cmd)}\n{proc.stderr}")
    if proc.returncode == 1:
        return []

    out: list[RgMatch] = []
    for line in (proc.stdout or "").splitlines():
        # Format: path:line:match_text
        m = re.match(r"^(?P<file>[^:]+):(?P<line>\d+):(?P<text>.*)$", line)
        if not m:
            continue
        file = m.group("file").replace("\\", "/")
        try:
            line_no = int(m.group("line"))
        except ValueError:
            continue
        out.append(RgMatch(file=file, line=line_no, text=m.group("text")))

    # Defensive determinism: sort again.
    out.sort(key=lambda r: (r.file, r.line, r.text))
    return out


def _filter_definition_hits(matches: list[RgMatch], *, callee_symbol: str) -> list[RgMatch]:
    # For plain functions, the pattern will also match the definition line:
    #   export function foo(
    #   function foo(
    # Remove those trivial false positives.
    if "." in callee_symbol:
        return matches
    name = callee_symbol
    def_re = re.compile(
        rf"\b(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+{re.escape(name)}\s*\("
    )
    out: list[RgMatch] = []
    for m in matches:
        if def_re.search(m.text):
            continue
        out.append(m)
    return out


def _read_lines(repo_root: Path, rel_file: str) -> list[str] | None:
    path = repo_root / rel_file
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


def _guess_enclosing_symbol(
    lines: list[str],
    *,
    line_no_1: int,
    scan_back: int = 250,
) -> str | None:
    # Cheap heuristic:
    # - scan backwards for the nearest function/arrow binding or method, while tracking the last class name.
    # - avoids parsing; this is intentionally imperfect.
    class_re = re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z0-9_]+)\b"
    )
    fn_re = re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)\s*\("
    )
    arrow_re = re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\(",
    )
    # Method signatures can be pretty varied; keep it conservative.
    method_re = re.compile(
        r"^\s*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:async\s+)?([A-Za-z0-9_]+)\s*\(",
    )
    bad_method_prefixes = ("if", "for", "while", "switch", "catch", "with")

    cur_class: str | None = None
    pending_method: str | None = None
    start = min(max(line_no_1 - 1, 0), len(lines) - 1)
    end = max(start - scan_back, 0)
    for idx in range(start, end - 1, -1):
        line = lines[idx]
        m = class_re.match(line)
        if m:
            cur_class = m.group(1)
            if pending_method:
                if pending_method == "constructor":
                    return f"{cur_class}.constructor"
                return f"{cur_class}.{pending_method}"
            continue
        m = fn_re.match(line)
        if m:
            return m.group(1)
        m = arrow_re.match(line)
        if m:
            return m.group(1)

        stripped = line.lstrip()
        head = stripped.split("(", 1)[0].strip()
        if head in bad_method_prefixes:
            continue
        if "=" in stripped:
            continue

        m = method_re.match(line)
        if not m:
            continue
        name = m.group(1)

        if cur_class:
            if name == "constructor":
                return f"{cur_class}.constructor"
            return f"{cur_class}.{name}"

        # When scanning backwards, we may see a method signature before we see
        # the class declaration. Record the first plausible method name and
        # attach it once we see a class above it.
        if pending_method is None:
            starts_like_method = stripped.startswith(
                ("public ", "private ", "protected ", "static ", "async ")
            )
            looks_like_sig = stripped.rstrip().endswith("{") or ":" in stripped
            if starts_like_method or looks_like_sig:
                pending_method = name
    return None


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


def _payload_for_strategy(
    *,
    strategy: str,
    repo_root: Path,
    matches: list[RgMatch],
    before: int,
    after: int,
) -> tuple[list[str], dict[str, Any]]:
    """Return (pieces, aux) where aux contains any strategy-specific info."""
    if strategy in ("match_only", "match_plus_enclosing_symbol"):
        aux: dict[str, Any] = {}
        if strategy == "match_plus_enclosing_symbol":
            guessed: list[tuple[str, int, str | None]] = []
            # Cache file reads per target to avoid O(m) file opens.
            file_cache: dict[str, list[str] | None] = {}
            for m in matches:
                lines = file_cache.get(m.file)
                if lines is None and m.file not in file_cache:
                    lines = _read_lines(repo_root, m.file)
                    file_cache[m.file] = lines
                sym = None
                if lines:
                    sym = _guess_enclosing_symbol(lines, line_no_1=m.line)
                guessed.append((m.file, m.line, sym))
            aux["guessed_callers"] = guessed

            pieces = [
                f"{m.file}:{m.line}:{(sym or '<unknown>')}:{m.text}"
                for m, (_, _, sym) in zip(matches, guessed, strict=False)
            ]
            return pieces, aux

        pieces = [f"{m.file}:{m.line}:{m.text}" for m in matches]
        return pieces, aux

    if strategy == "match_plus_context":
        file_lines: dict[str, list[str] | None] = {}
        windows_by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for m in matches:
            start = max(m.line - before, 1)
            end = m.line + after
            windows_by_file[m.file].append((start, end))

        pieces: list[str] = []
        for file in sorted(windows_by_file.keys()):
            lines = file_lines.get(file)
            if lines is None and file not in file_lines:
                lines = _read_lines(repo_root, file)
                file_lines[file] = lines
            if not lines:
                continue
            merged = _merge_windows(windows_by_file[file])
            for start, end in merged:
                end = min(end, len(lines))
                header = f"# {file}:{start}-{end}"
                body = "\n".join(f"{ln}: {lines[ln - 1].rstrip()}" for ln in range(start, end + 1))
                pieces.append(f"{header}\n{body}")
        return pieces, {}

    raise ValueError(f"Unknown strategy: {strategy}")


def _apply_budget(pieces: list[str], budget_tokens: int) -> tuple[str, int, int, int]:
    """Greedy deterministic prefix selection by tokens. Returns payload + counts + pieces_used."""
    payload_parts: list[str] = []
    used = 0
    for piece in pieces:
        candidate = "\n\n".join([*payload_parts, piece]) if payload_parts else piece
        toks = count_tokens(candidate)
        if toks > budget_tokens:
            break
        payload_parts.append(piece)
        used += 1
    payload = "\n\n".join(payload_parts)
    payload_bytes = len(payload.encode("utf-8"))
    payload_tokens = count_tokens(payload)
    return payload, payload_tokens, payload_bytes, used


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic rg baselines for 'what calls X?'")
    ap.add_argument("--repo-root", required=True, help="Path to the repo root to search with rg.")
    ap.add_argument("--curated", required=True, help="Path to curated edges JSON.")
    ap.add_argument(
        "--strategy",
        choices=["match_only", "match_plus_context", "match_plus_enclosing_symbol"],
        default="match_only",
    )
    ap.add_argument(
        "--budgets",
        default="500,1000,2000,5000,10000",
        help="Comma-separated token budgets for payload materialization.",
    )
    ap.add_argument("--before", type=int, default=3, help="Context lines before each hit (match_plus_context).")
    ap.add_argument("--after", type=int, default=3, help="Context lines after each hit (match_plus_context).")
    ap.add_argument(
        "--out",
        default=None,
        help="Write a JSON report to this path (default: benchmark/runs/<ts>-rg-impact-baseline-<corpus>.json).",
    )
    args = ap.parse_args()

    tldr_repo_root = get_repo_root()
    repo_root = Path(args.repo_root).resolve()
    curated_path = Path(args.curated).resolve()

    budgets: list[int] = []
    for part in str(args.budgets).split(","):
        part = part.strip()
        if not part:
            continue
        budgets.append(int(part))
    budgets = sorted(set(budgets))
    if not budgets:
        raise SystemExit("error: no budgets provided")

    curated_corpus_id, curated_edges = _load_curated_edges(curated_path)
    corpus_id = curated_corpus_id or repo_root.name

    expected_by_target: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    rg_pattern_by_target: dict[tuple[str, str], str] = {}
    for e in curated_edges:
        key = (e.callee_file, e.callee_symbol)
        expected_by_target[key].add((e.caller_file, e.caller_symbol))
        if e.rg_pattern:
            prev = rg_pattern_by_target.get(key)
            if prev is not None and prev != e.rg_pattern:
                raise SystemExit(f"error: conflicting rg_pattern for {key}: {prev!r} vs {e.rg_pattern!r}")
            rg_pattern_by_target[key] = e.rg_pattern

    protocol = {
        "strategy": args.strategy,
        "budgets": budgets,
        "before": int(args.before),
        "after": int(args.after),
        "curated": str(curated_path),
        "language": "typescript",
        "notes": [
            "This is a deterministic baseline that does not use AST, tsserver, or TLDR call graphs.",
            "Scoring compares rg-derived caller sets to curated expected callers.",
        ],
    }

    # Per-budget totals (micro-average across targets)
    totals_by_budget: dict[int, dict[str, int]] = {b: {"tp": 0, "fp": 0, "fn": 0} for b in budgets}
    payload_totals_by_budget: dict[int, dict[str, float]] = {
        b: {"payload_tokens_sum": 0.0, "payload_bytes_sum": 0.0, "targets": 0.0}
        for b in budgets
    }

    per_target: list[dict[str, Any]] = []

    for (callee_file, callee_symbol), expected_callers in sorted(expected_by_target.items()):
        rg_pattern = rg_pattern_by_target.get((callee_file, callee_symbol)) or _derive_rg_pattern(callee_symbol)
        matches = _run_rg(repo_root, pattern=rg_pattern)
        matches = _filter_definition_hits(matches, callee_symbol=callee_symbol)

        pieces, aux = _payload_for_strategy(
            strategy=args.strategy,
            repo_root=repo_root,
            matches=matches,
            before=int(args.before),
            after=int(args.after),
        )

        expected_files = {cf for (cf, _) in expected_callers}

        target_entry: dict[str, Any] = {
            "target": f"{callee_file}:{callee_symbol}",
            "callee_file": callee_file,
            "callee_symbol": callee_symbol,
            "rg_pattern": rg_pattern,
            "expected_callers": len(expected_callers),
            "expected_files": len(expected_files),
            "matches_total": len(matches),
            "pieces_total": len(pieces),
            "budgets": [],
        }

        guessed_callers: list[tuple[str, int, str | None]] | None = None
        if args.strategy == "match_plus_enclosing_symbol":
            raw = aux.get("guessed_callers")
            if isinstance(raw, list):
                guessed_callers = raw

        for budget in budgets:
            payload, payload_tokens, payload_bytes, pieces_used = _apply_budget(pieces, budget)

            # Determine found set under this budget.
            found_files: set[str] = set()
            found_callers: set[tuple[str, str]] = set()

            if args.strategy == "match_plus_context":
                # For context pieces, each piece starts with "# file:start-end".
                for piece in pieces[:pieces_used]:
                    if piece.startswith("# "):
                        header = piece.splitlines()[0]
                        hdr = header[2:].strip()
                        file = hdr.split(":", 1)[0]
                        found_files.add(file)
            else:
                # match lines carry file prefix
                for piece in pieces[:pieces_used]:
                    file = piece.split(":", 1)[0]
                    found_files.add(file)

            if args.strategy == "match_plus_enclosing_symbol" and guessed_callers is not None:
                for file, line_no, sym in guessed_callers[:pieces_used]:
                    if not sym:
                        continue
                    found_callers.add((file, sym))

            # Score.
            if args.strategy == "match_plus_enclosing_symbol":
                tp = len(expected_callers & found_callers)
                fp = len(found_callers - expected_callers)
                fn = len(expected_callers - found_callers)
            else:
                tp = len(expected_files & found_files)
                fp = len(found_files - expected_files)
                fn = len(expected_files - found_files)

            totals_by_budget[budget]["tp"] += tp
            totals_by_budget[budget]["fp"] += fp
            totals_by_budget[budget]["fn"] += fn
            payload_totals_by_budget[budget]["payload_tokens_sum"] += payload_tokens
            payload_totals_by_budget[budget]["payload_bytes_sum"] += payload_bytes
            payload_totals_by_budget[budget]["targets"] += 1

            entry = {
                "budget_tokens": budget,
                "payload_tokens": payload_tokens,
                "payload_bytes": payload_bytes,
                "pieces_used": pieces_used,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "metrics": _prf(tp, fp, fn),
            }
            # Keep the payload itself out of the report (it can be huge).
            target_entry["budgets"].append(entry)

        per_target.append(target_entry)

    summary: list[dict[str, Any]] = []
    for budget in budgets:
        tp = totals_by_budget[budget]["tp"]
        fp = totals_by_budget[budget]["fp"]
        fn = totals_by_budget[budget]["fn"]
        metrics = _prf(tp, fp, fn)
        tot = payload_totals_by_budget[budget]
        targets = int(tot["targets"])
        summary.append(
            {
                "budget_tokens": budget,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "metrics": metrics,
                "payload_tokens_sum": int(tot["payload_tokens_sum"]),
                "payload_bytes_sum": int(tot["payload_bytes_sum"]),
                "payload_tokens_mean": round((tot["payload_tokens_sum"] / targets) if targets else 0.0, 2),
                "payload_bytes_mean": round((tot["payload_bytes_sum"] / targets) if targets else 0.0, 2),
                "targets": targets,
            }
        )

    report = make_report(
        phase="phase2_rg_impact_baseline",
        meta=gather_meta(
            tldr_repo_root=tldr_repo_root,
            corpus_id=corpus_id,
            corpus_root=repo_root,
        ),
        protocol=protocol,
        results={
            "summary": summary,
            "targets": per_target,
        },
    )

    if args.out:
        out_path = Path(args.out)
    else:
        ts = now_utc_compact()
        out_path = bench_runs_root(tldr_repo_root) / f"{ts}-rg-impact-baseline-{corpus_id}-{args.strategy}.json"
    write_report(out_path, report)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
