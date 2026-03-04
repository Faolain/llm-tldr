#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _normalize_path(value: str, repo_root: Path) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if raw.startswith("./"):
        raw = raw[2:]
    if not raw:
        return raw

    p = Path(raw)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
        except Exception:
            return str(p).replace("\\", "/")
    return str(p).replace("\\", "/")


def _ranked_files_from_rg_json(stdout: str, repo_root: Path, top_k: int) -> list[str]:
    counts: dict[str, int] = {}
    first_line: dict[str, int] = {}

    for raw_line in str(stdout or "").splitlines():
        try:
            event = json.loads(raw_line)
        except Exception:
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
        path = _normalize_path(path_text, repo_root)
        if not path:
            continue

        submatches = data.get("submatches")
        hit_count = len(submatches) if isinstance(submatches, list) and submatches else 1
        counts[path] = counts.get(path, 0) + max(1, int(hit_count))

        line_number = data.get("line_number")
        line_value = int(line_number) if isinstance(line_number, int) and line_number > 0 else 1_000_000_000
        prev = first_line.get(path)
        if prev is None or line_value < prev:
            first_line[path] = line_value

    ranked = sorted(counts.keys(), key=lambda p: (-counts[p], first_line.get(p, 1_000_000_000), p))
    if top_k > 0:
        ranked = ranked[:top_k]
    return ranked


def _ranked_files_from_grep(stdout: str, repo_root: Path, top_k: int) -> list[str]:
    counts: dict[str, int] = {}
    first_line: dict[str, int] = {}

    for raw_line in str(stdout or "").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        file_part, _, rest = line.partition(":")
        path = _normalize_path(file_part, repo_root)
        if not path:
            continue

        counts[path] = counts.get(path, 0) + 1
        line_no = None
        if ":" in rest:
            line_part, _, _ = rest.partition(":")
            try:
                line_no = int(line_part)
            except Exception:
                line_no = None
        line_value = line_no if isinstance(line_no, int) and line_no > 0 else 1_000_000_000
        prev = first_line.get(path)
        if prev is None or line_value < prev:
            first_line[path] = line_value

    ranked = sorted(counts.keys(), key=lambda p: (-counts[p], first_line.get(p, 1_000_000_000), p))
    if top_k > 0:
        ranked = ranked[:top_k]
    return ranked


def _run_lexical_search(
    *,
    engine: str,
    repo_root: Path,
    pattern: str,
    timeout_s: float,
    top_k: int,
) -> list[str]:
    use_rg = engine == "rg" or (engine == "auto" and shutil.which("rg") is not None)
    if use_rg:
        argv = ["rg", "--json", "--line-number", "--no-messages", "--smart-case", "--", pattern, "."]
        proc = subprocess.run(
            argv,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode not in (0, 1):
            stderr = str(proc.stderr or "").strip()
            raise RuntimeError(f"rg failed with exit={proc.returncode}: {stderr}")
        return _ranked_files_from_rg_json(proc.stdout, repo_root, top_k)

    if shutil.which("grep") is None:
        raise RuntimeError("grep binary not found")
    argv = ["grep", "-R", "-n", "-E", "--", pattern, "."]
    proc = subprocess.run(
        argv,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode not in (0, 1):
        stderr = str(proc.stderr or "").strip()
        raise RuntimeError(f"grep failed with exit={proc.returncode}: {stderr}")
    return _ranked_files_from_grep(proc.stdout, repo_root, top_k)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Native lexical retrieval adapter (rg/grep only).")
    ap.add_argument("--repo", required=True, help="Repository root to search.")
    ap.add_argument("--pattern", default="", help="Preferred regex pattern (typically deterministic rg_pattern).")
    ap.add_argument("--query", default="", help="Fallback search text when --pattern is empty.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--timeout-s", type=float, default=25.0)
    ap.add_argument("--engine", choices=("auto", "rg", "grep"), default="auto")
    return ap


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(args.repo).resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        print(json.dumps({"error": f"repo root not found: {repo_root}"}))
        return 2

    pattern = str(args.pattern or "").strip() or str(args.query or "").strip()
    if not pattern:
        print(json.dumps({"ranked_files": []}))
        return 0

    try:
        ranked_files = _run_lexical_search(
            engine=str(args.engine),
            repo_root=repo_root,
            pattern=pattern,
            timeout_s=max(0.1, float(args.timeout_s)),
            top_k=max(0, int(args.top_k)),
        )
    except subprocess.TimeoutExpired:
        print(json.dumps({"error": f"search timed out after {float(args.timeout_s):.3f}s"}))
        return 124
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 2

    print(json.dumps({"ranked_files": ranked_files}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
