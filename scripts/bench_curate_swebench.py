#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_agentic_common import load_jsonl
from bench_util import get_repo_root, write_report

SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_S = 1800
DEFAULT_REPO = "django/django"


def _read_source(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("tasks", "instances", "data", "records"):
            value = data.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            raise ValueError(f"Unsupported SWE-bench source payload: {path}")
    else:
        raise ValueError(f"Unsupported SWE-bench source payload: {path}")
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
    return out


def _record_repo(row: dict[str, Any]) -> str:
    repo = row.get("repo")
    return str(repo).strip() if isinstance(repo, str) else ""


def _record_is_resolved(row: dict[str, Any]) -> bool:
    value = row.get("resolved")
    if isinstance(value, bool):
        return value
    for key in ("status", "instance_status", "resolution_status"):
        status = row.get(key)
        if isinstance(status, str):
            normalized = status.strip().lower()
            if normalized in {"resolved", "verified", "fixed", "pass"}:
                return True
            if normalized in {"unresolved", "failed", "open", "skip"}:
                return False
    patch = row.get("patch")
    return isinstance(patch, str) and bool(patch.strip())


def _coerce_test_cmd(row: dict[str, Any]) -> str | None:
    for key in ("test_cmd", "test_command", "runnable_test_cmd"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("test_cmds", "test_commands"):
        value = row.get(key)
        if isinstance(value, list):
            cmds = [str(item).strip() for item in value if str(item).strip()]
            if cmds:
                return " && ".join(cmds)
    targets: list[str] = []
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = row.get(key)
        if isinstance(value, list):
            targets.extend(str(item).strip() for item in value if str(item).strip())
    if targets and all(".py" in target or "::" in target for target in targets):
        return "python -m pytest " + " ".join(targets)
    return None


def _coerce_timeout_s(row: dict[str, Any], default_timeout_s: int) -> int:
    for key in ("timeout_s", "timeout", "test_timeout_s"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return int(value)
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                continue
            if parsed > 0:
                return parsed
    return int(default_timeout_s)


def _curate_task(row: dict[str, Any], *, default_timeout_s: int) -> dict[str, Any] | None:
    instance_id = row.get("instance_id")
    repo = _record_repo(row)
    patch = row.get("patch")
    base_commit = row.get("base_commit")
    test_cmd = _coerce_test_cmd(row)
    if not isinstance(instance_id, str) or not instance_id.strip():
        return None
    if not isinstance(base_commit, str) or not base_commit.strip():
        return None
    if not isinstance(patch, str) or not patch.strip():
        return None
    if not isinstance(test_cmd, str) or not test_cmd.strip():
        return None

    hints_text = row.get("hints_text")
    problem_statement = row.get("problem_statement")
    fail_to_pass = row.get("FAIL_TO_PASS") if isinstance(row.get("FAIL_TO_PASS"), list) else []
    pass_to_pass = row.get("PASS_TO_PASS") if isinstance(row.get("PASS_TO_PASS"), list) else []

    return {
        "instance_id": instance_id.strip(),
        "repo": repo,
        "base_commit": base_commit.strip(),
        "patch": patch,
        "problem_statement": str(problem_statement).strip() if isinstance(problem_statement, str) else "",
        "hints_text": str(hints_text).strip() if isinstance(hints_text, str) else "",
        "test_cmd": test_cmd.strip(),
        "timeout_s": _coerce_timeout_s(row, default_timeout_s),
        "fail_to_pass": [str(item) for item in fail_to_pass],
        "pass_to_pass": [str(item) for item in pass_to_pass],
    }


def curate_subset(
    rows: list[dict[str, Any]],
    *,
    repo: str,
    count: int,
    default_timeout_s: int,
) -> list[dict[str, Any]]:
    curated: list[dict[str, Any]] = []
    for row in rows:
        if _record_repo(row) != repo:
            continue
        if not _record_is_resolved(row):
            continue
        task = _curate_task(row, default_timeout_s=default_timeout_s)
        if task is None:
            continue
        curated.append(task)
    curated.sort(key=lambda item: item["instance_id"])
    return curated[: max(0, int(count))]


def build_subset_doc(
    *,
    rows: list[dict[str, Any]],
    source_path: Path,
    repo: str,
    count: int,
    default_timeout_s: int,
) -> dict[str, Any]:
    tasks = curate_subset(rows, repo=repo, count=count, default_timeout_s=default_timeout_s)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "swebench_verified",
        "source_path": str(source_path),
        "repo": repo,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(tasks),
        "tasks": tasks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Curate a deterministic SWE-bench Verified subset for agentic benchmarks.")
    ap.add_argument("--source", required=True, help="Path to the source SWE-bench JSON/JSONL payload.")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"Repository slug to keep (default: {DEFAULT_REPO}).")
    ap.add_argument("--count", type=int, default=30, help="Maximum tasks to keep (default: 30).")
    ap.add_argument(
        "--default-timeout-s",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"Fallback timeout for curated tasks (default: {DEFAULT_TIMEOUT_S}).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Write curated subset JSON here (default: benchmarks/agentic/swebench_subset.json).",
    )
    args = ap.parse_args()

    repo_root = get_repo_root()
    source_path = Path(args.source).resolve()
    out_path = (
        Path(args.out).resolve()
        if args.out
        else (repo_root / "benchmarks" / "agentic" / "swebench_subset.json").resolve()
    )
    rows = _read_source(source_path)
    doc = build_subset_doc(
        rows=rows,
        source_path=source_path,
        repo=str(args.repo),
        count=int(args.count),
        default_timeout_s=int(args.default_timeout_s),
    )
    write_report(out_path, doc)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
