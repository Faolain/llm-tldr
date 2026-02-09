#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_util import (
    bench_corpora_root,
    gather_meta,
    get_repo_root,
    load_corpora_manifest,
    write_json,
)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")


def _run_out(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return (proc.stdout or "").strip()


def _ensure_repo(repo_dir: Path, git_url: str) -> None:
    if (repo_dir / ".git").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    # Use a partial clone to reduce initial IO; checkout will fetch needed blobs.
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", git_url, str(repo_dir)])


def _checkout_pinned(repo_dir: Path, pinned_ref: str) -> dict[str, Any]:
    # Fetch updates to make sure tags/refs resolve; keep it conservative (no submodules).
    _run(["git", "fetch", "--tags", "--prune", "origin"], cwd=repo_dir)
    _run(["git", "checkout", "--force", pinned_ref], cwd=repo_dir)
    sha = _run_out(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    desc = _run_out(["git", "describe", "--tags", "--always", "--dirty"], cwd=repo_dir)
    return {"git_sha": sha, "git_describe": desc}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch pinned benchmark corpora into benchmark/corpora/<id>/")
    ap.add_argument("--manifest", default="benchmarks/corpora.json", help="Path to corpora manifest JSON")
    ap.add_argument("--corpus", action="append", default=[], help="Corpus id(s) to fetch (repeatable)")
    ap.add_argument("--all", action="store_true", help="Fetch all corpora from the manifest")
    ap.add_argument(
        "--out",
        default=None,
        help="Write a JSON report to this path (default: benchmark/runs/fetch-corpora.json)",
    )
    args = ap.parse_args()

    repo_root = get_repo_root()
    manifest_path = (repo_root / args.manifest).resolve()
    data = load_corpora_manifest(manifest_path)
    corpora: list[dict[str, Any]] = data["corpora"]

    wanted = set(args.corpus or [])
    if args.all:
        wanted = {c.get("id") for c in corpora if isinstance(c.get("id"), str)}
    if not wanted:
        raise SystemExit("error: pass --all or at least one --corpus <id>")

    bench_corpora = bench_corpora_root(repo_root)
    results: list[dict[str, Any]] = []

    for c in corpora:
        cid = c.get("id")
        if not isinstance(cid, str) or cid not in wanted:
            continue
        git_url = c.get("git_url")
        pinned_ref = c.get("pinned_ref")
        pinned_sha = c.get("pinned_sha")

        if not isinstance(git_url, str) or not git_url:
            raise SystemExit(f"error: corpus {cid} missing git_url")
        if not isinstance(pinned_ref, str) or not pinned_ref:
            raise SystemExit(f"error: corpus {cid} missing pinned_ref")

        dest = bench_corpora / cid
        _ensure_repo(dest, git_url)
        checkout = _checkout_pinned(dest, pinned_ref)

        entry: dict[str, Any] = {
            "id": cid,
            "git_url": git_url,
            "pinned_ref": pinned_ref,
            "pinned_sha": pinned_sha,
            "checkout": checkout,
            "path": str(dest),
            "ok": True,
        }
        if isinstance(pinned_sha, str) and pinned_sha and checkout["git_sha"] != pinned_sha:
            entry["ok"] = False
            entry["error"] = f"pinned_sha mismatch: got {checkout['git_sha']}, expected {pinned_sha}"
        results.append(entry)

    report = {
        "meta": gather_meta(tldr_repo_root=repo_root),
        "manifest": str(manifest_path),
        "results": results,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        out_path = repo_root / "benchmark" / "runs" / f"{ts}-fetch-corpora.json"
    write_json(out_path, report)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
