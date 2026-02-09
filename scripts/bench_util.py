from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, cwd: Path | None = None) -> str | None:
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
    return (proc.stdout or "").strip()


def _git_sha(repo_root: Path) -> str | None:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo_root)


def _git_describe(repo_root: Path) -> str | None:
    # Avoid failing when there are no tags.
    return _run(["git", "describe", "--tags", "--always", "--dirty"], cwd=repo_root)


def _cpu_brand() -> str | None:
    # Best-effort across platforms. Keep it lightweight and dependency-free.
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        return out or None
    return platform.processor() or None


def _python_version() -> str:
    return sys.version.split()[0]


def _tool_version(tool: str, args: list[str] | None = None) -> str | None:
    argv = [tool]
    if args:
        argv.extend(args)
    else:
        argv.append("--version")
    return _run(argv)


@dataclass(frozen=True)
class BenchMeta:
    date_utc: str
    tldr_git_sha: str | None
    tldr_git_describe: str | None
    corpus_id: str | None
    corpus_git_sha: str | None
    corpus_git_describe: str | None
    platform: str
    python: str
    node: str | None
    pnpm: str | None
    cpu: str | None


def gather_meta(
    *,
    tldr_repo_root: Path,
    corpus_id: str | None = None,
    corpus_root: Path | None = None,
) -> dict[str, Any]:
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    corpus_sha = _git_sha(corpus_root) if corpus_root is not None else None
    corpus_desc = _git_describe(corpus_root) if corpus_root is not None else None

    meta = BenchMeta(
        date_utc=date_utc,
        tldr_git_sha=_git_sha(tldr_repo_root),
        tldr_git_describe=_git_describe(tldr_repo_root),
        corpus_id=corpus_id,
        corpus_git_sha=corpus_sha,
        corpus_git_describe=corpus_desc,
        platform=f"{platform.system().lower()}-{platform.machine().lower()}",
        python=_python_version(),
        node=_tool_version("node", ["-v"]),
        pnpm=_tool_version("pnpm", ["-v"]),
        cpu=_cpu_brand(),
    )
    return asdict(meta)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def percentiles(values: list[float], ps: list[float] | None = None) -> dict[str, float]:
    """Compute simple nearest-rank percentiles for already-collected timings."""
    if ps is None:
        ps = [0.5, 0.95]
    if not values:
        return {}
    xs = sorted(values)
    out: dict[str, float] = {}
    n = len(xs)
    for p in ps:
        if p < 0 or p > 1:
            continue
        # Nearest-rank: 1-indexed rank.
        k = int(round(p * (n - 1)))
        out[f"p{int(p * 100)}"] = xs[k]
    return out


def write_report(path: Path, obj: Any) -> None:
    """Write a benchmark report JSON (stable key ordering for diffability)."""
    write_json(path, obj)


def bench_root(tldr_repo_root: Path) -> Path:
    # gitignored by default in this repo
    return tldr_repo_root / "benchmark"


def bench_corpora_root(tldr_repo_root: Path) -> Path:
    return bench_root(tldr_repo_root) / "corpora"


def bench_cache_root(tldr_repo_root: Path) -> Path:
    return bench_root(tldr_repo_root) / "cache-root"


def bench_runs_root(tldr_repo_root: Path) -> Path:
    return bench_root(tldr_repo_root) / "runs"


def get_repo_root() -> Path:
    # Prefer git root; fallback to cwd.
    out = _run(["git", "rev-parse", "--show-toplevel"])
    if out:
        return Path(out).resolve()
    return Path.cwd().resolve()


def load_corpora_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "corpora" not in data:
        raise ValueError(f"Bad corpora manifest: {path}")
    if not isinstance(data.get("corpora"), list):
        raise ValueError(f"Bad corpora manifest 'corpora' field: {path}")
    return data
