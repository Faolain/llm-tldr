from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class TsResolverError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TsCallGraphEdgeEndpoint:
    file: str
    symbol: str
    line: int | None = None
    col: int | None = None


@dataclass(frozen=True)
class TsCallGraphEdge:
    caller: TsCallGraphEdgeEndpoint
    callee: TsCallGraphEdgeEndpoint
    callsite: TsCallGraphEdgeEndpoint | None = None

    def to_tuple(self) -> tuple[str, str, str, str]:
        return (self.caller.file, self.caller.symbol, self.callee.file, self.callee.symbol)


def _canonical_relpath(root: Path, path: Path) -> str:
    # Use forward slashes for determinism across platforms.
    return str(path.resolve().relative_to(root.resolve())).replace(os.sep, "/")


def _nearest_tsconfig_for_file(
    root: Path,
    file_path: Path,
    *,
    dir_cache: dict[Path, Path | None],
) -> Path | None:
    """Return the nearest tsconfig.json above file_path (inclusive), bounded by root."""
    root = root.resolve()
    cur = file_path.resolve().parent
    visited: list[Path] = []
    while True:
        if cur in dir_cache:
            res = dir_cache[cur]
            for v in visited:
                dir_cache[v] = res
            return res

        visited.append(cur)
        candidate = cur / "tsconfig.json"
        if candidate.exists():
            for v in visited:
                dir_cache[v] = candidate
            return candidate

        if cur == root:
            for v in visited:
                dir_cache[v] = None
            return None

        cur = cur.parent


def _group_ts_files_by_nearest_tsconfig(
    root: Path,
    *,
    allow_files: Iterable[str | Path],
) -> tuple[dict[Path, list[Path]], list[Path]]:
    """Group scanned TS files by their nearest tsconfig.json under root."""
    root = root.resolve()
    dir_cache: dict[Path, Path | None] = {}
    groups: dict[Path, list[Path]] = {}
    unassigned: list[Path] = []

    for p in allow_files:
        pp = Path(p).resolve()
        try:
            pp.relative_to(root)
        except ValueError:
            continue
        tsconfig = _nearest_tsconfig_for_file(root, pp, dir_cache=dir_cache)
        if tsconfig is None:
            unassigned.append(pp)
        else:
            groups.setdefault(tsconfig, []).append(pp)

    for files in groups.values():
        files.sort()
    unassigned.sort()
    return groups, unassigned


def _prefer_edge(existing: TsCallGraphEdge, candidate: TsCallGraphEdge) -> TsCallGraphEdge:
    """Deterministically pick the better edge when deduping (prefer smallest callsite)."""
    if (
        candidate.callsite is not None
        and candidate.callsite.line is not None
        and candidate.callsite.col is not None
    ):
        if (
            existing.callsite is None
            or existing.callsite.line is None
            or existing.callsite.col is None
        ):
            return candidate
        if (candidate.callsite.line, candidate.callsite.col) < (existing.callsite.line, existing.callsite.col):
            return candidate
    return existing


def build_ts_resolved_call_graph_multi_tsconfig(
    root: str | Path,
    *,
    allow_files: Iterable[str | Path],
    trace: bool = False,
    timeout_s: int = 120,
) -> tuple[list[TsCallGraphEdge], dict[str, Any]]:
    """Build a TS-resolved call graph by splitting inputs across nearest tsconfig.json files."""
    root = Path(root).resolve()
    groups, unassigned = _group_ts_files_by_nearest_tsconfig(root, allow_files=allow_files)
    if not groups:
        raise TsResolverError("No tsconfig.json found for any TypeScript files under root", code="tsconfig_missing")

    edges_by_key: dict[tuple[str, str, str, str], TsCallGraphEdge] = {}
    ts_projects: list[dict[str, Any]] = []
    skipped_combined: list[dict[str, Any]] = []
    skipped_total = 0
    skipped_limit_total = 5000

    def _safe_rel(p: Path) -> str:
        try:
            return _canonical_relpath(root, p)
        except Exception:
            return str(p)

    for tsconfig_path in sorted(groups.keys(), key=lambda p: _safe_rel(p)):
        files = groups[tsconfig_path]
        proj: dict[str, Any] = {
            "tsconfig": _safe_rel(tsconfig_path),
            "allowlist_count": len(files),
        }
        try:
            edges, meta = build_ts_resolved_call_graph(
                root,
                allow_files=files,
                tsconfig=tsconfig_path,
                trace=trace,
                timeout_s=timeout_s,
            )
            proj.update(
                {
                    "status": "ok",
                    "processed_files": meta.get("processed_files"),
                    "edge_count": len(edges),
                }
            )
            if meta.get("typescript_version"):
                proj["typescript_version"] = meta.get("typescript_version")
            if meta.get("typescript_source"):
                proj["typescript_source"] = meta.get("typescript_source")

            if trace:
                # Track counts even when the per-project skip list is truncated.
                try:
                    skipped_total += int(meta.get("skipped_count") or 0)
                except Exception:
                    skipped_total += 0
                if isinstance(meta.get("skipped"), list):
                    proj["skipped_count"] = meta.get("skipped_count")
                    proj["skipped_truncated"] = meta.get("skipped_truncated")
                    for item in meta.get("skipped") or []:
                        if len(skipped_combined) >= skipped_limit_total:
                            break
                        if not isinstance(item, dict):
                            continue
                        d = dict(item)
                        d["tsconfig"] = proj["tsconfig"]
                        skipped_combined.append(d)

            for edge in edges:
                key = edge.to_tuple()
                existing = edges_by_key.get(key)
                if existing is None:
                    edges_by_key[key] = edge
                else:
                    edges_by_key[key] = _prefer_edge(existing, edge)
        except TsResolverError as exc:
            proj.update(
                {
                    "status": "error",
                    "error_code": exc.code or "resolver_error",
                    "error_message": str(exc),
                }
            )

        ts_projects.append(proj)

    ok_projects = sum(1 for p in ts_projects if p.get("status") == "ok")
    if ok_projects == 0:
        first_err = next((p for p in ts_projects if p.get("status") == "error"), None)
        code = (first_err or {}).get("error_code") or "resolver_error"
        msg = "All tsconfig projects failed"
        if first_err:
            msg = f"All tsconfig projects failed (first: {first_err.get('tsconfig')}: {first_err.get('error_message')})"
        raise TsResolverError(msg, code=str(code))

    edges = list(edges_by_key.values())
    edges.sort(
        key=lambda e: (
            e.caller.file,
            e.caller.symbol,
            e.callee.file,
            e.callee.symbol,
            e.callsite.line if e.callsite else -1,
            e.callsite.col if e.callsite else -1,
        )
    )

    meta_out: dict[str, Any] = {
        "resolver": "ts-compiler-api-multi",
        "root": str(root),
        "ok_projects": ok_projects,
        "error_projects": sum(1 for p in ts_projects if p.get("status") == "error"),
        "ts_projects": ts_projects,
        "unassigned_count": len(unassigned),
    }
    if unassigned:
        meta_out["unassigned_sample"] = [_safe_rel(p) for p in unassigned[:25]]
    if trace:
        skipped_combined.sort(
            key=lambda s: (
                str(s.get("tsconfig") or ""),
                str((s.get("callsite") or {}).get("file") or ""),
                int((s.get("callsite") or {}).get("line") or 0),
                int((s.get("callsite") or {}).get("col") or 0),
                str(s.get("reason") or ""),
            )
        )
        meta_out["skipped"] = skipped_combined
        meta_out["skipped_count"] = skipped_total
        meta_out["skipped_limit"] = skipped_limit_total
        meta_out["skipped_truncated"] = skipped_total > len(skipped_combined)

    return edges, meta_out


def _find_tsconfig(root: Path) -> Path | None:
    """Find a deterministic tsconfig for the project root.

    Rules:
    - Prefer <root>/tsconfig.json if present.
    - Otherwise, if exactly one tsconfig.json exists under root (excluding node_modules/.tldr), use it.
    - If none, return None.
    - If multiple, raise TsResolverError(code="tsconfig_ambiguous").
    """
    root = root.resolve()
    direct = root / "tsconfig.json"
    if direct.exists():
        return direct

    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune heavyweight/irrelevant dirs.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in ("node_modules", ".tldr", "dist", "build", ".git")
        ]
        if "tsconfig.json" in filenames:
            found.append(Path(dirpath) / "tsconfig.json")
            if len(found) > 1:
                break

    if not found:
        return None
    if len(found) > 1:
        raise TsResolverError(
            f"Ambiguous tsconfig selection under {root} (found multiple tsconfig.json files).",
            code="tsconfig_ambiguous",
        )
    return found[0]


def _node_exe() -> str | None:
    return shutil.which("node")


def build_ts_resolved_call_graph(
    root: str | Path,
    *,
    allow_files: Iterable[str | Path] | None = None,
    tsconfig: str | Path | None = None,
    trace: bool = False,
    timeout_s: int = 120,
) -> tuple[list[TsCallGraphEdge], dict[str, Any]]:
    """Build a TS-resolved call graph using a Node helper + TypeScript compiler API.

    Returns (edges, meta). Raises TsResolverError on resolver/toolchain problems.
    """
    root = Path(root).resolve()

    node = _node_exe()
    if node is None:
        raise TsResolverError("node not found in PATH", code="node_missing")

    helper = Path(__file__).with_name("ts_callgraph_node.js")
    if not helper.exists():
        raise TsResolverError(f"Missing Node helper: {helper}", code="helper_missing")

    if tsconfig is None:
        tsconfig_path = _find_tsconfig(root)
    else:
        tsconfig_path = Path(tsconfig).resolve()
        if not tsconfig_path.exists():
            raise TsResolverError(f"tsconfig does not exist: {tsconfig_path}", code="tsconfig_missing")

    if tsconfig_path is None:
        raise TsResolverError("No tsconfig.json found", code="tsconfig_missing")

    allowlist_path: Path | None = None
    if allow_files is not None:
        allow_abs: list[str] = []
        for p in allow_files:
            pp = Path(p).resolve()
            try:
                # Keep allowlist scoped to root. Anything else is ignored.
                pp.relative_to(root)
            except ValueError:
                continue
            allow_abs.append(str(pp))

        # Deterministic order for Node-side processing.
        allow_abs.sort()

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        )
        try:
            json.dump(allow_abs, tmp)
            tmp.flush()
            allowlist_path = Path(tmp.name)
        finally:
            tmp.close()

    cmd: list[str] = [
        node,
        str(helper),
        "--root",
        str(root),
        "--tsconfig",
        str(tsconfig_path),
    ]
    if allowlist_path is not None:
        cmd.extend(["--allowlist", str(allowlist_path)])
    if trace:
        cmd.append("--trace")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    finally:
        if allowlist_path is not None:
            try:
                allowlist_path.unlink()
            except OSError:
                pass

    stdout = (proc.stdout or "").strip()
    if not stdout:
        raise TsResolverError(
            f"TS resolver produced no output (exit={proc.returncode}): {proc.stderr.strip()}",
            code="resolver_no_output",
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise TsResolverError(
            f"Failed to parse TS resolver JSON: {exc}. stderr={proc.stderr.strip()!r}",
            code="resolver_bad_json",
        ) from exc

    if payload.get("status") != "ok":
        err = payload.get("error") or payload.get("message") or "TS resolver failed"
        code = payload.get("code")
        raise TsResolverError(str(err), code=str(code) if code else "resolver_error")

    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    if trace and isinstance(payload.get("skipped"), list):
        # Keep raw skipped entries for CLI trace mode. This can be large, so
        # callers should summarize/bound output.
        meta["skipped"] = payload.get("skipped")
    edges_raw = payload.get("edges", [])

    edges: list[TsCallGraphEdge] = []
    for item in edges_raw:
        try:
            caller = item["caller"]
            callee = item["callee"]
        except Exception:
            continue

        caller_ep = TsCallGraphEdgeEndpoint(
            file=str(caller.get("file")),
            symbol=str(caller.get("symbol")),
            line=caller.get("line"),
            col=caller.get("col"),
        )
        callee_ep = TsCallGraphEdgeEndpoint(
            file=str(callee.get("file")),
            symbol=str(callee.get("symbol")),
            line=callee.get("line"),
            col=callee.get("col"),
        )

        callsite_ep = None
        callsite = item.get("callsite")
        if isinstance(callsite, dict):
            callsite_ep = TsCallGraphEdgeEndpoint(
                file=str(callsite.get("file")),
                symbol=str(callsite.get("symbol") or ""),
                line=callsite.get("line"),
                col=callsite.get("col"),
            )

        edges.append(TsCallGraphEdge(caller=caller_ep, callee=callee_ep, callsite=callsite_ep))

    # Ensure deterministic ordering at the API boundary.
    edges.sort(key=lambda e: (e.caller.file, e.caller.symbol, e.callee.file, e.callee.symbol, e.callsite.line if e.callsite else -1, e.callsite.col if e.callsite else -1))

    return edges, meta
