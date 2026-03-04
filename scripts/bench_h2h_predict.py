#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import socket
import string
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench_util import bench_runs_root, get_repo_root, now_utc_compact, write_report

try:
    from tldr.stats import count_tokens as _count_tokens
except Exception:  # pragma: no cover - fallback if optional deps are unavailable

    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(str(text).split())


SCHEMA_VERSION = 1
CATEGORY_KEYS = ("retrieval", "impact", "slice", "complexity", "data_flow", "context")
PREFLIGHT_SEMANTIC_INDEX_MISSING_CLASS = "preflight_semantic_index_missing"
PREFLIGHT_SEMANTIC_INDEX_MISSING_REASON_MARKERS = ("semantic index not found",)


@dataclass(frozen=True)
class CommandAttempt:
    status: str
    returncode: int | None
    latency_ms: float
    stdout: str
    stderr: str
    error: str | None
    argv: list[str]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_rel_path(path: str) -> str:
    p = path.replace("\\", "/")
    if p.startswith("./"):
        return p[2:]
    return p


def _safe_component(value: str) -> str:
    return str(value).replace("/", "__").replace("\\", "__")


def _raw_log_path(repo_root: Path, *, tool_id: str, trial: int, task_id: str) -> Path:
    return (
        bench_runs_root(repo_root)
        / "raw_logs"
        / _safe_component(tool_id)
        / str(int(trial))
        / f"{_safe_component(task_id)}.log"
    )


def _required_categories(suite: dict[str, Any], capabilities: dict[str, bool]) -> set[str]:
    required: set[str] = set()
    lanes = suite.get("lanes")
    if not isinstance(lanes, list):
        return set(CATEGORY_KEYS)
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        categories = lane.get("categories")
        if not isinstance(categories, list):
            continue
        req_for_all = bool(lane.get("required_for_all_tools"))
        for category in categories:
            if category not in CATEGORY_KEYS:
                continue
            if req_for_all:
                required.add(category)
            elif bool(capabilities.get(category, False)):
                required.add(category)
    return required


def _validate_tool_profile(profile: dict[str, Any], suite_id: str) -> list[str]:
    errors: list[str] = []
    if int(profile.get("schema_version", 0)) != SCHEMA_VERSION:
        errors.append("tool profile schema_version must be 1")
    if profile.get("suite_id") != suite_id:
        errors.append("tool profile suite_id mismatch")

    capabilities = profile.get("capabilities")
    if not isinstance(capabilities, dict):
        errors.append("tool profile capabilities must be an object")
        capabilities = {}

    commands = profile.get("commands")
    if not isinstance(commands, dict):
        errors.append("tool profile commands must be an object")
        commands = {}

    feature_set_id = profile.get("feature_set_id")
    if feature_set_id is not None and (not isinstance(feature_set_id, str) or not feature_set_id.strip()):
        errors.append("tool profile feature_set_id must be a non-empty string when provided")

    for category in CATEGORY_KEYS:
        supported = capabilities.get(category)
        if not isinstance(supported, bool):
            errors.append(f"capabilities.{category} must be boolean")
            continue
        if not supported:
            continue
        entry = commands.get(category)
        if not isinstance(entry, dict):
            errors.append(f"commands.{category} must be present when capability is true")
            continue
        tmpl = entry.get("template")
        if not isinstance(tmpl, str) or not tmpl.strip():
            errors.append(f"commands.{category}.template must be a non-empty string")

    return errors


def _template_placeholders(template: str) -> list[str]:
    out: list[str] = []
    for _, field_name, _, _ in string.Formatter().parse(template):
        if not field_name:
            continue
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        out.append(root)
    return out


def _render_command_template(template: str, values: dict[str, Any]) -> str:
    missing = sorted({name for name in _template_placeholders(template) if name not in values})
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"missing template placeholders: {missing_str}")
    try:
        return template.format_map(values)
    except KeyError as exc:
        name = str(exc.args[0])
        raise ValueError(f"missing template placeholder: {name}") from exc


def _extract_json_from_text(text: str) -> Any | None:
    s = str(text or "").strip()
    if not s:
        return None

    # Strip optional markdown fences.
    if s.startswith("```"):
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
        s = "\n".join(lines).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    starts = [idx for idx in (s.find("{"), s.find("[")) if idx != -1]
    if not starts:
        return None
    start = min(starts)
    end = max(s.rfind("}"), s.rfind("]"))
    if end < start:
        return None
    frag = s[start : end + 1]
    try:
        return json.loads(frag)
    except Exception:
        return None


def _payload_stats(payload: str) -> tuple[int, int]:
    text = str(payload or "")
    return int(_count_tokens(text)), len(text.encode("utf-8"))


def _result_payload_text(result: dict[str, Any]) -> str:
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def _result_payload_stats(result: dict[str, Any]) -> tuple[int, int]:
    text = _result_payload_text(result)
    return _payload_stats(text)


def _default_result_for_category(category: str) -> dict[str, Any]:
    if category == "retrieval":
        return {"ranked_files": []}
    if category == "impact":
        return {"callers": []}
    if category == "slice":
        return {"lines": []}
    if category == "complexity":
        return {"cyclomatic_complexity": 0}
    if category == "data_flow":
        return {"origin_line": None, "flow_lines": []}
    if category == "context":
        return {"functions": [], "entry_point": None, "depth": 0}
    return {}


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _parse_retrieval_result(text: str, parsed: Any) -> dict[str, Any]:
    ranked: list[str] = []

    if isinstance(parsed, dict):
        rf = parsed.get("ranked_files")
        if isinstance(rf, list):
            ranked.extend([_normalize_rel_path(x) for x in rf if isinstance(x, str)])
        paths = parsed.get("paths")
        if isinstance(paths, list):
            ranked.extend([_normalize_rel_path(x) for x in paths if isinstance(x, str)])
        results = parsed.get("results")
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                file_path = item.get("file") or item.get("path")
                if isinstance(file_path, str):
                    ranked.append(_normalize_rel_path(file_path))
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                ranked.append(_normalize_rel_path(item))
                continue
            if not isinstance(item, dict):
                continue
            file_path = item.get("file") or item.get("path")
            if isinstance(file_path, str):
                ranked.append(_normalize_rel_path(file_path))

    if ranked:
        return {"ranked_files": _dedupe_preserve(ranked)}

    for line in str(text or "").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        candidate = raw
        if ":" in raw:
            left = raw.split(":", 1)[0].strip()
            if "/" in left or "." in Path(left).name:
                candidate = left
        candidate = _normalize_rel_path(candidate)
        if candidate:
            ranked.append(candidate)

    return {"ranked_files": _dedupe_preserve(ranked)}


def _parse_impact_result(parsed: Any) -> dict[str, Any]:
    callers: list[dict[str, str]] = []

    if isinstance(parsed, dict):
        raw_callers = parsed.get("callers")
        if isinstance(raw_callers, list):
            for item in raw_callers:
                if not isinstance(item, dict):
                    continue
                fp = item.get("file")
                fn = item.get("function") or item.get("caller")
                if isinstance(fp, str) and isinstance(fn, str):
                    callers.append({"file": _normalize_rel_path(fp), "function": fn})

        targets = parsed.get("targets")
        if isinstance(targets, dict):
            for _, value in targets.items():
                if not isinstance(value, dict):
                    continue
                nested = value.get("callers")
                if not isinstance(nested, list):
                    continue
                for item in nested:
                    if not isinstance(item, dict):
                        continue
                    fp = item.get("file")
                    fn = item.get("function") or item.get("caller")
                    if isinstance(fp, str) and isinstance(fn, str):
                        callers.append({"file": _normalize_rel_path(fp), "function": fn})

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            fp = item.get("file")
            fn = item.get("function") or item.get("caller")
            if isinstance(fp, str) and isinstance(fn, str):
                callers.append({"file": _normalize_rel_path(fp), "function": fn})

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for caller in callers:
        key = (caller["file"], caller["function"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(caller)
    return {"callers": deduped}


def _parse_slice_result(text: str, parsed: Any) -> dict[str, Any]:
    lines: list[int] = []
    if isinstance(parsed, dict) and isinstance(parsed.get("lines"), list):
        lines = [int(x) for x in parsed["lines"] if isinstance(x, int)]
    elif isinstance(parsed, list):
        lines = [int(x) for x in parsed if isinstance(x, int)]
    else:
        for token in str(text or "").replace(",", " ").split():
            token = token.strip()
            if token.isdigit():
                lines.append(int(token))
    return {"lines": sorted(set(lines))}


def _parse_complexity_result(text: str, parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, dict):
        for key in ("cyclomatic_complexity", "complexity", "cc"):
            value = parsed.get(key)
            if isinstance(value, int):
                return {"cyclomatic_complexity": int(value)}
    if isinstance(parsed, int):
        return {"cyclomatic_complexity": int(parsed)}
    for token in str(text or "").replace(",", " ").split():
        if token.isdigit():
            return {"cyclomatic_complexity": int(token)}
    return {"cyclomatic_complexity": 0}


def _parse_data_flow_result(parsed: Any, *, variable: str | None = None) -> dict[str, Any]:
    var_filter = variable.strip() if isinstance(variable, str) and variable.strip() else None

    def _as_line(value: Any) -> int | None:
        if isinstance(value, int):
            return int(value)
        return None

    # Current `tldrf dfg` schema: {refs, edges, variables}.
    has_dfg_shape = isinstance(parsed, dict) and any(k in parsed for k in ("refs", "edges", "variables"))
    dfg_flow_lines: set[int] = set()
    dfg_edge_def_lines: list[int] = []
    dfg_ref_def_lines: list[int] = []

    if isinstance(parsed, dict):
        edges = parsed.get("edges")
        if isinstance(edges, list):
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                if var_filter is not None and edge.get("var") != var_filter:
                    continue
                def_line = _as_line(edge.get("def_line"))
                use_line = _as_line(edge.get("use_line"))
                if def_line is not None:
                    dfg_flow_lines.add(def_line)
                    dfg_edge_def_lines.append(def_line)
                if use_line is not None:
                    dfg_flow_lines.add(use_line)

        refs = parsed.get("refs")
        if isinstance(refs, list):
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                if var_filter is not None and ref.get("name") != var_filter:
                    continue
                line = _as_line(ref.get("line"))
                if line is None:
                    continue
                dfg_flow_lines.add(line)
                ref_type = ref.get("type")
                if not isinstance(ref_type, str):
                    ref_type = ref.get("ref_type")
                if isinstance(ref_type, str) and ref_type.lower() == "definition":
                    dfg_ref_def_lines.append(line)

    dfg_origin_line: int | None = None
    if dfg_edge_def_lines:
        dfg_origin_line = min(dfg_edge_def_lines)
    elif dfg_ref_def_lines:
        dfg_origin_line = min(dfg_ref_def_lines)

    if has_dfg_shape:
        if dfg_flow_lines or dfg_origin_line is not None:
            return {
                "origin_line": dfg_origin_line,
                "flow_lines": sorted(dfg_flow_lines),
            }
        # Prefer variable-specific extraction over unfiltered fallbacks.
        if var_filter is not None:
            return {"origin_line": None, "flow_lines": []}

    # Legacy compatibility:
    # - {"origin_line": int, "flow_lines": [..]}
    # - {"flow": [{"line": int, "event": "defined"|"used"}, ...]}
    # - [{"line": int, "event": ...}, ...]
    legacy_flow_lines: set[int] = set()
    legacy_origin_line: int | None = None
    legacy_defined_lines: list[int] = []

    if isinstance(parsed, dict):
        fl = parsed.get("flow_lines")
        if isinstance(fl, list):
            for item in fl:
                line = _as_line(item)
                if line is not None:
                    legacy_flow_lines.add(line)

        origin = _as_line(parsed.get("origin_line"))
        if origin is not None:
            legacy_origin_line = origin

        flow = parsed.get("flow")
        if isinstance(flow, list):
            for event in flow:
                if not isinstance(event, dict):
                    continue
                line = _as_line(event.get("line"))
                if line is None:
                    continue
                legacy_flow_lines.add(line)
                ev = event.get("event")
                if isinstance(ev, str) and ev.lower() == "defined":
                    legacy_defined_lines.append(line)

    if isinstance(parsed, list):
        for event in parsed:
            if not isinstance(event, dict):
                continue
            line = _as_line(event.get("line"))
            if line is None:
                continue
            legacy_flow_lines.add(line)
            ev = event.get("event")
            if isinstance(ev, str) and ev.lower() == "defined":
                legacy_defined_lines.append(line)

    if legacy_origin_line is None and legacy_defined_lines:
        legacy_origin_line = min(legacy_defined_lines)
    return {"origin_line": legacy_origin_line, "flow_lines": sorted(legacy_flow_lines)}


def _normalize_context_name(name: str) -> str:
    """Strip module/class prefix from a qualified name to get the bare function name."""
    # "checks.check_dependencies" -> "check_dependencies"
    # "ModelAdmin.get_search_results" -> "get_search_results"
    return name.rsplit(".", 1)[-1] if "." in name else name


def _parse_context_result(text: str, parsed: Any) -> dict[str, Any]:
    funcs: list[dict[str, str]] = []

    # JSON mode (daemon or direct JSON output)
    if isinstance(parsed, dict):
        fn_list = parsed.get("functions")
        if isinstance(fn_list, list):
            for item in fn_list:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                fp = item.get("file")
                if isinstance(name, str) and isinstance(fp, str):
                    funcs.append({
                        "name": _normalize_context_name(name),
                        "file": _normalize_rel_path(fp),
                    })
            if funcs:
                return {"functions": funcs}

    # Text mode: parse to_llm_string() output lines like "📍 func_name (file.py:42)"
    for line in str(text or "").splitlines():
        m = re.search(r"📍\s+(?:\S+\.)?(\w+)\s+\(([^:)]+)", line)
        if m:
            name = m.group(1)
            fp = m.group(2).strip()
            funcs.append({"name": name, "file": _normalize_rel_path(fp)})

    return {"functions": funcs}


def _result_from_output(
    category: str,
    text: str,
    *,
    task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = _extract_json_from_text(text)
    if category == "retrieval":
        return _parse_retrieval_result(text, parsed)
    if category == "impact":
        return _parse_impact_result(parsed)
    if category == "slice":
        return _parse_slice_result(text, parsed)
    if category == "complexity":
        return _parse_complexity_result(text, parsed)
    if category == "data_flow":
        variable = None
        if isinstance(task, dict):
            input_obj = task.get("input")
            if isinstance(input_obj, dict):
                raw_variable = input_obj.get("variable")
                if isinstance(raw_variable, str) and raw_variable.strip():
                    variable = raw_variable.strip()
        return _parse_data_flow_result(parsed, variable=variable)
    if category == "context":
        return _parse_context_result(text, parsed)
    return _default_result_for_category(category)


def _trim_result_once(category: str, result: dict[str, Any]) -> bool:
    if category == "retrieval":
        ranked = result.get("ranked_files")
        if isinstance(ranked, list) and ranked:
            result["ranked_files"] = ranked[:-1]
            return True
        return False

    if category == "impact":
        callers = result.get("callers")
        if isinstance(callers, list) and callers:
            result["callers"] = callers[:-1]
            return True
        return False

    if category == "slice":
        lines = result.get("lines")
        if isinstance(lines, list) and lines:
            result["lines"] = lines[:-1]
            return True
        return False

    if category == "data_flow":
        flow_lines = result.get("flow_lines")
        if isinstance(flow_lines, list) and flow_lines:
            result["flow_lines"] = flow_lines[:-1]
            return True
        if result.get("origin_line") is not None:
            result["origin_line"] = None
            return True
        return False

    if category == "context":
        funcs = result.get("functions")
        if isinstance(funcs, list) and funcs:
            result["functions"] = funcs[:-1]
            return True
        return False

    # complexity is already minimal scalar output.
    return False


def _enforce_result_payload_caps(
    *,
    category: str,
    result: dict[str, Any],
    budget_tokens: int,
    max_payload_tokens_hard: int,
    max_payload_bytes_hard: int,
) -> tuple[dict[str, Any], int, int]:
    token_cap = int(budget_tokens)
    hard_token_cap = int(max_payload_tokens_hard)
    if hard_token_cap > 0:
        token_cap = min(token_cap, hard_token_cap)
    token_cap = max(0, token_cap)

    byte_cap = int(max_payload_bytes_hard)
    if byte_cap <= 0:
        byte_cap = 65536

    try:
        capped = json.loads(json.dumps(result))
    except Exception:
        capped = _default_result_for_category(category)

    # Iteratively trim deterministic tail fields until the serialized payload fits.
    for _ in range(20000):
        tokens, size_bytes = _result_payload_stats(capped)
        if tokens <= token_cap and size_bytes <= byte_cap:
            return capped, int(tokens), int(size_bytes)
        if not _trim_result_once(category, capped):
            fallback = _default_result_for_category(category)
            if capped == fallback:
                break
            capped = fallback

    final_tokens, final_size_bytes = _result_payload_stats(capped)
    return capped, int(final_tokens), int(final_size_bytes)


def _retrieval_pattern_has_lexical_hits(
    *,
    corpus_root: Path,
    pattern: str,
    pattern_hit_cache: dict[str, bool],
) -> bool:
    cached = pattern_hit_cache.get(pattern)
    if isinstance(cached, bool):
        return cached

    try:
        proc = subprocess.run(
            ["rg", "-m", "1", "-l", "--no-messages", "--", pattern, str(corpus_root)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        # If lexical check cannot run, do not force-empty retrieval output.
        pattern_hit_cache[pattern] = True
        return True

    if proc.returncode == 0:
        pattern_hit_cache[pattern] = True
        return True
    if proc.returncode == 1:
        pattern_hit_cache[pattern] = False
        return False

    # Invalid regex or runtime error; keep result unchanged rather than forcing empty.
    pattern_hit_cache[pattern] = True
    return True


def _apply_retrieval_rg_pattern_guard(
    *,
    task: dict[str, Any],
    corpus_root: Path,
    result: dict[str, Any],
    pattern_hit_cache: dict[str, bool],
) -> dict[str, Any]:
    if str(task.get("category")) != "retrieval":
        return result

    input_obj = task.get("input")
    if not isinstance(input_obj, dict):
        return result

    pattern = input_obj.get("rg_pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return result

    has_hits = _retrieval_pattern_has_lexical_hits(
        corpus_root=corpus_root,
        pattern=pattern,
        pattern_hit_cache=pattern_hit_cache,
    )
    if has_hits:
        return result
    return {"ranked_files": []}


def _run_command_once(*, argv: list[str], cwd: Path, timeout_s: float) -> CommandAttempt:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
        return CommandAttempt(
            status="timeout",
            returncode=None,
            latency_ms=latency_ms,
            stdout=stdout,
            stderr=stderr,
            error=f"timed out after {float(timeout_s):.3f}s",
            argv=list(argv),
        )
    except OSError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return CommandAttempt(
            status="error",
            returncode=None,
            latency_ms=latency_ms,
            stdout="",
            stderr="",
            error=str(exc),
            argv=list(argv),
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0
    status = "ok" if proc.returncode == 0 else "error"
    return CommandAttempt(
        status=status,
        returncode=int(proc.returncode),
        latency_ms=latency_ms,
        stdout=str(proc.stdout or ""),
        stderr=str(proc.stderr or ""),
        error=None,
        argv=list(argv),
    )


# ---------------------------------------------------------------------------
# Daemon execution path
# ---------------------------------------------------------------------------

_DAEMON_SUPPORTED_CATEGORIES = frozenset({"retrieval", "impact", "complexity", "data_flow", "slice", "context"})

# Templates that invoke non-daemon executables (fall back to subprocess).
_NON_DAEMON_TEMPLATE_PREFIXES = ("contextplus", "rg ")


def _extract_template_flag(template: str, flag: str) -> str | None:
    """Extract the literal value following *flag* in a template string.

    >>> _extract_template_flag("--abstain-threshold 0.35 --rerank", "--abstain-threshold")
    '0.35'
    >>> _extract_template_flag("--rerank --k 10", "--rerank") is None
    True
    """
    m = re.search(rf"{re.escape(flag)}\s+(\S+)", template)
    return m.group(1) if m else None


def _cli_template_to_daemon_command(
    category: str,
    template: str,
    values: dict[str, Any],
) -> dict[str, Any] | None:
    """Map a category + CLI template + rendered values to a daemon command dict.

    Returns ``None`` when the template cannot be served by the daemon (e.g.
    contextplus or rg-native templates), in which case the caller should fall
    back to subprocess execution.
    """
    if category not in _DAEMON_SUPPORTED_CATEGORIES:
        return None

    tmpl_lower = template.lower()
    for prefix in _NON_DAEMON_TEMPLATE_PREFIXES:
        if tmpl_lower.lstrip().startswith(prefix):
            return None

    if category == "retrieval":
        cmd: dict[str, Any] = {
            "cmd": "semantic",
            "action": "search",
            "query": values.get("query", ""),
            "k": int(values.get("top_k", 10)),
        }
        if "--hybrid" in template:
            cmd["retrieval_mode"] = "hybrid"
        rg_pattern = values.get("rg_pattern")
        if rg_pattern is not None:
            cmd["rg_pattern"] = str(rg_pattern)
        if "--rerank" in template:
            cmd["rerank"] = True
            rerank_top_n = _extract_template_flag(template, "--rerank-top-n")
            if rerank_top_n is not None:
                cmd["rerank_top_n"] = int(rerank_top_n)
        abstain_thresh = _extract_template_flag(template, "--abstain-threshold")
        if abstain_thresh is not None:
            cmd["abstain_threshold"] = float(abstain_thresh)
        if "--abstain-empty" in template:
            cmd["abstain_empty"] = True
        budget_tokens = values.get("budget_tokens")
        if budget_tokens is not None:
            cmd["budget_tokens"] = int(budget_tokens)
        no_result_guard = _extract_template_flag(template, "--no-result-guard")
        if no_result_guard is not None:
            cmd["no_result_guard"] = no_result_guard
        max_latency = _extract_template_flag(template, "--max-latency-ms-p50-ratio")
        if max_latency is not None:
            cmd["max_latency_ms_p50_ratio"] = float(max_latency)
        max_payload = _extract_template_flag(template, "--max-payload-tokens-median-ratio")
        if max_payload is not None:
            cmd["max_payload_tokens_median_ratio"] = float(max_payload)
        return cmd

    if category == "impact":
        return {
            "cmd": "impact",
            "func": values.get("function", ""),
            "file": values.get("file", ""),
        }

    if category == "complexity":
        return {
            "cmd": "cfg",
            "file": values.get("file_abs", values.get("file", "")),
            "function": values.get("function", ""),
        }

    if category == "data_flow":
        return {
            "cmd": "dfg",
            "file": values.get("file_abs", values.get("file", "")),
            "function": values.get("function", ""),
        }

    if category == "slice":
        return {
            "cmd": "slice",
            "file": values.get("file_abs", values.get("file", "")),
            "function": values.get("function", ""),
            "line": int(values.get("target_line", 0)),
        }

    if category == "context":
        return {
            "cmd": "context",
            "entry": values.get("entry_point", values.get("function", "")),
            "language": "python",
            "depth": int(values.get("depth", 2)),
        }

    return None


def _daemon_response_to_stdout(category: str, response: dict[str, Any]) -> str:
    """Convert a daemon JSON response into the stdout text that ``_result_from_output`` expects.

    The downstream parsing pipeline (``_result_from_output``) calls
    ``_extract_json_from_text()`` on stdout, so we emit JSON that the existing
    parsers already know how to handle.
    """
    if response.get("status") != "ok":
        return json.dumps(response)

    if category == "retrieval":
        # Daemon returns {"status":"ok","results":[...]} — pass the full response;
        # _parse_retrieval_result handles dicts with a "results" key.
        return json.dumps(response)

    if category == "impact":
        # Daemon returns {"status":"ok","callers":[{"caller":...}],"result":{"targets":{...}}}.
        # Canonicalize top-level "callers" to use "function" for downstream consistency.
        callers = response.get("callers")
        if isinstance(callers, list):
            normalized = []
            for entry in callers:
                if isinstance(entry, dict) and "caller" in entry and "function" not in entry:
                    new_entry = {k: v for k, v in entry.items() if k != "caller"}
                    new_entry["function"] = entry["caller"]
                    entry = new_entry
                normalized.append(entry)
            response = {**response, "callers": normalized}
        return json.dumps(response)

    if category == "slice":
        # Daemon returns {"status":"ok","lines":[...],"count":N}.
        return json.dumps(response)

    if category in ("complexity", "data_flow"):
        # Daemon wraps the analysis in a "result" key — unwrap it so the
        # category-specific parsers see the shape they expect.
        inner = response.get("result")
        if isinstance(inner, dict):
            return json.dumps(inner)
        return json.dumps(response)

    if category == "context":
        # Daemon returns {"status":"ok","result":{"entry_point":...,"functions":[...]}}.
        # Unwrap inner "result" so the context parser sees the shape it expects.
        inner = response.get("result", {})
        if isinstance(inner, str):
            return inner
        if isinstance(inner, dict):
            return json.dumps(inner)
        return json.dumps(response)

    return json.dumps(response)


def _run_daemon_query_once(
    *,
    command: dict[str, Any],
    category: str,
    corpus_root: Path,
    timeout_s: float,
    argv_for_log: list[str],
) -> CommandAttempt:
    """Execute a single query via the daemon, returning a ``CommandAttempt``."""
    from tldr.daemon.startup import query_daemon

    t0 = time.perf_counter()
    try:
        response = query_daemon(
            project_path=corpus_root,
            command=command,
            timeout=timeout_s,
        )
    except socket.timeout:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return CommandAttempt(
            status="timeout",
            returncode=None,
            latency_ms=latency_ms,
            stdout="",
            stderr="",
            error=f"daemon query timed out after {timeout_s:.3f}s",
            argv=argv_for_log,
        )
    except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return CommandAttempt(
            status="error",
            returncode=None,
            latency_ms=latency_ms,
            stdout="",
            stderr="",
            error=f"daemon connection error: {exc}",
            argv=argv_for_log,
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0
    stdout_text = _daemon_response_to_stdout(category, response)

    if response.get("status") == "ok":
        return CommandAttempt(
            status="ok",
            returncode=0,
            latency_ms=latency_ms,
            stdout=stdout_text,
            stderr="",
            error=None,
            argv=argv_for_log,
        )

    return CommandAttempt(
        status="error",
        returncode=1,
        latency_ms=latency_ms,
        stdout=stdout_text,
        stderr=response.get("message", ""),
        error=response.get("message", "daemon returned error"),
        argv=argv_for_log,
    )


def _write_raw_log(
    *,
    path: Path,
    task_id: str,
    category: str,
    budget_tokens: int,
    trial: int,
    attempts: list[CommandAttempt],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"task_id={task_id}",
        f"category={category}",
        f"budget_tokens={int(budget_tokens)}",
        f"trial={int(trial)}",
        f"attempt_count={len(attempts)}",
        "",
    ]
    for idx, attempt in enumerate(attempts, start=1):
        lines.extend(
            [
                f"[attempt {idx}]",
                f"status={attempt.status}",
                f"returncode={attempt.returncode}",
                f"latency_ms={attempt.latency_ms:.3f}",
                f"command={shlex.join(attempt.argv)}",
                f"error={attempt.error}",
                "[stdout]",
                attempt.stdout,
                "[stderr]",
                attempt.stderr,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _template_values(
    *,
    task: dict[str, Any],
    budget_tokens: int,
    trial: int,
    corpus_root: Path,
    retrieval_top_k: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "budget_tokens": int(budget_tokens),
        "trial": int(trial),
        "top_k": int(retrieval_top_k),
        "repo_root": str(corpus_root),
        "task_id": str(task.get("task_id") or ""),
        "category": str(task.get("category") or ""),
    }
    input_obj = task.get("input")
    if not isinstance(input_obj, dict):
        return out

    query = input_obj.get("query")
    if isinstance(query, str):
        out["query"] = query
    function = input_obj.get("function")
    if isinstance(function, str):
        out["function"] = function
    file_rel = input_obj.get("file")
    if isinstance(file_rel, str):
        normalized = _normalize_rel_path(file_rel)
        out["file"] = normalized
        out["file_abs"] = str((corpus_root / normalized).resolve())
    target_line = input_obj.get("target_line")
    if isinstance(target_line, int):
        out["target_line"] = int(target_line)
    variable = input_obj.get("variable")
    if isinstance(variable, str):
        out["variable"] = variable
    rg_pattern = input_obj.get("rg_pattern")
    if isinstance(rg_pattern, str):
        out["rg_pattern"] = rg_pattern
    entry_point = input_obj.get("entry_point")
    if isinstance(entry_point, str):
        out["entry_point"] = entry_point
    depth = input_obj.get("depth")
    if isinstance(depth, int):
        out["depth"] = int(depth)
    return out


def _validate_no_duplicate_prediction_rows(predictions: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, int, int]] = set()
    for row in predictions:
        if not isinstance(row, dict):
            continue
        task_id = row.get("task_id")
        budget = row.get("budget_tokens")
        trial = row.get("trial")
        if not isinstance(task_id, str) or not isinstance(budget, int) or not isinstance(trial, int):
            continue
        key = (task_id, int(budget), int(trial))
        if key in seen:
            raise ValueError(
                f"duplicate prediction row for (task_id={task_id}, budget_tokens={budget}, trial={trial})"
            )
        seen.add(key)


def _failure_class(status: str, reason: str | None = None) -> str:
    reason_norm = str(reason or "").lower()
    if status == "error" and any(marker in reason_norm for marker in PREFLIGHT_SEMANTIC_INDEX_MISSING_REASON_MARKERS):
        return PREFLIGHT_SEMANTIC_INDEX_MISSING_CLASS
    if status == "timeout":
        return "provider_transport_runtime"
    if status == "error":
        return "product_failure"
    if status == "pending":
        return "unclassified"
    return "none"


def _parse_filter_values(raw_values: list[str] | None) -> list[str]:
    values: list[str] = []
    for raw in raw_values or []:
        for token in str(raw).split(","):
            item = token.strip()
            if item:
                values.append(item)
    return values


def _parse_positive_int_filter(raw_values: list[str] | None, *, label: str) -> list[int]:
    out: list[int] = []
    for raw in _parse_filter_values(raw_values):
        try:
            value = int(raw)
        except Exception as exc:
            raise ValueError(f"{label} values must be integers: {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{label} values must be positive: {raw!r}")
        out.append(value)
    return sorted(set(out))


def _normalize_segment_filters(
    *,
    categories_raw: list[str] | None,
    task_ids_raw: list[str] | None,
    trials_raw: list[str] | None,
    budget_tokens_raw: list[str] | None,
) -> dict[str, list[Any]]:
    categories = sorted(set(_parse_filter_values(categories_raw)))
    unknown_categories = sorted(set(categories) - set(CATEGORY_KEYS))
    if unknown_categories:
        raise ValueError(f"unknown category filters: {', '.join(unknown_categories)}")

    return {
        "categories": categories,
        "task_ids": sorted(set(_parse_filter_values(task_ids_raw))),
        "trials": _parse_positive_int_filter(trials_raw, label="trial"),
        "budget_tokens": _parse_positive_int_filter(budget_tokens_raw, label="budget_tokens"),
    }


def _apply_segment_filters(
    *,
    tasks_sorted: list[dict[str, Any]],
    budgets: list[int],
    trials: int,
    segment_filters: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    selected_tasks = list(tasks_sorted)
    category_filters = {str(x) for x in segment_filters.get("categories", []) if isinstance(x, str)}
    task_id_filters = {str(x) for x in segment_filters.get("task_ids", []) if isinstance(x, str)}

    if category_filters:
        present_categories = {str(task.get("category")) for task in tasks_sorted if isinstance(task, dict)}
        missing_categories = sorted(category_filters - present_categories)
        if missing_categories:
            raise ValueError(
                "category filters not present in runnable task set: " + ", ".join(missing_categories)
            )
        selected_tasks = [
            task for task in selected_tasks if isinstance(task.get("category"), str) and task["category"] in category_filters
        ]

    if task_id_filters:
        present_task_ids = {str(task.get("task_id")) for task in tasks_sorted if isinstance(task, dict)}
        missing_task_ids = sorted(task_id_filters - present_task_ids)
        if missing_task_ids:
            raise ValueError(
                "task_id filters not present in task manifest: " + ", ".join(missing_task_ids)
            )
        selected_tasks = [
            task for task in selected_tasks if isinstance(task.get("task_id"), str) and task["task_id"] in task_id_filters
        ]

    if (category_filters or task_id_filters) and not selected_tasks:
        raise ValueError("segment filters selected no runnable tasks")

    trial_filters = {int(x) for x in segment_filters.get("trials", []) if isinstance(x, int)}
    selected_trials = list(range(1, int(trials) + 1))
    if trial_filters:
        missing_trials = sorted(trial_filters - set(selected_trials))
        if missing_trials:
            raise ValueError(
                "trial filters outside configured trial range: " + ", ".join(str(x) for x in missing_trials)
            )
        selected_trials = [trial for trial in selected_trials if trial in trial_filters]

    budget_filters = {int(x) for x in segment_filters.get("budget_tokens", []) if isinstance(x, int)}
    selected_budgets = [int(budget) for budget in budgets]
    if budget_filters:
        missing_budgets = sorted(budget_filters - set(selected_budgets))
        if missing_budgets:
            raise ValueError(
                "budget_tokens filters outside configured budgets: " + ", ".join(str(x) for x in missing_budgets)
            )
        selected_budgets = [budget for budget in selected_budgets if budget in budget_filters]

    return selected_tasks, selected_budgets, selected_trials


def _segment_filter_audit_doc(
    *,
    segment_filters: dict[str, list[Any]],
    selected_tasks: list[dict[str, Any]],
    selected_budgets: list[int],
    selected_trials: list[int],
) -> dict[str, Any]:
    selected_task_ids = [
        str(task.get("task_id")) for task in selected_tasks if isinstance(task, dict) and isinstance(task.get("task_id"), str)
    ]
    selected_categories = sorted(
        {
            str(task.get("category"))
            for task in selected_tasks
            if isinstance(task, dict) and isinstance(task.get("category"), str)
        }
    )
    return {
        "categories": [str(x) for x in segment_filters.get("categories", []) if isinstance(x, str)],
        "task_ids": [str(x) for x in segment_filters.get("task_ids", []) if isinstance(x, str)],
        "trials": [int(x) for x in segment_filters.get("trials", []) if isinstance(x, int)],
        "budget_tokens": [int(x) for x in segment_filters.get("budget_tokens", []) if isinstance(x, int)],
        "selected_categories": selected_categories,
        "selected_task_ids": selected_task_ids,
        "selected_trials": [int(x) for x in selected_trials],
        "selected_budget_tokens": [int(x) for x in selected_budgets],
        "selected_identity_count": len(selected_task_ids) * len(selected_trials) * len(selected_budgets),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run head-to-head task manifest through one tool profile.")
    ap.add_argument("--suite", default="benchmarks/head_to_head/suite.v1.json")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--tool-profile", required=True)
    ap.add_argument("--corpus-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--classification-out", default=None)
    ap.add_argument("--run-metadata-out", default=None)
    ap.add_argument(
        "--category",
        "--categories",
        dest="categories",
        action="append",
        default=None,
        help="Optional rerun filter; repeat or use comma-separated values.",
    )
    ap.add_argument(
        "--task-id",
        "--task-ids",
        dest="task_ids",
        action="append",
        default=None,
        help="Optional rerun filter; repeat or use comma-separated values.",
    )
    ap.add_argument(
        "--trial",
        "--trials",
        dest="trial_filters",
        action="append",
        default=None,
        help="Optional rerun filter; repeat or use comma-separated integers.",
    )
    ap.add_argument(
        "--budget-tokens",
        dest="budget_token_filters",
        action="append",
        default=None,
        help="Optional rerun filter; repeat or use comma-separated integers.",
    )
    ap.add_argument(
        "--use-daemon",
        dest="use_daemon",
        action="store_true",
        default=False,
        help="Use the llm-tldr daemon instead of spawning subprocesses per query.",
    )
    ap.add_argument(
        "--daemon-keep-alive",
        dest="daemon_keep_alive",
        action="store_true",
        default=False,
        help="Leave the daemon running after predictions (default: stop it).",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()

    repo_root = get_repo_root()
    suite_path = Path(args.suite).resolve()
    tasks_path = Path(args.tasks).resolve()
    profile_path = Path(args.tool_profile).resolve()

    suite = _read_json(suite_path)
    tasks_doc = _read_json(tasks_path)
    profile = _read_json(profile_path)

    if not isinstance(suite, dict) or not isinstance(tasks_doc, dict) or not isinstance(profile, dict):
        raise SystemExit("error: suite, task manifest, and tool profile must be JSON objects")

    suite_id = suite.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id:
        raise SystemExit("error: suite.suite_id must be a non-empty string")

    if tasks_doc.get("suite_id") != suite_id:
        raise SystemExit("error: task manifest suite_id mismatch")
    if profile.get("suite_id") != suite_id:
        raise SystemExit("error: tool profile suite_id mismatch")

    profile_errors = _validate_tool_profile(profile, suite_id)
    if profile_errors:
        raise SystemExit("error: invalid tool profile:\n- " + "\n- ".join(profile_errors))

    tasks = tasks_doc.get("tasks")
    if not isinstance(tasks, list):
        raise SystemExit("error: task manifest must contain a 'tasks' list")
    actual_task_manifest_sha = _sha256_json(tasks)
    declared_task_manifest_sha = tasks_doc.get("task_manifest_sha256")
    if declared_task_manifest_sha is not None and declared_task_manifest_sha != actual_task_manifest_sha:
        raise SystemExit("error: task manifest hash mismatch (file may have been edited)")

    tokenizer = suite.get("budgets", {}).get("tokenizer")
    if not isinstance(tokenizer, str) or not tokenizer:
        tokenizer = "cl100k_base"

    budgets_raw = suite.get("budgets", {}).get("token_budgets", [2000])
    budgets = sorted({int(x) for x in budgets_raw if isinstance(x, int) and x > 0})
    if not budgets:
        budgets = [2000]

    retrieval_top_k = int(suite.get("budgets", {}).get("retrieval_top_k", 10))
    max_payload_tokens_hard = int(suite.get("budgets", {}).get("max_payload_tokens_hard", 5000))
    if max_payload_tokens_hard <= 0:
        max_payload_tokens_hard = 5000
    max_payload_bytes_hard = int(suite.get("budgets", {}).get("max_payload_bytes_hard", 65536))
    if max_payload_bytes_hard <= 0:
        max_payload_bytes_hard = 65536
    trials = int(suite.get("protocol", {}).get("trials", 1))
    if trials <= 0:
        trials = 1
    timeout_s = float(suite.get("protocol", {}).get("timeout_s_per_query", 30))
    if timeout_s <= 0:
        timeout_s = 30.0
    retry_on_timeout = int(suite.get("protocol", {}).get("retry_on_timeout", 0))
    if retry_on_timeout < 0:
        retry_on_timeout = 0

    capabilities = profile.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    capabilities_by_category = {c: bool(capabilities.get(c, False)) for c in CATEGORY_KEYS}
    required_categories = _required_categories(suite, capabilities_by_category)

    tasks_sorted = [
        t
        for t in tasks
        if isinstance(t, dict)
        and isinstance(t.get("task_id"), str)
        and isinstance(t.get("category"), str)
        and str(t.get("category")) in required_categories
    ]
    tasks_sorted.sort(key=lambda t: str(t.get("task_id")))

    try:
        segment_filters = _normalize_segment_filters(
            categories_raw=args.categories,
            task_ids_raw=args.task_ids,
            trials_raw=args.trial_filters,
            budget_tokens_raw=args.budget_token_filters,
        )
        selected_tasks, selected_budgets, selected_trials = _apply_segment_filters(
            tasks_sorted=tasks_sorted,
            budgets=budgets,
            trials=trials,
            segment_filters=segment_filters,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    segment_filter_audit = _segment_filter_audit_doc(
        segment_filters=segment_filters,
        selected_tasks=selected_tasks,
        selected_budgets=selected_budgets,
        selected_trials=selected_trials,
    )

    if args.corpus_root:
        corpus_root = Path(args.corpus_root).resolve()
    else:
        dataset = tasks_doc.get("dataset")
        corpus_root_s = dataset.get("corpus_root") if isinstance(dataset, dict) else None
        if isinstance(corpus_root_s, str) and corpus_root_s:
            corpus_root = Path(corpus_root_s).resolve()
        else:
            corpus_root = repo_root

    commands = profile.get("commands")
    if not isinstance(commands, dict):
        commands = {}

    tool_id = profile.get("tool_id")
    if not isinstance(tool_id, str) or not tool_id:
        raise SystemExit("error: tool profile requires non-empty tool_id")
    feature_set_id = profile.get("feature_set_id")
    if not isinstance(feature_set_id, str) or not feature_set_id.strip():
        feature_set_id = "unspecified"
    else:
        feature_set_id = feature_set_id.strip()

    # ---- daemon lifecycle ----
    use_daemon = bool(getattr(args, "use_daemon", False))
    daemon_keep_alive = bool(getattr(args, "daemon_keep_alive", False))

    if use_daemon:
        if tool_id != "llm-tldr":
            raise SystemExit("error: --use-daemon is only supported for tool_id='llm-tldr'")
        from tldr.daemon.startup import start_daemon, stop_daemon, query_daemon as _ping_daemon

        start_daemon(corpus_root)
        # Ping to confirm the daemon is alive.
        try:
            ping_resp = _ping_daemon(corpus_root, {"cmd": "ping"}, timeout=5.0)
            if ping_resp.get("status") != "ok":
                raise SystemExit(f"error: daemon ping failed: {ping_resp}")
        except Exception as exc:
            raise SystemExit(f"error: daemon not reachable after start: {exc}") from exc

    predictions: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    retrieval_pattern_hit_cache: dict[str, bool] = {}

    for task in selected_tasks:
        task_id = str(task["task_id"])
        category = str(task["category"])
        command_entry = commands.get(category) if isinstance(commands.get(category), dict) else None
        template = command_entry.get("template") if isinstance(command_entry, dict) else None
        supported = bool(capabilities_by_category.get(category, False))

        for budget_tokens in selected_budgets:
            for trial in selected_trials:
                raw_log = _raw_log_path(repo_root, tool_id=tool_id, trial=trial, task_id=task_id)

                if not supported or not isinstance(template, str) or not template.strip():
                    row = {
                        "task_id": task_id,
                        "trial": int(trial),
                        "budget_tokens": int(budget_tokens),
                        "status": "unsupported",
                        "latency_ms": 0.0,
                        "payload_tokens": 0,
                        "payload_bytes": 0,
                        "result": _default_result_for_category(category),
                    }
                    predictions.append(row)
                    classification_rows.append(
                        {
                            "task_id": task_id,
                            "trial": int(trial),
                            "budget_tokens": int(budget_tokens),
                            "status": "unsupported",
                            "failure_class": _failure_class("unsupported"),
                            "reason": "tool capability disabled for category",
                            "raw_log": str(raw_log),
                        }
                    )
                    _write_raw_log(
                        path=raw_log,
                        task_id=task_id,
                        category=category,
                        budget_tokens=budget_tokens,
                        trial=trial,
                        attempts=[
                            CommandAttempt(
                                status="unsupported",
                                returncode=None,
                                latency_ms=0.0,
                                stdout="",
                                stderr="",
                                error="tool capability disabled for category",
                                argv=[],
                            )
                        ],
                    )
                    continue

                context = _template_values(
                    task=task,
                    budget_tokens=budget_tokens,
                    trial=trial,
                    corpus_root=corpus_root,
                    retrieval_top_k=retrieval_top_k,
                )
                attempts: list[CommandAttempt] = []
                final_status = "error"
                final_stdout = ""
                final_stderr = ""
                final_error: str | None = None
                total_latency_ms = 0.0
                argv: list[str] = []

                try:
                    rendered = _render_command_template(template, context)
                    argv = shlex.split(rendered)
                except Exception as exc:
                    final_error = str(exc)
                    attempts.append(
                        CommandAttempt(
                            status="error",
                            returncode=None,
                            latency_ms=0.0,
                            stdout="",
                            stderr="",
                            error=str(exc),
                            argv=[],
                        )
                    )
                else:
                    daemon_cmd = (
                        _cli_template_to_daemon_command(category, template, context)
                        if use_daemon
                        else None
                    )
                    max_attempts = 1 + int(retry_on_timeout)
                    for _ in range(max_attempts):
                        if daemon_cmd is not None:
                            attempt = _run_daemon_query_once(
                                command=daemon_cmd,
                                category=category,
                                corpus_root=corpus_root,
                                timeout_s=timeout_s,
                                argv_for_log=argv,
                            )
                        else:
                            attempt = _run_command_once(argv=argv, cwd=corpus_root, timeout_s=timeout_s)
                        attempts.append(attempt)
                        total_latency_ms += float(attempt.latency_ms)
                        final_status = attempt.status
                        final_stdout = attempt.stdout
                        final_stderr = attempt.stderr
                        final_error = attempt.error
                        if attempt.status != "timeout":
                            break

                _write_raw_log(
                    path=raw_log,
                    task_id=task_id,
                    category=category,
                    budget_tokens=budget_tokens,
                    trial=trial,
                    attempts=attempts,
                )

                if final_status == "ok":
                    result = _result_from_output(category, final_stdout, task=task)
                    result = _apply_retrieval_rg_pattern_guard(
                        task=task,
                        corpus_root=corpus_root,
                        result=result,
                        pattern_hit_cache=retrieval_pattern_hit_cache,
                    )
                    result, payload_tokens, payload_bytes = _enforce_result_payload_caps(
                        category=category,
                        result=result,
                        budget_tokens=int(budget_tokens),
                        max_payload_tokens_hard=max_payload_tokens_hard,
                        max_payload_bytes_hard=max_payload_bytes_hard,
                    )
                else:
                    result = _default_result_for_category(category)
                    payload_tokens, payload_bytes = _result_payload_stats(result)
                row = {
                    "task_id": task_id,
                    "trial": int(trial),
                    "budget_tokens": int(budget_tokens),
                    "status": final_status if final_status in {"ok", "timeout", "error"} else "error",
                    "latency_ms": round(float(total_latency_ms), 3),
                    "payload_tokens": int(payload_tokens),
                    "payload_bytes": int(payload_bytes),
                    "result": result,
                }
                predictions.append(row)
                classification_rows.append(
                    {
                        "task_id": task_id,
                        "trial": int(trial),
                        "budget_tokens": int(budget_tokens),
                        "status": row["status"],
                        "failure_class": _failure_class(
                            str(row["status"]),
                            final_error or (final_stderr.strip()[:500] if isinstance(final_stderr, str) else ""),
                        ),
                        "reason": final_error or (final_stderr.strip()[:500] if isinstance(final_stderr, str) else ""),
                        "raw_log": str(raw_log),
                    }
                )

    # ---- daemon teardown ----
    if use_daemon and not daemon_keep_alive:
        from tldr.daemon.startup import stop_daemon
        stop_daemon(corpus_root)

    _validate_no_duplicate_prediction_rows(predictions)

    out_doc = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "tool_id": tool_id,
        "feature_set_id": feature_set_id,
        "task_manifest_sha256": actual_task_manifest_sha,
        "tokenizer": tokenizer,
        "segment_filters": segment_filter_audit,
        "predictions": predictions,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = bench_runs_root(repo_root) / f"{now_utc_compact()}-h2h-{tool_id}-predictions.json"
    write_report(out_path, out_doc)

    if args.classification_out:
        class_path = Path(args.classification_out)
        class_doc = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": suite_id,
            "tool_id": tool_id,
            "feature_set_id": feature_set_id,
            "task_manifest_sha256": actual_task_manifest_sha,
            "segment_filters": segment_filter_audit,
            "rows": sorted(
                classification_rows,
                key=lambda r: (
                    str(r.get("task_id", "")),
                    int(r.get("budget_tokens", 0)),
                    int(r.get("trial", 0)),
                ),
            ),
        }
        write_report(class_path, class_doc)

    if args.run_metadata_out:
        metadata_path = Path(args.run_metadata_out)
        metadata_doc = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": suite_id,
            "tool_id": tool_id,
            "feature_set_id": feature_set_id,
            "suite_sha256": _sha256_file(suite_path),
            "task_manifest_sha256": actual_task_manifest_sha,
            "tool_profile_sha256": _sha256_file(profile_path),
            "tokenizer": tokenizer,
            "token_budgets": selected_budgets,
            "suite_token_budgets": budgets,
            "trials": trials,
            "selected_trials": selected_trials,
            "segment_filters": segment_filter_audit,
            "seeds": suite.get("protocol", {}).get("seeds"),
            "timeout_s_per_query": timeout_s,
            "retry_on_timeout": retry_on_timeout,
            "retrieval_top_k": retrieval_top_k,
            "corpus_root": str(corpus_root),
            "required_categories": sorted(required_categories),
            "prediction_count": len(predictions),
            "execution_mode": "daemon" if use_daemon else "subprocess",
        }
        write_report(metadata_path, metadata_doc)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
