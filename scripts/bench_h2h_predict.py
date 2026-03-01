#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
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
CATEGORY_KEYS = ("retrieval", "impact", "slice", "complexity", "data_flow")


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
        ranked.extend([_normalize_rel_path(x) for x in parsed if isinstance(x, str)])

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
                fn = item.get("function")
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
                    fn = item.get("function")
                    if isinstance(fp, str) and isinstance(fn, str):
                        callers.append({"file": _normalize_rel_path(fp), "function": fn})

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            fp = item.get("file")
            fn = item.get("function")
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


def _parse_data_flow_result(parsed: Any) -> dict[str, Any]:
    flow_lines: list[int] = []
    origin_line: int | None = None

    if isinstance(parsed, dict):
        fl = parsed.get("flow_lines")
        if isinstance(fl, list):
            flow_lines.extend([int(x) for x in fl if isinstance(x, int)])

        origin = parsed.get("origin_line")
        if isinstance(origin, int):
            origin_line = int(origin)

        flow = parsed.get("flow")
        if isinstance(flow, list):
            for event in flow:
                if not isinstance(event, dict):
                    continue
                ln = event.get("line")
                if isinstance(ln, int):
                    flow_lines.append(int(ln))
                    if origin_line is None and event.get("event") == "defined":
                        origin_line = int(ln)

    if isinstance(parsed, list):
        for event in parsed:
            if not isinstance(event, dict):
                continue
            ln = event.get("line")
            if isinstance(ln, int):
                flow_lines.append(int(ln))
                if origin_line is None and event.get("event") == "defined":
                    origin_line = int(ln)

    return {"origin_line": origin_line, "flow_lines": sorted(set(flow_lines))}


def _result_from_output(category: str, text: str) -> dict[str, Any]:
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
        return _parse_data_flow_result(parsed)
    return _default_result_for_category(category)


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


def _failure_class(status: str) -> str:
    if status == "timeout":
        return "provider_transport_runtime"
    if status == "error":
        return "product_failure"
    if status == "pending":
        return "unclassified"
    return "none"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run head-to-head task manifest through one tool profile.")
    ap.add_argument("--suite", default="benchmarks/head_to_head/suite.v1.json")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--tool-profile", required=True)
    ap.add_argument("--corpus-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--classification-out", default=None)
    ap.add_argument("--run-metadata-out", default=None)
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

    predictions: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []

    for task in tasks_sorted:
        task_id = str(task["task_id"])
        category = str(task["category"])
        command_entry = commands.get(category) if isinstance(commands.get(category), dict) else None
        template = command_entry.get("template") if isinstance(command_entry, dict) else None
        supported = bool(capabilities_by_category.get(category, False))

        for budget_tokens in budgets:
            for trial in range(1, trials + 1):
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
                    max_attempts = 1 + int(retry_on_timeout)
                    for _ in range(max_attempts):
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
                    result = _result_from_output(category, final_stdout)
                else:
                    result = _default_result_for_category(category)

                payload_tokens, payload_bytes = _payload_stats(final_stdout if final_status == "ok" else "")
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
                        "failure_class": _failure_class(str(row["status"])),
                        "reason": final_error or (final_stderr.strip()[:500] if isinstance(final_stderr, str) else ""),
                        "raw_log": str(raw_log),
                    }
                )

    _validate_no_duplicate_prediction_rows(predictions)

    out_doc = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "tool_id": tool_id,
        "task_manifest_sha256": actual_task_manifest_sha,
        "tokenizer": tokenizer,
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
            "task_manifest_sha256": actual_task_manifest_sha,
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
            "suite_sha256": _sha256_file(suite_path),
            "task_manifest_sha256": actual_task_manifest_sha,
            "tool_profile_sha256": _sha256_file(profile_path),
            "tokenizer": tokenizer,
            "token_budgets": budgets,
            "trials": trials,
            "seeds": suite.get("protocol", {}).get("seeds"),
            "timeout_s_per_query": timeout_s,
            "retry_on_timeout": retry_on_timeout,
            "retrieval_top_k": retrieval_top_k,
            "corpus_root": str(corpus_root),
            "required_categories": sorted(required_categories),
            "prediction_count": len(predictions),
        }
        write_report(metadata_path, metadata_doc)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
