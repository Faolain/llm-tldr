#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench_agentic_common import (
    ProviderRuntime,
    call_provider,
    container_workspace_path,
    docker_run,
    load_jsonl,
    make_provider_runtime,
    normalize_tool_name,
    sha256_file,
)
from bench_util import bench_root, gather_meta, get_repo_root, make_report, now_utc_compact, write_report


MODEL_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["tool", "final"]},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "answer": {"type": "string"},
    },
    "required": ["kind"],
    "additionalProperties": False,
}

MAX_OBSERVATION_CHARS = 5000
DEFAULT_TOOL_MAX_LINES = 80


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    args: dict[str, Any]
    observation: str
    ok: bool


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _clip(text: str, *, limit: int = MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 17] + "\n...[truncated]"


def _format_read_window(path: Path, *, start_line: int | None, end_line: int | None) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(start_line or 1))
    end = min(len(lines), int(end_line or len(lines)))
    body = "\n".join(f"{idx + 1:>5}: {lines[idx]}" for idx in range(start - 1, end))
    return _clip(body)


def _resolve_task_workspace(task: dict[str, Any], *, cli_repo_root: Path | None, repo_root: Path) -> Path:
    for key in ("repo_path", "workspace", "workspace_root"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = (repo_root / candidate).resolve()
            return candidate
    if cli_repo_root is not None:
        return cli_repo_root.resolve()
    repo_name = task.get("target_repo") or task.get("repo")
    if isinstance(repo_name, str) and repo_name.strip():
        candidate = (bench_root(repo_root) / "corpora" / repo_name).resolve()
        if candidate.exists():
            return candidate
    return repo_root


def resolve_workspace_path(workspace_root: Path, user_path: str) -> Path:
    candidate = (workspace_root / user_path).resolve()
    try:
        candidate.relative_to(workspace_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {user_path}") from exc
    return candidate


def _copy_workspace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)


def _run_cmd(
    runtime: ProviderRuntime,
    cmd: list[str] | str,
    *,
    cwd: Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
    shell: bool = False,
    network_disabled: bool = True,
    input_text: str | None = None,
) -> tuple[int, str, str]:
    if shell:
        command = ["/bin/sh", "-lc", str(cmd)]
    elif isinstance(cmd, str):
        command = [str(cmd)]
    else:
        command = [str(part) for part in cmd]
    proc = docker_run(
        runtime=runtime,
        workspace_root=cwd,
        command=command,
        timeout_s=timeout_s,
        input_text=input_text,
        extra_env=env,
        mount_repo=True,
        workspace_read_only=False,
        network_disabled=network_disabled,
    )
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""


def _tool_env(runtime: ProviderRuntime, workspace_root: Path) -> dict[str, str]:
    env = {
        "PYTHONUNBUFFERED": "1",
        "NO_COLOR": "1",
        "CI": "1",
        "TLDR_DEVICE": "cpu",
        "TLDR_CACHE_ROOT": f"{runtime.container_workspace_root}/.tldr-cache",
        "TLDR_INDEX": f"repo:{workspace_root.name}",
    }
    return env


def default_tools(*, arm: str, allow_edits: bool = False) -> dict[str, str]:
    tools = {
        "rg": "Exact lexical lookup. Args: pattern (required), glob (optional), max_results (optional).",
        "grep": "Recursive grep fallback. Args: pattern (required), max_results (optional).",
        "read_file": "Read a file window. Args: path (required), start_line (optional), end_line (optional).",
        "run_tests": "Run a deterministic test command in the workspace. Args: command (optional).",
    }
    if allow_edits:
        tools.update(
            {
                "replace_text": "Replace exact text in a file. Args: path, old, new, count (optional).",
                "write_file": "Write full file contents. Args: path, content.",
                "git_diff": "Show git diff or changed files. Args: names_only (optional bool).",
            }
        )
    if arm == "augmented":
        tools.update(
            {
                "tldrf_context": "Token-efficient context lookup. Args: entry, depth (optional), lang (optional).",
                "tldrf_impact": "Reverse-call impact lookup. Args: func, depth (optional), file (optional), lang (optional).",
                "tldrf_slice": "Program slice lookup. Args: file, function, line, direction (optional), var (optional), lang (optional).",
                "tldrf_dfg": "Data flow lookup. Args: file, function, lang (optional).",
                "tldrf_semantic_search": "Semantic code search. Args: query, k (optional), lang (optional).",
                "tldrf_hybrid_search": "Hybrid lexical+semantic search. Args: query, k (optional), rg_pattern (optional), lang (optional).",
            }
        )
    return tools


def _tool_descriptions(tool_map: dict[str, str]) -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in tool_map.items())


def build_loop_prompt(
    *,
    instruction_text: str,
    task: dict[str, Any],
    turn_index: int,
    tools: dict[str, str],
    history: list[dict[str, Any]],
) -> str:
    history_lines: list[str] = []
    for item in history[-8:]:
        if item.get("kind") == "tool_result":
            history_lines.append(
                "Observation from "
                f"{item.get('tool_name')}: {item.get('observation', '')}"
            )
        elif item.get("kind") == "assistant_decision":
            history_lines.append(f"Previous decision: {item.get('raw_text', '')}")
    task_json = json.dumps(
        {
            "id": task.get("id"),
            "workflow_class": task.get("workflow_class"),
            "question": task.get("question"),
            "expected_first_tool": task.get("expected_first_tool"),
            "expected_tool_set": task.get("expected_tool_set"),
            "forbidden_first_tool": task.get("forbidden_first_tool"),
            "max_allowed_dead_end_turns": task.get("max_allowed_dead_end_turns"),
        },
        sort_keys=True,
    )
    history_blob = "\n".join(history_lines) if history_lines else "(no prior tool observations)"
    return (
        "You are a benchmark-controlled coding assistant. "
        "You must choose exactly one next action and reply with JSON only.\n\n"
        "Rules:\n"
        "1. Use only the listed tools.\n"
        "2. Prefer exact lexical tools for exact symbol/path lookups.\n"
        "3. Use tldrf tools for concept lookup, impact, slicing, or data-flow when available.\n"
        "4. Do not invent files, symbols, or tool arguments.\n"
        "5. When you have enough evidence, return kind=final with a concise answer.\n\n"
        f"Instruction surface:\n{instruction_text[:4000]}\n\n"
        f"Task:\n{task_json}\n\n"
        f"Available tools:\n{_tool_descriptions(tools)}\n\n"
        f"Turn: {turn_index}\n"
        f"History:\n{history_blob}\n"
    )


def execute_tool(
    *,
    runtime: ProviderRuntime,
    tool_name: str,
    args: dict[str, Any],
    workspace_root: Path,
    default_test_command: str | None = None,
    timeout_s: float = 120.0,
) -> ToolInvocation:
    tool = normalize_tool_name(tool_name)
    env = _tool_env(runtime, workspace_root)
    max_results = max(1, int(args.get("max_results", DEFAULT_TOOL_MAX_LINES)))
    workspace_in_container = runtime.container_workspace_root
    sandbox_fs_script = f"{runtime.container_repo_root}/scripts/bench_sandbox_fs.py"

    try:
        if tool == "rg":
            pattern = str(args.get("pattern") or "")
            if not pattern:
                raise ValueError("pattern is required")
            cmd = ["rg", "-n", "--no-heading", "--color", "never", pattern, "."]
            glob = args.get("glob")
            if isinstance(glob, str) and glob.strip():
                cmd[1:1] = ["--glob", glob]
            code, stdout, stderr = _run_cmd(runtime, cmd, cwd=workspace_root, timeout_s=timeout_s, env=env)
            lines = (stdout.splitlines() or stderr.splitlines())[:max_results]
            obs = "\n".join(lines) or f"(exit={code})"
            return ToolInvocation(tool, dict(args), _clip(obs), code == 0)
        if tool == "grep":
            pattern = str(args.get("pattern") or "")
            if not pattern:
                raise ValueError("pattern is required")
            cmd = ["grep", "-R", "-n", "-E", pattern, "."]
            code, stdout, stderr = _run_cmd(runtime, cmd, cwd=workspace_root, timeout_s=timeout_s, env=env)
            lines = (stdout.splitlines() or stderr.splitlines())[:max_results]
            obs = "\n".join(lines) or f"(exit={code})"
            return ToolInvocation(tool, dict(args), _clip(obs), code == 0)
        if tool == "read_file":
            path = resolve_workspace_path(workspace_root, str(args.get("path") or ""))
            payload = {
                "path": container_workspace_path(runtime, workspace_root=workspace_root, path=path),
                "start_line": int(args["start_line"]) if args.get("start_line") is not None else None,
                "end_line": int(args["end_line"]) if args.get("end_line") is not None else None,
            }
            code, stdout, stderr = _run_cmd(
                runtime,
                ["python", sandbox_fs_script, "read-window"],
                cwd=workspace_root,
                timeout_s=timeout_s,
                env=env,
                input_text=json.dumps(payload, sort_keys=True),
            )
            obs = stdout or stderr or f"(exit={code})"
            return ToolInvocation(tool, dict(args), _clip(obs), code == 0)
        if tool == "run_tests":
            command = str(args.get("command") or default_test_command or "").strip()
            if not command:
                raise ValueError("command is required")
            code, stdout, stderr = _run_cmd(
                runtime,
                command,
                cwd=workspace_root,
                timeout_s=timeout_s,
                env=env,
                shell=True,
            )
            obs = f"exit_code={code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            return ToolInvocation(tool, dict(args), _clip(obs), code == 0)
        if tool == "replace_text":
            path = resolve_workspace_path(workspace_root, str(args.get("path") or ""))
            payload = {
                "path": container_workspace_path(runtime, workspace_root=workspace_root, path=path),
                "old": str(args.get("old") or ""),
                "new": str(args.get("new") or ""),
                "count": int(args.get("count", 1)),
            }
            code, stdout, stderr = _run_cmd(
                runtime,
                ["python", sandbox_fs_script, "replace-text"],
                cwd=workspace_root,
                timeout_s=timeout_s,
                env=env,
                input_text=json.dumps(payload, sort_keys=True),
            )
            obs = stdout or stderr or f"(exit={code})"
            return ToolInvocation(tool, dict(args), _clip(obs), code == 0)
        if tool == "write_file":
            path = resolve_workspace_path(workspace_root, str(args.get("path") or ""))
            payload = {
                "path": container_workspace_path(runtime, workspace_root=workspace_root, path=path),
                "content": str(args.get("content") or ""),
            }
            code, stdout, stderr = _run_cmd(
                runtime,
                ["python", sandbox_fs_script, "write-file"],
                cwd=workspace_root,
                timeout_s=timeout_s,
                env=env,
                input_text=json.dumps(payload, sort_keys=True),
            )
            obs = stdout or stderr or f"(exit={code})"
            return ToolInvocation(tool, dict(args), _clip(obs), code == 0)
        if tool == "git_diff":
            names_only = bool(args.get("names_only"))
            cmd = ["git", "diff", "--name-only"] if names_only else ["git", "diff", "--stat"]
            code, stdout, stderr = _run_cmd(runtime, cmd, cwd=workspace_root, timeout_s=timeout_s, env=env)
            obs = stdout or stderr or f"(exit={code})"
            return ToolInvocation(tool, dict(args), _clip(obs), code == 0)

        # tldrf tools
        cache_flags = ["--cache-root", f"{workspace_in_container}/.tldr-cache", "--index", f"repo:{workspace_root.name}"]
        if tool == "tldrf_context":
            entry = str(args.get("entry") or "")
            if not entry:
                raise ValueError("entry is required")
            cmd = [
                "tldrf",
                *cache_flags,
                "context",
                entry,
                "--project",
                workspace_in_container,
                "--depth",
                str(int(args.get("depth", 2))),
                "--lang",
                str(args.get("lang") or "python"),
            ]
        elif tool == "tldrf_impact":
            func = str(args.get("func") or args.get("function") or "")
            if not func:
                raise ValueError("func is required")
            cmd = [
                "tldrf",
                *cache_flags,
                "impact",
                func,
                "--project",
                workspace_in_container,
                "--depth",
                str(int(args.get("depth", 3))),
                "--lang",
                str(args.get("lang") or "python"),
            ]
            file_filter = args.get("file")
            if isinstance(file_filter, str) and file_filter.strip():
                file_path = resolve_workspace_path(workspace_root, file_filter)
                cmd.extend(["--file", str(file_path.relative_to(workspace_root).as_posix())])
        elif tool == "tldrf_slice":
            file_path = resolve_workspace_path(workspace_root, str(args.get("file") or ""))
            cmd = [
                "tldrf",
                *cache_flags,
                "slice",
                container_workspace_path(runtime, workspace_root=workspace_root, path=file_path),
                str(args.get("function") or ""),
                str(int(args.get("line"))),
                "--direction",
                str(args.get("direction") or "backward"),
            ]
            if args.get("var") is not None:
                cmd.extend(["--var", str(args.get("var"))])
            if args.get("lang") is not None:
                cmd.extend(["--lang", str(args.get("lang"))])
        elif tool == "tldrf_dfg":
            file_path = resolve_workspace_path(workspace_root, str(args.get("file") or ""))
            cmd = [
                "tldrf",
                *cache_flags,
                "dfg",
                container_workspace_path(runtime, workspace_root=workspace_root, path=file_path),
                str(args.get("function") or ""),
            ]
            if args.get("lang") is not None:
                cmd.extend(["--lang", str(args.get("lang"))])
        elif tool in ("tldrf_semantic_search", "tldrf_hybrid_search"):
            query = str(args.get("query") or "")
            if not query:
                raise ValueError("query is required")
            cmd = [
                "tldrf",
                *cache_flags,
                "semantic",
                "search",
                query,
                "--path",
                workspace_in_container,
                "--k",
                str(int(args.get("k", 5))),
                "--lang",
                str(args.get("lang") or "python"),
            ]
            if tool == "tldrf_hybrid_search":
                cmd.append("--hybrid")
            rg_pattern = args.get("rg_pattern")
            if isinstance(rg_pattern, str) and rg_pattern.strip():
                cmd.extend(["--rg-pattern", rg_pattern])
        else:
            raise ValueError(f"unsupported tool: {tool_name}")

        code, stdout, stderr = _run_cmd(runtime, cmd, cwd=workspace_root, timeout_s=timeout_s, env=env)
        obs = stdout or stderr or f"(exit={code})"
        return ToolInvocation(tool, dict(args), _clip(obs), code == 0)
    except Exception as exc:
        return ToolInvocation(tool, dict(args), f"{type(exc).__name__}: {exc}", False)


def evaluate_tool_choice_metrics(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(per_task)
    if total == 0:
        return {}
    first_tool_hits = 0
    compliant_hits = 0
    required_tldrf_total = 0
    required_tldrf_hits = 0
    exact_total = 0
    exact_rg_hits = 0
    unnecessary_rate_values: list[float] = []
    dead_end_values: list[int] = []
    recovery_candidates = 0
    recovery_hits = 0
    turns_to_expected: list[int] = []

    for row in per_task:
        first = row.get("first_tool")
        expected_first = row.get("expected_first_tool")
        if first == expected_first:
            first_tool_hits += 1
        if row.get("workflow_compliant"):
            compliant_hits += 1
        tool_calls = [normalize_tool_name(x) for x in row.get("tool_calls", []) if isinstance(x, str)]
        expected_tool_set = {
            normalize_tool_name(x) for x in row.get("expected_tool_set", []) if isinstance(x, str)
        }
        if any(name.startswith("tldrf_") for name in expected_tool_set):
            required_tldrf_total += 1
            if any(name in expected_tool_set for name in tool_calls):
                required_tldrf_hits += 1
        if row.get("workflow_class") in {"exact_lookup", "exact_symbol_definition"}:
            exact_total += 1
            if first == "rg":
                exact_rg_hits += 1
        if tool_calls:
            unnecessary = [name for name in tool_calls if expected_tool_set and name not in expected_tool_set]
            unnecessary_rate_values.append(len(unnecessary) / len(tool_calls))
        dead_end_values.append(int(row.get("dead_end_turns", 0)))
        if expected_tool_set:
            for idx, name in enumerate(tool_calls, start=1):
                if name in expected_tool_set:
                    turns_to_expected.append(idx)
                    if first != expected_first:
                        recovery_candidates += 1
                        recovery_hits += 1
                    break
            else:
                if first != expected_first:
                    recovery_candidates += 1

    return {
        "correct_first_tool_rate": first_tool_hits / total,
        "workflow_compliance_rate": compliant_hits / total,
        "tldrf_usage_on_required_rate": (
            required_tldrf_hits / required_tldrf_total if required_tldrf_total else None
        ),
        "rg_first_on_exact_rate": exact_rg_hits / exact_total if exact_total else None,
        "tool_choice_accuracy": first_tool_hits / total,
        "unnecessary_tool_call_rate": statistics.mean(unnecessary_rate_values) if unnecessary_rate_values else 0.0,
        "dead_end_turn_rate": statistics.mean(dead_end_values) if dead_end_values else 0.0,
        "recovery_after_wrong_first_tool_rate": (
            recovery_hits / recovery_candidates if recovery_candidates else None
        ),
        "median_turns_before_first_appropriate_tool_use": (
            statistics.median(turns_to_expected) if turns_to_expected else None
        ),
    }


def run_tool_choice_task(
    *,
    task: dict[str, Any],
    provider: str,
    model: str,
    runtime: Any,
    instruction_text: str,
    workspace_root: Path,
    tool_map: dict[str, str],
    model_sandbox_dir: Path,
    transcript_path: Path,
    max_turns: int,
    timeout_s: float,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    tool_calls: list[str] = []
    t0 = time.monotonic()
    final_answer = ""
    model_errors = 0

    for turn in range(1, max_turns + 1):
        prompt = build_loop_prompt(
            instruction_text=instruction_text,
            task=task,
            turn_index=turn,
            tools=tool_map,
            history=history,
        )
        raw_text, usage = call_provider(
            provider=provider,
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            max_tokens=800,
            temperature=0.0,
            json_schema=MODEL_ACTION_SCHEMA,
            runtime=runtime,
            kimi_work_dir=model_sandbox_dir,
        )
        decision = _extract_json_object(raw_text)
        history.append({"kind": "assistant_decision", "turn": turn, "raw_text": raw_text, "usage": usage})
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "assistant_decision",
                        "task_id": task.get("id"),
                        "turn": turn,
                        "raw_text": raw_text,
                        "parsed": decision,
                        "usage": usage,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if not decision:
            model_errors += 1
            continue
        if decision.get("kind") == "final":
            final_answer = str(decision.get("answer") or "")
            break
        tool_name = normalize_tool_name(str(decision.get("tool") or ""))
        if tool_name not in tool_map:
            model_errors += 1
            history.append(
                {
                    "kind": "tool_result",
                    "turn": turn,
                    "tool_name": tool_name,
                    "observation": f"unsupported tool: {tool_name}",
                }
            )
            continue
        args = decision.get("args")
        invocation = execute_tool(
            runtime=runtime,
            tool_name=tool_name,
            args=args if isinstance(args, dict) else {},
            workspace_root=workspace_root,
            timeout_s=timeout_s,
        )
        tool_calls.append(invocation.name)
        history.append(
            {
                "kind": "tool_result",
                "turn": turn,
                "tool_name": invocation.name,
                "observation": invocation.observation,
                "ok": invocation.ok,
            }
        )
        with transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "tool_result",
                        "task_id": task.get("id"),
                        "turn": turn,
                        "tool_name": invocation.name,
                        "tool_args": invocation.args,
                        "observation": invocation.observation,
                        "ok": invocation.ok,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    elapsed = time.monotonic() - t0
    expected_tool_set = [normalize_tool_name(x) for x in task.get("expected_tool_set", []) if isinstance(x, str)]
    first_tool = tool_calls[0] if tool_calls else None
    forbidden_first = [normalize_tool_name(x) for x in task.get("forbidden_first_tool", []) if isinstance(x, str)]
    dead_end_turns = 0
    current_dead_end = 0
    for name in tool_calls:
        if expected_tool_set and name in expected_tool_set:
            current_dead_end = 0
        else:
            current_dead_end += 1
            dead_end_turns = max(dead_end_turns, current_dead_end)
    workflow_compliant = bool(tool_calls)
    if expected_tool_set:
        workflow_compliant = any(name in expected_tool_set for name in tool_calls)
    if forbidden_first and first_tool in forbidden_first:
        workflow_compliant = False
    max_allowed_dead_end_turns = int(task.get("max_allowed_dead_end_turns", 999))
    if dead_end_turns > max_allowed_dead_end_turns:
        workflow_compliant = False

    return {
        "task_id": task.get("id"),
        "workflow_class": task.get("workflow_class"),
        "question": task.get("question"),
        "transcript_path": str(transcript_path),
        "solve_rate": 1 if final_answer or tool_calls else 0,
        "turn_count": len(history),
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "first_tool": first_tool,
        "expected_first_tool": normalize_tool_name(str(task.get("expected_first_tool") or "")) or None,
        "expected_tool_set": expected_tool_set,
        "workflow_compliant": workflow_compliant,
        "dead_end_turns": dead_end_turns,
        "changed_files": [],
        "wall_clock_s": round(elapsed, 6),
        "first_pass_time_s": None,
        "final_answer": final_answer,
        "model_errors": model_errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run controlled multi-turn tool-choice benchmark tasks.")
    ap.add_argument("--tasks", required=True, help="Path to JSON/JSONL task suite.")
    ap.add_argument("--provider", required=True, help="Answer-model provider.")
    ap.add_argument("--model", required=True, help="Answer model id.")
    ap.add_argument("--instruction-source", required=True, help="Canonical instruction document path.")
    ap.add_argument("--arm", choices=["baseline", "augmented"], default="augmented")
    ap.add_argument("--repo-root", default=None, help="Override workspace root for all tasks.")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--out", required=True, help="Summary report output path.")
    ap.add_argument("--kimi-share-dir", default=None)
    ap.add_argument("--codex-home", default=None)
    ap.add_argument("--claude-home", default=None)
    ap.add_argument("--sandbox-image", default=None)
    ap.add_argument("--docker-binary", default="docker")
    args = ap.parse_args()

    repo_root = get_repo_root()
    tasks_path = Path(args.tasks).resolve()
    tasks_doc = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks = tasks_doc["tasks"] if isinstance(tasks_doc, dict) and isinstance(tasks_doc.get("tasks"), list) else load_jsonl(tasks_path)
    instruction_path = Path(args.instruction_source).resolve()
    instruction_text = instruction_path.read_text(encoding="utf-8", errors="replace")
    runtime = make_provider_runtime(
        repo_root=repo_root,
        provider=str(args.provider),
        codex_home=args.codex_home,
        claude_home=args.claude_home,
        kimi_share_dir=args.kimi_share_dir,
        docker_binary=str(args.docker_binary),
        sandbox_image=str(args.sandbox_image) if args.sandbox_image else None,
    )
    cli_repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    ts = now_utc_compact()
    out_path = Path(args.out).resolve()
    transcripts_root = out_path.parent / f"{out_path.stem}.transcripts"
    tools_available = list(default_tools(arm=str(args.arm)).keys())
    workspaces_root = bench_root(repo_root) / "agentic" / "workspaces" / "tool-choice" / ts
    per_task: list[dict[str, Any]] = []

    for task in tasks:
        if not isinstance(task, dict):
            continue
        for trial in range(max(1, int(args.trials))):
            source_workspace_root = _resolve_task_workspace(task, cli_repo_root=cli_repo_root, repo_root=repo_root)
            workspace_root = workspaces_root / str(task.get("id", "task")) / f"trial-{trial}"
            _copy_workspace(source_workspace_root, workspace_root)
            transcript_path = transcripts_root / f"{task.get('id', 'task')}-trial{trial}.jsonl"
            model_sandbox_dir = bench_root(repo_root) / "agentic" / "model-sandboxes" / "tool-choice" / str(
                task.get("id", "task")
            ) / f"trial-{trial}"
            if model_sandbox_dir.exists():
                shutil.rmtree(model_sandbox_dir)
            model_sandbox_dir.mkdir(parents=True, exist_ok=True)
            result = run_tool_choice_task(
                task=task,
                provider=str(args.provider),
                model=str(args.model),
                runtime=runtime,
                instruction_text=instruction_text,
                workspace_root=workspace_root,
                tool_map=default_tools(arm=str(args.arm)),
                model_sandbox_dir=model_sandbox_dir,
                transcript_path=transcript_path,
                max_turns=int(args.max_turns),
                timeout_s=float(args.timeout_s),
            )
            result["trial"] = trial
            per_task.append(result)

    metrics = evaluate_tool_choice_metrics(per_task)
    report = make_report(
        phase="agentic_tool_choice",
        meta=gather_meta(tldr_repo_root=repo_root),
        protocol={
            "tasks": str(tasks_path),
            "provider": str(args.provider),
            "model": str(args.model),
            "arm": str(args.arm),
                "max_turns": int(args.max_turns),
                "timeout_s": float(args.timeout_s),
                "trials": int(args.trials),
                "instruction_source": str(instruction_path),
                "sandbox_image": str(getattr(runtime, "sandbox_image", None)),
                "docker_binary": str(getattr(runtime, "docker_binary", None)),
            },
        results={
            "task_count": len(per_task),
            **metrics,
            "per_task": per_task,
        },
    )
    report["arm"] = str(args.arm)
    report["instruction_surface"] = {
        "path": str(instruction_path),
        "sha256": sha256_file(instruction_path),
    }
    report["tools_available"] = tools_available
    report["task_suite_sha256"] = sha256_file(tasks_path)
    report["judge_config_hash"] = None
    write_report(out_path, report)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
