#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from bench_agentic_common import ProviderRuntime, call_provider, load_jsonl, make_provider_runtime, sha256_file
from bench_tool_choice import (
    MODEL_ACTION_SCHEMA,
    _extract_json_object,
    _resolve_task_workspace,
    default_tools,
    execute_tool,
)
from bench_util import bench_root, gather_meta, get_repo_root, make_report, now_utc_compact, write_report


def _copy_workspace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)


def _changed_files(*, runtime: ProviderRuntime, workspace_root: Path) -> list[str]:
    from bench_tool_choice import _run_cmd, _tool_env

    code, stdout, _ = _run_cmd(
        runtime,
        ["git", "diff", "--name-only"],
        cwd=workspace_root,
        timeout_s=60.0,
        env=_tool_env(runtime, workspace_root),
    )
    if code != 0:
        return []
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _set_metrics(expected: list[str], actual: list[str]) -> tuple[float, float]:
    expected_set = set(expected)
    actual_set = set(actual)
    if not expected_set and not actual_set:
        return 1.0, 1.0
    tp = len(expected_set & actual_set)
    precision = tp / len(actual_set) if actual_set else 1.0
    recall = tp / len(expected_set) if expected_set else 1.0
    return precision, recall


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for row in load_jsonl(path):
        task_id = row.get("task_id")
        if isinstance(task_id, str):
            completed.add(task_id)
    return completed


def _expected_exit_code(task: dict[str, Any]) -> int:
    value = task.get("expected_test_result", 0)
    if isinstance(value, dict):
        exit_code = value.get("exit_code", 0)
        return int(exit_code) if isinstance(exit_code, (int, float)) else 0
    return int(value)


def _build_patch_prompt(
    *,
    instruction_text: str,
    task: dict[str, Any],
    tools: dict[str, str],
    history: list[dict[str, Any]],
    turn_index: int,
) -> str:
    history_lines: list[str] = []
    for item in history[-10:]:
        if item.get("kind") == "tool_result":
            history_lines.append(f"{item.get('tool_name')}: {item.get('observation')}")
        elif item.get("kind") == "assistant_decision":
            history_lines.append(f"decision: {item.get('raw_text')}")
    history_blob = "\n".join(history_lines) if history_lines else "(no prior tool observations)"
    task_blob = json.dumps(
        {
            "id": task.get("id"),
            "category": task.get("category"),
            "issue_description": task.get("issue_description"),
            "expected_changed_files": task.get("expected_changed_files"),
            "hidden_test_command": task.get("hidden_test_command"),
        },
        sort_keys=True,
    )
    tool_lines = "\n".join(f"- {name}: {desc}" for name, desc in tools.items())
    return (
        "You are a benchmark-controlled patching assistant. "
        "Reply with JSON only. Choose exactly one next action.\n\n"
        "Rules:\n"
        "1. Use read/search tools to gather evidence.\n"
        "2. Use replace_text or write_file for edits.\n"
        "3. Run tests before finalizing.\n"
        "4. Return kind=final only after the patch is ready.\n\n"
        f"Instruction surface:\n{instruction_text[:4000]}\n\n"
        f"Task:\n{task_blob}\n\n"
        f"Available tools:\n{tool_lines}\n\n"
        f"Turn: {turn_index}\n"
        f"History:\n{history_blob}\n"
    )


def run_patch_task(
    *,
    task: dict[str, Any],
    provider: str,
    model: str,
    runtime: Any,
    instruction_text: str,
    workspace_root: Path,
    model_sandbox_dir: Path,
    transcript_path: Path,
    max_turns: int,
    timeout_s: float,
    arm: str,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    tool_map = default_tools(arm=arm, allow_edits=True)
    tool_calls: list[str] = []
    started = time.monotonic()
    first_pass_time_s: float | None = None
    last_test_ok = False
    error: str | None = None

    for turn in range(1, max_turns + 1):
        prompt = _build_patch_prompt(
            instruction_text=instruction_text,
            task=task,
            tools=tool_map,
            history=history,
            turn_index=turn,
        )
        raw_text, usage = call_provider(
            provider=provider,
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            max_tokens=1200,
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
            error = "invalid_json_decision"
            continue
        if decision.get("kind") == "final":
            break
        tool_name = str(decision.get("tool") or "")
        args = decision.get("args")
        invocation = execute_tool(
            runtime=runtime,
            tool_name=tool_name,
            args=args if isinstance(args, dict) else {},
            workspace_root=workspace_root,
            default_test_command=str(task.get("hidden_test_command") or ""),
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
        if invocation.name == "run_tests":
            expected_exit = _expected_exit_code(task)
            last_test_ok = invocation.observation.startswith(f"exit_code={expected_exit}")
            if last_test_ok and first_pass_time_s is None:
                first_pass_time_s = round(time.monotonic() - started, 6)

    if not last_test_ok and isinstance(task.get("hidden_test_command"), str):
        invocation = execute_tool(
            runtime=runtime,
            tool_name="run_tests",
            args={"command": task["hidden_test_command"]},
            workspace_root=workspace_root,
            default_test_command=str(task["hidden_test_command"]),
            timeout_s=timeout_s,
        )
        tool_calls.append(invocation.name)
        if invocation.ok and first_pass_time_s is None:
            first_pass_time_s = round(time.monotonic() - started, 6)
        expected_exit = _expected_exit_code(task)
        last_test_ok = invocation.observation.startswith(f"exit_code={expected_exit}")
        with transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "tool_result",
                        "task_id": task.get("id"),
                        "turn": max_turns + 1,
                        "tool_name": invocation.name,
                        "tool_args": invocation.args,
                        "observation": invocation.observation,
                        "ok": invocation.ok,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    changed_files = _changed_files(runtime=runtime, workspace_root=workspace_root)
    precision, recall = _set_metrics(
        [str(x) for x in task.get("expected_changed_files", []) if isinstance(x, str)],
        changed_files,
    )
    return {
        "task_id": task.get("id"),
        "category": task.get("category"),
        "transcript_path": str(transcript_path),
        "solve_rate": 1 if last_test_ok else 0,
        "turn_count": len(history),
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "changed_files": changed_files,
        "changed_file_precision": round(precision, 6),
        "changed_file_recall": round(recall, 6),
        "wall_clock_s": round(time.monotonic() - started, 6),
        "first_pass_time_s": first_pass_time_s,
        "workspace_root": str(workspace_root),
        "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run controlled patch/test benchmark tasks.")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=["baseline", "augmented"], required=True)
    ap.add_argument("--instruction-source", required=True)
    ap.add_argument("--repo-root", default=None, help="Override workspace root for all tasks.")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None, help="Skip completed task ids from a prior per-task JSONL.")
    ap.add_argument("--max-consecutive-errors", type=int, default=5)
    ap.add_argument("--max-error-rate-abort", type=float, default=0.30)
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
    out_path = Path(args.out).resolve()
    ts = now_utc_compact()
    workspaces_root = bench_root(repo_root) / "agentic" / "workspaces" / "patch" / ts
    transcripts_root = out_path.parent / f"{out_path.stem}.transcripts"
    answers_path = out_path.parent / f"{out_path.stem}.answers.jsonl"
    completed_ids = _load_completed_ids(Path(args.resume).resolve()) if args.resume else set()

    per_task: list[dict[str, Any]] = []
    consecutive_errors = 0
    error_count = 0

    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id") or "")
        if task_id in completed_ids:
            continue
        source_root = _resolve_task_workspace(task, cli_repo_root=cli_repo_root, repo_root=repo_root)
        workspace_root = workspaces_root / task_id
        _copy_workspace(source_root, workspace_root)
        model_sandbox_dir = bench_root(repo_root) / "agentic" / "model-sandboxes" / "patch" / task_id
        if model_sandbox_dir.exists():
            shutil.rmtree(model_sandbox_dir)
        model_sandbox_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcripts_root / f"{task_id}.jsonl"
        result = run_patch_task(
            task=task,
            provider=str(args.provider),
            model=str(args.model),
            runtime=runtime,
            instruction_text=instruction_text,
            workspace_root=workspace_root,
            model_sandbox_dir=model_sandbox_dir,
            transcript_path=transcript_path,
            max_turns=int(task.get("max_turns") or args.max_turns),
            timeout_s=float(task.get("timeout_s") or args.timeout_s),
            arm=str(args.arm),
        )
        per_task.append(result)
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        with answers_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        if result.get("error"):
            error_count += 1
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        processed = len(per_task)
        if consecutive_errors >= int(args.max_consecutive_errors):
            break
        if processed and (error_count / processed) > float(args.max_error_rate_abort):
            break

    report = make_report(
        phase="agentic_patch_tasks",
        meta=gather_meta(tldr_repo_root=repo_root),
        protocol={
            "tasks": str(tasks_path),
            "provider": str(args.provider),
            "model": str(args.model),
            "arm": str(args.arm),
            "max_turns": int(args.max_turns),
                "timeout_s": float(args.timeout_s),
                "instruction_source": str(instruction_path),
                "answers_path": str(answers_path),
                "resume": str(args.resume) if args.resume else None,
                "sandbox_image": str(getattr(runtime, "sandbox_image", None)),
                "docker_binary": str(getattr(runtime, "docker_binary", None)),
            },
        results={
            "task_count": len(per_task),
            "solve_rate": statistics.mean([float(row["solve_rate"]) for row in per_task]) if per_task else 0.0,
            "changed_file_precision_mean": (
                statistics.mean([float(row["changed_file_precision"]) for row in per_task]) if per_task else 0.0
            ),
            "changed_file_recall_mean": (
                statistics.mean([float(row["changed_file_recall"]) for row in per_task]) if per_task else 0.0
            ),
            "error_count": error_count,
            "per_task": per_task,
        },
    )
    report["arm"] = str(args.arm)
    report["instruction_surface"] = {
        "path": str(instruction_path),
        "sha256": sha256_file(instruction_path),
    }
    report["tools_available"] = list(default_tools(arm=str(args.arm), allow_edits=True).keys())
    report["task_suite_sha256"] = sha256_file(tasks_path)
    report["judge_config_hash"] = None
    write_report(out_path, report)
    print(out_path)
    print(answers_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
