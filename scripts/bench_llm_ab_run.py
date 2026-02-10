#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

from bench_util import (
    bench_root,
    bench_runs_root,
    gather_meta,
    get_repo_root,
    make_report,
    now_utc_compact,
    percentiles,
    write_report,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Score:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        # Treat "no predicted positives" as perfect precision (no false positives).
        # This also makes the empty/empty case score as perfect overall.
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        # Treat "no actual positives" as perfect recall (no false negatives).
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r) / (p + r) if (p + r) else 0.0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


_FENCE_RE = re.compile(r"^```(?:json)?\s*|```$", re.IGNORECASE | re.MULTILINE)


def _extract_json_from_text(text: str) -> Any | None:
    s = (text or "").strip()
    if not s:
        return None
    s = _FENCE_RE.sub("", s).strip()

    # If the whole thing parses, great.
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to locate a JSON object/array substring.
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
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


def _score_sets(expected: set[Any], got: set[Any]) -> Score:
    tp = len(expected & got)
    fp = len(got - expected)
    fn = len(expected - got)
    return Score(tp=tp, fp=fp, fn=fn)


def _expected_set(record: dict[str, Any]) -> set[Any] | None:
    exp = record.get("expected")
    cat = record.get("category")
    if cat == "impact":
        callers = exp
        if isinstance(exp, dict) and isinstance(exp.get("callers"), list):
            callers = exp.get("callers")
        if not isinstance(callers, list):
            return None
        out = set()
        for x in callers:
            if not isinstance(x, dict):
                continue
            f = x.get("file")
            fn = x.get("function")
            if isinstance(f, str) and isinstance(fn, str):
                out.add((f, fn))
        return out
    if cat == "slice":
        if not isinstance(exp, dict):
            return None
        lines = exp.get("lines")
        if not isinstance(lines, list):
            return None
        return {int(x) for x in lines if isinstance(x, int)}
    if cat == "data_flow":
        if not isinstance(exp, dict):
            return None
        flow = exp.get("flow")
        if not isinstance(flow, list):
            return None
        out = set()
        for x in flow:
            if not isinstance(x, dict):
                continue
            ln = x.get("line")
            ev = x.get("event")
            if isinstance(ln, int) and isinstance(ev, str):
                out.add((int(ln), ev))
        return out
    if cat == "retrieval":
        paths_obj = exp
        if isinstance(exp, dict) and isinstance(exp.get("paths"), list):
            paths_obj = exp.get("paths")
        if not isinstance(paths_obj, list):
            return None
        out: set[str] = set()
        for p in paths_obj:
            if not isinstance(p, str):
                continue
            s = p.strip().replace("\\", "/")
            if s.startswith("./"):
                s = s[2:]
            if s:
                out.add(s)
        return out
    return None


def _got_set(category: str, parsed: Any) -> set[Any] | None:
    if category == "impact":
        callers = parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("callers"), list):
            callers = parsed.get("callers")
        if not isinstance(callers, list):
            return None
        out = set()
        for x in callers:
            if not isinstance(x, dict):
                continue
            f = x.get("file")
            fn = x.get("function")
            if isinstance(f, str) and isinstance(fn, str):
                out.add((f, fn))
        return out
    if category == "slice":
        if isinstance(parsed, dict) and isinstance(parsed.get("lines"), list):
            return {int(x) for x in parsed["lines"] if isinstance(x, int)}
        return None
    if category == "data_flow":
        if isinstance(parsed, dict) and isinstance(parsed.get("flow"), list):
            out = set()
            for x in parsed["flow"]:
                if not isinstance(x, dict):
                    continue
                ln = x.get("line")
                ev = x.get("event")
                if isinstance(ln, int) and isinstance(ev, str):
                    out.add((int(ln), ev))
            return out
        return None
    if category == "retrieval":
        paths_obj = parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("paths"), list):
            paths_obj = parsed.get("paths")
        if not isinstance(paths_obj, list):
            return None
        out: set[str] = set()
        for p in paths_obj:
            if not isinstance(p, str):
                continue
            s = p.strip().replace("\\", "/")
            if s.startswith("./"):
                s = s[2:]
            if s:
                out.add(s)
        return out
    return None


def _json_schema_for_category(category: str) -> dict[str, Any] | None:
    if category == "impact":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "callers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"file": {"type": "string"}, "function": {"type": "string"}},
                        "required": ["file", "function"],
                    },
                }
            },
            "required": ["callers"],
        }
    if category == "slice":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {"lines": {"type": "array", "items": {"type": "integer"}}},
            "required": ["lines"],
        }
    if category == "data_flow":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "flow": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "line": {"type": "integer"},
                            "event": {"type": "string", "enum": ["defined", "used"]},
                        },
                        "required": ["line", "event"],
                    },
                }
            },
            "required": ["flow"],
        }
    if category == "retrieval":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
            "required": ["paths"],
        }
    return None


def _json_schema_for_judge_verdict() -> dict[str, Any]:
    score_obj = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "correctness": {"type": "integer", "minimum": 0, "maximum": 5},
            "groundedness": {"type": "integer", "minimum": 0, "maximum": 5},
            "completeness": {"type": "integer", "minimum": 0, "maximum": 5},
            "clarity": {"type": "integer", "minimum": 0, "maximum": 5},
            "actionability": {"type": "integer", "minimum": 0, "maximum": 5},
        },
        "required": ["correctness", "groundedness", "completeness", "clarity", "actionability"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B", "tie"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"A": score_obj, "B": score_obj},
                "required": ["A", "B"],
            },
            "notes": {"type": "string"},
        },
        "required": ["winner", "scores", "notes"],
    }


def _judge_prompt(
    *,
    question: str,
    rubric: str,
    context_a: str,
    answer_a: str,
    context_b: str,
    answer_b: str,
) -> str:
    schema_hint = (
        '{\n'
        '  "winner": "A" | "B" | "tie",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "scores": {\n'
        '    "A": {"correctness":0-5,"groundedness":0-5,"completeness":0-5,"clarity":0-5,"actionability":0-5},\n'
        '    "B": {"correctness":0-5,"groundedness":0-5,"completeness":0-5,"clarity":0-5,"actionability":0-5}\n'
        "  },\n"
        '  "notes": "short explanation"\n'
        "}"
    )
    return (
        "You are an impartial judge. Compare two answers (A vs B) to the same question.\n"
        "Use ONLY the provided contexts when judging correctness/groundedness.\n\n"
        f"Question:\n{question}\n\n"
        f"Rubric:\n{rubric}\n\n"
        "Variant A context:\n<BEGIN_CONTEXT_A>\n"
        f"{context_a}\n"
        "<END_CONTEXT_A>\n\n"
        "Variant A answer:\n<BEGIN_ANSWER_A>\n"
        f"{answer_a}\n"
        "<END_ANSWER_A>\n\n"
        "Variant B context:\n<BEGIN_CONTEXT_B>\n"
        f"{context_b}\n"
        "<END_CONTEXT_B>\n\n"
        "Variant B answer:\n<BEGIN_ANSWER_B>\n"
        f"{answer_b}\n"
        "<END_ANSWER_B>\n\n"
        f"Output JSON format:\n{schema_hint}\n\n"
        "Return ONLY the JSON verdict (no prose outside JSON)."
    )


def _extract_judge_winner(parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return None
    winner = parsed.get("winner")
    if winner in ("A", "B", "tie"):
        return str(winner)
    return None


def _anthropic_call(*, model: str, prompt: str, max_tokens: int, temperature: float) -> tuple[str, dict[str, Any]]:
    try:
        from anthropic import Anthropic
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("anthropic package not installed") from exc

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY env var")

    client = Anthropic(api_key=key)
    resp = client.messages.create(
        model=str(model),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        messages=[{"role": "user", "content": str(prompt)}],
    )
    text = ""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text += str(getattr(block, "text", "") or "")
    usage = getattr(resp, "usage", None)
    usage_dict: dict[str, Any] = {}
    if usage is not None:
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
    return text, usage_dict


def _codex_cli_call(
    *,
    model: str,
    prompt: str,
    timeout_s: float,
    output_schema: dict[str, Any] | None,
    profile: str | None,
    reasoning_effort: str | None,
    codex_home: Path | None,
) -> tuple[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="tldr-bench-codex-") as td:
        td_path = Path(td)
        last_msg_path = td_path / "last_message.txt"
        schema_path = td_path / "schema.json"

        cmd: list[str] = [
            "codex",
            "exec",
            "-m",
            str(model),
            "--sandbox",
            "read-only",
            "--output-last-message",
            str(last_msg_path),
        ]
        if profile:
            cmd.extend(["--profile", str(profile)])
        if reasoning_effort:
            cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        if output_schema is not None:
            schema_path.write_text(json.dumps(output_schema, sort_keys=True), encoding="utf-8")
            cmd.extend(["--output-schema", str(schema_path)])

        # Read prompt from stdin so we don't hit argv length limits.
        cmd.append("-")

        env = os.environ.copy()
        if codex_home is not None:
            env["CODEX_HOME"] = str(codex_home)
        proc = subprocess.run(
            cmd,
            input=str(prompt),
            text=True,
            capture_output=True,
            timeout=float(timeout_s),
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            msg = stderr or stdout or f"codex exec failed with exit code {proc.returncode}"
            raise RuntimeError(msg)

        text_out = ""
        if last_msg_path.exists():
            text_out = last_msg_path.read_text(encoding="utf-8", errors="replace")
        if not text_out.strip():
            # Best-effort fallback.
            text_out = proc.stdout or ""
        return text_out, {}


def _claude_sdk_result_to_text_and_usage(msg: Any) -> tuple[str, dict[str, Any]] | None:
    """Extract the final structured output from claude-agent-sdk messages.

    The SDK yields typed dataclass messages (UserMessage/AssistantMessage/SystemMessage/ResultMessage/StreamEvent).
    We only care about the terminal ResultMessage.
    """
    try:
        from claude_agent_sdk import ResultMessage
    except Exception:  # pragma: no cover
        return None

    if not isinstance(msg, ResultMessage):
        return None

    if getattr(msg, "is_error", False):
        err = str(getattr(msg, "result", "") or getattr(msg, "subtype", "") or "claude error")
        raise RuntimeError(err)

    structured = getattr(msg, "structured_output", None)
    if structured is not None:
        try:
            text_out = json.dumps(structured, sort_keys=True)
        except Exception:
            text_out = str(structured)
    else:
        text_out = str(getattr(msg, "result", "") or "")

    usage: dict[str, Any] = {}
    usage_obj = getattr(msg, "usage", None)
    if isinstance(usage_obj, dict):
        # Claude Code / SDK commonly uses camelCase keys.
        usage = {
            "input_tokens": usage_obj.get("inputTokens") or usage_obj.get("input_tokens"),
            "output_tokens": usage_obj.get("outputTokens") or usage_obj.get("output_tokens"),
        }
    total_cost_usd = getattr(msg, "total_cost_usd", None)
    if total_cost_usd is not None:
        usage["total_cost_usd"] = total_cost_usd
    return text_out, usage


def _claude_cli_call(
    *,
    model: str,
    prompt: str,
    timeout_s: float,
    json_schema: dict[str, Any] | None,
    env: dict[str, str] | None,
) -> tuple[str, dict[str, Any]]:
    # Disable tools so this behaves like a pure answer model over provided context.
    #
    # Note: Claude Code's --json-schema + --output-format=text produces an empty stdout
    # in practice; use --output-format=json and pull structured_output.
    output_format = "json" if json_schema is not None else "text"
    cmd: list[str] = [
        "claude",
        "--print",
        "--output-format",
        output_format,
        "--model",
        str(model),
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
    ]
    if json_schema is not None:
        cmd.extend(["--json-schema", json.dumps(json_schema, sort_keys=True)])

    # Pass as argv to avoid any stdin semantics ambiguity.
    cmd.append(str(prompt))

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=float(timeout_s),
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = stderr or stdout or f"claude failed with exit code {proc.returncode}"
        raise RuntimeError(msg)

    stdout = proc.stdout or ""
    if output_format != "json":
        return stdout, {}

    # output_format=json: stdout is a JSON envelope with usage + structured_output.
    try:
        obj = json.loads(stdout)
    except Exception:
        return stdout, {}

    structured = obj.get("structured_output")
    if structured is not None:
        try:
            text_out = json.dumps(structured, sort_keys=True)
        except Exception:
            text_out = str(structured)
    else:
        text_out = str(obj.get("result", "") or "")

    usage_obj = obj.get("usage") if isinstance(obj, dict) else None
    usage: dict[str, Any] = {}
    if isinstance(usage_obj, dict):
        usage = {
            "input_tokens": usage_obj.get("input_tokens"),
            "output_tokens": usage_obj.get("output_tokens"),
            "total_cost_usd": obj.get("total_cost_usd"),
        }
    return text_out, usage


async def _claude_agent_sdk_call_async(
    *,
    model: str,
    prompt: str,
    timeout_s: float,
    json_schema: dict[str, Any] | None,
    env: dict[str, str] | None,
) -> tuple[str, dict[str, Any]]:
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("claude-agent-sdk package not installed") from exc

    options = ClaudeAgentOptions(
        model=str(model),
        # Claude Code SDK often consumes a turn for internal/tool plumbing (even with tools=[]),
        # so max_turns=1 can terminate runs with subtype=error_max_turns and no output.
        max_turns=2,
        # This benchmark runner treats Claude as a pure answer model over provided context.
        tools=[],
        allowed_tools=[],
        disallowed_tools=[],
        include_partial_messages=False,
        permission_mode="bypassPermissions",
        output_format=({"type": "json_schema", "schema": json_schema} if json_schema is not None else None),
        env=dict(env or {}),
    )

    async def _run() -> tuple[str, dict[str, Any]]:
        text_out = ""
        usage: dict[str, Any] = {}
        async for msg in query(prompt=str(prompt), options=options):
            res = _claude_sdk_result_to_text_and_usage(msg)
            if res is None:
                continue
            text_out, usage = res
        return text_out, usage

    return await asyncio.wait_for(_run(), timeout=float(timeout_s))


def _claude_agent_sdk_call(
    *,
    model: str,
    prompt: str,
    timeout_s: float,
    json_schema: dict[str, Any] | None,
    env: dict[str, str] | None,
) -> tuple[str, dict[str, Any]]:
    return asyncio.run(
        _claude_agent_sdk_call_async(
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            json_schema=json_schema,
            env=env,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 7: run A/B prompt packets and score structured answers or judge open-ended tasks."
    )
    ap.add_argument("--prompts", required=True, help="Path to JSONL prompt packets (from bench_llm_ab_prompts.py).")
    ap.add_argument(
        "--mode",
        choices=["structured", "judge"],
        default="structured",
        help='Scoring mode: "structured" (deterministic PRF vs expected) or "judge" (open-ended tasks).',
    )
    ap.add_argument("--provider", choices=["codex", "claude_sdk", "claude_cli", "anthropic"], default="codex")
    ap.add_argument("--model", required=True, help="Answer model name (e.g., claude-3-5-sonnet-20241022).")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=1, help="Trials per variant per task (default: 1).")
    ap.add_argument("--timeout-s", type=float, default=180.0, help="Per-call timeout in seconds (default: 180).")
    ap.add_argument(
        "--judge-provider",
        choices=["codex", "claude_sdk", "claude_cli", "anthropic"],
        default=None,
        help='Judge provider (required for --mode judge). If omitted, judge mode errors out.',
    )
    ap.add_argument("--judge-model", default=None, help="Judge model name (required for --mode judge).")
    ap.add_argument(
        "--judge-timeout-s",
        type=float,
        default=None,
        help="Per-judge-call timeout in seconds (default: --timeout-s).",
    )
    ap.add_argument("--judge-max-tokens", type=int, default=800)
    ap.add_argument("--judge-temperature", type=float, default=0.0)
    ap.add_argument(
        "--enforce-json-schema",
        action="store_true",
        help="Pass a JSON Schema to the provider when supported (claude_sdk/claude_cli/codex).",
    )
    ap.add_argument(
        "--codex-profile",
        default=None,
        help='Codex CLI config profile name (e.g. "medium_think"). Only used for --provider codex.',
    )
    ap.add_argument(
        "--codex-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help='Codex reasoning effort override (only used for --provider codex). Example: "medium".',
    )
    ap.add_argument(
        "--codex-home",
        default=None,
        help="Path used as CODEX_HOME for Codex CLI (isolates session files). Default: benchmark/codex-home.",
    )
    ap.add_argument(
        "--judge-codex-profile",
        default=None,
        help='Codex CLI config profile name for the judge (only used for --judge-provider codex).',
    )
    ap.add_argument(
        "--judge-codex-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help='Codex reasoning effort override for the judge (only used for --judge-provider codex).',
    )
    ap.add_argument(
        "--claude-home",
        default=None,
        help=(
            "Home directory for Claude Code / Agent SDK state (isolates ~/.claude). "
            "Default: benchmark/claude-home (isolated; may require re-login). "
            'To reuse your existing Claude Code login, pass --claude-home "$HOME".'
        ),
    )
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N tasks (smoke testing).")
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs and exit without calling the model.")
    ap.add_argument("--out", default=None, help="Write report JSON to this path (default under benchmark/runs/).")
    ap.add_argument(
        "--answers-out",
        default=None,
        help="Write raw answers JSONL under benchmark/llm/ (default: benchmark/llm/<ts>-answers.jsonl).",
    )
    args = ap.parse_args()

    tldr_repo_root = get_repo_root()
    codex_home: Path | None = None
    needs_codex = args.provider == "codex" or args.judge_provider == "codex"
    if needs_codex:
        codex_home = (
            Path(args.codex_home).resolve()
            if args.codex_home
            else (bench_root(tldr_repo_root) / "codex-home").resolve()
        )
        codex_home.mkdir(parents=True, exist_ok=True)

        # Seed auth from the default home if missing. This avoids sandbox restrictions
        # on writing to ~/.codex while still using the existing login.
        auth_dst = codex_home / "auth.json"
        if not auth_dst.exists():
            auth_src = Path.home() / ".codex" / "auth.json"
            if auth_src.exists():
                shutil.copy2(auth_src, auth_dst)

    claude_home: Path | None = None
    claude_env: dict[str, str] | None = None
    needs_claude = args.provider in ("claude_cli", "claude_sdk") or args.judge_provider in ("claude_cli", "claude_sdk")
    if needs_claude:
        claude_home = (
            Path(args.claude_home).resolve()
            if args.claude_home
            else (bench_root(tldr_repo_root) / "claude-home").resolve()
        )
        claude_home.mkdir(parents=True, exist_ok=True)
        # Claude Code writes to ~/.claude and ~/.local/share/claude by default. In restricted
        # environments, redirect HOME/XDG_* into the repo to avoid EPERM.
        claude_env = os.environ.copy()
        claude_env.update(
            {
                "HOME": str(claude_home),
                "XDG_CONFIG_HOME": str(claude_home / ".config"),
                "XDG_DATA_HOME": str(claude_home / ".local" / "share"),
                "XDG_CACHE_HOME": str(claude_home / ".cache"),
            }
        )

    prompts_path = Path(args.prompts).resolve()
    records = _load_jsonl(prompts_path)
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]

    ts = now_utc_compact()
    llm_dir = bench_root(tldr_repo_root) / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    default_answers_name = (
        f"{ts}-llm-ab-answers.jsonl" if args.mode == "structured" else f"{ts}-llm-ab-answers-judge.jsonl"
    )
    answers_path = Path(args.answers_out).resolve() if args.answers_out else (llm_dir / default_answers_name)

    trials = max(1, int(args.trials))

    if args.mode == "judge" and (args.judge_provider is None or args.judge_model is None):
        raise SystemExit('error: --judge-provider and --judge-model are required for --mode "judge"')

    if args.dry_run:
        planned_tasks = 0
        planned_calls = 0
        for rec in records:
            variants = rec.get("variants")
            if not (isinstance(variants, list) and variants):
                continue
            if args.mode == "structured":
                if _expected_set(rec) is None:
                    continue
                planned_tasks += 1
                planned_calls += int(trials) * len(variants)
                continue
            if args.mode == "judge":
                if rec.get("task_type") != "open_ended":
                    continue
                planned_tasks += 1
                planned_calls += int(trials) * (len(variants) + 1)

        report = make_report(
            phase="phase7_llm_ab_run_structured" if args.mode == "structured" else "phase7_llm_ab_run_judge",
            meta=gather_meta(tldr_repo_root=tldr_repo_root),
            protocol={
                "schema_version": SCHEMA_VERSION,
                "prompts": str(prompts_path),
                "provider": args.provider,
                "model": str(args.model),
                "max_tokens": int(args.max_tokens),
                "temperature": float(args.temperature),
                "trials": trials,
                "timeout_s": float(args.timeout_s),
                "judge_provider": str(args.judge_provider) if args.judge_provider else None,
                "judge_model": str(args.judge_model) if args.judge_model else None,
                "judge_timeout_s": float(args.judge_timeout_s) if args.judge_timeout_s is not None else None,
                "judge_max_tokens": int(args.judge_max_tokens),
                "judge_temperature": float(args.judge_temperature),
                "enforce_json_schema": bool(args.enforce_json_schema),
                "dry_run": True,
                "limit": int(args.limit) if args.limit is not None else None,
                "codex_profile": str(args.codex_profile) if args.codex_profile else None,
                "codex_reasoning_effort": str(args.codex_reasoning_effort) if args.codex_reasoning_effort else None,
                "judge_codex_profile": str(args.judge_codex_profile) if args.judge_codex_profile else None,
                "judge_codex_reasoning_effort": str(args.judge_codex_reasoning_effort)
                if args.judge_codex_reasoning_effort
                else None,
                "codex_home": str(codex_home) if codex_home is not None else None,
                "claude_home": str(claude_home) if claude_home is not None else None,
                "answers_path": str(answers_path),
            },
            results={
                "tasks_planned": int(planned_tasks),
                "calls_planned": int(planned_calls),
                "notes": "Dry-run mode: no model calls were made; no answers JSONL was written.",
            },
        )

        if args.out:
            out_path = Path(args.out)
        else:
            suffix = "structured" if args.mode == "structured" else "judge"
            out_path = bench_runs_root(tldr_repo_root) / f"{ts}-llm-ab-run-{suffix}.json"
        write_report(out_path, report)
        print(out_path)
        return 0

    if args.mode == "judge":
        judge_provider = str(args.judge_provider)
        judge_model = str(args.judge_model)
        judge_timeout_s = float(args.judge_timeout_s) if args.judge_timeout_s is not None else float(args.timeout_s)

        per_task: list[dict[str, Any]] = []
        wins: list[float] = []
        time_s_by_source: dict[str, list[float]] = {"rg": [], "tldr": []}
        judge_time_s: list[float] = []

        answer_errors_total = 0
        answer_errors_by_source: dict[str, int] = {"rg": 0, "tldr": 0}
        judge_errors_total = 0
        judge_bad_json = 0

        score_by_source: dict[str, dict[str, list[float]]] = {"rg": {}, "tldr": {}}

        with answers_path.open("w", encoding="utf-8") as out_f:
            for rec in records:
                if rec.get("task_type") != "open_ended":
                    continue
                task_id = rec.get("task_id")
                category = rec.get("category")
                question = rec.get("question")
                rubric = rec.get("rubric")
                variants = rec.get("variants")
                if (
                    not isinstance(task_id, str)
                    or not isinstance(category, str)
                    or not isinstance(question, str)
                    or not isinstance(rubric, str)
                    or not isinstance(variants, list)
                ):
                    continue

                by_label: dict[str, dict[str, Any]] = {}
                for v in variants:
                    if not isinstance(v, dict):
                        continue
                    label = v.get("label")
                    if label in ("A", "B"):
                        by_label[str(label)] = v
                if "A" not in by_label or "B" not in by_label:
                    continue

                v_a = by_label["A"]
                v_b = by_label["B"]
                if not all(isinstance(v.get("prompt"), str) for v in (v_a, v_b)):
                    continue
                if not all(isinstance(v.get("context"), str) for v in (v_a, v_b)):
                    continue
                if not all(isinstance(v.get("source"), str) for v in (v_a, v_b)):
                    continue

                label_to_source = {"A": str(v_a["source"]), "B": str(v_b["source"])}

                task_trial_wins: list[float] = []
                trial_rows: list[dict[str, Any]] = []

                for trial in range(trials):
                    answers: dict[str, dict[str, Any]] = {}
                    for label, v in [("A", v_a), ("B", v_b)]:
                        source = str(v["source"])
                        prompt = str(v["prompt"])
                        t0 = time.monotonic()
                        error: str | None = None
                        try:
                            if args.provider == "codex":
                                text, usage = _codex_cli_call(
                                    model=str(args.model),
                                    prompt=prompt,
                                    timeout_s=float(args.timeout_s),
                                    output_schema=None,
                                    profile=str(args.codex_profile) if args.codex_profile else None,
                                    reasoning_effort=str(args.codex_reasoning_effort) if args.codex_reasoning_effort else None,
                                    codex_home=codex_home,
                                )
                            elif args.provider == "claude_sdk":
                                text, usage = _claude_agent_sdk_call(
                                    model=str(args.model),
                                    prompt=prompt,
                                    timeout_s=float(args.timeout_s),
                                    json_schema=None,
                                    env=claude_env,
                                )
                            elif args.provider == "claude_cli":
                                text, usage = _claude_cli_call(
                                    model=str(args.model),
                                    prompt=prompt,
                                    timeout_s=float(args.timeout_s),
                                    json_schema=None,
                                    env=claude_env,
                                )
                            elif args.provider == "anthropic":
                                text, usage = _anthropic_call(
                                    model=str(args.model),
                                    prompt=prompt,
                                    max_tokens=int(args.max_tokens),
                                    temperature=float(args.temperature),
                                )
                            else:  # pragma: no cover
                                raise RuntimeError(f"Unsupported provider: {args.provider}")
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            text = ""
                            usage = {"error": error}
                            answer_errors_total += 1
                            answer_errors_by_source[source] = int(answer_errors_by_source.get(source, 0)) + 1
                        dt = time.monotonic() - t0
                        time_s_by_source.setdefault(source, []).append(float(dt))

                        row = {
                            "kind": "answer",
                            "task_id": task_id,
                            "category": category,
                            "trial": trial,
                            "label": label,
                            "source": source,
                            "provider": str(args.provider),
                            "model": str(args.model),
                            "time_s": round(float(dt), 6),
                            "usage": usage,
                            "error": error,
                            "raw_text": text,
                        }
                        out_f.write(json.dumps(row, sort_keys=True) + "\n")
                        out_f.flush()
                        answers[label] = {"text": text, "usage": usage, "time_s": float(dt), "error": error}

                    judge_prompt = _judge_prompt(
                        question=question,
                        rubric=rubric,
                        context_a=str(v_a["context"]),
                        answer_a=str(answers.get("A", {}).get("text", "")),
                        context_b=str(v_b["context"]),
                        answer_b=str(answers.get("B", {}).get("text", "")),
                    )
                    judge_schema = _json_schema_for_judge_verdict() if args.enforce_json_schema else None

                    t0 = time.monotonic()
                    judge_error: str | None = None
                    try:
                        if judge_provider == "codex":
                            judge_text, judge_usage = _codex_cli_call(
                                model=judge_model,
                                prompt=judge_prompt,
                                timeout_s=judge_timeout_s,
                                output_schema=judge_schema,
                                profile=str(args.judge_codex_profile) if args.judge_codex_profile else None,
                                reasoning_effort=str(args.judge_codex_reasoning_effort)
                                if args.judge_codex_reasoning_effort
                                else None,
                                codex_home=codex_home,
                            )
                        elif judge_provider == "claude_sdk":
                            judge_text, judge_usage = _claude_agent_sdk_call(
                                model=judge_model,
                                prompt=judge_prompt,
                                timeout_s=judge_timeout_s,
                                json_schema=judge_schema,
                                env=claude_env,
                            )
                        elif judge_provider == "claude_cli":
                            judge_text, judge_usage = _claude_cli_call(
                                model=judge_model,
                                prompt=judge_prompt,
                                timeout_s=judge_timeout_s,
                                json_schema=judge_schema,
                                env=claude_env,
                            )
                        elif judge_provider == "anthropic":
                            judge_text, judge_usage = _anthropic_call(
                                model=judge_model,
                                prompt=judge_prompt,
                                max_tokens=int(args.judge_max_tokens),
                                temperature=float(args.judge_temperature),
                            )
                        else:  # pragma: no cover
                            raise RuntimeError(f"Unsupported judge provider: {judge_provider}")
                    except Exception as exc:
                        judge_error = f"{type(exc).__name__}: {exc}"
                        judge_text = ""
                        judge_usage = {"error": judge_error}
                        judge_errors_total += 1
                    judge_dt = time.monotonic() - t0
                    judge_time_s.append(float(judge_dt))

                    verdict_parsed = _extract_json_from_text(judge_text)
                    winner = _extract_judge_winner(verdict_parsed)
                    if winner is None:
                        judge_bad_json += 1
                        winner = "tie"

                    # Convert judge verdict into a TLDR-vs-rg win score.
                    if winner == "tie":
                        win = 0.5
                    else:
                        winner_source = label_to_source.get(winner)
                        if winner_source == "tldr":
                            win = 1.0
                        elif winner_source == "rg":
                            win = 0.0
                        else:
                            win = 0.5
                    task_trial_wins.append(float(win))

                    # Score distributions (optional): map A/B numeric scores to rg/tldr.
                    if isinstance(verdict_parsed, dict):
                        scores_obj = verdict_parsed.get("scores")
                        if isinstance(scores_obj, dict):
                            for label in ("A", "B"):
                                sc = scores_obj.get(label)
                                if not isinstance(sc, dict):
                                    continue
                                src = label_to_source.get(label)
                                if src not in ("rg", "tldr"):
                                    continue
                                for k, v in sc.items():
                                    if isinstance(v, (int, float)):
                                        score_by_source.setdefault(src, {}).setdefault(str(k), []).append(float(v))

                    judge_row = {
                        "kind": "judge",
                        "task_id": task_id,
                        "category": category,
                        "trial": trial,
                        "provider": judge_provider,
                        "model": judge_model,
                        "time_s": round(float(judge_dt), 6),
                        "usage": judge_usage,
                        "error": judge_error,
                        "winner": winner,
                        "label_to_source": label_to_source,
                        "win_tldr_over_rg": win,
                        "raw_text": judge_text,
                    }
                    out_f.write(json.dumps(judge_row, sort_keys=True) + "\n")
                    out_f.flush()

                    trial_rows.append(
                        {
                            "trial": trial,
                            "winner": winner,
                            "win_tldr_over_rg": win,
                            "judge_error": judge_error,
                        }
                    )

                win_mean = statistics.mean(task_trial_wins) if task_trial_wins else 0.5
                wins.append(float(win_mean))
                per_task.append(
                    {
                        "task_id": task_id,
                        "category": category,
                        "variants": [
                            {"label": "A", "source": label_to_source["A"]},
                            {"label": "B", "source": label_to_source["B"]},
                        ],
                        "win_mean_tldr_over_rg": float(win_mean),
                        "trials": trial_rows,
                    }
                )

        report = make_report(
            phase="phase7_llm_ab_run_judge",
            meta=gather_meta(tldr_repo_root=tldr_repo_root),
            protocol={
                "schema_version": SCHEMA_VERSION,
                "prompts": str(prompts_path),
                "provider": args.provider,
                "model": str(args.model),
                "trials": trials,
                "timeout_s": float(args.timeout_s),
                "judge_provider": judge_provider,
                "judge_model": judge_model,
                "judge_timeout_s": judge_timeout_s,
                "judge_max_tokens": int(args.judge_max_tokens),
                "judge_temperature": float(args.judge_temperature),
                "enforce_json_schema": bool(args.enforce_json_schema),
                "limit": int(args.limit) if args.limit is not None else None,
                "codex_profile": str(args.codex_profile) if args.codex_profile else None,
                "codex_reasoning_effort": str(args.codex_reasoning_effort) if args.codex_reasoning_effort else None,
                "judge_codex_profile": str(args.judge_codex_profile) if args.judge_codex_profile else None,
                "judge_codex_reasoning_effort": str(args.judge_codex_reasoning_effort)
                if args.judge_codex_reasoning_effort
                else None,
                "codex_home": str(codex_home) if codex_home is not None else None,
                "claude_home": str(claude_home) if claude_home is not None else None,
                "answers_path": str(answers_path),
            },
            results={
                "tasks_judged": len(per_task),
                "answer_errors_total": int(answer_errors_total),
                "answer_errors_by_source": {k: int(v) for k, v in answer_errors_by_source.items()},
                "judge_errors_total": int(judge_errors_total),
                "judge_bad_json": int(judge_bad_json),
                "win_rate_tldr_over_rg": (sum(wins) / len(wins)) if wins else None,
                "answer_time_s_percentiles": {k: percentiles(v) for k, v in time_s_by_source.items() if v},
                "judge_time_s_percentiles": percentiles(judge_time_s) if judge_time_s else None,
                "judge_score_mean": {
                    src: {k: (statistics.mean(v) if v else None) for k, v in dims.items()}
                    for src, dims in score_by_source.items()
                },
                "per_task": per_task,
            },
        )

        if args.out:
            out_path = Path(args.out)
        else:
            out_path = bench_runs_root(tldr_repo_root) / f"{ts}-llm-ab-run-judge.json"
        write_report(out_path, report)
        print(out_path)
        print(answers_path)
        return 0

    per_task: list[dict[str, Any]] = []
    wins: list[float] = []
    win_by_pair: dict[str, list[float]] = {}
    win_by_pair_by_category: dict[str, dict[str, list[float]]] = {}
    f1_by_source: dict[str, list[float]] = {"rg": [], "tldr": []}
    time_s_by_source: dict[str, list[float]] = {"rg": [], "tldr": []}
    bad_json = 0
    errors_total = 0
    errors_by_source: dict[str, int] = {"rg": 0, "tldr": 0}

    with answers_path.open("w", encoding="utf-8") as out_f:
        for rec in records:
            task_id = rec.get("task_id")
            category = rec.get("category")
            variants = rec.get("variants")
            if not isinstance(task_id, str) or not isinstance(category, str) or not isinstance(variants, list):
                continue
            expected = _expected_set(rec)
            if expected is None:
                continue

            json_schema = _json_schema_for_category(category) if args.enforce_json_schema else None
            task_row: dict[str, Any] = {"task_id": task_id, "category": category, "variants": []}
            scores: dict[str, float] = {}

            for v in variants:
                if not isinstance(v, dict):
                    continue
                label = v.get("label")
                source = v.get("source")
                prompt = v.get("prompt")
                if not isinstance(label, str) or not isinstance(source, str) or not isinstance(prompt, str):
                    continue

                trial_scores: list[float] = []
                trial_rows: list[dict[str, Any]] = []
                for trial in range(trials):
                    t0 = time.monotonic()
                    error: str | None = None
                    try:
                        if args.provider == "codex":
                            text, usage = _codex_cli_call(
                                model=str(args.model),
                                prompt=prompt,
                                timeout_s=float(args.timeout_s),
                                output_schema=json_schema,
                                profile=str(args.codex_profile) if args.codex_profile else None,
                                reasoning_effort=str(args.codex_reasoning_effort)
                                if args.codex_reasoning_effort
                                else None,
                                codex_home=codex_home,
                            )
                        elif args.provider == "claude_sdk":
                            text, usage = _claude_agent_sdk_call(
                                model=str(args.model),
                                prompt=prompt,
                                timeout_s=float(args.timeout_s),
                                json_schema=json_schema,
                                env=claude_env,
                            )
                        elif args.provider == "claude_cli":
                            text, usage = _claude_cli_call(
                                model=str(args.model),
                                prompt=prompt,
                                timeout_s=float(args.timeout_s),
                                json_schema=json_schema,
                                env=claude_env,
                            )
                        elif args.provider == "anthropic":
                            text, usage = _anthropic_call(
                                model=str(args.model),
                                prompt=prompt,
                                max_tokens=int(args.max_tokens),
                                temperature=float(args.temperature),
                            )
                        else:  # pragma: no cover
                            raise RuntimeError(f"Unsupported provider: {args.provider}")
                    except Exception as exc:
                        # Keep the run going and score this trial as a miss.
                        error = f"{type(exc).__name__}: {exc}"
                        text = ""
                        usage = {"error": error}
                        errors_total += 1
                        errors_by_source[source] = int(errors_by_source.get(source, 0)) + 1
                    dt = time.monotonic() - t0
                    time_s_by_source.setdefault(source, []).append(float(dt))

                    parsed = _extract_json_from_text(text)
                    got = _got_set(category, parsed)
                    if got is None:
                        bad_json += 1
                        sc = Score(tp=0, fp=0, fn=len(expected))
                    else:
                        sc = _score_sets(expected, got)
                    trial_scores.append(sc.f1)

                    row = {
                        "task_id": task_id,
                        "label": label,
                        "source": source,
                        "trial": trial,
                        "provider": str(args.provider),
                        "model": str(args.model),
                        "time_s": round(float(dt), 6),
                        "usage": usage,
                        "error": error,
                        "f1": round(sc.f1, 6),
                        "precision": round(sc.precision, 6),
                        "recall": round(sc.recall, 6),
                        "raw_text": text,
                    }
                    out_f.write(json.dumps(row, sort_keys=True) + "\n")
                    out_f.flush()
                    trial_rows.append(row)

                f1_mean = statistics.mean(trial_scores) if trial_scores else 0.0
                scores[source] = float(f1_mean)
                f1_by_source.setdefault(source, []).append(float(f1_mean))
                task_row["variants"].append(
                    {
                        "label": label,
                        "source": source,
                        "f1_mean": f1_mean,
                        "trials": trial_rows,
                    }
                )

            # Pairwise win signals across all sources present on this task.
            srcs = sorted(scores.keys())
            for a in srcs:
                for b in srcs:
                    if a == b:
                        continue
                    if scores[a] > scores[b]:
                        w = 1.0
                    elif scores[a] < scores[b]:
                        w = 0.0
                    else:
                        w = 0.5
                    key = f"{a}_over_{b}"
                    win_by_pair.setdefault(key, []).append(float(w))
                    win_by_pair_by_category.setdefault(category, {}).setdefault(key, []).append(float(w))

            # A/B win signal (0/0.5/1): compare TLDR vs rg when both present.
            if "tldr" in scores and "rg" in scores:
                if scores["tldr"] > scores["rg"]:
                    wins.append(1.0)
                elif scores["tldr"] < scores["rg"]:
                    wins.append(0.0)
                else:
                    wins.append(0.5)
            per_task.append(task_row)

    report = make_report(
        phase="phase7_llm_ab_run_structured",
        meta=gather_meta(tldr_repo_root=tldr_repo_root),
        protocol={
            "schema_version": SCHEMA_VERSION,
            "prompts": str(prompts_path),
            "provider": args.provider,
            "model": str(args.model),
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "trials": trials,
            "timeout_s": float(args.timeout_s),
            "enforce_json_schema": bool(args.enforce_json_schema),
            "dry_run": bool(args.dry_run),
            "limit": int(args.limit) if args.limit is not None else None,
            "codex_profile": str(args.codex_profile) if args.codex_profile else None,
            "codex_reasoning_effort": str(args.codex_reasoning_effort) if args.codex_reasoning_effort else None,
            "codex_home": str(codex_home) if codex_home is not None else None,
            "claude_home": str(claude_home) if claude_home is not None else None,
            "answers_path": str(answers_path),
        },
        results={
            "tasks_scored": len(per_task),
            "bad_json": int(bad_json),
            "errors_total": int(errors_total),
            "errors_by_source": {k: int(v) for k, v in errors_by_source.items()},
            "win_rate_tldr_over_rg": (sum(wins) / len(wins)) if wins else None,
            "win_rate_by_pair": {k: ((sum(v) / len(v)) if v else None) for k, v in win_by_pair.items()},
            "win_rate_by_pair_by_category": {
                cat: {k: ((sum(v) / len(v)) if v else None) for k, v in pairs.items()}
                for cat, pairs in win_by_pair_by_category.items()
            },
            "f1_mean": {k: (statistics.mean(v) if v else None) for k, v in f1_by_source.items()},
            "f1_percentiles": {k: percentiles(v) for k, v in f1_by_source.items() if v},
            "time_s_percentiles": {k: percentiles(v) for k, v in time_s_by_source.items() if v},
            "per_task": per_task,
        },
    )

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = bench_runs_root(tldr_repo_root) / f"{ts}-llm-ab-run-structured.json"
    write_report(out_path, report)
    print(out_path)
    print(answers_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
