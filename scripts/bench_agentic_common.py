from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from bench_util import bench_root
from tldr.stats import count_tokens


DEFAULT_KIMI_HEALTH_PROMPT = "Reply with exactly: OK"
DEFAULT_CLAUDE_HEALTH_PROMPT = "Reply with exactly: OK"
DEFAULT_KIMI_MODEL = "kimi-code/kimi-for-coding"
DEFAULT_SANDBOX_IMAGE = "llm-tldr-agentic-kimi:latest"
DEFAULT_DOCKER_BINARY = "docker"
DEFAULT_CONTAINER_REPO_ROOT = "/opt/llm-tldr-src"
DEFAULT_CONTAINER_WORKSPACE_ROOT = "/workspace"
DEFAULT_CONTAINER_KIMI_SEED_ROOT = "/run/kimi-seed"
DEFAULT_CONTAINER_CLAUDE_SEED_ROOT = "/run/claude-seed"
DEFAULT_CONTAINER_KIMI_SHARE_ROOT = "/home/agent/.kimi"
DEFAULT_CONTAINER_CLAUDE_HOME_ROOT = "/home/agent"
SANDBOX_MARKER_ENV = "TLDR_AGENTIC_SANDBOX"
CODEX_SOURCE_ENV = "TLDR_AGENTIC_CODEX_SOURCE_DIR"
CLAUDE_SOURCE_ENV = "TLDR_AGENTIC_CLAUDE_SOURCE_DIR"
KIMI_SOURCE_ENV = "TLDR_AGENTIC_KIMI_SOURCE_DIR"


@dataclass(frozen=True)
class ProviderRuntime:
    repo_root: Path
    codex_home: Path | None = None
    claude_home: Path | None = None
    claude_env: dict[str, str] | None = None
    kimi_share_dir: Path | None = None
    kimi_env: dict[str, str] | None = None
    docker_binary: str = DEFAULT_DOCKER_BINARY
    sandbox_image: str = DEFAULT_SANDBOX_IMAGE
    container_repo_root: str = DEFAULT_CONTAINER_REPO_ROOT
    container_workspace_root: str = DEFAULT_CONTAINER_WORKSPACE_ROOT
    container_kimi_seed_root: str = DEFAULT_CONTAINER_KIMI_SEED_ROOT
    container_claude_seed_root: str = DEFAULT_CONTAINER_CLAUDE_SEED_ROOT
    container_kimi_share_root: str = DEFAULT_CONTAINER_KIMI_SHARE_ROOT
    container_claude_home_root: str = DEFAULT_CONTAINER_CLAUDE_HOME_ROOT


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def load_json_obj(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"expected JSON object on line {lineno}: {path}")
            rows.append(obj)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_judge_config(path: Path) -> tuple[dict[str, Any], str]:
    obj = load_json_obj(path)
    return obj, sha256_json(obj)


def estimate_usage(prompt: str, text_out: str, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(usage or {})
    if merged.get("input_tokens") is None:
        merged["input_tokens"] = int(count_tokens(prompt))
        merged["input_tokens_estimated"] = True
    if merged.get("output_tokens") is None:
        merged["output_tokens"] = int(count_tokens(text_out))
        merged["output_tokens_estimated"] = True
    return merged


def aggregate_usage_totals(usages: list[dict[str, Any]]) -> dict[str, int | float | None]:
    input_total = 0
    output_total = 0
    cost_total = 0.0
    saw_input = False
    saw_output = False
    saw_cost = False
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        value = usage.get("input_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            input_total += int(value)
            saw_input = True
        value = usage.get("output_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            output_total += int(value)
            saw_output = True
        value = usage.get("total_cost_usd")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cost_total += float(value)
            saw_cost = True
    return {
        "total_input_tokens": input_total if saw_input else None,
        "total_output_tokens": output_total if saw_output else None,
        "total_cost_usd": round(cost_total, 6) if saw_cost else None,
    }


def _benchmark_owned_dir(*, repo_root: Path, path_value: str | None, default_name: str) -> Path:
    benchmark_root = bench_root(repo_root).resolve()
    candidate = Path(path_value).resolve() if path_value else (benchmark_root / default_name).resolve()
    try:
        candidate.relative_to(benchmark_root)
    except ValueError as exc:
        raise ValueError(f"runtime path must live under {benchmark_root}: {candidate}") from exc
    return candidate


def _require_benchmark_owned_path(*, repo_root: Path, path: Path, label: str) -> Path:
    benchmark_root = bench_root(repo_root).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(benchmark_root)
    except ValueError as exc:
        raise ValueError(f"{label} must live under {benchmark_root}: {resolved}") from exc
    return resolved


def prepare_codex_home(*, repo_root: Path, codex_home: str | None) -> Path:
    dst = _benchmark_owned_dir(repo_root=repo_root, path_value=codex_home, default_name="codex-home")
    dst.mkdir(parents=True, exist_ok=True)
    auth_dst = dst / "auth.json"
    if not auth_dst.exists():
        source_root = Path(os.environ.get(CODEX_SOURCE_ENV) or (Path.home() / ".codex"))
        auth_src = source_root / "auth.json"
        if auth_src.exists():
            shutil.copy2(auth_src, auth_dst)
    return dst


def prepare_claude_runtime(*, repo_root: Path, claude_home: str | None) -> tuple[Path, dict[str, str]]:
    dst = _benchmark_owned_dir(repo_root=repo_root, path_value=claude_home, default_name="claude-home")
    dst.mkdir(parents=True, exist_ok=True)
    bootstrap_claude_home_dir(dst)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(dst),
            "XDG_CONFIG_HOME": str(dst / ".config"),
            "XDG_DATA_HOME": str(dst / ".local" / "share"),
            "XDG_CACHE_HOME": str(dst / ".cache"),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_TELEMETRY": "1",
        }
    )
    return dst, env


def bootstrap_claude_home_dir(dst: Path, *, src_home: Path | None = None) -> None:
    src_root = (src_home or Path(os.environ.get(CLAUDE_SOURCE_ENV) or Path.home())).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    src_claude_json = src_root / ".claude.json"
    dst_claude_json = dst / ".claude.json"
    if src_claude_json.exists() and not dst_claude_json.exists():
        shutil.copy2(src_claude_json, dst_claude_json)
    src_claude_dir = src_root / ".claude"
    if not src_claude_dir.exists() and (src_root / "settings.json").exists():
        src_claude_dir = src_root
    dst_claude_dir = dst / ".claude"
    dst_claude_dir.mkdir(parents=True, exist_ok=True)
    for name in ("settings.json", "mcp-needs-auth-cache.json", "stats-cache.json"):
        src_path = src_claude_dir / name
        dst_path = dst_claude_dir / name
        if src_path.exists() and not dst_path.exists():
            shutil.copy2(src_path, dst_path)


def bootstrap_kimi_share_dir(dst: Path, *, src: Path | None = None) -> None:
    src_dir = src or Path(os.environ.get(KIMI_SOURCE_ENV) or (Path.home() / ".kimi"))
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.toml", "device_id", "kimi.json"):
        from_path = src_dir / name
        to_path = dst / name
        if from_path.exists() and not to_path.exists():
            shutil.copy2(from_path, to_path)
    src_credentials = src_dir / "credentials"
    dst_credentials = dst / "credentials"
    if src_credentials.exists() and not dst_credentials.exists():
        shutil.copytree(src_credentials, dst_credentials)


def prepare_kimi_runtime(*, repo_root: Path, kimi_share_dir: str | None) -> tuple[Path, dict[str, str]]:
    dst = _benchmark_owned_dir(repo_root=repo_root, path_value=kimi_share_dir, default_name="kimi-share")
    bootstrap_kimi_share_dir(dst)
    env = os.environ.copy()
    env["KIMI_SHARE_DIR"] = str(dst)
    return dst, env


def make_provider_runtime(
    *,
    repo_root: Path,
    provider: str,
    judge_provider: str | None = None,
    codex_home: str | None = None,
    claude_home: str | None = None,
    kimi_share_dir: str | None = None,
    docker_binary: str = DEFAULT_DOCKER_BINARY,
    sandbox_image: str = DEFAULT_SANDBOX_IMAGE,
) -> ProviderRuntime:
    judge = judge_provider or ""
    needs_codex = provider == "codex" or judge == "codex"
    needs_claude = provider in ("claude_cli", "claude_sdk") or judge in ("claude_cli", "claude_sdk")
    needs_kimi = provider == "kimi_cli" or judge == "kimi_cli"

    codex_dir = prepare_codex_home(repo_root=repo_root, codex_home=codex_home) if needs_codex else None
    claude_dir, claude_env = prepare_claude_runtime(repo_root=repo_root, claude_home=claude_home) if needs_claude else (None, None)
    kimi_dir, kimi_env = prepare_kimi_runtime(repo_root=repo_root, kimi_share_dir=kimi_share_dir) if needs_kimi else (None, None)

    return ProviderRuntime(
        repo_root=repo_root.resolve(),
        codex_home=codex_dir,
        claude_home=claude_dir,
        claude_env=claude_env,
        kimi_share_dir=kimi_dir,
        kimi_env=kimi_env,
        docker_binary=str(docker_binary or DEFAULT_DOCKER_BINARY),
        sandbox_image=str(sandbox_image or DEFAULT_SANDBOX_IMAGE),
    )


def _docker_mount(src: Path, dst: str, *, read_only: bool) -> list[str]:
    spec = f"type=bind,src={src.resolve()},dst={dst}"
    if read_only:
        spec += ",readonly"
    return [
        "--mount",
        spec,
    ]


def container_workspace_path(runtime: ProviderRuntime, *, workspace_root: Path, path: Path) -> str:
    rel = path.resolve().relative_to(workspace_root.resolve())
    return str(PurePosixPath(runtime.container_workspace_root) / PurePosixPath(rel.as_posix()))


def _select_auth_env(
    *keys: str,
    primary: dict[str, str] | None = None,
    secondary: dict[str, str] | None = None,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in (primary or {}, secondary or {}, os.environ):
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value:
                merged[key] = value
    return merged


def docker_run(
    *,
    runtime: ProviderRuntime,
    workspace_root: Path,
    command: list[str],
    timeout_s: float,
    input_text: str | None = None,
    extra_env: dict[str, str] | None = None,
    mount_repo: bool = True,
    workspace_read_only: bool = False,
    network_disabled: bool = False,
) -> subprocess.CompletedProcess[str]:
    workspace_root = _require_benchmark_owned_path(
        repo_root=runtime.repo_root,
        path=workspace_root,
        label="sandbox workspace",
    )
    workspace_root.mkdir(parents=True, exist_ok=True)
    repo_root = runtime.repo_root.resolve()
    uid = getattr(os, "getuid", lambda: 1000)()
    gid = getattr(os, "getgid", lambda: 1000)()
    cmd: list[str] = [
        runtime.docker_binary,
        "run",
        "--rm",
        "--interactive",
        "--user",
        f"{uid}:{gid}",
        "--workdir",
        runtime.container_workspace_root,
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "512",
        "--memory",
        "4g",
        "--cpus",
        "2",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=1g",
        "--tmpfs",
        "/home/agent:rw,nosuid,size=256m",
        "--env",
        f"{SANDBOX_MARKER_ENV}=1",
        "--env",
        "HOME=/home/agent",
        "--env",
        "XDG_CONFIG_HOME=/home/agent/.config",
        "--env",
        "XDG_DATA_HOME=/home/agent/.local/share",
        "--env",
        "XDG_CACHE_HOME=/home/agent/.cache",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        "NO_COLOR=1",
        "--env",
        "CI=1",
        "--env",
        f"PYTHONPATH={runtime.container_repo_root}",
        *_docker_mount(workspace_root, runtime.container_workspace_root, read_only=workspace_read_only),
    ]
    if network_disabled:
        cmd.extend(["--network", "none"])
    if mount_repo:
        cmd.extend(_docker_mount(repo_root, runtime.container_repo_root, read_only=True))
    for key, value in sorted((extra_env or {}).items()):
        cmd.extend(["--env", f"{key}={value}"])
    cmd.append(runtime.sandbox_image)
    cmd.extend(command)
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=float(timeout_s),
        check=False,
    )


def docker_run_kimi(
    *,
    runtime: ProviderRuntime,
    workspace_root: Path,
    command: list[str],
    timeout_s: float,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if runtime.kimi_share_dir is None:
        raise RuntimeError("Kimi runtime is not prepared")
    proc_env = {
        "KIMI_SHARE_DIR": runtime.container_kimi_share_root,
    }
    workspace_root = _require_benchmark_owned_path(
        repo_root=runtime.repo_root,
        path=workspace_root,
        label="kimi sandbox workspace",
    )
    workspace_root.mkdir(parents=True, exist_ok=True)
    kimi_share_dir = _require_benchmark_owned_path(
        repo_root=runtime.repo_root,
        path=runtime.kimi_share_dir,
        label="Kimi runtime directory",
    )
    container_command = [str(part) for part in command]
    for idx, part in enumerate(container_command[:-1]):
        if part in ("--work-dir", "-w"):
            container_command[idx + 1] = runtime.container_workspace_root
    uid = getattr(os, "getuid", lambda: 1000)()
    gid = getattr(os, "getgid", lambda: 1000)()
    cmd: list[str] = [
        runtime.docker_binary,
        "run",
        "--rm",
        "--interactive",
        "--user",
        f"{uid}:{gid}",
        "--workdir",
        runtime.container_workspace_root,
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "512",
        "--memory",
        "4g",
        "--cpus",
        "2",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=1g",
        "--tmpfs",
        "/home/agent:rw,nosuid,size=256m",
        "--env",
        f"{SANDBOX_MARKER_ENV}=1",
        "--env",
        "HOME=/home/agent",
        "--env",
        "XDG_CONFIG_HOME=/home/agent/.config",
        "--env",
        "XDG_DATA_HOME=/home/agent/.local/share",
        "--env",
        "XDG_CACHE_HOME=/home/agent/.cache",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        "NO_COLOR=1",
        "--env",
        "CI=1",
        *_docker_mount(workspace_root, runtime.container_workspace_root, read_only=False),
        *_docker_mount(kimi_share_dir, runtime.container_kimi_seed_root, read_only=True),
    ]
    for key, value in sorted(proc_env.items()):
        cmd.extend(["--env", f"{key}={value}"])
    cmd.append(runtime.sandbox_image)
    cmd.extend(container_command)
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=float(timeout_s),
        check=False,
    )


def docker_run_claude(
    *,
    runtime: ProviderRuntime,
    workspace_root: Path,
    command: list[str],
    timeout_s: float,
    input_text: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if runtime.claude_home is None:
        raise RuntimeError("Claude runtime is not prepared")
    passthrough_env = _select_auth_env(
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_CLIENT_ID",
        "CLAUDE_CODE_CUSTOM_OAUTH_URL",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "CLAUDE_CODE_DISABLE_TELEMETRY",
        primary=extra_env,
        secondary=runtime.claude_env,
    )
    workspace_root = _require_benchmark_owned_path(
        repo_root=runtime.repo_root,
        path=workspace_root,
        label="Claude sandbox workspace",
    )
    workspace_root.mkdir(parents=True, exist_ok=True)
    claude_home = _require_benchmark_owned_path(
        repo_root=runtime.repo_root,
        path=runtime.claude_home,
        label="Claude runtime directory",
    )
    uid = getattr(os, "getuid", lambda: 1000)()
    gid = getattr(os, "getgid", lambda: 1000)()
    cmd: list[str] = [
        runtime.docker_binary,
        "run",
        "--rm",
        "--interactive",
        "--user",
        f"{uid}:{gid}",
        "--workdir",
        runtime.container_workspace_root,
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "512",
        "--memory",
        "4g",
        "--cpus",
        "2",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=1g",
        "--env",
        f"{SANDBOX_MARKER_ENV}=1",
        "--env",
        f"HOME={runtime.container_claude_home_root}",
        "--env",
        f"XDG_CONFIG_HOME={runtime.container_claude_home_root}/.config",
        "--env",
        f"XDG_DATA_HOME={runtime.container_claude_home_root}/.local/share",
        "--env",
        f"XDG_CACHE_HOME={runtime.container_claude_home_root}/.cache",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        "NO_COLOR=1",
        "--env",
        "CI=1",
        "--env",
        f"PYTHONPATH={runtime.container_repo_root}",
        "--env",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
        "--env",
        "CLAUDE_CODE_DISABLE_TELEMETRY=1",
        *_docker_mount(workspace_root, runtime.container_workspace_root, read_only=False),
        *_docker_mount(claude_home, runtime.container_claude_home_root, read_only=False),
    ]
    for key, value in sorted(passthrough_env.items()):
        cmd.extend(["--env", f"{key}={value}"])
    cmd.append(runtime.sandbox_image)
    cmd.extend(command)
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=float(timeout_s),
        check=False,
    )


def anthropic_call(*, model: str, prompt: str, max_tokens: int, temperature: float) -> tuple[str, dict[str, Any]]:
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


def codex_cli_call(
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
            text_out = proc.stdout or ""
        return text_out, {}


def claude_sdk_result_to_text_and_usage(msg: Any) -> tuple[str, dict[str, Any]] | None:
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
        usage = {
            "input_tokens": usage_obj.get("inputTokens") or usage_obj.get("input_tokens"),
            "output_tokens": usage_obj.get("outputTokens") or usage_obj.get("output_tokens"),
        }
    total_cost_usd = getattr(msg, "total_cost_usd", None)
    if total_cost_usd is not None:
        usage["total_cost_usd"] = total_cost_usd
    return text_out, usage


def claude_cli_call(
    *,
    runtime: ProviderRuntime,
    model: str,
    prompt: str,
    timeout_s: float,
    json_schema: dict[str, Any] | None,
    env: dict[str, str] | None,
    work_dir: Path,
    effort: str = "medium",
) -> tuple[str, dict[str, Any]]:
    output_format = "json" if json_schema is not None else "text"
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        "claude",
        "-p",
        "--output-format",
        output_format,
        "--model",
        str(model),
        "--effort",
        str(effort),
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
    ]
    if json_schema is not None:
        cmd.extend(["--json-schema", json.dumps(json_schema, sort_keys=True)])
    cmd.append(str(prompt))

    proc = docker_run_claude(
        runtime=runtime,
        workspace_root=work_dir,
        command=cmd,
        timeout_s=timeout_s,
        extra_env=env,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = stderr or stdout or f"claude failed with exit code {proc.returncode}"
        raise RuntimeError(msg)

    stdout = proc.stdout or ""
    if output_format != "json":
        return stdout, {}

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


def claude_health_check(
    *,
    runtime: ProviderRuntime,
    model: str,
    timeout_s: float,
    work_dir: Path,
    env: dict[str, str] | None = None,
    effort: str = "medium",
) -> None:
    try:
        text, _ = claude_cli_call(
            runtime=runtime,
            model=model,
            prompt=DEFAULT_CLAUDE_HEALTH_PROMPT,
            timeout_s=timeout_s,
            json_schema=None,
            env=env,
            work_dir=work_dir,
            effort=effort,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Claude health check timed out inside the sandbox. Containerized Claude auth is not available. "
            "Set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY, or provide a repo-local Claude home that does not rely "
            "on the host keychain."
        ) from exc
    if text.strip() != "OK":
        raise RuntimeError(f"Claude health check failed: expected 'OK', got {text!r}")


async def claude_agent_sdk_call_async(
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
        max_turns=2,
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
            res = claude_sdk_result_to_text_and_usage(msg)
            if res is None:
                continue
            text_out, usage = res
        return text_out, usage

    return await asyncio.wait_for(_run(), timeout=float(timeout_s))


def claude_agent_sdk_call(
    *,
    model: str,
    prompt: str,
    timeout_s: float,
    json_schema: dict[str, Any] | None,
    env: dict[str, str] | None,
) -> tuple[str, dict[str, Any]]:
    return asyncio.run(
        claude_agent_sdk_call_async(
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            json_schema=json_schema,
            env=env,
        )
    )


def _parse_kimi_stream_json(stdout: str) -> tuple[str, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    messages: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        events.append(obj)
        if obj.get("role") == "assistant" and isinstance(obj.get("content"), str):
            messages.append(str(obj["content"]))
    return ("\n".join(messages).strip(), events)


def kimi_cli_call(
    *,
    runtime: ProviderRuntime,
    model: str,
    prompt: str,
    timeout_s: float,
    json_schema: dict[str, Any] | None,
    work_dir: Path,
    final_message_only: bool = True,
    stream_json: bool = True,
) -> tuple[str, dict[str, Any]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    if json_schema is not None:
        schema_note = (
            "\n\nReturn JSON that conforms to this schema exactly:\n"
            + json.dumps(json_schema, sort_keys=True, indent=2)
        )
        prompt = f"{prompt}{schema_note}"

    cmd: list[str] = [
        "kimi",
        "--work-dir",
        str(work_dir),
        "--model",
        str(model or DEFAULT_KIMI_MODEL),
        "--print",
        "--input-format",
        "text",
        "--output-format",
        "stream-json" if stream_json else "text",
    ]
    if final_message_only:
        cmd.append("--final-message-only")

    proc = docker_run_kimi(
        runtime=runtime,
        workspace_root=work_dir,
        command=cmd,
        timeout_s=timeout_s,
        input_text=str(prompt),
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = stderr or stdout or f"kimi failed with exit code {proc.returncode}"
        raise RuntimeError(msg)

    stdout = proc.stdout or ""
    if stream_json:
        text_out, events = _parse_kimi_stream_json(stdout)
        usage = estimate_usage(prompt, text_out, {"raw_stdout": stdout, "event_count": len(events), "estimated": True})
        return text_out, usage
    usage = estimate_usage(prompt, stdout, {"estimated": True})
    return stdout, usage


def kimi_health_check(
    *,
    runtime: ProviderRuntime,
    model: str,
    timeout_s: float,
    work_dir: Path,
) -> None:
    text, _ = kimi_cli_call(
        runtime=runtime,
        model=model or DEFAULT_KIMI_MODEL,
        prompt=DEFAULT_KIMI_HEALTH_PROMPT,
        timeout_s=timeout_s,
        json_schema=None,
        work_dir=work_dir,
        final_message_only=True,
        stream_json=True,
    )
    if text.strip() != "OK":
        raise RuntimeError(f"Kimi health check failed: expected 'OK', got {text!r}")


def call_provider(
    *,
    provider: str,
    model: str,
    prompt: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
    json_schema: dict[str, Any] | None,
    runtime: ProviderRuntime,
    codex_profile: str | None = None,
    codex_reasoning_effort: str | None = None,
    kimi_work_dir: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    if provider == "codex":
        return codex_cli_call(
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            output_schema=json_schema,
            profile=codex_profile,
            reasoning_effort=codex_reasoning_effort,
            codex_home=runtime.codex_home,
        )
    if provider == "claude_sdk":
        return claude_agent_sdk_call(
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            json_schema=json_schema,
            env=runtime.claude_env,
        )
    if provider == "claude_cli":
        sandbox_dir = kimi_work_dir or (bench_root(runtime.repo_root) / "agentic" / "model-sandboxes" / "claude")
        return claude_cli_call(
            runtime=runtime,
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            json_schema=json_schema,
            env=runtime.claude_env,
            work_dir=sandbox_dir,
        )
    if provider == "anthropic":
        return anthropic_call(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    if provider == "kimi_cli":
        sandbox_dir = kimi_work_dir
        if sandbox_dir is None:
            sandbox_dir = Path(tempfile.mkdtemp(prefix="tldr-kimi-answer-"))
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        return kimi_cli_call(
            runtime=runtime,
            model=model or DEFAULT_KIMI_MODEL,
            prompt=prompt,
            timeout_s=timeout_s,
            json_schema=json_schema,
            work_dir=sandbox_dir,
            final_message_only=True,
            stream_json=True,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


def normalize_tool_name(name: str) -> str:
    raw = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "rg": "rg",
        "grep": "grep",
        "read_file": "read_file",
        "file_read": "read_file",
        "cat": "read_file",
        "run_tests": "run_tests",
        "test": "run_tests",
        "replace_text": "replace_text",
        "write_file": "write_file",
        "git_diff": "git_diff",
        "tldrf_search": "tldrf_search",
        "tldrf_context": "tldrf_context",
        "tldrf_impact": "tldrf_impact",
        "tldrf_slice": "tldrf_slice",
        "tldrf_dfg": "tldrf_dfg",
        "tldrf_semantic": "tldrf_semantic_search",
        "tldrf_semantic_search": "tldrf_semantic_search",
        "tldrf_hybrid": "tldrf_hybrid_search",
        "tldrf_hybrid_search": "tldrf_hybrid_search",
    }
    return aliases.get(raw, raw)
