import subprocess
import sys
from pathlib import Path

import pytest


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import bench_agentic_common as mod

    return mod


def test_prepare_kimi_runtime_bootstraps_credentials(tmp_path: Path, monkeypatch):
    mod = _load_mod()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source = tmp_path / "source-kimi"
    source.mkdir()
    (source / "config.toml").write_text('default_model = "kimi-code/kimi-for-coding"\n', encoding="utf-8")
    (source / "device_id").write_text("device-123\n", encoding="utf-8")
    (source / "kimi.json").write_text("{}", encoding="utf-8")
    (source / "credentials").mkdir()
    (source / "credentials" / "kimi-code.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)
    source.rename(tmp_path / ".kimi")
    dst_root = repo_root / "benchmark" / "dst-kimi"

    dst, env = mod.prepare_kimi_runtime(repo_root=repo_root, kimi_share_dir=str(dst_root))

    assert dst == dst_root.resolve()
    assert env["KIMI_SHARE_DIR"] == str(dst)
    assert (dst / "config.toml").exists()
    assert (dst / "credentials" / "kimi-code.json").exists()


def test_prepare_claude_runtime_bootstraps_home(tmp_path: Path, monkeypatch):
    mod = _load_mod()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source_home = tmp_path / "source-home"
    source_home.mkdir()
    (source_home / ".claude.json").write_text('{"oauthAccount": {"emailAddress": "user@example.com"}}\n', encoding="utf-8")
    (source_home / ".claude").mkdir()
    (source_home / ".claude" / "settings.json").write_text('{"permissions": {"allow": []}}\n', encoding="utf-8")

    monkeypatch.setattr(mod.Path, "home", lambda: source_home)
    dst_root = repo_root / "benchmark" / "dst-claude"

    dst, env = mod.prepare_claude_runtime(
        repo_root=repo_root,
        claude_home=str(dst_root),
    )

    assert dst == dst_root.resolve()
    assert env["HOME"] == str(dst)
    assert (dst / ".claude.json").exists()
    assert (dst / ".claude" / "settings.json").exists()


def test_prepare_claude_runtime_rejects_host_home_target(tmp_path: Path):
    mod = _load_mod()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "host-home"
    outside.mkdir()

    try:
        mod.prepare_claude_runtime(repo_root=repo_root, claude_home=str(outside))
    except ValueError as exc:
        assert "benchmark" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_prepare_kimi_runtime_rejects_host_home_target(tmp_path: Path):
    mod = _load_mod()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "host-kimi"
    outside.mkdir()

    try:
        mod.prepare_kimi_runtime(repo_root=repo_root, kimi_share_dir=str(outside))
    except ValueError as exc:
        assert "benchmark" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_kimi_cli_call_uses_stream_json_and_work_dir(monkeypatch, tmp_path: Path):
    mod = _load_mod()
    seen: dict[str, object] = {}

    def fake_docker_run_kimi(*, runtime, workspace_root, command, timeout_s, input_text):
        seen["runtime"] = runtime
        seen["workspace_root"] = workspace_root
        seen["command"] = command
        seen["input"] = input_text
        seen["timeout"] = timeout_s
        return subprocess.CompletedProcess(command, 0, stdout='{"role":"assistant","content":"OK"}\n', stderr="")

    monkeypatch.setattr(mod, "docker_run_kimi", fake_docker_run_kimi)

    runtime = mod.ProviderRuntime(
        repo_root=tmp_path,
        kimi_share_dir=tmp_path / "share",
        sandbox_image="llm-tldr-agentic-kimi:test",
    )

    text, usage = mod.kimi_cli_call(
        runtime=runtime,
        model="kimi-code/kimi-for-coding",
        prompt="Reply with exactly: OK",
        timeout_s=15.0,
        json_schema=None,
        work_dir=tmp_path / "sandbox",
    )

    assert text == "OK"
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert seen["timeout"] == 15.0
    assert seen["workspace_root"] == tmp_path / "sandbox"
    assert seen["runtime"] == runtime
    assert seen["command"] == [
        "kimi",
        "--work-dir",
        str((tmp_path / "sandbox")),
        "--model",
        "kimi-code/kimi-for-coding",
        "--print",
        "--input-format",
        "text",
        "--output-format",
        "stream-json",
        "--final-message-only",
    ]


def test_docker_run_kimi_rewrites_work_dir_for_container(monkeypatch, tmp_path: Path):
    mod = _load_mod()
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    runtime = mod.ProviderRuntime(
        repo_root=tmp_path,
        kimi_share_dir=tmp_path / "benchmark" / "kimi-share",
        sandbox_image="llm-tldr-agentic-kimi:test",
    )
    workspace_root = tmp_path / "benchmark" / "model-sandboxes" / "llm-ab" / "answer-model"
    runtime.kimi_share_dir.mkdir(parents=True, exist_ok=True)

    mod.docker_run_kimi(
        runtime=runtime,
        workspace_root=workspace_root,
        command=[
            "kimi",
            "--work-dir",
            str(workspace_root),
            "--model",
            "kimi-code/kimi-for-coding",
        ],
        timeout_s=15.0,
        input_text="OK",
    )

    cmd = seen["cmd"]
    assert cmd[0:2] == ["docker", "run"]
    assert cmd[-5:] == [
        "kimi",
        "--work-dir",
        runtime.container_workspace_root,
        "--model",
        "kimi-code/kimi-for-coding",
    ]


def test_claude_health_check_accepts_ok(monkeypatch, tmp_path: Path):
    mod = _load_mod()

    def fake_claude_cli_call(**kwargs):
        return "OK", {}

    monkeypatch.setattr(mod, "claude_cli_call", fake_claude_cli_call)

    runtime = mod.ProviderRuntime(
        repo_root=tmp_path,
        claude_home=tmp_path / "benchmark" / "claude-home",
        sandbox_image="llm-tldr-agentic-kimi:test",
    )

    mod.claude_health_check(
        runtime=runtime,
        model="sonnet",
        timeout_s=15.0,
        work_dir=tmp_path / "benchmark" / "model-sandboxes" / "judge-model",
    )


def test_claude_health_check_surfaces_auth_timeout(monkeypatch, tmp_path: Path):
    mod = _load_mod()

    def fake_claude_cli_call(**kwargs):
        raise subprocess.TimeoutExpired(cmd=["claude"], timeout=30.0)

    monkeypatch.setattr(mod, "claude_cli_call", fake_claude_cli_call)

    runtime = mod.ProviderRuntime(
        repo_root=tmp_path,
        claude_home=tmp_path / "benchmark" / "claude-home",
        sandbox_image="llm-tldr-agentic-kimi:test",
    )

    with pytest.raises(RuntimeError, match="Containerized Claude auth is not available"):
        mod.claude_health_check(
            runtime=runtime,
            model="sonnet",
            timeout_s=30.0,
            work_dir=tmp_path / "benchmark" / "model-sandboxes" / "judge-model",
        )


def test_aggregate_usage_totals_ignores_missing_values():
    mod = _load_mod()
    totals = mod.aggregate_usage_totals(
        [
            {"input_tokens": 10, "output_tokens": 5},
            {"input_tokens": 7, "output_tokens": 3, "total_cost_usd": 0.5},
            {"error": "boom"},
        ]
    )
    assert totals == {
        "total_input_tokens": 17,
        "total_output_tokens": 8,
        "total_cost_usd": 0.5,
    }


def test_sha256_json_is_stable():
    mod = _load_mod()
    left = mod.sha256_json({"b": 2, "a": 1})
    right = mod.sha256_json({"a": 1, "b": 2})
    assert left == right
    assert len(left) == 64
