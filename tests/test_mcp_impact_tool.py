from __future__ import annotations

import tldr.mcp_server as mcp_server


def test_mcp_impact_forwards_file_and_depth(monkeypatch) -> None:
    captured: list[tuple[str | None, dict]] = []

    def _fake_send(project: str | None, command: dict) -> dict:
        captured.append((project, dict(command)))
        return {"status": "ok", "callers": [], "result": {"targets": {}, "total_targets": 0}}

    monkeypatch.setattr(mcp_server, "_send_command", _fake_send)

    out = mcp_server.impact(
        "repo",
        "main",
        file="tldr/cli.py",
        depth=4,
        ensure_warm=False,
    )

    assert out["status"] == "ok"
    assert captured == [
        (
            "repo",
            {
                "cmd": "impact",
                "func": "main",
                "depth": 4,
                "file": "tldr/cli.py",
            },
        )
    ]


def test_mcp_impact_warms_and_retries_when_function_missing(monkeypatch) -> None:
    captured: list[tuple[str | None, dict]] = []
    responses = [
        {
            "status": "ok",
            "callers": [],
            "result": {"error": "Function 'main' not found in call graph"},
        },
        {"status": "ok", "files": 10, "edges": 99},
        {
            "status": "ok",
            "callers": [{"caller": "test_main", "file": "tests/test_cli.py", "line": 1}],
            "result": {"targets": {"tldr/cli.py:main": {"caller_count": 1}}, "total_targets": 1},
        },
    ]

    def _fake_send(project: str | None, command: dict) -> dict:
        captured.append((project, dict(command)))
        return responses.pop(0)

    monkeypatch.setattr(mcp_server, "_send_command", _fake_send)

    out = mcp_server.impact(
        "repo",
        "main",
        file="tldr/cli.py",
        depth=3,
        language="python",
        ensure_warm=True,
    )

    assert out["status"] == "ok"
    assert out["result"]["total_targets"] == 1
    assert captured == [
        ("repo", {"cmd": "impact", "func": "main", "depth": 3, "file": "tldr/cli.py"}),
        ("repo", {"cmd": "warm", "language": "python"}),
        ("repo", {"cmd": "impact", "func": "main", "depth": 3, "file": "tldr/cli.py"}),
    ]


def test_mcp_impact_does_not_warm_for_non_missing_errors(monkeypatch) -> None:
    captured: list[tuple[str | None, dict]] = []

    def _fake_send(project: str | None, command: dict) -> dict:
        captured.append((project, dict(command)))
        return {"status": "error", "message": "daemon unavailable"}

    monkeypatch.setattr(mcp_server, "_send_command", _fake_send)

    out = mcp_server.impact(
        "repo",
        "main",
        ensure_warm=True,
    )

    assert out["status"] == "error"
    assert captured == [("repo", {"cmd": "impact", "func": "main", "depth": 3})]
