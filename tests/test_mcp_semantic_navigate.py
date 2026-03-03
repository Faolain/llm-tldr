from __future__ import annotations

import tldr.mcp_server as mcp_server


def test_mcp_semantic_navigate_forwards_lane5_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_send(project: str | None, command: dict) -> dict:
        captured["project"] = project
        captured["command"] = command
        return {"status": "ok"}

    monkeypatch.setattr(mcp_server, "_send_command", _fake_send)

    out = mcp_server.semantic(
        "repo",
        "navigate auth checks",
        k=7,
        hybrid=True,
        navigate_cluster=True,
        budget_tokens=2000,
    )

    assert out == {"status": "ok"}
    assert captured["project"] == "repo"
    command = captured["command"]
    assert isinstance(command, dict)
    assert command["cmd"] == "semantic"
    assert command["action"] == "search"
    assert command["query"] == "navigate auth checks"
    assert command["k"] == 7
    assert command["retrieval_mode"] == "hybrid"
    assert command["navigate_cluster"] is True
    assert command["budget_tokens"] == 2000
