from __future__ import annotations

import tldr.mcp_server as mcp_server


def test_mcp_semantic_compound_forwards_lane4_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_send(project: str | None, command: dict) -> dict:
        captured["project"] = project
        captured["command"] = command
        return {"status": "ok"}

    monkeypatch.setattr(mcp_server, "_send_command", _fake_send)

    out = mcp_server.semantic(
        "repo",
        "find auth checks",
        k=7,
        hybrid=True,
        compound_impact=True,
        impact_depth=4,
        impact_limit=2,
        impact_language="python",
    )

    assert out == {"status": "ok"}
    assert captured["project"] == "repo"
    command = captured["command"]
    assert isinstance(command, dict)
    assert command["cmd"] == "semantic"
    assert command["action"] == "search"
    assert command["query"] == "find auth checks"
    assert command["k"] == 7
    assert command["retrieval_mode"] == "hybrid"
    assert command["compound_impact"] is True
    assert command["impact_depth"] == 4
    assert command["impact_limit"] == 2
    assert command["impact_language"] == "python"
