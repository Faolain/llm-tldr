from __future__ import annotations

import json
from unittest.mock import patch

from tldr.api import FunctionContext, RelevantContext
from tldr.daemon.cached_queries import cached_context
from tldr.salsa import SalsaDB


def test_cached_context_returns_llm_string_not_dict_repr() -> None:
    fake_context = RelevantContext(
        entry_point="pkg.auth.login",
        depth=2,
        functions=[
            FunctionContext(
                name="pkg.auth.login",
                file="pkg/auth.py",
                line=42,
                signature="def login(user, password):",
                docstring="Authenticate a user.",
                calls=["pkg.auth.validate"],
            )
        ],
    )

    with patch("tldr.api.get_relevant_context", return_value=fake_context):
        response = cached_context(
            SalsaDB(),
            project=".",
            entry="pkg.auth.login",
            language="python",
            depth=2,
            ignore_spec=None,
            workspace_root=None,
        )

    assert response["status"] == "ok"
    assert isinstance(response["result"], str)
    assert "Code Context" in response["result"]
    assert "'entry_point'" not in response["result"]
    json.dumps(response)
