from __future__ import annotations

import json
from types import SimpleNamespace

from tldr.daemon.core import TLDRDaemon


def test_load_semantic_config_accepts_jina_model(tmp_path) -> None:
    tldr_config = tmp_path / "config.json"
    tldr_config.write_text(json.dumps({"semantic": {"model": "jina-code-0.5b"}}))

    dummy = SimpleNamespace(
        _claude_settings_path=tmp_path / "settings.json",
        _tldr_config_path=tldr_config,
    )

    config = TLDRDaemon._load_semantic_config(dummy)

    assert config["model"] == "jina-code-0.5b"


def test_load_semantic_config_keeps_bge_default(tmp_path) -> None:
    dummy = SimpleNamespace(
        _claude_settings_path=tmp_path / "settings.json",
        _tldr_config_path=tmp_path / "config.json",
    )

    config = TLDRDaemon._load_semantic_config(dummy)

    assert config["model"] == "bge-large-en-v1.5"


def test_daemon_semantic_index_uses_configured_model(monkeypatch, tmp_path) -> None:
    import tldr.semantic as semantic

    captured = {}

    def _fake_build_semantic_index(project_path, **kwargs):
        captured["project_path"] = project_path
        captured.update(kwargs)
        return 7

    monkeypatch.setattr(semantic, "build_semantic_index", _fake_build_semantic_index)

    dummy = SimpleNamespace(
        project=tmp_path,
        _semantic_config={"model": "jina-code-0.5b"},
        _ignore_spec=None,
        index_paths=None,
        index_config=None,
        _workspace_root=None,
    )

    out = TLDRDaemon._handle_semantic(
        dummy,
        {"action": "index", "language": "python"},
    )

    assert out == {"status": "ok", "indexed": 7}
    assert captured["project_path"] == str(tmp_path)
    assert captured["lang"] == "python"
    assert captured["model"] == "jina-code-0.5b"
