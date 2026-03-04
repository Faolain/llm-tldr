import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tldr import cli


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(cli, "_show_first_run_tip", lambda: None)
    monkeypatch.setattr("sys.argv", ["tldr", *argv])
    cli.main()


def test_parser_accepts_json_without_cmd():
    parser = cli.build_parser()
    args = parser.parse_args(["daemon", "query", "--json", '{"cmd":"ping"}'])

    assert args.command == "daemon"
    assert args.action == "query"
    assert args.cmd is None
    assert args.json_payload == '{"cmd":"ping"}'


def test_daemon_query_json_payload_calls_query_daemon(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()

    mock_query = MagicMock(return_value={"status": "ok", "message": "pong"})
    monkeypatch.setattr("tldr.daemon.query_daemon", mock_query)

    _run_cli(
        monkeypatch,
        [
            "daemon",
            "query",
            "--json",
            '{"cmd":"ping"}',
            "--project",
            str(project),
        ],
    )

    called_project, called_command = mock_query.call_args.args[:2]
    assert called_project == project.resolve()
    assert called_command == {"cmd": "ping"}

    kwargs = mock_query.call_args.kwargs
    assert kwargs["index_ctx"] is None
    assert kwargs["cache_root"] is None
    assert kwargs["index_id"] is None

    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"


def test_daemon_query_simple_cmd_still_works(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()

    mock_query = MagicMock(return_value={"status": "ok"})
    monkeypatch.setattr("tldr.daemon.query_daemon", mock_query)

    _run_cli(
        monkeypatch,
        ["daemon", "query", "ping", "--project", str(project)],
    )

    _, called_command = mock_query.call_args.args[:2]
    assert called_command == {"cmd": "ping"}


def test_daemon_query_invalid_json_exits_1(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()

    mock_query = MagicMock()
    monkeypatch.setattr("tldr.daemon.query_daemon", mock_query)

    with pytest.raises(SystemExit) as exc:
        _run_cli(
            monkeypatch,
            [
                "daemon",
                "query",
                "--json",
                '{"cmd":',
                "--project",
                str(project),
            ],
        )

    assert exc.value.code == 1
    assert "invalid JSON for --json" in capsys.readouterr().err
    mock_query.assert_not_called()


def test_daemon_query_requires_cmd_or_json(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()

    mock_query = MagicMock()
    monkeypatch.setattr("tldr.daemon.query_daemon", mock_query)

    with pytest.raises(SystemExit) as exc:
        _run_cli(
            monkeypatch,
            ["daemon", "query", "--project", str(project)],
        )

    assert exc.value.code == 1
    assert "either CMD or --json must be provided" in capsys.readouterr().err
    mock_query.assert_not_called()


def test_daemon_query_json_takes_precedence_over_cmd(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()

    mock_query = MagicMock(return_value={"status": "ok"})
    monkeypatch.setattr("tldr.daemon.query_daemon", mock_query)

    _run_cli(
        monkeypatch,
        [
            "daemon",
            "query",
            "ping",
            "--json",
            '{"cmd":"status"}',
            "--project",
            str(project),
        ],
    )

    _, called_command = mock_query.call_args.args[:2]
    assert called_command == {"cmd": "status"}


def test_daemon_query_rejects_non_object_json(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()

    mock_query = MagicMock()
    monkeypatch.setattr("tldr.daemon.query_daemon", mock_query)

    with pytest.raises(SystemExit) as exc:
        _run_cli(
            monkeypatch,
            [
                "daemon",
                "query",
                "--json",
                "[]",
                "--project",
                str(project),
            ],
        )

    assert exc.value.code == 1
    assert "JSON payload must be an object" in capsys.readouterr().err
    mock_query.assert_not_called()


def test_daemon_query_rejects_json_without_cmd_key(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()

    mock_query = MagicMock()
    monkeypatch.setattr("tldr.daemon.query_daemon", mock_query)

    with pytest.raises(SystemExit) as exc:
        _run_cli(
            monkeypatch,
            [
                "daemon",
                "query",
                "--json",
                '{"action":"status"}',
                "--project",
                str(project),
            ],
        )

    assert exc.value.code == 1
    assert "JSON payload must include a non-empty 'cmd'" in capsys.readouterr().err
    mock_query.assert_not_called()


def test_daemon_query_json_preserves_index_kwargs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    cache_root = tmp_path / "cache"
    project.mkdir()
    cache_root.mkdir()

    index_ctx = SimpleNamespace(cache_root=cache_root.resolve(), index_id="dep:test")
    monkeypatch.setattr("tldr.indexing.get_index_context", lambda **_kwargs: index_ctx)

    mock_query = MagicMock(return_value={"status": "ok"})
    monkeypatch.setattr("tldr.daemon.query_daemon", mock_query)

    _run_cli(
        monkeypatch,
        [
            "daemon",
            "query",
            "--json",
            '{"cmd":"status"}',
            "--project",
            str(project),
            "--cache-root",
            str(cache_root),
            "--index",
            "dep:test",
        ],
    )

    kwargs = mock_query.call_args.kwargs
    assert kwargs["index_ctx"] is index_ctx
    assert kwargs["cache_root"] == cache_root.resolve()
    assert kwargs["index_id"] == "dep:test"
