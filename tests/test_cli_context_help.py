import pytest

from tldr.cli import build_parser


def test_context_help_includes_mode_dispatch_depth_and_examples(capsys) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["context", "-h"])

    out = capsys.readouterr().out
    assert "Dispatch rule" in out
    assert "architecture/surface browsing (module)" in out
    assert "debugging/refactor impact flow (symbol)" in out
    assert "Call depth for symbol mode" in out
    assert "module mode" in out
    assert "depth 0" in out
    assert "tldrf context providers/auth --project ." in out
    assert "tldrf context login --project ." in out
    assert "tldrf context pkg/mod.py --project ." in out
