from __future__ import annotations

import argparse

from tldr.cli import build_parser


def _semantic_index_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    root_subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    semantic_parser = root_subparsers.choices["semantic"]
    semantic_subparsers = next(
        action
        for action in semantic_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return semantic_subparsers.choices["index"]


def test_semantic_index_model_flag_accepts_jina() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["semantic", "index", ".", "--model", "jina-code-0.5b"]
    )

    assert args.command == "semantic"
    assert args.action == "index"
    assert args.model == "jina-code-0.5b"


def test_semantic_index_help_mentions_jina() -> None:
    help_text = _semantic_index_parser().format_help()
    normalized = help_text.replace("-\n                        ", "-")
    assert "jina-code-0.5b" in normalized
