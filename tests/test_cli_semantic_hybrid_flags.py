from __future__ import annotations

from tldr.cli import build_parser


def test_semantic_search_hybrid_flags_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "semantic",
            "search",
            "query text",
            "--hybrid",
            "--no-result-guard",
            "rg_empty",
            "--rg-pattern",
            "^def\\s+items_for_result",
            "--rrf-k",
            "90",
            "--rg-glob",
            "*.py",
        ]
    )

    assert args.command == "semantic"
    assert args.action == "search"
    assert args.hybrid is True
    assert args.no_result_guard == "rg_empty"
    assert args.rg_pattern == "^def\\s+items_for_result"
    assert args.rg_glob == "*.py"
    assert args.rrf_k == 90


def test_semantic_search_hybrid_flags_default_to_legacy_behavior() -> None:
    parser = build_parser()
    args = parser.parse_args(["semantic", "search", "query text"])

    assert args.command == "semantic"
    assert args.action == "search"
    assert args.hybrid is False
    assert args.no_result_guard == "none"
    assert args.rg_pattern is None
    assert args.rg_glob is None
    assert args.rrf_k == 60
