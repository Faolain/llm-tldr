from __future__ import annotations

import argparse

from tldr.cli import build_parser


def _semantic_search_option_strings() -> set[str]:
    parser = build_parser()
    root_subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    semantic_parser = root_subparsers.choices["semantic"]
    semantic_subparsers = next(
        action for action in semantic_parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    search_parser = semantic_subparsers.choices["search"]
    option_strings: set[str] = set()
    for action in search_parser._actions:
        option_strings.update(action.option_strings)
    return option_strings


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
    assert args.budget_tokens is None


def test_semantic_search_lane2_confidence_and_rerank_flags_exposed() -> None:
    option_strings = _semantic_search_option_strings()
    expected = {
        "--abstain-threshold",
        "--abstain-empty",
        "--rerank",
        "--rerank-top-n",
    }
    missing = expected - option_strings
    assert not missing, f"missing lane2 semantic search flags: {sorted(missing)}"


def test_semantic_search_lane2_regression_bound_flags_exposed() -> None:
    option_strings = _semantic_search_option_strings()
    expected = {
        "--max-latency-ms-p50-ratio",
        "--max-payload-tokens-median-ratio",
    }
    missing = expected - option_strings
    assert not missing, f"missing lane2 regression bound flags: {sorted(missing)}"


def test_semantic_search_lane3_budget_flag_exposed() -> None:
    option_strings = _semantic_search_option_strings()
    assert "--budget-tokens" in option_strings


def test_semantic_search_lane2_flags_parse_values() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "semantic",
            "search",
            "query text",
            "--hybrid",
            "--abstain-threshold",
            "0.65",
            "--abstain-empty",
            "--rerank",
            "--rerank-top-n",
            "8",
            "--max-latency-ms-p50-ratio",
            "1.15",
            "--max-payload-tokens-median-ratio",
            "1.20",
        ]
    )

    assert args.abstain_threshold == 0.65
    assert args.abstain_empty is True
    assert args.rerank is True
    assert args.rerank_top_n == 8
    assert args.max_latency_ms_p50_ratio == 1.15
    assert args.max_payload_tokens_median_ratio == 1.20


def test_semantic_search_lane3_budget_flag_parses_value() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "semantic",
            "search",
            "query text",
            "--hybrid",
            "--budget-tokens",
            "1500",
        ]
    )

    assert args.budget_tokens == 1500
