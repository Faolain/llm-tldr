from tldr.cli import build_parser


def test_index_flags_parse_before_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        ["--cache-root", "/tmp/cache", "warm", "/tmp/project", "--index", "dep:test"]
    )
    assert args.cache_root == "/tmp/cache"
    assert args.index_id == "dep:test"
    assert args.command == "warm"


def test_index_flags_parse_after_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        ["warm", "/tmp/project", "--cache-root", "/tmp/cache", "--index", "dep:test"]
    )
    assert args.cache_root == "/tmp/cache"
    assert args.index_id == "dep:test"
    assert args.command == "warm"
