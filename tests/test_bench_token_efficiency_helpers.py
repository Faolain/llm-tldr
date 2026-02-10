import runpy
import sys
from pathlib import Path


def _load_mod():
    # Load script as a module dict without executing main().
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_token_efficiency.py")


def test_python_find_def_span_includes_decorators():
    mod = _load_mod()
    find_span = mod["_python_find_def_span"]
    lines = [
        "@decorator",
        "def foo(x):",
        "    a = 1",
        "    return a",
        "",
        "def bar():",
        "    pass",
    ]
    span = find_span(lines, function_name="foo")
    assert span == (1, 4)


def test_python_enclosing_def_span_scans_backwards():
    mod = _load_mod()
    enclosing = mod["_python_enclosing_def_span"]
    lines = [
        "def outer():",
        "    x = 1",
        "    def inner():",
        "        return x",
        "    return inner()",
        "",
        "def next_fn():",
        "    return 123",
    ]
    # Line inside outer but after nested inner.
    span = enclosing(lines, line_1based=5)
    assert span is not None
    start, end, name = span
    assert name == "outer"
    assert start == 1
    assert end == 5


def test_apply_budget_prefix_selection():
    mod = _load_mod()
    apply_budget = mod["_apply_budget"]

    pieces = ["one", "two", "three"]
    payload, toks, _bytes, used = apply_budget(pieces, budget_tokens=0)
    assert payload == ""
    assert toks == 0
    assert used == 0

    payload2, _toks2, _bytes2, used2 = apply_budget(pieces, budget_tokens=10_000)
    assert used2 == 3
    assert payload2 == "one\n\ntwo\n\nthree"


def test_select_window_within_span_respects_span_bounds():
    mod = _load_mod()
    select = mod["_select_window_within_span"]

    file_rel = "x.py"
    lines = [
        "def foo():",
        "    a = 1",
        "    b = 2",
        "    return a + b",
        "",
        "def bar():",
        "    return 0",
    ]
    payload, included, win = select(
        file_rel=file_rel,
        lines=lines,
        span=(1, 4),
        target_line=3,
        budget_tokens=10_000,
    )
    assert win["start"] == 1
    assert win["end"] == 4
    assert included == {1, 2, 3, 4}
    assert payload.startswith("# x.py:1-4")
