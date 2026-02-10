import runpy
import sys
from pathlib import Path


def _load_mod():
    # Load script as a module dict without executing main().
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_llm_ab_prompts.py")


def test_pack_open_ended_slice_context_adds_extra_windows_outside_target_window():
    mod = _load_mod()
    pack = mod["_pack_open_ended_slice_context"]

    file_rel = "x.py"
    lines = ["pass" for _ in range(1000)]
    target_line = 500
    slice_lines = [10, 990]

    meta, code = pack(
        file_rel=file_rel,
        lines=lines,
        function="foo",
        span=(1, 1000),
        target_line=target_line,
        slice_lines=slice_lines,
        budget_tokens=800,
    )

    tw = meta["target_window"]
    assert tw["start"] <= target_line <= tw["end"]
    assert meta["slice_window_radius"] == 3
    assert meta["strategy"] == "target_window_plus_slice_windows"

    extra = meta["extra_windows"]
    assert isinstance(extra, list)
    assert extra, "expected at least one extra window outside the target window"

    included = set(meta["slice_lines"])
    assert target_line in included
    assert 10 in included or 990 in included
    assert "10: pass" in code or "990: pass" in code

    # Prove we're including remote slice lines via extra windows, not because the target window swallowed the function.
    assert not (tw["start"] <= 10 <= tw["end"])
    assert not (tw["start"] <= 990 <= tw["end"])
    assert any(w["start"] <= 10 <= w["end"] for w in extra) or any(w["start"] <= 990 <= w["end"] for w in extra)


def test_pack_open_ended_slice_context_includes_related_definition_snippets():
    mod = _load_mod()
    pack = mod["_pack_open_ended_slice_context"]

    file_rel = "x.py"
    # helper() is outside foo() span and should be pulled in as a related definition.
    lines = [
        "def helper(x):",
        "    return x + 1",
        "",
        "def foo():",
        "    a = helper(1)",
        "    return a",
    ]
    meta, code = pack(
        file_rel=file_rel,
        lines=lines,
        function="foo",
        span=(4, 6),
        target_line=6,
        slice_lines=[5, 6],
        budget_tokens=800,
    )

    related = meta.get("related_definitions")
    assert isinstance(related, list) and related
    assert any(r.get("name") == "helper" and r.get("kind") in ("def", "class") for r in related)
    # The snippet should include the helper definition.
    assert "# x.py:1-2" in code
    assert "def helper" in code
