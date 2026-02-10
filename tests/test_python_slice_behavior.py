import textwrap

from tldr.api import get_slice


def test_slice_return_includes_predicates_but_not_other_returns() -> None:
    code = textwrap.dedent(
        """
        def mark_safe(s):
            if hasattr(s, "__html__"):
                return s
            if callable(s):
                return _safety_decorator(mark_safe, s)
            return SafeString(s)
        """
    ).lstrip("\n")

    # Line numbers: def=1, if1=2, return=3, if2=4, return=5, return=6
    lines = get_slice(code, "mark_safe", line=6, direction="backward", language="python")
    assert lines == {2, 4, 6}


def test_slice_return_excludes_raise_on_other_path() -> None:
    code = textwrap.dedent(
        """
        def get_valid_filename(name):
            s = str(name).strip().replace(" ", "_")
            s = re.sub(r"(?u)[^-\\w.]", "", s)
            if s in {"", ".", ".."}:
                raise ValueError("bad")
            return s
        """
    ).lstrip("\n")

    # Line numbers: def=1, s1=2, s2=3, if=4, raise=5, return=6
    lines = get_slice(code, "get_valid_filename", line=6, direction="backward", language="python")
    assert lines == {2, 3, 4, 6}


def test_slice_non_return_is_data_slice() -> None:
    code = textwrap.dedent(
        """
        def assign_only(exc):
            if exc:
                pass
            x = exc.__class__.__name__
            return x
        """
    ).lstrip("\n")

    # Line numbers: def=1, if=2, pass=3, assign=4, return=5
    lines = get_slice(code, "assign_only", line=4, direction="backward", language="python")
    assert lines == {4}


def test_slice_return_includes_assert_guard() -> None:
    code = textwrap.dedent(
        """
        def with_assert(a):
            if a:
                a = 1
            assert a
            return a
        """
    ).lstrip("\n")

    # Line numbers: def=1, if=2, assign=3, assert=4, return=5
    lines = get_slice(code, "with_assert", line=5, direction="backward", language="python")
    assert lines == {2, 3, 4, 5}

