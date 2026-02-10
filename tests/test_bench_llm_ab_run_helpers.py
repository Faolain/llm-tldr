import runpy
import sys
from pathlib import Path

import pytest


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_llm_ab_run.py")


def test_extract_json_from_text_plain():
    mod = _load_mod()
    fn = mod["_extract_json_from_text"]
    assert fn('[{"file":"a.py","function":"f"}]') == [{"file": "a.py", "function": "f"}]


def test_extract_json_from_text_code_fence():
    mod = _load_mod()
    fn = mod["_extract_json_from_text"]
    text = "```json\n{\"lines\": [1, 2, 3]}\n```"
    assert fn(text) == {"lines": [1, 2, 3]}


def test_extract_json_from_text_embedded():
    mod = _load_mod()
    fn = mod["_extract_json_from_text"]
    text = "Here you go:\n\n{\"flow\": [{\"line\": 1, \"event\": \"defined\"}]}\nThanks"
    assert fn(text) == {"flow": [{"line": 1, "event": "defined"}]}


def test_score_sets_f1():
    mod = _load_mod()
    score_sets = mod["_score_sets"]
    expected = {1, 2, 3}
    got = {2, 3, 4}
    sc = score_sets(expected, got)
    assert sc.tp == 2
    assert sc.fp == 1
    assert sc.fn == 1
    assert round(sc.f1, 4) == 0.6667


def test_claude_sdk_result_to_text_and_usage_structured():
    try:
        from claude_agent_sdk import ResultMessage
    except Exception:
        pytest.skip("claude-agent-sdk not installed")

    mod = _load_mod()
    fn = mod["_claude_sdk_result_to_text_and_usage"]

    msg = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        total_cost_usd=0.1,
        usage={"inputTokens": 10, "outputTokens": 20},
        result="",
        structured_output={"lines": [1, 2, 3]},
    )
    text, usage = fn(msg)
    assert isinstance(text, str)
    assert text
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 20
    assert usage["total_cost_usd"] == 0.1


def test_claude_sdk_result_to_text_and_usage_error_raises():
    try:
        from claude_agent_sdk import ResultMessage
    except Exception:
        pytest.skip("claude-agent-sdk not installed")

    mod = _load_mod()
    fn = mod["_claude_sdk_result_to_text_and_usage"]

    msg = ResultMessage(
        subtype="error",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=1,
        session_id="s",
        result="boom",
    )
    with pytest.raises(RuntimeError):
        fn(msg)
