import json
import runpy
import sys
from pathlib import Path

import pytest


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_llm_ab_run.py")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


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


def test_score_sets_empty_is_perfect():
    mod = _load_mod()
    score_sets = mod["_score_sets"]
    sc = score_sets(set(), set())
    assert sc.tp == 0
    assert sc.fp == 0
    assert sc.fn == 0
    assert sc.precision == 1.0
    assert sc.recall == 1.0
    assert sc.f1 == 1.0


def test_extract_judge_winner():
    mod = _load_mod()
    fn = mod["_extract_judge_winner"]
    assert fn({"winner": "A"}) == "A"
    assert fn({"winner": "B"}) == "B"
    assert fn({"winner": "tie"}) == "tie"
    assert fn({"winner": " a "}) == "A"
    assert fn({"winner": "Tie"}) == "tie"
    assert fn({"winner": "draw"}) == "tie"
    assert fn({"winner": "C"}) is None
    assert fn("nope") is None


def test_classify_structured_output_empty_and_malformed():
    mod = _load_mod()
    fn = mod["_classify_structured_output"]

    status_empty, got_empty = fn("slice", "  \n\t ")
    assert status_empty == "empty"
    assert got_empty is None

    status_malformed, got_malformed = fn("slice", '{"oops": [1, 2]}')
    assert status_malformed == "malformed"
    assert got_malformed is None


def test_classify_structured_output_invariant_and_per_source_split():
    mod = _load_mod()
    classify = mod["_classify_structured_output"]

    samples = [
        ("rg", "slice", "   "),
        ("tldr", "slice", '{"oops": [1]}'),
        ("rg", "slice", '{"lines": [1, 2, 3]}'),
    ]

    empty_output_total = 0
    malformed_output_total = 0
    empty_output_by_source = {"rg": 0, "tldr": 0}
    malformed_output_by_source = {"rg": 0, "tldr": 0}

    for source, category, text in samples:
        status, _ = classify(category, text)
        if status == "empty":
            empty_output_total += 1
            empty_output_by_source[source] += 1
        elif status == "malformed":
            malformed_output_total += 1
            malformed_output_by_source[source] += 1

    bad_json = empty_output_total + malformed_output_total
    assert bad_json == 2
    assert bad_json == empty_output_total + malformed_output_total
    assert empty_output_by_source == {"rg": 1, "tldr": 0}
    assert malformed_output_by_source == {"rg": 0, "tldr": 1}


def test_classify_judge_verdict_empty_and_malformed():
    mod = _load_mod()
    fn = mod["_classify_judge_verdict"]

    status_empty, winner_empty, _ = fn("   ")
    assert status_empty == "empty"
    assert winner_empty is None

    status_malformed, winner_malformed, _ = fn('{"winner": "C"}')
    assert status_malformed == "malformed"
    assert winner_malformed is None


def test_classify_judge_verdict_invariant():
    mod = _load_mod()
    classify = mod["_classify_judge_verdict"]

    statuses = [
        classify(" ")[0],
        classify('{"winner":"C","scores":{"A":{},"B":{}},"notes":"x"}')[0],
        classify('{"winner":"A"}')[0],
    ]

    judge_empty_verdict_total = statuses.count("empty")
    judge_malformed_verdict_total = statuses.count("malformed")
    judge_bad_json = judge_empty_verdict_total + judge_malformed_verdict_total

    assert statuses == ["empty", "malformed", "ok"]
    assert judge_bad_json == 2
    assert judge_bad_json == judge_empty_verdict_total + judge_malformed_verdict_total


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


def test_main_structured_report_serializes_bad_json_split(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mod = _load_mod()
    globals_dict = mod["main"].__globals__

    prompts_path = tmp_path / "prompts-structured.jsonl"
    _write_jsonl(
        prompts_path,
        [
            {
                "task_id": "slice-1",
                "category": "slice",
                "expected": {"lines": [1, 2]},
                "variants": [
                    {"label": "A", "source": "rg", "prompt": "EMPTY_OUTPUT"},
                    {"label": "B", "source": "tldr", "prompt": "MALFORMED_OUTPUT"},
                ],
            }
        ],
    )
    out_path = tmp_path / "report-structured.json"
    answers_out = tmp_path / "answers-structured.jsonl"

    def fake_anthropic_call(*, model: str, prompt: str, max_tokens: int, temperature: float):
        del model, max_tokens, temperature
        if prompt == "EMPTY_OUTPUT":
            return "   ", {"input_tokens": 1, "output_tokens": 0}
        if prompt == "MALFORMED_OUTPUT":
            return '{"oops": [1, 2]}', {"input_tokens": 1, "output_tokens": 1}
        return '{"lines": [1, 2]}', {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setitem(globals_dict, "_anthropic_call", fake_anthropic_call)
    monkeypatch.setitem(globals_dict, "get_repo_root", lambda: tmp_path)
    monkeypatch.setitem(globals_dict, "gather_meta", lambda **_: {"test_meta": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_llm_ab_run.py",
            "--prompts",
            str(prompts_path),
            "--provider",
            "anthropic",
            "--model",
            "fake-model",
            "--out",
            str(out_path),
            "--answers-out",
            str(answers_out),
        ],
    )

    assert mod["main"]() == 0

    report = json.loads(out_path.read_text(encoding="utf-8"))
    results = report["results"]
    assert results["empty_output_total"] == 1
    assert results["malformed_output_total"] == 1
    assert results["bad_json"] == 2
    assert results["bad_json"] == results["empty_output_total"] + results["malformed_output_total"]


def test_main_judge_report_serializes_bad_json_split(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mod = _load_mod()
    globals_dict = mod["main"].__globals__

    prompts_path = tmp_path / "prompts-judge.jsonl"
    _write_jsonl(
        prompts_path,
        [
            {
                "task_id": "judge-empty",
                "task_type": "open_ended",
                "category": "retrieval",
                "question": "Q_EMPTY",
                "rubric": "Use context only.",
                "variants": [
                    {"label": "A", "source": "rg", "context": "ctx-a", "prompt": "answer-a-empty"},
                    {"label": "B", "source": "tldr", "context": "ctx-b", "prompt": "answer-b-empty"},
                ],
            },
            {
                "task_id": "judge-malformed",
                "task_type": "open_ended",
                "category": "retrieval",
                "question": "Q_MALFORMED",
                "rubric": "Use context only.",
                "variants": [
                    {"label": "A", "source": "rg", "context": "ctx-a2", "prompt": "answer-a-malformed"},
                    {"label": "B", "source": "tldr", "context": "ctx-b2", "prompt": "answer-b-malformed"},
                ],
            },
        ],
    )
    out_path = tmp_path / "report-judge.json"
    answers_out = tmp_path / "answers-judge.jsonl"

    def fake_anthropic_call(*, model: str, prompt: str, max_tokens: int, temperature: float):
        del model, max_tokens, temperature
        if prompt.startswith("You are an impartial judge."):
            if "Q_EMPTY" in prompt:
                return " ", {"input_tokens": 1, "output_tokens": 0}
            if "Q_MALFORMED" in prompt:
                return '{"winner":"C"}', {"input_tokens": 1, "output_tokens": 1}
        return "deterministic answer", {"input_tokens": 2, "output_tokens": 3}

    monkeypatch.setitem(globals_dict, "_anthropic_call", fake_anthropic_call)
    monkeypatch.setitem(globals_dict, "get_repo_root", lambda: tmp_path)
    monkeypatch.setitem(globals_dict, "gather_meta", lambda **_: {"test_meta": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_llm_ab_run.py",
            "--prompts",
            str(prompts_path),
            "--mode",
            "judge",
            "--provider",
            "anthropic",
            "--model",
            "fake-answer-model",
            "--judge-provider",
            "anthropic",
            "--judge-model",
            "fake-judge-model",
            "--judge-retries",
            "0",
            "--out",
            str(out_path),
            "--answers-out",
            str(answers_out),
        ],
    )

    assert mod["main"]() == 0

    report = json.loads(out_path.read_text(encoding="utf-8"))
    results = report["results"]
    assert results["judge_empty_verdict_total"] == 1
    assert results["judge_malformed_verdict_total"] == 1
    assert results["judge_bad_json"] == 2
    assert results["judge_bad_json"] == (
        results["judge_empty_verdict_total"] + results["judge_malformed_verdict_total"]
    )
