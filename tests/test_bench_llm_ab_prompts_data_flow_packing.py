import json
import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_llm_ab_prompts.py")


def _split_payload(payload: str) -> tuple[dict, str]:
    head, sep, tail = payload.partition("\n\n")
    if sep:
        return json.loads(head), tail
    return json.loads(payload), ""


def test_data_flow_open_ended_context_emits_contract_and_semantic_roles(tmp_path):
    mod = _load_mod()
    ctx = mod["_data_flow_context_tldr_plus_code"]

    src = "\n".join(
        [
            "def foo(token):",
            "    mask = token[:4]",
            "    if len(token) > 4:",
            "        body = mask + token[4:]",
            "        masked = body[::-1]",
            "    else:",
            "        body = mask + 'x'",
            "        masked = body",
            "    sink = masked.upper()",
            "    _ = sink + mask",
            "    return mask + sink",
        ]
    )
    (tmp_path / "x.py").write_text(src, encoding="utf-8")

    ctx.__globals__["get_dfg_context"] = lambda *_args, **_kwargs: {
        "edges": [
            {"var": "mask", "def_line": 2, "use_line": 4},
            {"var": "mask", "def_line": 2, "use_line": 7},
            {"var": "mask", "def_line": 2, "use_line": 10},
            {"var": "mask", "def_line": 2, "use_line": 11},
        ],
        "refs": [
            {"name": "mask", "type": "definition", "line": 2, "column": 4},
            {"name": "mask", "type": "use", "line": 4, "column": 15},
            {"name": "mask", "type": "use", "line": 7, "column": 15},
            {"name": "mask", "type": "use", "line": 10, "column": 15},
            {"name": "mask", "type": "use", "line": 11, "column": 11},
        ],
    }

    payload, tok, _ = ctx(
        repo_root=tmp_path,
        file_rel="x.py",
        function="foo",
        variable="mask",
        budget_tokens=1800,
    )
    assert tok <= 1800

    meta, code = _split_payload(payload)
    assert code.strip()
    assert meta["strategy"] == "anchor_window_plus_bridge_plus_flow_windows"
    assert isinstance(meta.get("included_lines"), list) and meta["included_lines"]
    assert isinstance(meta.get("function_span_lines"), dict)
    assert isinstance(meta.get("truncated"), bool)

    span = meta["function_span_lines"]
    assert span.get("start") == 1
    assert span.get("end") == 11

    roles = meta.get("semantic_roles")
    assert isinstance(roles, list) and roles
    allowed = {"input", "predicate", "transform", "use", "return"}
    for role in roles:
        assert isinstance(role, dict)
        assert isinstance(role.get("line"), int)
        assert role.get("role") in allowed
        assert isinstance(role.get("rationale"), str) and role["rationale"].strip()

    roles_by_line = {r["line"]: r["role"] for r in roles}
    assert roles_by_line.get(2) == "input"
    assert roles_by_line.get(11) == "return"
    assert "input" in {r["role"] for r in roles}
    assert "return" in {r["role"] for r in roles}


def test_data_flow_budget_hard_cap_and_deterministic_drop_order(tmp_path):
    mod = _load_mod()
    ctx = mod["_data_flow_context_tldr_plus_code"]

    src = "\n".join(
        [
            "def foo(seed):",
            "    mask = seed.strip()",
            "    if seed:",
            "        a1 = mask + '1'",
            "        a2 = mask + '2'",
            "        a3 = mask + '3'",
            "        a4 = mask + '4'",
            "    else:",
            "        b1 = mask + '5'",
            "        b2 = mask + '6'",
            "        b3 = mask + '7'",
            "    c1 = mask + '8'",
            "    c2 = mask + '9'",
            "    c3 = mask + '10'",
            "    if len(mask) > 1:",
            "        out = c1 + c2 + c3",
            "    out2 = out + mask",
            "    out3 = out2 + mask",
            "    out4 = out3 + mask",
            "    out5 = out4 + mask",
            "    out6 = out5 + mask",
            "    return out6 + mask",
        ]
    )
    (tmp_path / "x.py").write_text(src, encoding="utf-8")

    ctx.__globals__["get_dfg_context"] = lambda *_args, **_kwargs: {
        "edges": [
            {"var": "mask", "def_line": 2, "use_line": 4},
            {"var": "mask", "def_line": 2, "use_line": 9},
            {"var": "mask", "def_line": 2, "use_line": 12},
            {"var": "mask", "def_line": 2, "use_line": 15},
            {"var": "mask", "def_line": 2, "use_line": 17},
            {"var": "mask", "def_line": 2, "use_line": 22},
        ],
        "refs": [{"name": "mask", "line": ln, "type": "use", "column": 8} for ln in range(2, 23)],
    }

    low_budget = 1000
    payload_low_1, tok_low_1, _ = ctx(
        repo_root=tmp_path,
        file_rel="x.py",
        function="foo",
        variable="mask",
        budget_tokens=low_budget,
    )
    payload_low_2, tok_low_2, _ = ctx(
        repo_root=tmp_path,
        file_rel="x.py",
        function="foo",
        variable="mask",
        budget_tokens=low_budget,
    )

    assert payload_low_1 == payload_low_2
    assert tok_low_1 == tok_low_2
    assert tok_low_1 <= low_budget

    meta_low, code_low = _split_payload(payload_low_1)
    assert isinstance(meta_low.get("truncated"), bool) and meta_low["truncated"]
    assert code_low.strip()
    assert isinstance(meta_low.get("included_lines"), list) and meta_low["included_lines"]

    high_budget = 1200
    payload_high_1, tok_high_1, _ = ctx(
        repo_root=tmp_path,
        file_rel="x.py",
        function="foo",
        variable="mask",
        budget_tokens=high_budget,
    )
    payload_high_2, tok_high_2, _ = ctx(
        repo_root=tmp_path,
        file_rel="x.py",
        function="foo",
        variable="mask",
        budget_tokens=high_budget,
    )
    assert payload_high_1 == payload_high_2
    assert tok_high_1 == tok_high_2
    assert tok_high_1 <= high_budget

    meta_high, code_high = _split_payload(payload_high_1)
    assert code_high.strip()
    assert meta_high["strategy"] == "anchor_window_plus_bridge_plus_flow_windows"
    assert isinstance(meta_high.get("bridge_lines"), list) and meta_high["bridge_lines"]
    assert isinstance(meta_high.get("included_lines"), list) and meta_high["included_lines"]


def test_data_flow_packer_adaptive_search_finds_non_empty_under_budget(monkeypatch):
    mod = _load_mod()
    pack = mod["_pack_open_ended_data_flow_context"]
    bte = pack.__globals__["bte"]

    def fake_count_tokens(text: str) -> int:
        return len(text.splitlines())

    def fake_select_window_within_span(*, file_rel, lines, span, target_line, budget_tokens):
        lo, hi = span if span is not None else (1, len(lines))
        lo = max(1, int(lo))
        hi = min(int(hi), len(lines))
        target = max(int(lo), min(int(hi), int(target_line)))
        if int(budget_tokens) >= 10:
            return "", set(), {"start": max(int(lo), int(target) - 10), "end": min(int(hi), int(target) + 10)}
        return "", set(), {"start": int(target), "end": int(target)}

    monkeypatch.setitem(pack.__globals__, "count_tokens", fake_count_tokens)
    monkeypatch.setattr(bte, "_select_window_within_span", fake_select_window_within_span)

    lines = [f"v = {i}" for i in range(1, 41)]
    meta, code = pack(
        file_rel="x.py",
        lines=lines,
        function="foo",
        variable="v",
        span=(1, 40),
        flow_events=[{"line": 20, "event": "defined"}],
        ref_lines=[20],
        budget_tokens=20,
    )

    assert code.strip()
    assert meta["strategy"] == "anchor_window_plus_bridge_plus_flow_windows"
    assert meta.get("included_lines") == [20]
    assert isinstance(meta.get("target_window"), dict)
    assert meta["target_window"]["start"] == 20
    assert meta["target_window"]["end"] == 20


def test_data_flow_packer_tiny_budget_cap_is_deterministic_in_meta_min_path(monkeypatch):
    mod = _load_mod()
    pack = mod["_pack_open_ended_data_flow_context"]
    bte = pack.__globals__["bte"]

    # Force the no-window fallback path so cap enforcement is exercised directly.
    def fake_select_window_within_span(*, file_rel, lines, span, target_line, budget_tokens):
        return "", set(), None

    monkeypatch.setattr(bte, "_select_window_within_span", fake_select_window_within_span)

    lines = [f"value = {i}" for i in range(1, 40)]
    kwargs = dict(
        file_rel="x.py",
        lines=lines,
        function="foo",
        variable="value",
        span=(1, len(lines)),
        flow_events=[{"line": 2, "event": "defined"}, {"line": 20, "event": "used"}],
        ref_lines=[2, 20],
    )

    observed: dict[int, tuple[dict, str, str, int]] = {}
    for budget in (64, 16, 1):
        meta1, code1 = pack(**kwargs, budget_tokens=budget)
        meta2, code2 = pack(**kwargs, budget_tokens=budget)
        assert meta1 == meta2
        assert code1 == code2

        payload = json.dumps(meta1, sort_keys=True)
        if code1:
            payload += "\n\n" + code1
        tok = int(mod["count_tokens"](payload))
        assert tok <= budget
        observed[budget] = (meta1, code1, payload, tok)

    # Lowest budget emits the guaranteed-fit sentinel.
    _meta1, code_budget_1, payload_budget_1, tok_budget_1 = observed[1]
    assert code_budget_1 == ""
    assert payload_budget_1 == "{}"
    assert tok_budget_1 == 1

    # Stable drop order across tighter budgets.
    keys_64 = set(observed[64][0].keys())
    keys_16 = set(observed[16][0].keys())
    keys_1 = set(observed[1][0].keys())
    assert keys_16.issubset(keys_64)
    assert keys_1.issubset(keys_16)


def test_data_flow_context_tldr_plus_code_tiny_budget_stays_under_cap(monkeypatch, tmp_path):
    mod = _load_mod()
    ctx = mod["_data_flow_context_tldr_plus_code"]
    bte = ctx.__globals__["bte"]

    src = "\n".join(
        [
            "def foo(seed):",
            "    value = seed.strip()",
            "    if value:",
            "        out = value + 'x'",
            "    return out if value else ''",
        ]
    )
    (tmp_path / "x.py").write_text(src, encoding="utf-8")

    # Force meta-min fallback path from the packer.
    def fake_select_window_within_span(*, file_rel, lines, span, target_line, budget_tokens):
        return "", set(), None

    monkeypatch.setattr(bte, "_select_window_within_span", fake_select_window_within_span)
    ctx.__globals__["get_dfg_context"] = lambda *_args, **_kwargs: {
        "edges": [{"var": "value", "def_line": 2, "use_line": 4}],
        "refs": [{"name": "value", "line": 2, "type": "definition", "column": 4}],
    }

    seen: dict[int, tuple[str, int]] = {}
    for budget in (64, 16, 1):
        payload_1, tok_1, _ = ctx(
            repo_root=tmp_path,
            file_rel="x.py",
            function="foo",
            variable="value",
            budget_tokens=budget,
        )
        payload_2, tok_2, _ = ctx(
            repo_root=tmp_path,
            file_rel="x.py",
            function="foo",
            variable="value",
            budget_tokens=budget,
        )
        assert payload_1 == payload_2
        assert tok_1 == tok_2
        assert tok_1 <= budget
        seen[budget] = (payload_1, tok_1)

    assert seen[1] == ("{}", 1)
