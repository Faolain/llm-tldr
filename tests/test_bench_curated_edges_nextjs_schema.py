import json
from pathlib import Path


def test_nextjs_curated_edges_schema_smoke():
    path = Path("benchmarks/ts/nextjs_curated_edges.json")
    data = json.loads(path.read_text())

    assert isinstance(data, dict)
    assert data.get("repo") == "nextjs"

    edges = data.get("edges")
    assert isinstance(edges, list)
    assert len(edges) >= 30

    seen = set()
    any_class_caller = False
    any_class_callee = False
    any_next_env = False

    for e in edges:
        assert isinstance(e, dict)
        caller = e.get("caller")
        callee = e.get("callee")
        assert isinstance(caller, dict)
        assert isinstance(callee, dict)
        cf = caller.get("file")
        cs = caller.get("symbol")
        tf = callee.get("file")
        ts = callee.get("symbol")
        assert isinstance(cf, str) and cf
        assert isinstance(cs, str) and cs
        assert isinstance(tf, str) and tf
        assert isinstance(ts, str) and ts

        any_class_caller = any_class_caller or ("." in cs)
        any_class_callee = any_class_callee or ("." in ts)
        any_next_env = any_next_env or (tf == "packages/next-env/index.ts")

        key = (cf, cs, tf, ts)
        assert key not in seen
        seen.add(key)

    assert any_class_caller
    assert any_class_callee
    assert any_next_env

