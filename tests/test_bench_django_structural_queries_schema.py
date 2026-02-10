import json
from pathlib import Path


def test_django_structural_queries_schema():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "benchmarks" / "python" / "django_structural_queries.json"
    data = json.loads(path.read_text())

    assert data.get("schema_version") == 1
    assert data.get("repo") == "django"

    queries = data.get("queries")
    assert isinstance(queries, list)
    assert queries, "expected at least 1 query"

    ids: set[str] = set()
    categories: set[str] = set()
    counts: dict[str, int] = {}

    for q in queries:
        assert isinstance(q, dict)
        qid = q.get("id")
        assert isinstance(qid, str) and qid
        assert qid not in ids, f"duplicate id: {qid}"
        ids.add(qid)

        cat = q.get("category")
        assert isinstance(cat, str) and cat
        categories.add(cat)
        counts[cat] = counts.get(cat, 0) + 1

        if cat == "impact":
            assert isinstance(q.get("function"), str)
            assert isinstance(q.get("file"), str)
            expected = q.get("expected_callers")
            assert isinstance(expected, list)
            assert expected, "impact queries should include at least 1 expected caller"
            for c in expected:
                assert isinstance(c, dict)
                assert isinstance(c.get("file"), str)
                assert isinstance(c.get("function"), str)
        elif cat == "slice":
            assert isinstance(q.get("file"), str)
            assert isinstance(q.get("function"), str)
            assert isinstance(q.get("target_line"), int)
            expected_lines = q.get("expected_slice_lines")
            assert isinstance(expected_lines, list)
            assert all(isinstance(x, int) for x in expected_lines)
        elif cat == "complexity":
            assert isinstance(q.get("file"), str)
            assert isinstance(q.get("function"), str)
        elif cat == "data_flow":
            assert isinstance(q.get("file"), str)
            assert isinstance(q.get("function"), str)
            assert isinstance(q.get("variable"), str)
            expected_flow = q.get("expected_flow")
            assert isinstance(expected_flow, list)
            assert expected_flow, "data_flow queries should include expected_flow events"
            for ev in expected_flow:
                assert isinstance(ev, dict)
                assert isinstance(ev.get("line"), int)
                assert isinstance(ev.get("event"), str)

    # Keep at least one query per core category.
    assert {"impact", "slice", "complexity", "data_flow"}.issubset(categories)

    # Keep the suite at spec size so results aren't dominated by a tiny starter set.
    assert counts.get("impact", 0) >= 15
    assert counts.get("slice", 0) >= 10
    assert counts.get("complexity", 0) >= 10
    assert counts.get("data_flow", 0) >= 10
