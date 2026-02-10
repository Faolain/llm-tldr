import json
from pathlib import Path


def test_django_retrieval_queries_schema():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "benchmarks" / "retrieval" / "django_queries.json"
    data = json.loads(path.read_text())

    assert data.get("schema_version") == 1
    assert data.get("repo") == "django"

    queries = data.get("queries")
    assert isinstance(queries, list)
    assert len(queries) >= 50

    ids: set[str] = set()
    for q in queries:
        assert isinstance(q, dict)
        qid = q.get("id")
        assert isinstance(qid, str) and qid
        assert qid not in ids, f"duplicate id: {qid}"
        ids.add(qid)

        assert isinstance(q.get("query"), str)
        relevant = q.get("relevant_files")
        assert isinstance(relevant, list)
        assert all(isinstance(x, str) for x in relevant)

        rg_pattern = q.get("rg_pattern")
        assert rg_pattern is None or isinstance(rg_pattern, str)
