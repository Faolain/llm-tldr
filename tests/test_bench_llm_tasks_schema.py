import json
from pathlib import Path


def test_bench_llm_tasks_schema_and_refs():
    repo_root = Path(__file__).resolve().parents[1]
    tasks_path = repo_root / "benchmarks" / "llm" / "tasks.json"
    structural_path = repo_root / "benchmarks" / "python" / "django_structural_queries.json"

    tasks = json.loads(tasks_path.read_text())
    assert isinstance(tasks, dict)
    assert tasks.get("schema_version") == 1
    rows = tasks.get("tasks")
    assert isinstance(rows, list)
    assert len(rows) >= 30

    structural = json.loads(structural_path.read_text())
    queries = structural.get("queries") if isinstance(structural, dict) else structural
    assert isinstance(queries, list)
    by_id = {q.get("id"): q for q in queries if isinstance(q, dict) and isinstance(q.get("id"), str)}

    seen_ids: set[str] = set()
    allowed = {"impact", "slice", "data_flow"}
    for t in rows:
        assert isinstance(t, dict)
        tid = t.get("id")
        assert isinstance(tid, str)
        assert tid not in seen_ids
        seen_ids.add(tid)

        cat = t.get("category")
        assert cat in allowed

        qid = t.get("query_id")
        assert isinstance(qid, str)
        q = by_id.get(qid)
        assert isinstance(q, dict), f"task {tid} references missing query_id {qid}"
        assert q.get("category") == cat

        question = t.get("question")
        assert isinstance(question, str)
        assert question.strip()

