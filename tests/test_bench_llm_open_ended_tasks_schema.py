import json
import re
from pathlib import Path


def _question_mentions_target_line(question: str, target_line: int) -> bool:
    return bool(
        re.search(
            rf"\btarget[_\s-]*line\s*[:=]?\s*{int(target_line)}\b",
            question,
            flags=re.IGNORECASE,
        )
    )


def test_bench_llm_open_ended_tasks_schema_and_refs():
    repo_root = Path(__file__).resolve().parents[1]
    tasks_path = repo_root / "benchmarks" / "llm" / "open_ended_tasks.json"
    structural_path = repo_root / "benchmarks" / "python" / "django_structural_queries.json"

    tasks = json.loads(tasks_path.read_text())
    assert isinstance(tasks, dict)
    assert tasks.get("schema_version") == 1
    rows = tasks.get("tasks")
    assert isinstance(rows, list)
    assert len(rows) >= 6

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

        assert t.get("task_type") == "open_ended"

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
        q_lower = question.lower()

        rubric = t.get("rubric")
        assert isinstance(rubric, str)
        assert rubric.strip()

        q_file = q.get("file")
        q_function = q.get("function")
        assert isinstance(q_file, str), f"task {tid}: mapped query {qid} missing file anchor"
        assert isinstance(q_function, str), f"task {tid}: mapped query {qid} missing function anchor"
        assert q_file in question, f"task {tid}: question missing file anchor {q_file!r}"
        assert q_function in question, f"task {tid}: question missing function anchor {q_function!r}"

        if cat == "slice":
            target_line = q.get("target_line")
            assert isinstance(target_line, int), f"task {tid}: slice query {qid} must define int target_line"
            assert _question_mentions_target_line(question, int(target_line)), (
                f"task {tid}: question must mention target_line={target_line}"
            )
        elif cat == "data_flow":
            variable = q.get("variable")
            assert isinstance(variable, str) and variable.strip(), f"task {tid}: data_flow query {qid} missing variable"
            assert re.search(rf"\b{re.escape(variable)}\b", question), (
                f"task {tid}: question missing data_flow variable anchor {variable!r}"
            )
        elif cat == "impact":
            # Impact questions should explicitly include both symbol and file path anchors.
            assert q_function.lower() in q_lower, f"task {tid}: impact question missing function anchor {q_function!r}"
            assert q_file.lower() in q_lower, f"task {tid}: impact question missing file anchor {q_file!r}"


def test_open_ended_task_query_alignment_and_anchor_consistency():
    # Explicit test entrypoint required by Phase 0 implementation plan.
    test_bench_llm_open_ended_tasks_schema_and_refs()


def test_oe08_regression_guard_maps_to_b10_configure():
    repo_root = Path(__file__).resolve().parents[1]
    tasks_path = repo_root / "benchmarks" / "llm" / "open_ended_tasks.json"
    structural_path = repo_root / "benchmarks" / "python" / "django_structural_queries.json"

    tasks_doc = json.loads(tasks_path.read_text())
    tasks = tasks_doc.get("tasks")
    assert isinstance(tasks, list)

    structural_doc = json.loads(structural_path.read_text())
    queries = structural_doc.get("queries") if isinstance(structural_doc, dict) else structural_doc
    assert isinstance(queries, list)
    by_id = {q.get("id"): q for q in queries if isinstance(q, dict) and isinstance(q.get("id"), str)}

    oe08 = next((t for t in tasks if isinstance(t, dict) and t.get("id") == "OE08"), None)
    assert isinstance(oe08, dict)
    assert oe08.get("query_id") == "B10"

    q = by_id.get("B10")
    assert isinstance(q, dict)
    assert q.get("category") == "slice"
    assert q.get("function") == "configure"
    assert q.get("file") == "django/conf/__init__.py"
    assert q.get("target_line") == 124

    question = oe08.get("question")
    assert isinstance(question, str)
    assert "configure" in question
    assert "django/conf/__init__.py" in question
    assert _question_mentions_target_line(question, 124)
