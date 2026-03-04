import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_retrieval_quality.py")


def test_rrf_fuse_boosts_docs_supported_by_multiple_rankers():
    mod = _load_mod()
    fuse = mod["_rrf_fuse"]

    ranked = fuse(
        [
            ["a.py", "b.py", "c.py"],
            ["b.py", "d.py", "e.py"],
        ]
    )
    assert ranked[0] == "b.py"
    assert ranked.index("b.py") < ranked.index("a.py")


def test_rrf_fuse_tie_break_is_deterministic():
    mod = _load_mod()
    fuse = mod["_rrf_fuse"]

    ranked = fuse([["b.py"], ["a.py"]])
    assert ranked == ["a.py", "b.py"]


def test_effective_k_from_budget_tokens_is_deterministic_and_clamped():
    mod = _load_mod()
    effective_k = mod["_effective_k_from_budget_tokens"]

    assert effective_k(10, budget_tokens=2000) == 10
    assert effective_k(10, budget_tokens=1000) == 5
    assert effective_k(10, budget_tokens=500) == 3
    assert effective_k(10, budget_tokens=5000) == 25
    assert effective_k(10, budget_tokens=500000) == 50
    assert effective_k(10, budget_tokens=0) == 10
    assert effective_k(10, budget_tokens=-1) == 10
    assert effective_k(10, budget_tokens="bad") == 10


def test_budget_tokens_scales_semantic_k_and_is_recorded_in_protocol(tmp_path: Path, monkeypatch):
    mod = _load_mod()
    main = mod["main"]
    globals_dict = main.__globals__

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "q1",
                        "query": "needle",
                        "rg_pattern": "needle",
                        "relevant_files": ["pkg/a.py"],
                    }
                ]
            }
        )
    )

    semantic_faiss = tmp_path / "semantic.faiss"
    semantic_faiss.write_bytes(b"")
    semantic_meta = tmp_path / "semantic.meta.json"
    semantic_meta.write_text(json.dumps({"model": "fake", "dimension": 1, "count": 1}))

    fake_index_ctx = SimpleNamespace(
        paths=SimpleNamespace(
            semantic_faiss=semantic_faiss,
            semantic_metadata=semantic_meta,
        ),
        cache_root=tmp_path / "cache",
        index_id="repo:test",
        config=None,
    )

    semantic_ks: list[int] = []

    def _fake_semantic_rank_files(*args, **kwargs):
        semantic_ks.append(int(kwargs["k"]))
        return ["pkg/a.py"]

    monkeypatch.setitem(globals_dict, "get_index_context", lambda **_: fake_index_ctx)
    monkeypatch.setitem(globals_dict, "_rg_rank_files", lambda *_, **__: ["pkg/a.py"])
    monkeypatch.setitem(globals_dict, "_semantic_rank_files", _fake_semantic_rank_files)

    out_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_retrieval_quality.py",
            "--repo-root",
            str(repo_root),
            "--queries",
            str(queries_path),
            "--ks",
            "8",
            "--budget-tokens",
            "500",
            "--out",
            str(out_path),
        ],
    )

    assert main() == 0
    assert semantic_ks == [2]

    report = json.loads(out_path.read_text())
    assert report["protocol"]["budget_tokens"] == 500
    assert report["protocol"]["effective_k"] == 2


def test_no_result_guard_rg_empty_forces_empty_semantic_and_hybrid(tmp_path: Path, monkeypatch):
    mod = _load_mod()
    main = mod["main"]
    globals_dict = main.__globals__

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "q1",
                        "query": "missing-symbol",
                        "rg_pattern": "missing-symbol",
                        "relevant_files": [],
                    }
                ]
            }
        )
    )

    semantic_faiss = tmp_path / "semantic.faiss"
    semantic_faiss.write_bytes(b"")
    semantic_meta = tmp_path / "semantic.meta.json"
    semantic_meta.write_text(json.dumps({"model": "fake", "dimension": 1, "count": 0}))

    fake_index_ctx = SimpleNamespace(
        paths=SimpleNamespace(
            semantic_faiss=semantic_faiss,
            semantic_metadata=semantic_meta,
        ),
        cache_root=tmp_path / "cache",
        index_id="repo:test",
        config=None,
    )

    semantic_calls = {"count": 0}

    def _semantic_should_not_run(*args, **kwargs):
        semantic_calls["count"] += 1
        return ["unexpected.py"]

    monkeypatch.setitem(globals_dict, "get_index_context", lambda **_: fake_index_ctx)
    monkeypatch.setitem(globals_dict, "_rg_rank_files", lambda *_, **__: [])
    monkeypatch.setitem(globals_dict, "_semantic_rank_files", _semantic_should_not_run)

    out_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_retrieval_quality.py",
            "--repo-root",
            str(repo_root),
            "--queries",
            str(queries_path),
            "--no-result-guard",
            "rg_empty",
            "--out",
            str(out_path),
        ],
    )

    assert main() == 0
    assert semantic_calls["count"] == 0

    report = json.loads(out_path.read_text())
    query_row = report["results"]["per_query"][0]
    assert query_row["no_result_guard_triggered"] is True
    assert query_row["semantic"]["top_files"] == []
    assert query_row["hybrid_rrf"]["top_files"] == []
