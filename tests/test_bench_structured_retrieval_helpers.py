import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_structured_retrieval.py")


def test_score_structured_predictions_matches_rg_span_to_gold_symbol():
    mod = _load_mod()
    score = mod["_score_structured_predictions"]

    result = score(
        targets=[
            {
                "file": "pkg/mod.py",
                "qualified_symbol": "pkg.mod.py.Target",
                "symbol_kind": "class",
                "start_line": 20,
                "end_line": 20,
            }
        ],
        predictions=[
            {
                "file": "pkg/mod.py",
                "qualified_symbol": None,
                "symbol_kind": "unknown",
                "start_line": 20,
                "end_line": 20,
                "rank": 1,
            }
        ],
    )

    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["matched_pairs"][0]["mode"] == "span"


def test_score_structured_predictions_rejects_wrong_symbol_when_both_sides_have_symbols():
    mod = _load_mod()
    score = mod["_score_structured_predictions"]

    result = score(
        targets=[
            {
                "file": "pkg/mod.py",
                "qualified_symbol": "pkg.mod.py.Target",
                "symbol_kind": "class",
                "start_line": 20,
                "end_line": 20,
            }
        ],
        predictions=[
            {
                "file": "pkg/mod.py",
                "qualified_symbol": "pkg.mod.py.OtherTarget",
                "symbol_kind": "class",
                "start_line": 20,
                "end_line": 20,
                "rank": 1,
            }
        ],
    )

    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["matched_pairs"] == []


def test_dedupe_items_drops_duplicate_symbol_predictions():
    mod = _load_mod()
    dedupe = mod["_dedupe_items"]

    items, unscorable = dedupe(
        [
            {
                "file": "pkg/mod.py",
                "qualified_symbol": "pkg.mod.py.Target",
                "symbol_kind": "class",
                "start_line": 20,
                "end_line": 20,
                "rank": 1,
            },
            {
                "file": "pkg/mod.py",
                "qualified_symbol": "pkg.mod.py.Target",
                "symbol_kind": "class",
                "start_line": 20,
                "end_line": 20,
                "rank": 2,
            },
        ]
    )

    assert unscorable == 0
    assert len(items) == 1


def test_hybrid_structured_predictions_projects_semantic_units_within_ranked_files(tmp_path: Path):
    mod = _load_mod()
    hybrid = mod["_hybrid_structured_predictions"]
    backend = mod["BackendSpec"](
        backend_id="bge_hybrid",
        kind="semantic_hybrid",
        display_name="bge hybrid",
        projection_unit_k=10,
    )

    def fake_rows(*_, retrieval_mode: str = "semantic", **__):
        if retrieval_mode == "hybrid":
            return [
                {"file": "pkg/first.py", "rank": 1},
                {"file": "pkg/second.py", "rank": 2},
            ]
        return [
            {
                "file": "pkg/second.py",
                "qualified_name": "pkg.second.py.Second",
                "unit_type": "class",
                "line": 20,
            },
            {
                "file": "pkg/first.py",
                "qualified_name": "pkg.first.py.First",
                "unit_type": "class",
                "line": 10,
            },
            {
                "file": "pkg/first.py",
                "qualified_name": "pkg.first.py.Helper",
                "unit_type": "class",
                "line": 30,
            },
        ]

    hybrid.__globals__["_semantic_search_rows"] = fake_rows

    predictions = hybrid(
        tmp_path,
        backend=backend,
        query="Where is First implemented?",
        rg_pattern="First",
        top_k=2,
        use_daemon=False,
    )

    assert [item["qualified_symbol"] for item in predictions] == [
        "pkg.first.py.First",
        "pkg.first.py.Helper",
    ]
    assert [item["hybrid_file_rank"] for item in predictions] == [1, 1]
    assert [item["rank"] for item in predictions] == [1, 2]


def test_main_writes_backend_micro_f1_report(tmp_path: Path, monkeypatch):
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
                        "id": "SR1",
                        "query": "Where is Target implemented?",
                        "rg_pattern": "^class\\s+Target\\b",
                        "targets": [
                            {
                                "file": "pkg/mod.py",
                                "qualified_symbol": "pkg.mod.py.Target",
                                "symbol_kind": "class",
                                "start_line": 20,
                                "end_line": 20,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    semantic_meta = tmp_path / "semantic-meta.json"
    semantic_meta.write_text(json.dumps({"model": "fake-model", "dimension": 128}), encoding="utf-8")

    fake_index_ctx = SimpleNamespace(
        paths=SimpleNamespace(semantic_metadata=semantic_meta),
        cache_root=tmp_path / "cache",
        index_id="repo:test-bge",
        config=None,
    )

    monkeypatch.setitem(globals_dict, "get_index_context", lambda **_: fake_index_ctx)
    monkeypatch.setitem(
        globals_dict,
        "_rg_structured_predictions",
        lambda *_, **__: [
            {
                "file": "pkg/mod.py",
                "qualified_symbol": None,
                "symbol_kind": "unknown",
                "start_line": 20,
                "end_line": 20,
                "rank": 1,
            }
        ],
    )
    monkeypatch.setitem(
        globals_dict,
        "_semantic_structured_predictions",
        lambda *_, **__: [
            {
                "file": "pkg/mod.py",
                "qualified_symbol": "pkg.mod.py.Target",
                "symbol_kind": "class",
                "start_line": 20,
                "end_line": 20,
                "rank": 1,
            }
        ],
    )

    out_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_structured_retrieval.py",
            "--repo-root",
            str(repo_root),
            "--queries",
            str(queries_path),
            "--semantic-backend",
            "bge=repo:test-bge",
            "--out",
            str(out_path),
        ],
    )

    assert main() == 0

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["results"]["by_backend"]["rg_native"]["micro"]["f1"] == 1.0
    assert report["results"]["by_backend"]["bge"]["micro"]["f1"] == 1.0
    assert report["results"]["comparisons"][0]["lhs"] == "bge"
