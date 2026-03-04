import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_head_to_head.py")


def _fixture_suite() -> dict[str, dict[str, str]]:
    return {
        "sources": {
            "retrieval_queries": "tests/fixtures/head_to_head/materialize/retrieval_queries.json",
            "structural_queries": "tests/fixtures/head_to_head/materialize/structural_queries.json",
        }
    }


def test_materialize_tasks_valid_fixture_has_zero_warnings_and_stable_hash():
    mod = _load_mod()
    repo_root = Path(__file__).resolve().parents[1]
    corpus_root = repo_root / "tests" / "fixtures" / "head_to_head" / "materialize" / "corpus"
    suite = _fixture_suite()

    tasks_a, warnings_a, source_hashes_a = mod["_materialize_tasks"](
        suite=suite,
        repo_root=repo_root,
        corpus_root=corpus_root,
    )
    tasks_b, warnings_b, source_hashes_b = mod["_materialize_tasks"](
        suite=suite,
        repo_root=repo_root,
        corpus_root=corpus_root,
    )

    assert warnings_a == []
    assert warnings_b == []
    assert len(tasks_a) == 5
    assert tasks_a == tasks_b
    assert source_hashes_a == source_hashes_b

    manifest_hash_a = mod["_sha256_json"](tasks_a)
    manifest_hash_b = mod["_sha256_json"](tasks_b)

    assert manifest_hash_a == manifest_hash_b
    assert manifest_hash_a == "c61d089141308ac56362ad4b2447016491b6286f047d68f561d651184b1c932c"
