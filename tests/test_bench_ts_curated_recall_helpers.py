import json
import runpy
import sys
from pathlib import Path


def _load_mod():
    scripts_dir = (Path(__file__).resolve().parents[1] / "scripts").as_posix()
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return runpy.run_path("scripts/bench_ts_curated_recall.py")


def test_load_graph_cache_rejects_language_mismatch(tmp_path: Path):
    mod = _load_mod()
    load_graph_cache = mod["_load_graph_cache"]

    cache_path = tmp_path / "call_graph.json"
    cache_path.write_text(
        json.dumps(
            {
                "languages": ["python"],
                "edges": [
                    {
                        "from_file": "a.py",
                        "from_func": "fa",
                        "to_file": "b.py",
                        "to_func": "fb",
                    }
                ],
                "meta": {"source": "test"},
            }
        )
    )

    assert load_graph_cache(cache_path, lang="typescript") is None
