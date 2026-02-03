from pathlib import Path

import numpy as np
import pytest

from tldr.indexing import get_index_context
from tldr.semantic import build_semantic_index, semantic_search


class DummyModel:
    def encode(self, texts, batch_size=None, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), 3), dtype=np.float32)


def test_semantic_indexes_isolated(tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    cache_root = tmp_path / "repo"
    cache_root.mkdir()

    scan_a = tmp_path / "dep_a"
    scan_b = tmp_path / "dep_b"
    scan_a.mkdir()
    scan_b.mkdir()

    (scan_a / "a.py").write_text("def func_a():\n    return 1\n")
    (scan_b / "b.py").write_text("def func_b():\n    return 2\n")

    monkeypatch.setattr("tldr.semantic.get_model", lambda *_args, **_kwargs: DummyModel())

    ctx_a = get_index_context(
        scan_root=scan_a,
        cache_root_arg=cache_root,
        index_id_arg="dep:a",
        allow_create=True,
    )
    ctx_b = get_index_context(
        scan_root=scan_b,
        cache_root_arg=cache_root,
        index_id_arg="dep:b",
        allow_create=True,
    )

    build_semantic_index(
        str(scan_a),
        lang="python",
        model="dummy-model",
        show_progress=False,
        index_paths=ctx_a.paths,
        index_config=ctx_a.config,
    )
    build_semantic_index(
        str(scan_b),
        lang="python",
        model="dummy-model",
        show_progress=False,
        index_paths=ctx_b.paths,
        index_config=ctx_b.config,
    )

    assert ctx_a.paths.semantic_metadata.exists()
    assert ctx_b.paths.semantic_metadata.exists()
    assert ctx_a.paths.semantic_metadata != ctx_b.paths.semantic_metadata

    results = semantic_search(
        str(scan_a),
        "function",
        model="dummy-model",
        index_paths=ctx_a.paths,
        index_config=ctx_a.config,
    )
    assert results
    assert all("a.py" in r.get("file", "") for r in results)
