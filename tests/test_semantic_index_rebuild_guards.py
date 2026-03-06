from __future__ import annotations

import numpy as np
import pytest

from tldr.semantic import build_semantic_index, semantic_search


class DummyModel:
    def __init__(self, dim: int):
        self.dim = dim

    def encode(
        self,
        texts,
        batch_size=None,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), self.dim), dtype=np.float32)


def _make_repo(tmp_path):
    scan_root = tmp_path / "repo"
    scan_root.mkdir()
    (scan_root / "auth.py").write_text("def verify_access_token(token):\n    return token\n")
    return scan_root


def test_switching_to_jina_requires_rebuild(tmp_path, monkeypatch) -> None:
    pytest.importorskip("faiss")
    scan_root = _make_repo(tmp_path)

    monkeypatch.setattr(
        "tldr.semantic.get_model",
        lambda model_name=None, **_kwargs: DummyModel(
            896 if model_name == "jina-code-0.5b" else 1024
        ),
    )

    build_semantic_index(
        str(scan_root),
        lang="python",
        model="bge-large-en-v1.5",
        show_progress=False,
    )

    with pytest.raises(ValueError, match="Semantic index model mismatch"):
        build_semantic_index(
            str(scan_root),
            lang="python",
            model="jina-code-0.5b",
            show_progress=False,
        )


def test_searching_with_jina_against_bge_index_raises_model_mismatch(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("faiss")
    scan_root = _make_repo(tmp_path)

    monkeypatch.setattr(
        "tldr.semantic.get_model",
        lambda model_name=None, **_kwargs: DummyModel(
            896 if model_name == "jina-code-0.5b" else 1024
        ),
    )

    build_semantic_index(
        str(scan_root),
        lang="python",
        model="bge-large-en-v1.5",
        show_progress=False,
    )

    with pytest.raises(
        ValueError, match="Semantic search model mismatch with index"
    ):
        semantic_search(
            str(scan_root),
            "find jwt validation",
            model="jina-code-0.5b",
        )
