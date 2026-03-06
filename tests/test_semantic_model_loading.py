from __future__ import annotations

import pytest

from tldr import semantic


class FakeSentenceTransformer:
    instances = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        FakeSentenceTransformer.instances.append(self)

    def encode(
        self,
        texts,
        batch_size=None,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        return [[0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture(autouse=True)
def reset_model_cache() -> None:
    semantic._reset_cached_model()
    try:
        yield
    finally:
        semantic._reset_cached_model()


def test_jina_loading_uses_left_padding(monkeypatch) -> None:
    import sentence_transformers

    FakeSentenceTransformer.instances.clear()
    semantic._reset_cached_model()
    monkeypatch.setattr(semantic, "_model_exists_locally", lambda _hf_name: True)
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", FakeSentenceTransformer
    )

    model = semantic.get_model("jina-code-0.5b", device="cpu")

    assert model is FakeSentenceTransformer.instances[0]
    assert model.model_name == "jinaai/jina-code-embeddings-0.5b"
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["tokenizer_kwargs"] == {"padding_side": "left"}


def test_bge_and_minilm_do_not_force_left_padding(monkeypatch) -> None:
    import sentence_transformers

    FakeSentenceTransformer.instances.clear()
    semantic._reset_cached_model()
    monkeypatch.setattr(semantic, "_model_exists_locally", lambda _hf_name: True)
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", FakeSentenceTransformer
    )

    semantic.get_model("bge-large-en-v1.5", device="cpu")
    semantic._reset_cached_model()
    semantic.get_model("all-MiniLM-L6-v2", device="cpu")

    assert len(FakeSentenceTransformer.instances) == 2
    assert "tokenizer_kwargs" not in FakeSentenceTransformer.instances[0].kwargs
    assert "tokenizer_kwargs" not in FakeSentenceTransformer.instances[1].kwargs


def test_model_cache_invalidates_when_model_or_device_changes(monkeypatch) -> None:
    import sentence_transformers

    FakeSentenceTransformer.instances.clear()
    semantic._reset_cached_model()
    monkeypatch.setattr(semantic, "_model_exists_locally", lambda _hf_name: True)
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", FakeSentenceTransformer
    )

    first = semantic.get_model("jina-code-0.5b", device="cpu")
    second = semantic.get_model("jina-code-0.5b", device="cpu")
    third = semantic.get_model("jina-code-0.5b", device="mps")
    fourth = semantic.get_model("bge-large-en-v1.5", device="mps")

    assert first is second
    assert first is not third
    assert third is not fourth
    assert len(FakeSentenceTransformer.instances) == 3
