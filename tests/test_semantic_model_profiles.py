from __future__ import annotations

from tldr.semantic import SUPPORTED_MODELS


def test_jina_code_profile_is_registered() -> None:
    profile = SUPPORTED_MODELS["jina-code-0.5b"]

    assert profile["hf_name"] == "jinaai/jina-code-embeddings-0.5b"
    assert profile["dimension"] == 896
    assert (
        profile["query_prefix"]
        == "Find the most relevant code snippet given the following query:\n"
    )
    assert profile["document_prefix"] == "Candidate code snippet:\n"
    assert profile["tokenizer_kwargs"]["padding_side"] == "left"
