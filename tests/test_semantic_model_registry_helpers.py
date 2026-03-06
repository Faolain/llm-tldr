from __future__ import annotations

from tldr.semantic import (
    _canonical_model_id,
    _model_dimension,
    _query_prefix_for_model,
    _resolve_hf_model_name,
)


def test_model_registry_helpers_round_trip_supported_keys_and_hf_ids() -> None:
    jina_hf = "jinaai/jina-code-embeddings-0.5b"

    assert _resolve_hf_model_name("jina-code-0.5b") == jina_hf
    assert _canonical_model_id("jina-code-0.5b") == jina_hf
    assert _canonical_model_id(jina_hf) == jina_hf
    assert _model_dimension("jina-code-0.5b") == 896


def test_bge_registry_contract_remains_unchanged() -> None:
    assert _resolve_hf_model_name("bge-large-en-v1.5") == "BAAI/bge-large-en-v1.5"
    assert _canonical_model_id("bge-large-en-v1.5") == "BAAI/bge-large-en-v1.5"
    assert _model_dimension("bge-large-en-v1.5") == 1024
    assert (
        _query_prefix_for_model("bge-large-en-v1.5")
        == "Represent this code search query: "
    )
