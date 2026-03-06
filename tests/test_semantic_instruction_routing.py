from __future__ import annotations

from tldr.semantic import (
    EmbeddingUnit,
    build_document_embedding_text,
    build_embedding_text,
    build_query_embedding_text,
)


def _unit() -> EmbeddingUnit:
    return EmbeddingUnit(
        name="verify_access_token",
        qualified_name="auth.verify_access_token",
        file="auth.py",
        line=1,
        language="python",
        unit_type="function",
        signature="def verify_access_token(token: str) -> dict",
        docstring="Validate a JWT and return claims.",
        code_preview="claims = decode(token)\nreturn claims",
    )


def test_jina_document_embedding_uses_candidate_prefix() -> None:
    text = build_document_embedding_text(_unit(), "jina-code-0.5b")
    assert text.startswith("Candidate code snippet:\n")
    assert "Function: verify_access_token" in text


def test_jina_query_embedding_uses_nl2code_prefix() -> None:
    assert (
        build_query_embedding_text("find jwt validation", "jina-code-0.5b")
        == "Find the most relevant code snippet given the following query:\n"
        "find jwt validation"
    )


def test_bge_query_embedding_keeps_existing_prefix() -> None:
    assert (
        build_query_embedding_text("find jwt validation", "bge-large-en-v1.5")
        == "Represent this code search query: find jwt validation"
    )


def test_bge_document_embedding_remains_unprefixed() -> None:
    unit = _unit()
    assert build_document_embedding_text(unit, "bge-large-en-v1.5") == build_embedding_text(unit)
