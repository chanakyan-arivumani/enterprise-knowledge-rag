import psycopg
import pytest
from unittest.mock import Mock
from src.rag import pipeline as rag_pipeline
from rag.retrieval import RetrievalResult
from rag.generation import REFUSAL_ANSWER, CitationValidationError, ContextBudgetError


@pytest.mark.parametrize(
    ("query", "k"),
    [
        ("", 3),
        ("   \n\t", 3),
        ("What is the learning allowance?", 0),
        ("What is the learning allowance?", -1),
    ],
)
def test_invalid_inputs_fail_before_dependencies(monkeypatch, query, k):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Invalid input reached a database or model dependency")

    monkeypatch.setattr(rag_pipeline, "generate_embeddings", unexpected_call)
    monkeypatch.setattr(rag_pipeline, "retrieve_chunks_by_vector", unexpected_call)
    monkeypatch.setattr(rag_pipeline, "generate_answer", unexpected_call)

    with pytest.raises(ValueError):
        rag_pipeline.answer_question(
            conn=object(),
            query=query,
            k=k,
            max_context_chars=10,
        )


def test_empty_retrieval_skips_generation(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline,
        "generate_embeddings",
        lambda *args, **kwargs: [0.1, 0.2],
    )
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_chunks_by_vector",
        lambda *args, **kwargs: [],
    )

    def unexpected_generation(*args, **kwargs):
        pytest.fail("Generation must not run when retrieval returns no evidence")

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        unexpected_generation,
    )

    answer, citation_map = rag_pipeline.answer_question(
        conn=object(),
        query="What is the learning allowance?",
        k=3,
        max_context_chars=50,
    )

    assert answer == "I don't know based on the provided context."
    assert citation_map == []


def test_database_failure_propagates(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline,
        "generate_embeddings",
        lambda *args, **kwargs: [0.1, 0.2],
    )

    def failing_retrieval(*args, **kwargs):
        raise psycopg.OperationalError("Database unavailable")

    def unexpected_generation(*args, **kwargs):
        pytest.fail("Generation must not run after retrieval fails")

    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_chunks_by_vector",
        failing_retrieval,
    )
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        unexpected_generation,
    )

    with pytest.raises(
        psycopg.OperationalError,
        match="Database unavailable",
    ):
        rag_pipeline.answer_question(
            conn=object(),
            query="What is the learning allowance?",
            max_context_chars=50,
        )


def test_embedding_failure_propagates(monkeypatch):
    def failing_embedding(*args, **kwargs):
        raise RuntimeError("Embedding unavailable")

    def unexpected_call(*args, **kwargs):
        pytest.fail("Pipeline continued after embedding failure")

    monkeypatch.setattr(rag_pipeline, "generate_embeddings", failing_embedding)
    monkeypatch.setattr(rag_pipeline, "retrieve_chunks_by_vector", unexpected_call)
    monkeypatch.setattr(rag_pipeline, "generate_answer", unexpected_call)

    with pytest.raises(RuntimeError, match="Embedding unavailable"):
        rag_pipeline.answer_question(
            conn=object(),
            query="What is the learning allowance?",
            max_context_chars=50,
        )


def test_generation_failure_propagates(monkeypatch):
    results = [
        RetrievalResult(
            chunk_id="chunk-1",
            text="The annual learning allowance is INR 30000.",
            score=0.9,
        )
    ]

    monkeypatch.setattr(
        rag_pipeline,
        "generate_embeddings",
        lambda *args, **kwargs: [0.1, 0.2],
    )
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_chunks_by_vector",
        lambda *args, **kwargs: results,
    )

    def failing_generation(*args, **kwargs):
        raise RuntimeError("Generation unavailable")

    monkeypatch.setattr(rag_pipeline, "generate_answer", failing_generation)

    with pytest.raises(RuntimeError, match="Generation unavailable"):
        rag_pipeline.answer_question(
            conn=object(),
            query="What is the learning allowance?",
            max_context_chars=100,
        )


@pytest.mark.parametrize(
    ("generated_answer", "should_raise", "expected_labels"),
    [
        ("The allowance is INR 30000. (Chunk 1)", False, [1]),
        ("The allowance is INR 30000. (Chunk 1) (Chunk 99)", True, []),
        ("The allowance is INR 30000.", True, []),
        (REFUSAL_ANSWER, False, []),
    ],
    ids=["valid", "unknown-label", "missing-citation", "refusal"],
)
def test_pipeline_citation_policy(
    monkeypatch,
    generated_answer,
    should_raise,
    expected_labels,
):
    result = RetrievalResult(
        chunk_id="chunk-1",
        text="The annual learning allowance is INR 30000.",
        score=0.9,
        document_id="doc-1",
        source="policy.txt",
    )

    monkeypatch.setattr(
        rag_pipeline,
        "generate_embeddings",
        lambda *args, **kwargs: [0.1, 0.2],
    )
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_chunks_by_vector",
        lambda *args, **kwargs: [result],
    )
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda *args, **kwargs: generated_answer,
    )

    if should_raise:
        with pytest.raises(CitationValidationError):
            rag_pipeline.answer_question(
                conn=object(),
                query="What is the learning allowance?",
                max_context_chars=100,
            )
        return

    answer, citations = rag_pipeline.answer_question(
        conn=object(),
        query="What is the learning allowance?",
        max_context_chars=100,
    )

    assert answer == generated_answer
    assert [citation.label for citation in citations] == expected_labels


def test_answer_question_raises_when_no_context_fits(monkeypatch):
    results = [
        RetrievalResult(
            chunk_id="chunk-a",
            document_id="doc-a",
            source="test.txt",
            text="Alpha.",
            score=0.9,
        )
    ]

    monkeypatch.setattr(
        rag_pipeline,
        "generate_embeddings",
        lambda *args, **kwargs: [0.1, 0.2],
    )
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve_chunks_by_vector",
        lambda *args, **kwargs: results,
    )
    generate_answer_mock = Mock()

    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        generate_answer_mock,
    )
    with pytest.raises(ContextBudgetError):
        rag_pipeline.answer_question(
            conn=object(),
            query="What does the document say?",
            max_context_chars=13,
        )

    generate_answer_mock.assert_not_called()
