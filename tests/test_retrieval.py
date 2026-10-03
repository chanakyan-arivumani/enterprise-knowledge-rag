import pytest
import psycopg

from ingestion.models import Document, Chunk
from src.rag.retrieval import retrieve_chunks_by_vector
from persistence.repository import (
    insert_document,
    get_document,
    insert_chunks,
    get_chunks,
    update_document,
    delete_chunks,
    update_chunk_embedding,
)
from persistence.service import persist_document
from tests.conftest import db_conn


def test_retrieve_chunks_by_vector_ranks_chunks(db_conn):
    original_document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-123",
        text="A. B. C.",
        metadata={"version": 1},
    )

    chunks = [
        Chunk(
            chunk_id="chunk-c",
            document_id=original_document.document_id,
            chunk_index=2,
            text="C.",
            metadata={"sentence_ids": [3]},
        ),
        Chunk(
            chunk_id="chunk-a",
            document_id=original_document.document_id,
            chunk_index=0,
            text="A.",
            metadata={"sentence_ids": [1]},
        ),
        Chunk(
            chunk_id="chunk-b",
            document_id=original_document.document_id,
            chunk_index=1,
            text="B.",
            metadata={"sentence_ids": [2]},
        ),
    ]

    persist_document(db_conn, original_document, chunks)
    embedding_a = [1.0, 0.0] + [0.0] * 1022
    embedding_b = [0.6, 0.8] + [0.0] * 1022
    embedding_c = [0.0, 1.0] + [0.0] * 1022
    update_chunk_embedding(
        db_conn,
        chunk_id="chunk-c",
        embedding=embedding_c,
        embedding_model="test-model",
    )
    update_chunk_embedding(
        db_conn,
        chunk_id="chunk-a",
        embedding=embedding_a,
        embedding_model="test-model",
    )
    update_chunk_embedding(
        db_conn,
        chunk_id="chunk-b",
        embedding=embedding_b,
        embedding_model="test-model",
    )
    query_embedding = [1.0, 0.0] + [0.0] * 1022
    results = retrieve_chunks_by_vector(db_conn, query_embedding, "test-model", 2)
    assert [result.chunk_id for result in results] == ["chunk-a", "chunk-b"]
    assert [result.score for result in results] == pytest.approx([1.0, 0.6])
    assert results[0].document_id == original_document.document_id
    assert results[0].source == original_document.source


def test_retrieve_chunks_by_vector_filters_ineligible_chunks(db_conn):
    original_document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-123",
        text="A. B. C.",
        metadata={"version": 1},
    )

    chunks = [
        Chunk(
            chunk_id="chunk-valid",
            document_id=original_document.document_id,
            chunk_index=2,
            text="C.",
            metadata={"sentence_ids": [3]},
        ),
        Chunk(
            chunk_id="chunk-other-model",
            document_id=original_document.document_id,
            chunk_index=0,
            text="A.",
            metadata={"sentence_ids": [1]},
        ),
        Chunk(
            chunk_id="chunk-b",
            document_id=original_document.document_id,
            chunk_index=1,
            text="B.",
            metadata={"sentence_ids": [2]},
        ),
    ]
    persist_document(db_conn, original_document, chunks)
    update_chunk_embedding(
        db_conn,
        chunk_id="chunk-valid",
        embedding=[0.6, 0.8] + [0.0] * 1022,
        embedding_model="test-model",
    )
    update_chunk_embedding(
        db_conn,
        chunk_id="chunk-other-model",
        embedding=[1.0, 0.0] + [0.0] * 1022,
        embedding_model="other-model",
    )

    query_embedding = [1.0, 0.0] + [0.0] * 1022

    results = retrieve_chunks_by_vector(
        db_conn,
        query_embedding,
        "test-model",
        k=3,
    )

    assert [result.chunk_id for result in results] == ["chunk-valid"]
    assert results[0].score == pytest.approx(0.6)
    assert results[0].document_id == original_document.document_id
    assert results[0].source == original_document.source

    results = retrieve_chunks_by_vector(
        db_conn,
        query_embedding,
        embedding_model="missing-model",
        k=3,
    )

    assert results == []
