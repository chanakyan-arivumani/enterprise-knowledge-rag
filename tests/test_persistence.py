import pytest
import psycopg
from psycopg.errors import ForeignKeyViolation
from tests.conftest import db_conn
from ingestion.models import Document, Chunk
from persistence.repository import (
    insert_document,
    get_document,
    insert_chunks,
    get_chunks,
    update_document,
    delete_chunks,
)
from persistence.service import persist_document


def test_insert_and_get_document(db_conn):
    document = Document(
        document_id="1234",
        source="/test/doc.txt",
        document_type="text",
        content_hash="erdvfdvffd",
        text="Hello world!",
        metadata={"path": "/test/doc.txt"},
    )

    insert_document(db_conn, document)
    result = get_document(db_conn, document.document_id)

    assert result == document


def test_insert_and_get_document_with_chunks(db_conn):
    document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-123",
        text="A. B.",
        metadata={"path": "/test/doc.txt"},
    )

    chunks = [
        Chunk(
            chunk_id="chunk-1",
            document_id=document.document_id,
            chunk_index=0,
            text="A.",
            metadata={"sentence_ids": [1]},
        ),
        Chunk(
            chunk_id="chunk-2",
            document_id=document.document_id,
            chunk_index=1,
            text="B.",
            metadata={"sentence_ids": [2]},
        ),
    ]

    insert_document(db_conn, document)
    insert_chunks(db_conn, chunks)

    stored_document = get_document(db_conn, document.document_id)
    stored_chunks = get_chunks(db_conn, document.document_id)

    assert stored_document == document
    assert stored_chunks == chunks


def test_insert_orphan_chunk(db_conn):
    chunks = [
        Chunk(
            chunk_id="chunk-1",
            document_id="Document_001",
            chunk_index=0,
            text="A.",
            metadata={"sentence_ids": [1]},
        ),
        Chunk(
            chunk_id="chunk-2",
            document_id="Document_001",
            chunk_index=1,
            text="B.",
            metadata={"sentence_ids": [2]},
        ),
    ]
    with pytest.raises(ForeignKeyViolation):
        insert_chunks(db_conn, chunks)


def test_update_document(db_conn):
    original_document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-123",
        text="A. B.",
        metadata={"path": "/test/doc.txt"},
    )
    insert_document(db_conn, original_document)
    original_document.content_hash = "hash-456"
    original_document.text = "A. B. C. D."
    update_document(db_conn, original_document)
    document_from_db = get_document(db_conn, original_document.document_id)
    assert original_document == document_from_db
    original_document.document_id == document_from_db.document_id
    original_document.content_hash == document_from_db.content_hash
    original_document.text == document_from_db.text
    original_document.metadata == document_from_db.metadata


def test_delete_chunks(db_conn):
    document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-123",
        text="A. B.",
        metadata={"path": "/test/doc.txt"},
    )
    insert_document(db_conn, document)

    chunks = [
        Chunk(
            chunk_id="chunk-1",
            document_id=document.document_id,
            chunk_index=0,
            text="A.",
            metadata={"sentence_ids": [1]},
        ),
        Chunk(
            chunk_id="chunk-2",
            document_id=document.document_id,
            chunk_index=1,
            text="B.",
            metadata={"sentence_ids": [2]},
        ),
    ]
    insert_chunks(db_conn, chunks)
    delete_chunks(db_conn, document.document_id)
    assert get_chunks(db_conn, document.document_id) == []
    assert get_document(db_conn, document.document_id) is not None


def test_persist_new_document(db_conn):
    document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-123",
        text="A. B.",
        metadata={"path": "/test/doc.txt"},
    )
    chunks = [
        Chunk(
            chunk_id="chunk-1",
            document_id=document.document_id,
            chunk_index=0,
            text="A.",
            metadata={"sentence_ids": [1]},
        ),
        Chunk(
            chunk_id="chunk-2",
            document_id=document.document_id,
            chunk_index=1,
            text="B.",
            metadata={"sentence_ids": [2]},
        ),
    ]
    persist_document(db_conn, document, chunks)
    chunks_from_db = get_chunks(db_conn, document.document_id)
    document_from_db = get_document(db_conn, document.document_id)
    assert document == document_from_db
    assert chunks == chunks_from_db


def test_persist_unchanged_document(db_conn):
    original_document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-123",
        text="A. B.",
        metadata={"version": 1},
    )

    chunks = [
        Chunk(
            chunk_id="chunk-1",
            document_id=original_document.document_id,
            chunk_index=0,
            text="A.",
            metadata={"sentence_ids": [1]},
        ),
        Chunk(
            chunk_id="chunk-2",
            document_id=original_document.document_id,
            chunk_index=1,
            text="B.",
            metadata={"sentence_ids": [2]},
        ),
    ]

    persist_document(db_conn, original_document, chunks)

    updated_document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-123",  # unchanged
        text="A. B.",  # unchanged
        metadata={"version": 2},
    )

    persist_document(db_conn, updated_document, chunks)

    document_from_db = get_document(db_conn, updated_document.document_id)
    chunks_from_db = get_chunks(db_conn, updated_document.document_id)

    assert document_from_db == updated_document
    assert chunks_from_db == chunks


def test_persist_changed_document_replaces_chunks(db_conn):
    original_document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-old",
        text="A. B.",
        metadata={"version": 1},
    )

    original_chunks = [
        Chunk(
            chunk_id="chunk-old-1",
            document_id=original_document.document_id,
            chunk_index=0,
            text="A.",
            metadata={"sentence_ids": [1]},
        ),
        Chunk(
            chunk_id="chunk-old-2",
            document_id=original_document.document_id,
            chunk_index=1,
            text="B.",
            metadata={"sentence_ids": [2]},
        ),
    ]

    persist_document(
        db_conn,
        original_document,
        original_chunks,
    )

    updated_document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-new",
        text="C.",
        metadata={"version": 2},
    )

    updated_chunks = [
        Chunk(
            chunk_id="chunk-new-1",
            document_id=updated_document.document_id,
            chunk_index=0,
            text="C.",
            metadata={"sentence_ids": [1]},
        ),
    ]

    persist_document(
        db_conn,
        updated_document,
        updated_chunks,
    )

    document_from_db = get_document(
        db_conn,
        updated_document.document_id,
    )
    chunks_from_db = get_chunks(
        db_conn,
        updated_document.document_id,
    )

    assert document_from_db == updated_document
    assert chunks_from_db == updated_chunks


def test_transaction_rollback_on_failure(db_conn):
    original_document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-old",
        text="A. B.",
        metadata={"version": 1},
    )

    original_chunks = [
        Chunk(
            chunk_id="chunk-old-1",
            document_id=original_document.document_id,
            chunk_index=0,
            text="A.",
            metadata={"sentence_ids": [1]},
        ),
        Chunk(
            chunk_id="chunk-old-2",
            document_id=original_document.document_id,
            chunk_index=1,
            text="B.",
            metadata={"sentence_ids": [2]},
        ),
    ]

    persist_document(
        db_conn,
        original_document,
        original_chunks,
    )

    with db_conn.cursor() as cursor:
        cursor.execute("SAVEPOINT before_replacement")

    updated_document = Document(
        document_id="doc-123",
        source="/test/doc.txt",
        document_type="text",
        content_hash="hash-new",
        text="C. D.",
        metadata={"version": 2},
    )

    updated_chunks = [
        Chunk(
            chunk_id="chunk-new-1",
            document_id=updated_document.document_id,
            chunk_index=0,
            text="C.",
            metadata={"sentence_ids": [1]},
        ),
        Chunk(
            chunk_id="chunk-new-2",
            document_id=updated_document.document_id,
            chunk_index=0,
            text="D.",
            metadata={"sentence_ids": [2]},
        ),
    ]

    with pytest.raises(psycopg.IntegrityError):
        persist_document(
            db_conn,
            updated_document,
            updated_chunks,
        )

    with db_conn.cursor() as cursor:
        cursor.execute("ROLLBACK TO SAVEPOINT before_replacement")

    assert (
        get_document(
            db_conn,
            original_document.document_id,
        )
        == original_document
    )

    assert (
        get_chunks(
            db_conn,
            original_document.document_id,
        )
        == original_chunks
    )
