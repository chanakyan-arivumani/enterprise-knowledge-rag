import json
import pytest
import psycopg
from psycopg.types.json import Jsonb
from ingestion.models import Document, Chunk


def insert_document(
    conn: psycopg.Connection,
    document: Document,
) -> None:
    query = """
        INSERT INTO documents (
            document_id,
            source,
            document_type,
            content_hash,
            text,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s);
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                document.document_id,
                document.source,
                document.document_type,
                document.content_hash,
                document.text,
                Jsonb(document.metadata),
            ),
        )


def get_document(conn: psycopg.Connection, document_id: str) -> Document | None:
    query = """
        SELECT document_id,
            source,
            document_type,
            content_hash,
            text,
            metadata
        FROM documents 
        WHERE document_id=%s;
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (document_id,))
        row = cursor.fetchone()
    if row is None:
        return None
    return Document(
        document_id=row[0],
        source=row[1],
        document_type=row[2],
        content_hash=row[3],
        text=row[4],
        metadata=row[5],
    )


def insert_chunks(
    conn: psycopg.Connection,
    chunks: list[Chunk],
) -> None:
    query = """
        INSERT INTO chunks (
            chunk_id,
            document_id,
            chunk_index,
            text,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s);
    """
    with conn.cursor() as cursor:
        chunks_to_insert = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.text,
                Jsonb(chunk.metadata),
            )
            for chunk in chunks
        ]
        cursor.executemany(query, chunks_to_insert)


def get_chunks(
    conn: psycopg.Connection,
    document_id: str,
) -> list[Chunk]:
    query = """
        SELECT 
            chunk_id, 
            document_id, 
            chunk_index, 
            text, 
            metadata,
            embedding::text as embedding,
            embedding_model
        FROM chunks
        WHERE document_id=%s
        ORDER BY chunk_index;
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (document_id,))
        rows = cursor.fetchall()

    chunks = [
        Chunk(
            chunk_id=row[0],
            document_id=row[1],
            chunk_index=row[2],
            text=row[3],
            metadata=row[4],
            embedding=(json.loads(row[5]) if row[5] is not None else None),
            embedding_model=row[6],
        )
        for row in rows
    ]
    return chunks


def update_document(
    conn: psycopg.Connection,
    document: Document,
) -> None:
    query = """
        UPDATE documents
        SET
            source = %s,
            document_type = %s,
            content_hash = %s,
            text = %s,
            metadata = %s
        WHERE document_id = %s;
    """
    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                document.source,
                document.document_type,
                document.content_hash,
                document.text,
                Jsonb(document.metadata),
                document.document_id,
            ),
        )


def delete_chunks(
    conn: psycopg.Connection,
    document_id: str,
) -> None:
    query = """
        DELETE from chunks
        WHERE document_id=%s
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (document_id,))


def update_chunk_embedding(
    conn: psycopg.Connection,
    chunk_id: str,
    embedding: list[float],
    embedding_model: str,
) -> None:
    if len(embedding) != 1024:
        raise ValueError("Embeddings must be 1024 in length")

    if not embedding_model.strip():
        raise ValueError("Embedding model cannot be empty")

    query = """
        UPDATE chunks
        SET embedding = %s::vector,
            embedding_model = %s
        WHERE chunk_id = %s;
    """
    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                str(embedding),
                embedding_model,
                chunk_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Chunk {chunk_id} not found")


@pytest.mark.parametrize(
    "embedding, embedding_model, expected_error",
    [
        ([0.1] * 1023, "qwen3-embedding:0.6b", "1024"),
        ([0.1] * 1025, "qwen3-embedding:0.6b", "1024"),
        ([0.1] * 1024, "", "model"),
        ([0.1] * 1024, "   ", "model"),
    ],
)
def test_update_chunk_embedding_invalid_input(
    db_conn,
    embedding,
    embedding_model,
    expected_error,
):
    with pytest.raises(ValueError, match=expected_error):
        update_chunk_embedding(
            db_conn,
            chunk_id="unused-chunk",
            embedding=embedding,
            embedding_model=embedding_model,
        )
