import psycopg
from ingestion.models import Document, Chunk
from persistence.repository import (
    get_document,
    insert_document,
    update_document,
    get_chunks,
    insert_chunks,
    delete_chunks,
)


def persist_document(
    conn: psycopg.Connection,
    document: Document,
    chunks: list[Chunk],
) -> None:
    if any(chunk.document_id != document.document_id for chunk in chunks):
        raise ValueError("All chunks must belong to the document")

    existing_doc = get_document(conn, document.document_id)

    if existing_doc is None:
        insert_document(conn, document)
        insert_chunks(conn, chunks)
        return

    if existing_doc.content_hash == document.content_hash:
        stored_chunks = get_chunks(conn, existing_doc.document_id)
        stored_ids = [
            chunk.chunk_id
            for chunk in sorted(stored_chunks, key=lambda c: c.chunk_index)
        ]
        incoming_ids = [
            chunk.chunk_id for chunk in sorted(chunks, key=lambda c: c.chunk_index)
        ]

        if stored_ids == incoming_ids:
            # Same content and chunking: preserve stored embeddings.
            update_document(conn, document)
            return

    # Content changed OR chunking changed.
    update_document(conn, document)
    delete_chunks(conn, document.document_id)
    insert_chunks(conn, chunks)
