import psycopg
from ingestion.models import Document, Chunk
from persistence.repository import (
    get_document,
    insert_document,
    insert_chunks,
    update_document,
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
        update_document(conn, document)
        return

    update_document(conn, document)
    delete_chunks(conn, document.document_id)
    insert_chunks(conn, chunks)
