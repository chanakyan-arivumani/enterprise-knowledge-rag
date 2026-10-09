from pathlib import Path
import psycopg
from ingestion.models import Chunk
from ingestion.parsers import parse_file
from rag.chunking import chunk_document
from rag.embeddings import generate_embeddings, EMBEDDING_MODEL
from persistence.repository import update_chunk_embedding
from persistence.service import persist_document


def ingest_file(
    path: Path,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    document = parse_file(path)
    return chunk_document(
        document,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def ingest_directory(
    path: Path,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file in sorted(path.iterdir()):
        if file.is_file():
            chunks.extend(
                ingest_file(path=file, chunk_size=chunk_size, overlap=overlap)
            )
    return chunks


def ingest_file_to_db(
    conn: psycopg.Connection,
    path: Path,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
) -> None:
    document = parse_file(path)

    if not document.text.strip():
        raise ValueError("Document contains no usable text")

    chunks = chunk_document(
        document,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    chunk_embeddings = [
        generate_embeddings(chunk.text, model=embedding_model) for chunk in chunks
    ]

    with conn.transaction():
        persist_document(conn, document, chunks)
        for chunk, embedding in zip(chunks, chunk_embeddings, strict=True):
            update_chunk_embedding(conn, chunk.chunk_id, embedding, embedding_model)
