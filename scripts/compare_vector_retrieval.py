import math
import os
import pytest
import psycopg
from pathlib import Path

from ingestion.parsers import parse_file
from rag.chunking import chunk_document
from rag.embeddings import generate_embeddings, EMBEDDING_MODEL
from rag.retrieval import rank_chunks_by_vector, retrieve_chunks_by_vector
from persistence.service import persist_document
from persistence.repository import get_chunks, update_chunk_embedding


def main() -> None:
    document = parse_file(
        Path("/home/chan/projects/ai-rag/evals/datasets/sample_company_policy.txt")
    )
    chunks = chunk_document(document, chunk_size=2, overlap=0)

    """****************** Writing document and chunks ******************"""

    chunk_embeddings = [
        generate_embeddings(chunk.text, model=EMBEDDING_MODEL) for chunk in chunks
    ]
    query = "How much can an employee spend on professional learning each year?"
    query_embedding = generate_embeddings(
        query,
        model=EMBEDDING_MODEL,
    )

    with psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    ) as conn:
        persist_document(conn, document, chunks)

        for chunk, embedding in zip(chunks, chunk_embeddings, strict=True):
            update_chunk_embedding(
                conn,
                chunk_id=chunk.chunk_id,
                embedding=embedding,
                embedding_model=EMBEDDING_MODEL,
            )

    """****************** Verifying stored chunks ******************"""

    with psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    ) as conn:
        stored_chunks = get_chunks(conn, document.document_id)

        assert len(stored_chunks) == len(chunks) == 6
        assert {chunk.chunk_id for chunk in stored_chunks} == {
            chunk.chunk_id for chunk in chunks
        }

        for chunk in stored_chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1024
            assert chunk.embedding_model == EMBEDDING_MODEL

        print(f"Verified {len(stored_chunks)} embedded chunks after reconnecting.")

        python_chunks = [
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": chunk.embedding,
            }
            for chunk in stored_chunks
        ]

        k = 3

        python_results = rank_chunks_by_vector(query_embedding, python_chunks, k)
        db_results = retrieve_chunks_by_vector(
            conn,
            query_embedding,
            EMBEDDING_MODEL,
            k=k,
            document_id=document.document_id,
        )

        for python_result, db_result in zip(python_results, db_results, strict=True):
            print(
                f"Same chunk: {python_result.chunk_id == db_result.chunk_id} | "
                f"Python: {python_result.score:.8f} | "
                f"PostgreSQL: {db_result.score:.8f}"
            )
            print(db_result.text)

        assert len(python_results) == len(db_results) == k

        for python_result, db_result in zip(python_results, db_results, strict=True):
            assert python_result.chunk_id == db_result.chunk_id
            assert math.isclose(
                python_result.score,
                db_result.score,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )


if __name__ == "__main__":
    main()
