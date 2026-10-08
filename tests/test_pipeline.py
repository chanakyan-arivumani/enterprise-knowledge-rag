import pymupdf
import pytest
from ingestion.identity import generate_document_id, generate_content_hash
from ingestion.models import Chunk
from ingestion.pipeline import ingest_file, ingest_directory, ingest_file_to_db
from src.rag import pipeline as rag_pipeline
from rag.embeddings import EMBEDDING_MODEL
import ingestion.pipeline as ingestion_pipeline
from persistence.repository import get_document, get_chunks


def test_ingest_file_with_temporary_text_file(tmp_path):
    file_path = tmp_path / "document.txt"
    file_path.write_text(
        "\ufeff\r\nFirst sentence. Second sentence. Third sentence. Fourth sentence.  \r\n",
        encoding="utf-8",
    )

    chunks = ingest_file(file_path, chunk_size=3, overlap=1)

    assert len(chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.text for chunk in chunks] == [
        "First sentence. Second sentence. Third sentence.",
        "Third sentence. Fourth sentence.",
    ]
    expected_content_hash = generate_content_hash(
        "First sentence. Second sentence. Third sentence. Fourth sentence."
    )
    expected_document_id = generate_document_id(str(file_path))
    assert all(chunk.document_id == expected_document_id for chunk in chunks)
    assert [chunk.metadata["sentence_ids"] for chunk in chunks] == [[1, 2, 3], [3, 4]]
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == str(file_path)
        assert chunk.metadata["path"] == str(file_path)
        assert chunk.metadata["document_type"] == "text"
        assert chunk.chunk_index == index


def test_ingest_file_with_temporary_pdf(tmp_path):
    file_path = tmp_path / "document.pdf"
    text = "First sentence. Second sentence."
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), text)
        pdf.save(file_path)

    chunks = ingest_file(file_path, chunk_size=1, overlap=0)

    assert len(chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.text for chunk in chunks] == ["First sentence.", "Second sentence."]

    expected_document_id = generate_document_id(str(file_path))
    expected_content_hash = generate_content_hash(text)
    for chunk in chunks:
        assert chunk.document_id == expected_document_id
        assert chunk.metadata["source"] == str(file_path)
        assert chunk.metadata["path"] == str(file_path)
        assert chunk.metadata["document_type"] == "pdf"


def test_ingest_files_in_directory(tmp_path):
    file_1_path = tmp_path / "document1.txt"
    file_1_path.write_text(
        "\ufeff\r\nFirst sentence. Second sentence. Third sentence. Fourth sentence.  \r\n",
        encoding="utf-8",
    )
    file_2_path = tmp_path / "document2.txt"
    file_2_path.write_text(
        "Fifth sentence. Sixth sentence. Seventh sentence. Eighth sentence.",
        encoding="utf-8",
    )
    chunks = ingest_directory(tmp_path, chunk_size=3, overlap=1)
    assert len(chunks) == 4
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    sources = {chunk.metadata["source"] for chunk in chunks}

    assert sources == {
        str(file_1_path),
        str(file_2_path),
    }


def test_ingest_file_to_db_rolls_back_failed_update(db_conn, tmp_path, monkeypatch):
    file_path = tmp_path / "document_1.txt"
    file_path.write_text(
        "\ufeff\r\nFirst sentence. Second sentence.  \r\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ingestion_pipeline,
        "generate_embeddings",
        lambda *args, **kwargs: [0.2] * 1024,
    )

    ingest_file_to_db(
        conn=db_conn,
        path=file_path,
        chunk_size=1,
        overlap=0,
        embedding_model=EMBEDDING_MODEL,
    )

    real_update = ingestion_pipeline.update_chunk_embedding

    document_id = generate_document_id(str(file_path))
    original_document = get_document(db_conn, document_id)
    original_chunks = get_chunks(db_conn, document_id)
    assert original_document.document_id == document_id
    assert original_chunks[0].text == "First sentence."
    assert original_chunks[1].text == "Second sentence."

    file_path.write_text(
        "\ufeff\r\nNew first sentence. New second sentence.  \r\n",
        encoding="utf-8",
    )

    write_calls = 0

    def fail_second_write(*args, **kwargs):
        nonlocal write_calls
        write_calls += 1

        if write_calls == 2:
            raise RuntimeError("Simulated embedding write failure")

        return real_update(*args, **kwargs)

    monkeypatch.setattr(
        ingestion_pipeline,
        "update_chunk_embedding",
        fail_second_write,
    )
    with pytest.raises(
        RuntimeError,
        match="Simulated embedding write failure",
    ):
        ingest_file_to_db(
            conn=db_conn,
            path=file_path,
            chunk_size=1,
            overlap=0,
            embedding_model=EMBEDDING_MODEL,
        )
    assert write_calls == 2
    assert get_document(db_conn, document_id) == original_document
    assert get_chunks(db_conn, document_id) == original_chunks
