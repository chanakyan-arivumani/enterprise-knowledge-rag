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


def test_ingest_file_to_db_rejects_empty_update(db_conn, tmp_path, monkeypatch):
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
    document_id = generate_document_id(str(file_path))
    original_document = get_document(db_conn, document_id)
    original_chunks = get_chunks(db_conn, document_id)
    assert len(original_chunks) == 2
    assert all(chunk.embedding is not None for chunk in original_chunks)
    assert all(chunk.embedding_model == EMBEDDING_MODEL for chunk in original_chunks)

    file_path.write_text(" \n\t  \n", encoding="utf-8")

    with pytest.raises(ValueError):
        ingest_file_to_db(
            conn=db_conn,
            path=file_path,
            chunk_size=1,
            overlap=0,
            embedding_model=EMBEDDING_MODEL,
        )

    assert get_document(db_conn, document_id) == original_document
    assert get_chunks(db_conn, document_id) == original_chunks


def test_ingest_file_to_db_reembeds_when_model_changes(db_conn, tmp_path, monkeypatch):
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
        embedding_model="model-a",
    )
    document_id = generate_document_id(str(file_path))
    original_document = get_document(db_conn, document_id)
    original_chunks = get_chunks(db_conn, document_id)

    monkeypatch.setattr(
        ingestion_pipeline,
        "generate_embeddings",
        lambda *args, **kwargs: [0.1] * 1024,
    )
    ingest_file_to_db(
        conn=db_conn,
        path=file_path,
        chunk_size=1,
        overlap=0,
        embedding_model="model-b",
    )

    updated_document = get_document(db_conn, document_id)
    updated_chunks = get_chunks(db_conn, document_id)
    assert document_id == updated_document.document_id

    for updated_chunk, original_chunk in zip(
        updated_chunks, original_chunks, strict=True
    ):
        assert updated_chunk.chunk_id == original_chunk.chunk_id
        assert updated_chunk.embedding_model == "model-b"
        assert updated_chunk.embedding == [0.1] * 1024


@pytest.mark.parametrize(
    ("updated_text", "updated_chunk_size", "expected_chunk_texts"),
    [
        pytest.param(
            "Updated first sentence. Updated second sentence. Third sentence.",
            1,
            [
                "Updated first sentence.",
                "Updated second sentence.",
                "Third sentence.",
            ],
            id="content_changed",
        ),
        pytest.param(
            "First sentence. Second sentence.",
            2,
            [
                "First sentence. Second sentence.",
            ],
            id="chunking_changed",
        ),
    ],
)
def test_ingest_file_to_db_rebuilds_changed_content_or_chunking(
    db_conn,
    tmp_path,
    monkeypatch,
    updated_text,
    updated_chunk_size,
    expected_chunk_texts,
):
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
        embedding_model="model-a",
    )

    file_path.write_text(updated_text, encoding="utf-8")
    ingest_file_to_db(
        conn=db_conn,
        path=file_path,
        chunk_size=updated_chunk_size,
        overlap=0,
        embedding_model=EMBEDDING_MODEL,
    )
    document_id = generate_document_id(str(file_path))
    stored_chunks = get_chunks(db_conn, document_id)

    assert [chunk.text for chunk in stored_chunks] == expected_chunk_texts
