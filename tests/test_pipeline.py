import pymupdf

from ingestion.identity import generate_document_id, generate_content_hash
from ingestion.models import Chunk
from ingestion.pipeline import ingest_file, ingest_directory


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
        assert chunk.metadata["chunk_index"] == index


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
