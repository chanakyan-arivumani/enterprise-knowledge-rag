import pytest
import pymupdf
from ingestion.parsers import (
    parse_text_file,
    parse_markdown_file,
    parse_pdf_file,
    parse_file,
)
from ingestion.identity import generate_document_id

# Sets the source correctly
# Sets the document type correctly


def test_parse_text_file(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text(
        "Hello! We are testing text parsing with a mock", encoding="utf-8"
    )

    document = parse_text_file(file_path)

    assert document.text == "Hello! We are testing text parsing with a mock"
    assert document.source == str(file_path)
    assert document.document_type == "text"


def test_parse_text_file_normalizes_and_generates_id(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("\r\nHello\nWorld!  \r\n", encoding="utf-8")

    document = parse_text_file(file_path)

    expected_normalized_text = "Hello\nWorld!"
    assert document.text == expected_normalized_text
    assert document.document_id == generate_document_id(expected_normalized_text)


def test_parse_markdown_file(tmp_path):
    file_path = tmp_path / "markdown.md"
    file_path.write_text(
        """# Bengaluru
            **Bengaluru** is the capital of Karnataka.
        """,
        encoding="utf-8",
    )

    document = parse_markdown_file(file_path)

    assert "# Bengaluru" in document.text
    assert "**Bengaluru**" in document.text
    assert document.source == str(file_path)
    assert document.document_type == "markdown"


def test_parse_markdown_file_normalizes_and_generates_id(tmp_path):
    file_path = tmp_path / "markdown.md"
    file_path.write_text(
        """# Bengaluru
            **Bengaluru** is the capital of Karnataka.
        """,
        encoding="utf-8",
    )

    document = parse_markdown_file(file_path)

    expected_normalized_text = """# Bengaluru
            **Bengaluru** is the capital of Karnataka."""
    assert document.text == expected_normalized_text
    assert document.document_id == generate_document_id(expected_normalized_text)


def test_parse_pdf_file(tmp_path):
    file_path = tmp_path / "test.pdf"

    with pymupdf.open() as pdf:
        page_1 = pdf.new_page()
        page_1.insert_text((72, 72), "Text from page 1")

        page_2 = pdf.new_page()
        page_2.insert_text((72, 72), "Text from page 2")

        pdf.save(file_path)

    document = parse_pdf_file(file_path)

    expected_text = "Text from page 1\n\nText from page 2"

    assert document.text == expected_text
    assert document.document_id == generate_document_id(expected_text)
    assert document.source == str(file_path)
    assert document.document_type == "pdf"


def test_parse_file_dispatches_text(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("Hello", encoding="utf-8")

    document = parse_file(file_path)

    assert document.document_type == "text"


def test_parse_file_dispatches_markdown(tmp_path):
    file_path = tmp_path / "README.md"
    file_path.write_text("# Hello", encoding="utf-8")

    document = parse_file(file_path)

    assert document.document_type == "markdown"


def test_parse_file_unsupported_extension(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_file(file_path)


def test_parse_file_dispatches_pdf(tmp_path):
    file_path = tmp_path / "TEST.PDF"

    with pymupdf.open() as pdf:
        page_1 = pdf.new_page()
        page_1.insert_text((72, 72), "Text from page 1")
        pdf.save(file_path)

    document = parse_file(file_path)

    assert document.document_type == "pdf"
