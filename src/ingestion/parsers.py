import pymupdf
from pathlib import Path
from ingestion.models import Document
from ingestion.normalization import normalize_text
from ingestion.identity import generate_document_id, generate_content_hash


def parse_text_file(path: Path) -> Document:
    """
    Read UTF-8 text, normalize it, generate a document ID,
    and return a Document object.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    document_id = generate_document_id(str(path))
    normalized_text = normalize_text(text)
    content_hash = generate_content_hash(normalized_text)

    return Document(
        document_id=document_id,
        content_hash=content_hash,
        source=str(path),
        document_type="text",
        text=normalized_text,
        metadata={"path": str(path)},
    )


def parse_markdown_file(path: Path) -> Document:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    document_id = generate_document_id(str(path))
    normalized_text = normalize_text(text)
    content_hash = generate_content_hash(normalized_text)

    return Document(
        document_id=document_id,
        content_hash=content_hash,
        source=str(path),
        document_type="markdown",
        text=normalized_text,
        metadata={"path": str(path)},
    )


def parse_pdf_file(path: Path) -> Document:
    page_texts = []
    with pymupdf.open(path) as doc:
        for page in doc:
            page_texts.append(page.get_text())
    pages = "\n".join(page_texts)
    document_id = generate_document_id(str(path))
    normalized_text = normalize_text(pages)
    content_hash = generate_content_hash(normalized_text)

    return Document(
        document_id=document_id,
        content_hash=content_hash,
        source=str(path),
        document_type="pdf",
        text=normalized_text,
        metadata={"path": str(path)},
    )


def parse_file(path: Path) -> Document:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return parse_text_file(path)

    if suffix == ".md":
        return parse_markdown_file(path)

    if suffix == ".pdf":
        return parse_pdf_file(path)

    raise ValueError(f"Unsupported file type: {suffix}")
