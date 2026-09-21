from pathlib import Path
from ingestion.models import Chunk
from ingestion.parsers import parse_file
from rag.chunking import chunk_document


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
