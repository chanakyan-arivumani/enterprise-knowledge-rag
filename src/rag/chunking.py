import re
from ingestion.identity import generate_chunk_id
from ingestion.models import Chunk, Document


def chunk_by_sentences(
    sentences: list[str], chunk_size: int, overlap: int
) -> list[dict]:
    """
    [{"id": 1, "text": "Sentence one. Sentence two.", "sentence_ids": {1,2}]
    """
    chunk_id = 1
    result = []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    step = chunk_size - overlap
    for idx in range(0, len(sentences), step):
        chunk_sentences = sentences[idx : idx + chunk_size]
        text = " ".join(chunk_sentences)
        result.append(
            {
                "id": chunk_id,
                "text": text,
                "sentence_ids": set(range(idx + 1, idx + len(chunk_sentences) + 1)),
                # "embedding": generate_embeddings(text),
            }
        )
        chunk_id += 1

    return result


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_document(
    document: Document,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    sentences = split_into_sentences(document.text)

    raw_chunks = chunk_by_sentences(
        sentences,
        chunk_size,
        overlap,
    )

    chunks = []

    for chunk_index, raw_chunk in enumerate(raw_chunks):
        chunk = Chunk(
            chunk_id=generate_chunk_id(
                document.document_id,
                chunk_index,
                raw_chunk["text"],
            ),
            document_id=document.document_id,
            text=raw_chunk["text"],
            metadata={
                "source": document.source,
                "document_type": document.document_type,
                "chunk_index": chunk_index,
                "sentence_ids": sorted(raw_chunk["sentence_ids"]),
            },
        )
        chunks.append(chunk)
    return chunks
