import pytest
from ingestion.identity import generate_chunk_id
from ingestion.models import Chunk, Document
from rag.chunking import chunk_by_sentences, split_into_sentences, chunk_document


def test_zero_overlap_chunking():
    # overlap = 0
    sentences = ["A.", "B.", "C.", "D."]
    chunk_size = 2
    overlap = 0
    expected = ["A. B.", "C. D."]
    result = chunk_by_sentences(sentences, chunk_size, overlap)
    assert len(result) == 2
    assert result[0]["text"] == "A. B."
    assert result[0]["sentence_ids"] == {1, 2}
    assert result[1]["text"] == "C. D."
    assert result[1]["sentence_ids"] == {3, 4}


def test_chunking_with_overlap():
    sentences = ["A.", "B.", "C.", "D."]
    chunk_size = 3
    overlap = 1
    expected = ["A. B. C.", "C. D."]
    result = chunk_by_sentences(sentences, chunk_size, overlap)
    assert len(result) == 2
    assert result[0]["text"] == "A. B. C."
    assert result[0]["sentence_ids"] == {1, 2, 3}
    assert result[1]["text"] == "C. D."
    assert result[1]["sentence_ids"] == {3, 4}


def test_chunking_with_invalid_chunk_size():
    chunk_size = 0
    sentences = ["A.", "B.", "C.", "D."]
    overlap = 1
    with pytest.raises(ValueError, match="chunk_size must be greater than 0") as ex:
        chunk_by_sentences(sentences, chunk_size, overlap)


def test_chunking_with_invalid_overlap():
    chunk_size = 2
    overlap = -1
    sentences = ["A.", "B.", "C.", "D."]
    with pytest.raises(ValueError, match="overlap must be >= 0 and < chunk_size") as ex:
        chunk_by_sentences(sentences, chunk_size, overlap)


def test_chunking_with_overlap_equal_to_chunk_size():
    chunk_size = 2
    overlap = 2
    sentences = ["A.", "B.", "C.", "D."]
    with pytest.raises(ValueError, match="overlap must be >= 0 and < chunk_size"):
        chunk_by_sentences(
            sentences,
            chunk_size,
            overlap,
        )


def test_split_into_sentences():
    text = "Hello world. How are you? I am fine!"

    result = split_into_sentences(text)

    assert result == [
        "Hello world.",
        "How are you?",
        "I am fine!",
    ]


def test_split_into_sentences_across_newlines():
    text = "Hello world.\nHow are you?\nI am fine!"

    result = split_into_sentences(text)

    assert result == [
        "Hello world.",
        "How are you?",
        "I am fine!",
    ]


def test_split_into_sentences_strips_whitespace():
    text = "   Hello world.   How are you?   "

    result = split_into_sentences(text)

    assert result == [
        "Hello world.",
        "How are you?",
    ]


def test_split_into_sentences_empty_text():
    assert split_into_sentences("") == []


def test_chunk_document():
    document = Document(
        document_id="doc-123",
        content_hash="3224567",
        source="docs/test.txt",
        document_type="text",
        text="A. B. C. D.",
        metadata={"category": "test"},
    )
    result = chunk_document(document=document, chunk_size=2, overlap=0)
    assert len(result) == 2
    assert all(isinstance(chunk, Chunk) for chunk in result)
    assert result[0].document_id == "doc-123"
    assert result[0].text == "A. B."
    assert result[0].chunk_index == 0
    assert result[0].metadata["sentence_ids"] == [1, 2]
    assert result[0].metadata["category"] == "test"
    assert result[1].text == "C. D."
    assert result[1].chunk_index == 1
    assert result[1].metadata["sentence_ids"] == [3, 4]

    assert result[0].chunk_id == generate_chunk_id(
        "doc-123",
        0,
        "A. B.",
    )


def test_chunk_document_determinism():
    document = Document(
        document_id="doc-123",
        content_hash="76543456",
        source="docs/test.txt",
        document_type="text",
        text="A. B. C. D.",
        metadata={},
    )
    result_1 = chunk_document(document, chunk_size=2, overlap=0)
    result_2 = chunk_document(document, chunk_size=2, overlap=0)

    assert [chunk.chunk_id for chunk in result_1] == [
        chunk.chunk_id for chunk in result_2
    ]
