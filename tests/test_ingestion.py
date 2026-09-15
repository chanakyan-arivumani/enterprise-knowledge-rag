import pytest
from ingestion.identity import generate_document_id


def test_generate_document_id_same_text_same_id():
    text_1 = "testing document id generation"
    text_2 = "testing document id generation"
    assert generate_document_id(text_1) == generate_document_id(text_2)


def test_generate_document_id_diff_text_diff_id():
    text_1 = "testing document id generation"
    text_2 = "testing document id generation with different text"
    assert generate_document_id(text_1) != generate_document_id(text_2)


def test_generate_document_id_known_text():
    text = "testing document id generation for known text"
    assert (
        generate_document_id(text)
        == "5c4bb1a1a88b0c5777e89b206aafec0b7b8fc8ad1d009e98afbd1553fd2ee3d6"
    )
