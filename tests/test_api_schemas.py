import pytest
from pydantic import ValidationError

from src.api.schemas import QuestionRequest


def test_query_is_trimmed_before_length_validation():
    query = "x" * 2000

    request = QuestionRequest(query=f"  {query}  ")

    assert request.query == query


@pytest.mark.parametrize(
    "query",
    [
        "",  # Empty
        " \t\n ",  # Empty after trimming
        "x" * 2001,  # Too long after trimming
    ],
)
def test_invalid_query_is_rejected(query):
    with pytest.raises(ValidationError):
        QuestionRequest(query=query)
