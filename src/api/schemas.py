from typing import Any, Annotated
from pydantic import BaseModel, Field, StringConstraints
from rag.models import Citation


class QuestionRequest(BaseModel):
    query: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=2000,
        ),
    ]
    k: int = Field(default=3, gt=0, le=10)
    document_id: str | None = None


class QuestionResponse(BaseModel):
    answer: str
    citations: list[Citation]
