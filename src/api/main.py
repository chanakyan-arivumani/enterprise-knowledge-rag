from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, HTTPException

from src.api.dependencies import get_db
from src.api.schemas import QuestionRequest, QuestionResponse
from src.rag.pipeline import answer_question
from src.rag.generation import ContextBudgetError, CitationValidationError

app = FastAPI()

MAX_CONTEXT_CHARS = 2000


@app.post("/questions", response_model=QuestionResponse)
def create_question(
    request: QuestionRequest,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
) -> QuestionResponse:
    try:
        answer, citations = answer_question(
            conn=conn,
            query=request.query,
            k=request.k,
            document_id=request.document_id,
            max_context_chars=MAX_CONTEXT_CHARS,
        )
    except ContextBudgetError as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to construct answer context",
        ) from exc
    except CitationValidationError as exc:
        raise HTTPException(
            status_code=502, detail="Unable to generate a valid cited answer"
        ) from exc
    return QuestionResponse(answer=answer, citations=citations)
