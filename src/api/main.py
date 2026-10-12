import psycopg

from typing import Annotated
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException

from src.config import load_settings, Settings
from src.api.dependencies import get_db, get_settings
from src.api.schemas import QuestionRequest, QuestionResponse
from src.rag.pipeline import answer_question
from src.rag.generation import ContextBudgetError, CitationValidationError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.settings = load_settings()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/questions", response_model=QuestionResponse)
def create_question(
    request: QuestionRequest,
    conn: Annotated[psycopg.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> QuestionResponse:
    try:
        answer, citations = answer_question(
            conn=conn,
            query=request.query,
            k=request.k,
            document_id=request.document_id,
            max_context_chars=settings.max_context_chars,
            embedding_model=settings.embedding_model,
            generation_model=settings.generation_model,
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
