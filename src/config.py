import os
from typing import Annotated
from pydantic import BaseModel, Field, StringConstraints


class Settings(BaseModel):
    database_url: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
        ),
    ]
    embedding_model: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
        ),
    ] = "qwen3-embedding:0.6b"

    generation_model: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
        ),
    ] = "qwen3.5:9b"
    max_context_chars: int = Field(default=2000, gt=0)


def load_settings() -> Settings:
    return Settings.model_validate(
        {
            "database_url": os.environ.get("DATABASE_URL"),
            "embedding_model": os.environ.get(
                "EMBEDDING_MODEL", "qwen3-embedding:0.6b"
            ),
            "generation_model": os.environ.get("GENERATION_MODEL", "qwen3.5:9b"),
            "max_context_chars": os.environ.get("MAX_CONTEXT_CHARS", "2000"),
        }
    )
