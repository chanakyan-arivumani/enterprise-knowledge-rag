import psycopg

from typing import Annotated
from collections.abc import Iterator
from fastapi import Depends, Request

from src.config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Iterator[psycopg.Connection]:
    with psycopg.connect(
        settings.database_url,
        autocommit=True,
    ) as conn:
        yield conn
