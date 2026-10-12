import os
from collections.abc import Iterator

import psycopg


def get_db() -> Iterator[psycopg.Connection]:
    with psycopg.connect(
        os.environ["DATABASE_URL"],
        autocommit=True,
    ) as conn:
        yield conn
