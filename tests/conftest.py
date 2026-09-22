import os
import pytest
import psycopg


@pytest.fixture
def db_conn():
    conn = psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )

    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()
