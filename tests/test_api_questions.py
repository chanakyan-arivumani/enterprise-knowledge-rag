import pytest
from pydantic import ValidationError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api import main as entrypoint
from src.api.dependencies import get_db

app = entrypoint.app


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test_rag")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("GENERATION_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "2000")


def test_read_main(monkeypatch):
    fake_conn = object()
    captured = {}

    def override_get_db():
        yield fake_conn

    def fake_answer_question(**kwargs):
        captured.update(kwargs)
        return "Test answer", []

    monkeypatch.setattr(entrypoint, "answer_question", fake_answer_question)
    app.dependency_overrides[get_db] = override_get_db

    try:
        monkeypatch.setenv("MAX_CONTEXT_CHARS", "3000")
        monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
        monkeypatch.setenv("GENERATION_MODEL", "test-generation-model")
        with TestClient(app) as client:
            response = client.post(
                "/questions",
                json={
                    "query": "  What is the learning allowance?  ",
                    "k": 2,
                    "document_id": "doc-123",
                },
            )
        assert response.status_code == 200
        assert response.json() == {"answer": "Test answer", "citations": []}

        assert captured["conn"] is fake_conn
        assert captured["query"] == "What is the learning allowance?"
        assert captured["k"] == 2
        assert captured["document_id"] == "doc-123"
        assert captured["max_context_chars"] == 3000
        assert captured["embedding_model"] == "test-embedding-model"
        assert captured["generation_model"] == "test-generation-model"
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_create_question_rejects_blank_query(monkeypatch):
    def override_get_db():
        yield object()

    def unexpected_pipeline_call(**kwargs):
        pytest.fail("Pipeline must not run for an invalid request")

    monkeypatch.setattr(
        entrypoint,
        "answer_question",
        unexpected_pipeline_call,
    )
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            response = client.post(
                "/questions",
                json={"query": " \t\n "},
            )

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(error["loc"] == ["body", "query"] for error in errors)
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.parametrize(
    "exception_type, expected_status, expected_detail",
    [
        (
            entrypoint.ContextBudgetError,
            500,
            "Unable to construct answer context",
        ),
        (
            entrypoint.CitationValidationError,
            502,
            "Unable to generate a valid cited answer",
        ),
    ],
)
def test_create_question_handles_pipeline_error(
    monkeypatch,
    exception_type,
    expected_status,
    expected_detail,
):
    def failing_answer_question(**kwargs):
        raise exception_type("Internal diagnostic: must not reach the client")

    def override_get_db():
        yield object()

    monkeypatch.setattr(
        entrypoint,
        "answer_question",
        failing_answer_question,
    )
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            response = client.post(
                "/questions",
                json={"query": "What is the learning allowance?"},
            )

        assert response.status_code == expected_status
        assert response.json() == {"detail": expected_detail}
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_startup_rejects_missing_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        with TestClient(app):
            pass

    assert any(error["loc"] == ("database_url",) for error in exc_info.value.errors())
