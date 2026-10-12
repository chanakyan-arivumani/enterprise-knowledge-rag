import pytest
from pydantic import ValidationError

from src.config import load_settings


def test_load_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "  postgresql://localhost/test_rag  ")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("GENERATION_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "3000")

    settings = load_settings()

    assert settings.database_url == "postgresql://localhost/test_rag"
    assert settings.max_context_chars == 3000


def test_load_settings_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        load_settings()

    assert any(error["loc"] == ("database_url",) for error in exc_info.value.errors())
