from pydantic import BaseModel
from typing import Any


class Document(BaseModel):
    document_id: str  # logical identity
    content_hash: str  # content fingerprint
    source: str
    document_type: str
    text: str
    metadata: dict[str, Any]


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: dict[str, Any]
