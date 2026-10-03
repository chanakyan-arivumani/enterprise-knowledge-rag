from pydantic import BaseModel


class Citation(BaseModel):
    label: int
    chunk_id: str
    document_id: str
    source: str
    excerpt: str
