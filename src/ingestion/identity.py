import hashlib


def generate_document_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    value = f"{document_id}:{chunk_index}:{text}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
