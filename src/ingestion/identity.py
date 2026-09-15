import hashlib


def generate_document_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
