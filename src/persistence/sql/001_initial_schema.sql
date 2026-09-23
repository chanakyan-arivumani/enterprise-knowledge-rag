CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    document_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL
        REFERENCES documents(document_id)
        ON DELETE CASCADE,
    text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1024),
    embedding_model TEXT,

    UNIQUE (document_id, chunk_index),
    CONSTRAINT chunks_embedding_model_check
        CHECK (
            (embedding IS NULL) = (embedding_model IS NULL)
        ),
    CONSTRAINT chunks_chunk_index_check
        CHECK (chunk_index >= 0)
);
