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

    UNIQUE (document_id, chunk_index)
);
