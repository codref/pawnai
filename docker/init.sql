-- Enable pgvector extension in the pawnai database
CREATE EXTENSION IF NOT EXISTS vector;

-- Speaker segment embeddings
-- The embedding column uses pgvector vector(512) for cosine-similarity search.
CREATE TABLE IF NOT EXISTS embeddings (
    id                  TEXT PRIMARY KEY,
    audio_file          TEXT NOT NULL,
    local_speaker_label TEXT NOT NULL,
    start_time          DOUBLE PRECISION NOT NULL,
    end_time            DOUBLE PRECISION NOT NULL,
    embedding           vector(512) NOT NULL
);

-- Human-readable speaker name mappings
CREATE TABLE IF NOT EXISTS speaker_names (
    id                  TEXT PRIMARY KEY,
    audio_file          TEXT NOT NULL,
    local_speaker_label TEXT NOT NULL,
    speaker_name        TEXT NOT NULL,
    labeled_at          TIMESTAMP WITH TIME ZONE
);

-- Index for fast ANN cosine-distance search on embeddings
CREATE INDEX IF NOT EXISTS embeddings_embedding_idx
    ON embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
