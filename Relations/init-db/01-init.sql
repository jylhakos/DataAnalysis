-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table with vector column
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(384),  -- Dimension for all-MiniLM-L6-v2 model
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search using HNSW
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING hnsw (embedding vector_cosine_ops);

-- Create GIN index for metadata queries
CREATE INDEX IF NOT EXISTS documents_metadata_idx 
ON documents USING GIN (metadata);

-- Create index for created_at for time-based queries
CREATE INDEX IF NOT EXISTS documents_created_at_idx 
ON documents (created_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a helper function for similarity search
CREATE OR REPLACE FUNCTION search_documents(
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id INT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Insert sample document for testing
INSERT INTO documents (content, metadata, embedding) 
VALUES (
    'Vector databases are specialized systems designed to store and query high-dimensional vector embeddings efficiently.',
    '{"source": "system", "type": "documentation", "topic": "vector-databases"}'::jsonb,
    NULL  -- Embedding will be added by the application
) ON CONFLICT DO NOTHING;

-- Display summary
SELECT 
    'Database initialization complete' AS status,
    COUNT(*) AS document_count,
    (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'documents') AS index_count
FROM documents;
