"""
Load sample documents into the vector database.
This script reads text files, generates embeddings, and stores them in PostgreSQL.
"""

import os
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}

# Embedding model
MODEL_NAME = 'all-MiniLM-L6-v2'  # 384 dimensions


def load_documents_from_directory(directory: str):
    """Load documents from a directory."""
    documents = []
    doc_dir = Path(directory)
    
    if not doc_dir.exists():
        print(f"Creating directory: {directory}")
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample documents
        sample_docs = [
            ("vector_databases.txt", """Vector databases are specialized systems designed to store and query high-dimensional vector embeddings efficiently. They enable semantic search, where similar items are found based on meaning rather than exact keyword matches. Common algorithms include HNSW (Hierarchical Navigable Small World) and IVF (Inverted File Index)."""),
            
            ("llm_integration.txt", """Large Language Models (LLMs) integrate with vector databases through Retrieval Augmented Generation (RAG). This approach allows LLMs to retrieve relevant context from external knowledge bases before generating responses. The workflow includes: converting queries to embeddings, searching the vector database, retrieving relevant documents, and augmenting the LLM prompt with this context."""),
            
            ("postgresql_pgvector.txt", """PostgreSQL with the pgvector extension provides native vector search capabilities within a traditional relational database. This allows storing structured data and vector embeddings together, enabling hybrid queries that combine SQL filtering with semantic similarity search. It's ideal for applications needing both transactional integrity and vector operations."""),
            
            ("embeddings_explained.txt", """Embeddings are numerical representations of data that capture semantic meaning. Text, images, and other content are converted into high-dimensional vectors where similar items are positioned close together in vector space. Common embedding models include Sentence-BERT, OpenAI's text-embedding-ada-002, and various multimodal models."""),
            
            ("rag_workflow.txt", """Retrieval Augmented Generation (RAG) enhances LLM responses by retrieving relevant information from external sources. The process involves: 1) User query 2) Query embedding generation 3) Vector similarity search 4) Context retrieval 5) Prompt augmentation 6) LLM generation. This reduces hallucinations and provides up-to-date information beyond the model's training data."""),
        ]
        
        for filename, content in sample_docs:
            filepath = doc_dir / filename
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created sample document: {filename}")
    
    for filepath in doc_dir.glob('*.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                documents.append({
                    'content': content,
                    'filename': filepath.name,
                    'source': str(filepath)
                })
    
    return documents


def generate_embeddings(documents, model):
    """Generate embeddings for documents."""
    print(f"Generating embeddings using {MODEL_NAME}...")
    texts = [doc['content'] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def store_documents(documents, embeddings):
    """Store documents and embeddings in PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        print(f"\nStoring {len(documents)} documents...")
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Convert embedding to list for PostgreSQL
            embedding_list = embedding.tolist()
            
            # Prepare metadata
            metadata = json.dumps({
                'filename': doc['filename'],
                'source': doc['source'],
                'length': len(doc['content'])
            })
            
            # Insert document
            cur.execute("""
                INSERT INTO documents (content, metadata, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (doc['content'], metadata, embedding_list))
            
            print(f"  ✓ Stored: {doc['filename']}")
        
        conn.commit()
        
        # Get total count
        cur.execute("SELECT COUNT(*) FROM documents;")
        total = cur.fetchone()[0]
        print(f"\n✓ Total documents in database: {total}")
        
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error storing documents: {e}")
        return False


def main():
    print("=" * 60)
    print("Load Documents Script")
    print("=" * 60)
    
    # Define data directory
    data_dir = Path(__file__).parent.parent / 'data' / 'sample_documents'
    
    # Load documents
    print(f"\nLoading documents from: {data_dir}")
    documents = load_documents_from_directory(str(data_dir))
    
    if not documents:
        print("✗ No documents found!")
        return False
    
    print(f"✓ Loaded {len(documents)} documents")
    
    # Load embedding model
    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"✓ Model loaded (embedding dimension: {model.get_sentence_embedding_dimension()})")
    
    # Generate embeddings
    embeddings = generate_embeddings(documents, model)
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    # Store in database
    success = store_documents(documents, embeddings)
    
    if success:
        print("\n✓ Documents loaded successfully!")
    else:
        print("\n✗ Failed to load documents")
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
