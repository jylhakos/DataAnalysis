"""
Flask REST API server for RAG chatbot.
Provides endpoints for querying the vector database and generating responses with Ollama.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from sentence_transformers import SentenceTransformer
import ollama
import json

app = Flask(__name__)
CORS(app)

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}

MODEL_NAME = 'all-MiniLM-L6-v2'
OLLAMA_MODEL = 'llama3.2:3b'
TOP_K = 3


# Initialize embedding model
print(f"Loading embedding model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("✓ Model loaded")


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(**DB_CONFIG)


def search_similar_documents(query: str, top_k: int = TOP_K):
    """Search for similar documents in the vector database."""
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search database
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        documents = []
        for row in results:
            doc_id, content, metadata, similarity = row
            documents.append({
                'id': doc_id,
                'content': content,
                'metadata': metadata,
                'similarity': float(similarity)
            })
        
        return documents
        
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


def generate_response(query: str, context_docs: list):
    """Generate response using Ollama with retrieved context."""
    try:
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Document {i+1} (similarity: {doc['similarity']:.2f}):\n{doc['content']}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Build prompt
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response with Ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: Could not generate response. Make sure Ollama is running with model '{OLLAMA_MODEL}'."


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        # Check database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents;")
        doc_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        # Check Ollama
        ollama_status = "available"
        try:
            ollama.list()
        except:
            ollama_status = "unavailable"
        
        return jsonify({
            'status': 'healthy',
            'database': {
                'connected': True,
                'document_count': doc_count
            },
            'embedding_model': MODEL_NAME,
            'ollama': {
                'status': ollama_status,
                'model': OLLAMA_MODEL
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/search', methods=['POST'])
def search():
    """Search for similar documents."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', TOP_K)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        documents = search_similar_documents(query, top_k)
        
        return jsonify({
            'query': query,
            'documents': documents,
            'count': len(documents)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint with RAG."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', TOP_K)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Search for relevant documents
        print(f"Query: {query}")
        documents = search_similar_documents(query, top_k)
        print(f"Found {len(documents)} relevant documents")
        
        # Generate response
        response = generate_response(query, documents)
        
        return jsonify({
            'query': query,
            'response': response,
            'sources': documents,
            'model': OLLAMA_MODEL
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/documents', methods=['GET'])
def list_documents():
    """List all documents."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, content, metadata, created_at
            FROM documents
            ORDER BY created_at DESC
        """)
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        documents = []
        for row in results:
            doc_id, content, metadata, created_at = row
            documents.append({
                'id': doc_id,
                'content': content[:200] + '...' if len(content) > 200 else content,
                'metadata': metadata,
                'created_at': created_at.isoformat() if created_at else None
            })
        
        return jsonify({
            'documents': documents,
            'count': len(documents)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("RAG Chat Server")
    print("=" * 60)
    print(f"Embedding Model: {MODEL_NAME}")
    print(f"LLM Model: {OLLAMA_MODEL}")
    print(f"Starting server on http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
