"""
Interactive chat client for RAG chatbot.
Connects to the Flask server and provides a command-line interface for chatting.
"""

import requests
import sys
from typing import Optional


class ChatClient:
    """Client for interacting with the RAG chat server."""
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url
        self.session = requests.Session()
    
    def check_health(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("\n" + "=" * 60)
                print("Server Status:")
                print("=" * 60)
                print(f"Status: {data['status']}")
                print(f"Database documents: {data['database']['document_count']}")
                print(f"Embedding model: {data['embedding_model']}")
                print(f"LLM model: {data['ollama']['model']}")
                print(f"Ollama status: {data['ollama']['status']}")
                print("=" * 60 + "\n")
                return data['status'] == 'healthy'
            return False
        except requests.exceptions.RequestException as e:
            print(f"\n✗ Cannot connect to server: {e}")
            print("Make sure the server is running:")
            print("  python app/server.py\n")
            return False
    
    def chat(self, query: str, top_k: int = 3) -> Optional[dict]:
        """Send a chat message and get response."""
        try:
            response = self.session.post(
                f"{self.server_url}/chat",
                json={'query': query, 'top_k': top_k},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json()
                print(f"\n✗ Error: {error_data.get('error', 'Unknown error')}")
                return None
                
        except requests.exceptions.Timeout:
            print("\n✗ Request timed out. The server might be busy.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"\n✗ Request failed: {e}")
            return None
    
    def search_documents(self, query: str, top_k: int = 5) -> Optional[dict]:
        """Search for similar documents."""
        try:
            response = self.session.post(
                f"{self.server_url}/search",
                json={'query': query, 'top_k': top_k},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"\n✗ Search failed: {e}")
            return None
    
    def list_documents(self) -> Optional[dict]:
        """List all documents in the database."""
        try:
            response = self.session.get(f"{self.server_url}/documents", timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"\n✗ Failed to list documents: {e}")
            return None


def print_response(data: dict, show_sources: bool = True):
    """Pretty print chat response."""
    print("\n" + "-" * 60)
    print("Assistant:", data['response'])
    
    if show_sources and data.get('sources'):
        print("\n" + "-" * 60)
        print("Sources:")
        for i, source in enumerate(data['sources'], 1):
            print(f"\n{i}. Similarity: {source['similarity']:.3f}")
            content = source['content']
            if len(content) > 150:
                content = content[:150] + "..."
            print(f"   {content}")
    print("-" * 60 + "\n")


def print_search_results(data: dict):
    """Pretty print search results."""
    print("\n" + "=" * 60)
    print(f"Found {data['count']} documents:")
    print("=" * 60)
    
    for i, doc in enumerate(data['documents'], 1):
        print(f"\n{i}. Similarity: {doc['similarity']:.3f}")
        print(f"   ID: {doc['id']}")
        content = doc['content']
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"   Content: {content}")
        if doc.get('metadata'):
            print(f"   Metadata: {doc['metadata']}")
    
    print("=" * 60 + "\n")


def print_help():
    """Print help message."""
    print("\n" + "=" * 60)
    print("Available Commands:")
    print("=" * 60)
    print("  /help      - Show this help message")
    print("  /list      - List all documents in database")
    print("  /search    - Search for similar documents")
    print("  /sources   - Toggle showing sources with responses")
    print("  /clear     - Clear the screen")
    print("  /quit      - Exit the chat")
    print("\nJust type your question to chat with the assistant!")
    print("=" * 60 + "\n")


def main():
    """Main chat loop."""
    print("\n" + "=" * 60)
    print("RAG Chat Client")
    print("=" * 60)
    print("Type '/help' for commands or '/quit' to exit")
    print("=" * 60)
    
    client = ChatClient()
    
    # Check server health
    if not client.check_health():
        print("Cannot start chat client without a healthy server.")
        return 1
    
    show_sources = True
    
    while True:
        try:
            # Get user input
            query = input("\nYou: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.startswith('/'):
                command = query.lower()
                
                if command == '/quit' or command == '/exit':
                    print("\nGoodbye!")
                    break
                
                elif command == '/help':
                    print_help()
                    continue
                
                elif command == '/list':
                    data = client.list_documents()
                    if data:
                        print(f"\n✓ Found {data['count']} documents")
                        for doc in data['documents']:
                            print(f"\nID: {doc['id']}")
                            print(f"Content: {doc['content']}")
                            print(f"Metadata: {doc['metadata']}")
                            print(f"Created: {doc['created_at']}")
                    continue
                
                elif command.startswith('/search'):
                    search_query = input("Enter search query: ").strip()
                    if search_query:
                        data = client.search_documents(search_query)
                        if data:
                            print_search_results(data)
                    continue
                
                elif command == '/sources':
                    show_sources = not show_sources
                    status = "enabled" if show_sources else "disabled"
                    print(f"\n✓ Source display {status}")
                    continue
                
                elif command == '/clear':
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                else:
                    print(f"\n✗ Unknown command: {query}")
                    print("Type '/help' for available commands")
                    continue
            
            # Send chat message
            print("\nThinking...")
            data = client.chat(query)
            
            if data:
                print_response(data, show_sources=show_sources)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type '/quit' to exit or continue chatting.")
            continue
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            continue
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
