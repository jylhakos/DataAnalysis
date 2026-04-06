#!/bin/bash

# Test script for RAG Chat System
# Tests all components: database, search, and chat

echo "===================================================================="
echo "RAG Chat System - Testing Script"
echo "===================================================================="
echo ""

# Check if server is running
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✗ Server is not running"
    echo "  Please start the server first: ./run_server.sh"
    exit 1
fi

# Test health endpoint
echo "Test 1: Health Check"
echo "--------------------------------------------------------------------"
curl -s http://localhost:5000/health | python3 -m json.tool
echo ""

# Test document listing
echo ""
echo "Test 2: List Documents"
echo "--------------------------------------------------------------------"
curl -s http://localhost:5000/documents | python3 -m json.tool
echo ""

# Test search endpoint
echo ""
echo "Test 3: Vector Search"
echo "--------------------------------------------------------------------"
echo "Query: 'What is pgvector?'"
curl -s -X POST http://localhost:5000/search \
    -H "Content-Type: application/json" \
    -d '{"query":"What is pgvector?","top_k":2}' | python3 -m json.tool
echo ""

echo ""
echo "===================================================================="
echo "Test 4: RAG Chat (this may take 10-20 seconds)"
echo "===================================================================="
echo "Query: 'Explain how vector embeddings work'"
echo ""

# Save chat response to temp file
RESPONSE=$(mktemp)
curl -s -X POST http://localhost:5000/chat \
    -H "Content-Type: application/json" \
    -d '{"query":"Explain how vector embeddings work"}' > "$RESPONSE"

# Display response
echo "Response:"
echo "--------------------------------------------------------------------"
python3 -c "
import json
with open('$RESPONSE') as f:
    data = json.load(f)
    print(data.get('response', 'Error: No response'))
    print()
    print(f\"Sources used: {len(data.get('sources', []))}\")
"

# Cleanup
rm -f "$RESPONSE"

echo ""
echo "===================================================================="
echo "All tests completed!"
echo "===================================================================="
echo ""
echo "You can now use the interactive client with: ./run_client.sh"
echo ""
