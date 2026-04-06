#!/bin/bash

# Run server script
# Starts the Flask REST API server

echo "===================================================================="
echo "Starting RAG Chat Server"
echo "===================================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "✗ Virtual environment not found"
    echo "  Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if PostgreSQL is running
if ! docker ps | grep -q postgres-vectordb; then
    echo "✗ PostgreSQL container is not running"
    echo "  Starting PostgreSQL..."
    docker-compose up -d 2>/dev/null || docker compose up -d
    echo "  Waiting for PostgreSQL to be ready..."
    sleep 10
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Start server
echo "Starting server on http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
python app/server.py
