#!/bin/bash

# Run client script
# Starts the interactive chat client

echo "===================================================================="
echo "Starting RAG Chat Client"
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

# Check if server is running
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✗ Server is not running"
    echo "  Please start the server first:"
    echo "    ./run_server.sh"
    echo ""
    exit 1
fi

# Start client
python app/client.py
