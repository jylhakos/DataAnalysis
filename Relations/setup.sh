#!/bin/bash

# Setup script for RAG Chat with PostgreSQL and pgvector
# This script sets up the entire environment

set -e  # Exit on error

echo "===================================================================="
echo "RAG Chat Setup Script"
echo "===================================================================="
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: This script is designed for Linux. You may need to adapt it."
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "Checking prerequisites..."
echo ""

# Check Docker
if ! command_exists docker; then
    echo "✗ Docker is not installed"
    echo "  Please install Docker: https://docs.docker.com/engine/install/"
    exit 1
else
    echo "✓ Docker is installed"
fi

# Check Docker Compose
if command_exists docker-compose; then
    COMPOSE_CMD="docker-compose"
    echo "✓ Docker Compose (standalone) is available"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
    echo "✓ Docker Compose (plugin) is available"
else
    echo "✗ Docker Compose is not available"
    echo "  Please install Docker Compose"
    exit 1
fi

# Check Python
if ! command_exists python3; then
    echo "✗ Python 3 is not installed"
    echo "  Please install Python 3.8 or higher"
    exit 1
else
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✓ Python $PYTHON_VERSION is installed"
fi

# Check Ollama
if ! command_exists ollama; then
    echo "✗ Ollama is not installed"
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama is installed"
fi

echo ""
echo "===================================================================="
echo "Step 1: Setting up Docker containers"
echo "===================================================================="
echo ""

# Start Docker containers
echo "Starting PostgreSQL with pgvector..."
$COMPOSE_CMD up -d

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 15

# Check if PostgreSQL is ready
if docker exec postgres-vectordb pg_isready -U postgres -d vectordb >/dev/null 2>&1; then
    echo "✓ PostgreSQL is ready"
else
    echo "✗ PostgreSQL is not ready. Check logs with: $COMPOSE_CMD logs postgres"
    exit 1
fi

echo ""
echo "===================================================================="
echo "Step 2: Setting up Python virtual environment"
echo "===================================================================="
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip >/dev/null 2>&1
pip install -r requirements.txt

echo "✓ Dependencies installed"

echo ""
echo "===================================================================="
echo "Step 3: Setting up database"
echo "===================================================================="
echo ""

# Run database setup
python scripts/setup_database.py

echo ""
echo "===================================================================="
echo "Step 4: Setting up Ollama"
echo "===================================================================="
echo ""

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Pull model
echo "Pulling Ollama model (llama3.2)..."
echo "This may take a few minutes..."
ollama pull llama3.2

echo "✓ Ollama model ready"

echo ""
echo "===================================================================="
echo "Step 5: Loading sample documents"
echo "===================================================================="
echo ""

# Load documents
python scripts/load_documents.py

echo ""
echo "===================================================================="
echo "Setup Complete!"
echo "===================================================================="
echo ""
echo "To start using the chat application:"
echo ""
echo "1. Start the server (in one terminal):"
echo "   source venv/bin/activate"
echo "   python app/server.py"
echo ""
echo "2. Start the client (in another terminal):"
echo "   source venv/bin/activate"
echo "   python app/client.py"
echo ""
echo "Or use the convenience scripts:"
echo "   ./run_server.sh"
echo "   ./run_client.sh"
echo ""
echo "===================================================================="
