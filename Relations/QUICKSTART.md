# Quick Start Guide

This document provides step-by-step instructions to set up PostgreSQL with pgvector, install Ollama for local LLM inference, and build a RAG (Retrieval Augmented Generation) chat application using Python and LangChain.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Install Docker](#step-1-install-docker)
- [Step 2: Set Up PostgreSQL with pgvector](#step-2-set-up-postgresql-with-pgvector)
- [Step 3: Install Ollama](#step-3-install-ollama)
- [Step 4: Set Up Python Virtual Environment](#step-4-set-up-python-virtual-environment)
- [Step 5: Install Dependencies](#step-5-install-dependencies)
- [Step 6: Create Database and Tables](#step-6-create-database-and-tables)
- [Step 7: Run the Example Application](#step-7-run-the-example-application)
- [Step 8: Test the Chat Interface](#step-8-test-the-chat-interface)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Linux operating system (Ubuntu 20.04+ recommended)
- At least 8GB RAM
- 10GB free disk space
- Internet connection for downloading dependencies
- Basic familiarity with command line

## Step 1: Install Docker

Docker is required to run PostgreSQL and pgvector in containers.

### Install Docker on Ubuntu/Linux

```bash
# Update package index
sudo apt-get update

# Install required packages
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your user to the docker group
sudo usermod -aG docker $USER

# Apply group changes (or log out and log back in)
newgrp docker

# Verify installation
docker --version
docker compose version
```

## Step 2: Set Up PostgreSQL with pgvector

We'll use Docker Compose to set up PostgreSQL with the pgvector extension.

### Start PostgreSQL Container

```bash
# Navigate to project directory
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Relations

# Start PostgreSQL with pgvector using Docker Compose
docker compose up -d

# Verify containers are running
docker ps

# Check PostgreSQL logs
docker compose logs postgres

# Wait for PostgreSQL to be ready (usually takes 10-20 seconds)
sleep 20

# Test connection
docker exec -it postgres-vectordb psql -U postgres -d vectordb -c "SELECT version();"
```

### Verify pgvector Extension

```bash
# Check if pgvector extension is available
docker exec -it postgres-vectordb psql -U postgres -d vectordb -c "SELECT * FROM pg_available_extensions WHERE name = 'vector';"

# The extension is created automatically by init.sql
# Verify it's enabled
docker exec -it postgres-vectordb psql -U postgres -d vectordb -c "\dx"
```

## Step 3: Install Ollama

Ollama allows you to run open-source LLMs locally.

### Install Ollama on Linux

```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Start Ollama service (if not already running)
ollama serve &

# Pull a lightweight model for testing (llama3.2 is recommended)
ollama pull llama3.2

# Alternative: Pull a smaller model if you have limited resources
# ollama pull phi3

# Verify model is available
ollama list

# Test the model
ollama run llama3.2 "Hello, how are you?"
```

### Ollama Model Recommendations

| Model | Size | RAM Required | Best For |
|-------|------|--------------|----------|
| llama3.2 | 2GB | 8GB | General purpose, fast |
| phi3 | 1.5GB | 4GB | Resource-constrained systems |
| mistral | 4GB | 16GB | Better quality responses |
| llama3 | 4.7GB | 16GB | Production use |

## Step 4: Set Up Python Virtual Environment

Create an isolated Python environment for the project.

```bash
# Navigate to project directory
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Relations

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Verify Python version (should be 3.8 or higher)
python --version
```

**Important**: Always activate the virtual environment before running Python scripts:
```bash
source venv/bin/activate
```

To deactivate the virtual environment:
```bash
deactivate
```

## Step 5: Install Dependencies

With the virtual environment activated, install required Python packages.

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Verify installations
pip list | grep -E "langchain|psycopg2|ollama|chromadb"
```

### Key Dependencies

- **langchain**: Framework for building LLM applications
- **langchain-community**: Community integrations
- **langchain-postgres**: PostgreSQL vector store integration
- **psycopg2-binary**: PostgreSQL adapter for Python
- **ollama**: Python client for Ollama
- **sentence-transformers**: For generating embeddings
- **chromadb**: Optional alternative vector store

## Step 6: Create Database and Tables

Initialize the database schema and create necessary tables.

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Run the database setup script
python scripts/setup_database.py

# Verify tables were created
docker exec -it postgres-vectordb psql -U postgres -d vectordb -c "\dt"

# Check the documents table structure
docker exec -it postgres-vectordb psql -U postgres -d vectordb -c "\d documents"
```

The setup script will:
1. Enable the pgvector extension
2. Create a `documents` table with vector column
3. Create indexes for efficient similarity search
4. Insert sample documents

## Step 7: Run the Example Application

### Load Sample Documents

```bash
# Activate virtual environment
source venv/bin/activate

# Load sample documents into the vector database
python scripts/load_documents.py

# This will:
# - Read documents from data/sample_documents/
# - Generate embeddings using sentence-transformers
# - Store embeddings in PostgreSQL with pgvector
```

### Start the Chat Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the Flask REST API server
python app/server.py

# Server will start on http://localhost:5000
# Keep this terminal open
```

### Run the Chat Client (in a new terminal)

```bash
# Open a new terminal
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Relations

# Activate virtual environment
source venv/bin/activate

# Start the interactive chat client
python app/client.py
```

## Step 8: Test the Chat Interface

Once the client is running, you can interact with the system:

### Example Interactions

```
You: What documents do you have?
Assistant: I have access to documents about...

You: Tell me about vector databases
Assistant: Based on the documents, vector databases are...

You: How do embeddings work?
Assistant: Embeddings are numerical representations...

You: Compare PostgreSQL and vector databases
Assistant: PostgreSQL is a relational database that... Vector databases specialize in...
```

### How It Works

1. **User Input**: You type a question
2. **Embedding**: Question is converted to a vector embedding
3. **Vector Search**: PostgreSQL pgvector finds similar documents
4. **Context Retrieval**: Top K relevant documents are retrieved
5. **LLM Generation**: Ollama generates response using retrieved context
6. **Response**: Answer is displayed to user

## Troubleshooting

### PostgreSQL Connection Issues

```bash
# Check if PostgreSQL container is running
docker ps | grep postgres

# Restart PostgreSQL container
docker compose restart postgres

# Check logs for errors
docker compose logs postgres

# Test direct connection
docker exec -it postgres-vectordb psql -U postgres -d vectordb
```

### Ollama Not Responding

```bash
# Check if Ollama is running
ps aux | grep ollama

# Restart Ollama
pkill ollama
ollama serve &

# Test Ollama connection
curl http://localhost:11434/api/tags

# Pull model again if needed
ollama pull llama3.2
```

### Python Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install --upgrade -r requirements.txt

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Port Already in Use

```bash
# Find process using port 5000
sudo lsof -i :5000

# Kill the process (replace PID)
kill -9 <PID>

# Or use a different port
# Edit app/server.py and change the port number
```

### Out of Memory

```bash
# Check memory usage
free -h

# Use a smaller Ollama model
ollama pull phi3

# Reduce batch size in scripts
# Edit scripts/load_documents.py: batch_size = 10
```

### Vector Dimension Mismatch

```bash
# Check embedding model dimensions
python -c "from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('all-MiniLM-L6-v2'); \
print(f'Dimensions: {model.get_sentence_embedding_dimension()}')"

# Ensure database vector column matches
# Default for all-MiniLM-L6-v2 is 384 dimensions
```

## Stopping the Services

```bash
# Stop the Flask server (Ctrl+C in server terminal)

# Stop the chat client (type 'exit' or Ctrl+C)

# Stop and remove PostgreSQL containers
docker compose down

# Stop Ollama
pkill ollama

# Deactivate virtual environment
deactivate
```

## Next Steps

After completing this quickstart:

1. **Explore the Code**:
   - Review `app/server.py` for REST API implementation
   - Examine `app/client.py` for chat interface
   - Study `scripts/setup_database.py` for database schema

2. **Customize**:
   - Add your own documents to `data/sample_documents/`
   - Modify the embedding model in configuration
   - Adjust the number of retrieved documents (top K)
   - Change the LLM model in Ollama

3. **Scale Up**:
   - Add more documents to the vector database
   - Implement user authentication
   - Deploy to production servers
   - Add web UI instead of CLI

4. **Learn More**:
   - Read [README.md](README.md) for detailed concepts
   - Explore LangChain documentation
   - Study pgvector advanced features
   - Experiment with different LLM models

## Useful Commands Reference

```bash
# Docker
docker compose up -d              # Start services
docker compose down               # Stop services
docker compose logs postgres      # View PostgreSQL logs
docker ps                         # List running containers

# PostgreSQL
docker exec -it postgres-vectordb psql -U postgres -d vectordb  # Connect to DB
docker exec -it postgres-vectordb psql -U postgres -d vectordb -c "SELECT COUNT(*) FROM documents;"  # Run query

# Ollama
ollama list                       # List installed models
ollama pull <model>               # Download a model
ollama run <model> "<prompt>"     # Test a model
ollama serve                      # Start Ollama server

# Python Virtual Environment
source venv/bin/activate          # Activate
deactivate                        # Deactivate
pip list                          # List installed packages
pip freeze > requirements.txt     # Export dependencies

# Project
python scripts/setup_database.py  # Initialize database
python scripts/load_documents.py  # Load documents
python app/server.py              # Start API server
python app/client.py              # Start chat client
```

## Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pgvector GitHub Repository](https://github.com/pgvector/pgvector)
- [Ollama Documentation](https://ollama.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)

---

**Need Help?** Check the main [README.md](README.md).
