#!/bin/bash

# Setup Script for Vectors Project
# This script sets up the virtual environment and installs dependencies

echo "=============================================="
echo "Vectors Project Setup"
echo "=============================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Display Python version
PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    
    echo "Virtual environment created successfully!"
fi

echo ""
echo "Activating virtual environment..."

# Activate virtual environment
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

echo "Virtual environment activated!"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

if [ $? -ne 0 ]; then
    echo "Warning: Failed to upgrade pip, continuing anyway..."
fi

echo ""

# Install requirements
echo "Installing Python packages..."
echo "This may take several minutes..."
echo ""

pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements."
    echo "Please check requirements.txt and try again."
    exit 1
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run demo scripts:"
echo "   python src/embeddings.py"
echo "   python src/attention_demo.py"
echo "   python src/qa_demo.py"
echo ""
echo "3. To deactivate the virtual environment:"
echo "   deactivate"
echo ""
echo "=============================================="
