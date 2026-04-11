#!/bin/bash

# Setup script for AI Data Analysis Project
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on error

echo "====================================="
echo "AI Data Analysis - Environment Setup"
echo "====================================="
echo ""

# Check Python version
echo "Step 1: Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
echo "Found Python $PYTHON_VERSION"

# Check if version is 3.8 or higher
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Create virtual environment
echo "Step 2: Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Warning: Virtual environment already exists"
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Keeping existing virtual environment"
        echo "Skipping to dependency installation..."
    fi
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Using existing virtual environment"
fi
echo ""

# Activate virtual environment
echo "Step 3: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Step 4: Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Step 5: Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed successfully"
else
    echo "✗ Error installing dependencies"
    echo "Please check the error messages above"
    exit 1
fi
echo ""

# Verify installation
echo "Step 6: Verifying installation..."
python3 -c "import langchain; import pandas; import streamlit; import matplotlib; import seaborn; print('✓ Core packages verified')"
echo ""

# Create necessary directories
echo "Step 7: Creating project directories..."
mkdir -p use_case_1
mkdir -p use_case_2
mkdir -p tests
echo "✓ Directories created"
echo ""

# Display environment info
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "Virtual environment location: $(pwd)/venv"
echo "Python version: $(python --version)"
echo "pip version: $(pip --version | cut -d ' ' -f 2)"
echo ""
echo "Installed packages:"
pip list | grep -E "langchain|pandas|streamlit|matplotlib|seaborn|deepagents|openai|anthropic" || echo "Core packages installed"
echo ""
echo "====================================="
echo "Next Steps:"
echo "====================================="
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Set your API keys:"
echo "   export OPENAI_API_KEY='your-key-here'"
echo "   export ANTHROPIC_API_KEY='your-key-here' (optional)"
echo ""
echo "3. Run Use Case 1 (Data Agent):"
echo "   cd use_case_1"
echo "   streamlit run data_agent.py"
echo ""
echo "4. Run Use Case 2 (Deep Agent):"
echo "   cd use_case_2"
echo "   python deep_agent.py"
echo ""
echo "For more information, see README.md"
echo ""
