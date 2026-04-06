#!/bin/bash

# Script to create and activate virtual environment for LLM gradient descent project
# Usage: source setup_venv.sh

echo "Creating virtual environment 'llm_env'..."
python3 -m venv llm_env

echo "Activating virtual environment..."
source llm_env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch and dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Installing additional packages..."
pip install -r requirements.txt

echo ""
echo "Virtual environment setup complete!"
echo "Environment is now activated."
echo ""
echo "To activate in the future, run:"
echo "  source llm_env/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
