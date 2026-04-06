#!/bin/bash

# Gradient Descent LLM Project Setup Script

echo "Setting up Gradient Descent LLM Project..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv llm_env

# Activate virtual environment
echo "Activating virtual environment..."
source llm_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source llm_env/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Run gradient descent examples: python src/gradient_descent.py"
echo "  2. Train a neural network: python src/neural_network.py"
echo "  3. Explore Jupyter notebooks: jupyter notebook"
