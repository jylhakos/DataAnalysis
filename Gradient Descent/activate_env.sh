#!/bin/bash

# Activate the virtual environment
# Usage: source activate_env.sh

if [ -d "llm_env" ]; then
    echo "Activating virtual environment 'llm_env'..."
    source llm_env/bin/activate
    echo "Virtual environment activated!"
    echo ""
    echo "Python version:"
    python --version
    echo ""
    echo "To deactivate, run: deactivate"
else
    echo "Virtual environment 'llm_env' not found."
    echo "Please run setup_venv.sh first:"
    echo "  source setup_venv.sh"
fi
