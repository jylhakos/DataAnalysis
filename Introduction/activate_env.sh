#!/bin/bash

# Activation helper script for AI Data Analysis Project
# This script activates the virtual environment and displays helpful information

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found"
    echo "Please run ./setup_env.sh first to create the environment"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Display information
echo "====================================="
echo "AI Data Analysis - Environment Active"
echo "====================================="
echo ""
echo "✓ Virtual environment activated"
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "====================================="
echo "Quick Commands:"
echo "====================================="
echo ""
echo "Use Case 1 - Data Agent (Streamlit):"
echo "  cd use_case_1 && streamlit run data_agent.py"
echo ""
echo "Use Case 2 - Deep Agent:"
echo "  cd use_case_2 && python deep_agent.py"
echo ""
echo "Use Case 2 - CSV Analyzer:"
echo "  cd use_case_2 && python csv_analyzer.py"
echo ""
echo "Run Tests:"
echo "  pytest tests/ -v"
echo ""
echo "Deactivate:"
echo "  deactivate"
echo ""
echo "====================================="
echo "Environment Variables:"
echo "====================================="
echo ""

# Check for API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠ OPENAI_API_KEY not set"
    echo "  Set it: export OPENAI_API_KEY='your-key-here'"
else
    echo "✓ OPENAI_API_KEY is set"
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠ ANTHROPIC_API_KEY not set (optional)"
else
    echo "✓ ANTHROPIC_API_KEY is set"
fi

if [ -z "$LANGSMITH_API_KEY" ]; then
    echo "⚠ LANGSMITH_API_KEY not set (optional for tracing)"
else
    echo "✓ LANGSMITH_API_KEY is set"
    if [ -z "$LANGSMITH_TRACING" ]; then
        echo "  Note: Set LANGSMITH_TRACING=true to enable tracing"
    fi
fi

echo ""
echo "====================================="
echo ""
