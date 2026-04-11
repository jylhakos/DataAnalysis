# Use Case 1: Data Agent with Streamlit and LangChain

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Using the Interface](#using-the-interface)
  - [Example Prompt for Questions](#example-prompt-for-questions)
- [Code Example](#code-example)
- [Sample Data](#sample-data)
- [How It Works](#how-it-works)
  - [Security Note](#security-note)
- [Customization](#customization)
  - [Using Your Own Data](#using-your-own-data)
  - [Changing the Model](#changing-the-model)
- [Troubleshooting](#troubleshooting)
  - [Issue: API Key Not Found](#issue-api-key-not-found)
  - [Issue: Module Not Found](#issue-module-not-found)
  - [Issue: Port Already in Use](#issue-port-already-in-use)
  - [Issue: Agent Not Responding](#issue-agent-not-responding)
- [Performance Tips](#performance-tips)
- [Testing](#testing)
- [Limitations](#limitations)
- [Next Steps](#next-steps)
- [Resources](#resources)

## Overview

This use case demonstrates how to build a functional AI agent using Streamlit and LangChain that allows users to chat with their data using natural language queries.

## Features

- Interactive Streamlit web interface
- Natural language querying of CSV data
- Support for OpenAI GPT-4 and Anthropic Claude models
- Real-time data analysis and insights
- Chat history tracking
- Data preview and column information display
- Configurable model temperature
- Support for custom CSV file uploads

## Architecture

```
User Question (Natural Language)
        ↓
Streamlit Interface
        ↓
LangChain Agent
        ↓
LLM (GPT-4 / Claude)
        ↓
Python Code Generation
        ↓
Pandas DataFrame Execution
        ↓
Result Formatting
        ↓
Display to User
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment activated
- OpenAI API key or Anthropic API key

### Setup

```bash
# Navigate to project root
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Introduction

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### Environment Variables

Set your API key(s):

```bash
# For OpenAI
export OPENAI_API_KEY='your-openai-api-key-here'

# For Anthropic (optional)
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

Or create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

## Usage

### Running the Application

```bash
# Navigate to use case 1 directory
cd use_case_1

# Run the Streamlit application
streamlit run data_agent.py

# The app will open in your browser at http://localhost:8501
```

### Using the Interface

1. **Configure Settings** (Sidebar):
   - Choose LLM provider (OpenAI or Anthropic)
   - Adjust temperature (0 = deterministic, 1 = creative)
   - Select data source (default sales data or upload custom CSV)

2. **View Data**:
   - Check data preview with row/column counts
   - Expand sections to view sample data and column information
   - Verify data loaded correctly

3. **Initialize Agent**:
   - Agent initializes automatically on first load
   - Click "Reinitialize Agent" if you change settings

4. **Ask Questions**:
   - Type your question in the chat input
   - Wait for the agent to analyze and respond
   - Review the answer and ask follow-up questions

### Example Prompt for Questions

Try asking:

- "What was the average profit margin per product category?"
- "Show me the top 5 products by revenue"
- "What is the trend in sales over the last quarter?"
- "Which customer segment has the highest lifetime value?"
- "Calculate the correlation between price and sales volume"
- "What is the distribution of sales by region?"
- "Identify any outliers in the revenue data"
- "What is the total revenue by month?"
- "Which region has the best performance?"
- "Compare Electronics vs Furniture sales"

## Code Example

Basic usage without Streamlit:

```python
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Load your data
df = pd.read_csv("sales_data.csv")

# Initialize the AI Agent
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4"), 
    df, 
    verbose=True,
    allow_dangerous_code=True
)

# Ask a question in plain English
response = agent.run("What was the average profit margin per product category?")
print(response)
```

## Sample Data

The included `sales_data.csv` contains:

- **Columns**: date, product_category, product_name, region, customer_segment, sales_volume, revenue, cost, profit_margin
- **Rows**: 52 sample records
- **Date Range**: January 2024 - March 2024
- **Categories**: Electronics, Furniture, Office Supplies
- **Regions**: North America, Europe, Asia
- **Segments**: Enterprise, Consumer

## How It Works

1. **Data Loading**: The application loads CSV data using pandas
2. **Agent Initialization**: Creates a LangChain pandas dataframe agent with the selected LLM
3. **Query Processing**: User's natural language question is sent to the agent
4. **Code Generation**: The LLM generates Python/pandas code to answer the question
5. **Execution**: The generated code is executed against the dataframe
6. **Response**: Results are formatted and displayed to the user

### Security Note

The agent uses `allow_dangerous_code=True` to execute generated Python code. This is necessary for the agent to work but should only be used with trusted data sources in controlled environments.

## Customization

### Using Your Own Data

1. **Option 1**: Replace `sales_data.csv` with your own CSV file
2. **Option 2**: Use the file uploader in the sidebar
3. **Option 3**: Modify the code to load from a database:

```python
import sqlalchemy

# Connect to database
engine = sqlalchemy.create_engine('postgresql://user:pass@localhost/db')
df = pd.read_sql_query("SELECT * FROM your_table", engine)
```

### Changing the Model

Modify the model in the code:

```python
# Use GPT-3.5 Turbo (faster, cheaper)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Use GPT-4 Turbo
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")

# Use Claude 3 Opus
llm = ChatAnthropic(temperature=0, model="claude-3-opus-20240229")
```

## Troubleshooting

### Issue: API Key Not Found

```bash
# Error: OPENAI_API_KEY not found in environment variables

# Solution: Export the key
export OPENAI_API_KEY='your-key-here'

# Verify it's set
echo $OPENAI_API_KEY
```

### Issue: Module Not Found

```bash
# Error: ModuleNotFoundError: No module named 'langchain'

# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: Port Already in Use

```bash
# Error: Port 8501 is already in use

# Solution: Use a different port
streamlit run data_agent.py --server.port 8502
```

### Issue: Agent Not Responding

- Check your internet connection (API calls require network access)
- Verify your API key is valid and has available credits
- Try reducing the temperature setting
- Check the terminal for error messages

## Performance Tips

1. **Use GPT-3.5 for Simple Queries**: Faster and cheaper than GPT-4
2. **Set Temperature to 0**: More consistent results for data analysis
3. **Be Specific**: Clear questions get better answers
4. **Limit Data Size**: For large datasets, filter first or use sampling
5. **Cache Results**: Store frequently accessed insights

## Testing

Run the included tests:

```bash
# From project root
pytest tests/test_data_agent.py -v
```

## Limitations

- Requires internet connection for API calls
- API costs apply based on usage
- Code execution requires `allow_dangerous_code=True`
- May occasionally generate incorrect code (always verify results)
- Limited to pandas operations
- Performance depends on dataset size

## Next Steps

- Explore Use Case 2 for deep agent implementation
- Try different LLM models and compare results
- Integrate with your own data sources
- Add data visualization capabilities
- Implement caching for common queries
- Add user authentication for production use

## Resources

- LangChain Documentation: https://docs.langchain.com/
- Streamlit Documentation: https://docs.streamlit.io/
- Pandas Documentation: https://pandas.pydata.org/docs/
