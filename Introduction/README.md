# How to Use AI for Data Analysis

## Table of Contents
- [Quick Start](#quick-start)
- [Overview](#overview)
- [What is AI for Data Analysis?](#what-is-ai-for-data-analysis)
- [Key Concepts](#key-concepts)
- [Agentic AI Tools](#agentic-ai-tools)
- [Agent Libraries](#agent-libraries)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Use Cases](#use-cases)
- [Testing & Validation](#testing--validation)
- [AI Tools for Data Analysis](#ai-tools-for-data-analysis)
- [How AI Agents Work](#how-ai-agents-work)
- [Frameworks & Technologies](#frameworks--technologies)
- [DevOps Setup Guide](#devops-setup-guide)

## Quick Start

### Setup

```bash
# 1. Run setup script
./setup_env.sh

# 2. Set API keys
export OPENAI_API_KEY='your-openai-api-key-here'
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'  # Optional

# 3. Add to your shell profile (for persistence)
echo "export OPENAI_API_KEY='your-key'" >> ~/.bashrc
source ~/.bashrc
```

### Generate Sample CSV Files

Sample CSV files are not included in the repository. Generate them using the provided script:

```bash
# Run the sample CSV generator script
python generate_sample_csv.py
```

This script creates:
- `use_case_1/sales_data.csv` - Sales data with Date, Product, Quantity, Price, Total columns
- `use_case_2/sample_data.csv` - Customer data with Name, Age, Score, Category columns

**Alternative: Create your own CSV file**
```bash
# For use_case_1 - Create a sales_data.csv with columns like:
# Date, Product, Quantity, Price, Total
# Example: 2024-01-01,Widget A,10,25.50,255.00
```

**Alternative: Use Python to generate custom sample data**
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample sales data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
data = {
    'Date': dates,
    'Product': np.random.choice(['Widget A', 'Widget B', 'Widget C'], 100),
    'Quantity': np.random.randint(1, 50, 100),
    'Price': np.random.uniform(10, 100, 100).round(2),
}
df = pd.DataFrame(data)
df['Total'] = df['Quantity'] * df['Price']
df.to_csv('use_case_1/sales_data.csv', index=False)
print("sales_data.csv generated")

# Generate sample data for deep agent analysis
data2 = {
    'Name': [f'Customer {i}' for i in range(200)],
    'Age': np.random.randint(18, 80, 200),
    'Score': np.random.uniform(0, 100, 200).round(2),
    'Category': np.random.choice(['A', 'B', 'C', 'D'], 200),
}
df2 = pd.DataFrame(data2)
df2.to_csv('use_case_2/sample_data.csv', index=False)
print("sample_data.csv generated")
```

**Alternative: Use your own data**
- Place any CSV file in the use_case folders
- Update the file path in the application code or specify it when running

### Usage

```bash
# Activate virtual environment
source venv/bin/activate
# or
./activate_env.sh

# Deactivate when done
deactivate
```

### Run Use Case 1 (Data Agent - Streamlit)

```bash
cd use_case_1
streamlit run data_agent.py
# Opens browser at http://localhost:8501
```

### Run Use Case 2 (Deep Agent)

```bash
cd use_case_2

# Interactive mode
python deep_agent.py

# CSV Analyzer
python csv_analyzer.py --input sample_data.csv --output results
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_data_agent.py -v

# With coverage
pytest tests/ --cov=use_case_1 --cov=use_case_2
```

### Common Commands

```bash
# Update packages
pip install --upgrade -r requirements.txt

# Check installed packages
pip list | grep -E "langchain|pandas|streamlit"

# View logs (Streamlit)
streamlit run data_agent.py --logger.level debug
```

## Overview

AI for data analysis refers to the use of machine learning, natural language processing, and automation to help people explore, prepare, and interpret data without requiring deep technical expertise. This project demonstrates practical implementations of AI agents for data analysis using modern frameworks and tools.

## What is AI for Data Analysis?

AI data analysis combines:
- **Automation** of exploration and analysis work
- **Natural, conversational interaction** with data
- **Guided insight discovery** instead of manual dashboard building

### Core Capabilities

**Real-time Analysis**: Monitor changing conditions as they happen rather than reviewing yesterday's numbers tomorrow.

**Predictive Analytics**: Use historical data patterns to forecast future outcomes, enabling businesses to anticipate market shifts, customer behavior, and operational needs.

**Prescriptive Analytics**: Combine AI with optimization algorithms to determine the best course of action for achieving business goals.

**Natural Language Processing (NLP)**: Interpret unstructured data such as text or speech, extracting insights from customer feedback, social media interactions, and other text-based sources.

**Conversational Analytics**: Interact with your data through plain-language questions rather than code or query builders.

**Automated Data Preparation**: Streamline data cleaning, transformation, and integration processes, accelerating data preparation tasks and ensuring quality data for analysis.

**Autonomous Monitoring**: AI agents monitor data autonomously, detect anomalies, and trigger workflows without human prompt.

## Key Concepts

### AI Data Analysis Techniques

**Machine Learning Algorithms**: Identify patterns, correlations, and make predictions across large datasets.

**Deep Learning**: Apply neural networks to complex problems like time-series analysis, forecasting, or image and signal data.

**Natural Language Processing (NLP)**: Extract insights from unstructured text or allow users to query data in plain language.

### Analytics Types

**Descriptive Analytics**: Assess performance and interpret datasets to unveil hidden patterns and current hurdles. Retailers use AI to analyze purchasing trends and discover preferences of particular target groups.

**Predictive Analytics**: Estimate what will happen next by assessing historical trends to provide precise forecasts, scrutinize challenges, and streamline organizational tasks.

**Prescriptive Analytics**: Make weighted decisions using different types of algorithms to determine the best course of action.

## Agentic AI Tools

### What is an Agentic AI Tool?

An agentic AI tool is a system that can figure out how to accomplish a task or goal without having a human tell it what to do. It's essentially an AI agent that can act as a team member and complete tasks on your behalf.

Unlike traditional automation tools that follow rigid "if this, then that" logic, agentic AI tools can:
- Reason through problems
- Handle decision-making on their own
- Adapt their approach based on context

### How Agentic AI Tools Work

Agentic AI tools work by combining large language models with the ability to take real-world actions across your tech stack:

1. You give the agent a goal or task
2. The agent breaks that goal down into smaller steps
3. It figures out which tools or integrations it needs to use
4. It executes each step until the task is complete

### Agent Capabilities

Modern agents can:
- Connect to PostgreSQL databases
- Perform automated Exploratory Data Analysis (EDA)
- Draft executive summaries of their findings
- Run continuously in the background, watching for defined conditions
- Take action when those conditions occur

## Agent Libraries

An agent library is a collection of prebuilt software components, tools, and code modules designed to help developers create, manage, and deploy autonomous AI agents.

### How Agent Libraries Work

Agent libraries provide the foundational building blocks and operational framework for AI agents:
1. Receive input (user query or system data)
2. Process information using embedded AI models and logic
3. Execute tasks or provide responses
4. Retrieve information, update databases, initiate workflows, or communicate with users

### Popular Agent Libraries

**LangChain**: Open-source framework for building LLM-powered applications. Integrates external components and data sources.
- Documentation: https://docs.langchain.com/

**LangGraph**: Extends LangChain capabilities by allowing developers to create structured, state-aware agent workflows using graph-based architectures.

**Pydantic AI**: Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.
- Documentation: https://pydantic.dev/docs/ai/overview/
- Provides structured framework for building type-safe LLM applications
- Ensures data validation and consistent output formatting

**AutoGen**: Helps build multi-agent systems that can collaborate to solve complex tasks. Develop conversational agents that work together, share context, and execute multi-step workflows autonomously.
- Allows multiple agents (e.g., "Coder" and "Reviewer") to collaborate on analysis

**CrewAI**: Orchestrates multiple AI agents to handle complex tasks. Create agent-based workflows where specialized agents collaborate, delegate tasks, and share information.
- Orchestrates role-playing agents to complete complex, multi-step data pipelines

**DeepAgents**: Standalone library built on top of LangChain's core building blocks for agents. Uses the LangGraph runtime for durable execution, streaming, human-in-the-loop, and other features.
- PyPI: https://pypi.org/project/deepagents/
- Documentation: https://docs.langchain.com/oss/python/deepagents/overview

## Project Structure

```
📁 Introduction/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 📜 setup_env.sh                 # Virtual environment setup script
├── 📜 activate_env.sh              # Activation helper script
├── 📁 use_case_1/                  # Data Agent with Streamlit
│   ├── 📄 data_agent.py           # Main application
│   ├── 📄 sales_data.csv          # Sample dataset
│   └── 📄 README_UC1.md           # Use case 1 documentation
├── 📁 use_case_2/                  # Deep Agent for CSV Analysis
│   ├── 📄 deep_agent.py           # Deep agent implementation
│   ├── 📄 csv_analyzer.py         # CSV analysis tool
│   ├── 📄 sample_data.csv         # Sample dataset
│   └── 📄 README_UC2.md           # Use case 2 documentation
└── 📁 tests/                       # Test scripts
    ├── 📄 test_data_agent.py      # Tests for use case 1
    └── 📄 test_deep_agent.py      # Tests for use case 2
```

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- OpenAI API key or Anthropic API key (for LLM access)

### Step-by-Step Setup

#### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Introduction

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

Alternatively, use the provided setup script:

```bash
# Make the script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

#### 2. Install Dependencies

```bash
# Ensure virtual environment is activated
# You should see (venv) in your terminal prompt

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

#### 3. Configure API Keys

```bash
# Set OpenAI API key (for Use Case 1)
export OPENAI_API_KEY='your-openai-api-key-here'

# Optional: Set Anthropic API key
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'

# Optional: Enable LangSmith tracing
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY='your-langsmith-api-key-here'
```

To make these permanent, add them to your `~/.bashrc` or `~/.zshrc`:

```bash
echo "export OPENAI_API_KEY='your-openai-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

**Environment Variables Reference**

| Variable | Required | Description |
|----------|----------|-------------|
| OPENAI_API_KEY | Yes | OpenAI API key for GPT models |
| ANTHROPIC_API_KEY | Optional | Anthropic API key for Claude |
| LANGSMITH_API_KEY | Optional | LangSmith for tracing |
| LANGSMITH_TRACING | Optional | Set to 'true' to enable tracing |

#### 4. Verify Installation

```bash
# Check Python version
python --version

# Verify packages are installed
pip list | grep -E "langchain|pandas|streamlit|openai"

# Test import
python -c "import langchain; import pandas; import streamlit; print('All packages imported successfully!')"
```

### Activating the Virtual Environment

Every time you work on this project, activate the virtual environment:

```bash
# Navigate to project directory
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Introduction

# Activate virtual environment
source venv/bin/activate

# Or use the helper script
./activate_env.sh
```

To deactivate:

```bash
deactivate
```

## Use Cases

### Use Case 1: Data Agent with Streamlit and LangChain

Build a functional agent using Streamlit and LangChain to chat with your data.

#### Overview

This use case demonstrates how to create an interactive data analysis agent that allows users to ask questions about their data in plain English and receive intelligent responses.

#### Features

- Natural language querying of CSV data
- Interactive Streamlit interface
- Automated Python code generation and execution
- Real-time data analysis

#### How It Works

1. **Data Loading**: Load dataset using `pd.read_csv()`
2. **Agent Initialization**: Create a LangChain `create_pandas_dataframe_agent` using OpenAI or Anthropic model
3. **Interface**: Use Streamlit's `st.text_input` to capture user questions
4. **Execution**: Pass question to agent, which generates and executes Python code internally to return results

#### Running Use Case 1

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Navigate to use case 1 directory
cd use_case_1

# Run the Streamlit application
streamlit run data_agent.py
```

The application will open in your browser at `http://localhost:8501`

#### Example Questions

- "What was the average profit margin per product category?"
- "Show me the top 5 products by revenue"
- "What is the trend in sales over the last quarter?"
- "Which customer segment has the highest lifetime value?"

#### Code Example

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
    verbose=True
)

# Ask a question in plain English
response = agent.run("What was the average profit margin per product category?")
print(response)
```

### Use Case 2: Deep Agent for CSV Analysis

Build a deep agent that accepts CSV files, performs exploratory data analysis, and generates visualizations.

#### Overview

This use case demonstrates building a sophisticated agent using the DeepAgents library that can autonomously analyze CSV files, generate insights, and create visualizations.

#### Features

- Accept CSV file for analysis
- Perform exploratory data analysis (EDA)
- Generate visualizations automatically
- Share results and insights
- Durable execution with LangGraph runtime
- Streaming and human-in-the-loop capabilities

#### Deep Agents vs. Shallow Agents

Using an LLM to call tools in a loop is the simplest form of an agent. This architecture, however, can yield agents that are "shallow". Deep agents provide:
- Planning capabilities
- Code execution
- Working with artifacts (scripts, reports, plots)
- Multi-step reasoning
- State management

#### How It Works

1. **Agent Creation**: Initialize deep agent with custom tools
2. **CSV Processing**: Agent reads and analyzes CSV structure
3. **EDA Execution**: Performs statistical analysis and identifies patterns
4. **Visualization**: Generates charts and graphs
5. **Reporting**: Assembles findings into readable format

#### Running Use Case 2

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Navigate to use case 2 directory
cd use_case_2

# Run the deep agent
python deep_agent.py
```

#### Code Example

```python
# pip install -qU deepagents
from deepagents import create_deep_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_deep_agent(
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

#### LangSmith Tracing

Use LangSmith to trace requests, debug agent behavior, and evaluate outputs:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY='your-api-key'
```

Visit https://smith.langchain.com/ to observe, evaluate, and deploy your agents.

#### Specifying Models

You can specify different models and providers:
- Documentation: https://docs.langchain.com/oss/python/deepagents/models
- Ollama integrations: https://docs.langchain.com/oss/python/integrations/providers/ollama

#### Backends

Deep Agents expose a filesystem surface to the agent via tools like:
- `ls`: List directory contents
- `read_file`: Read file contents
- `write_file`: Write to files
- `edit_file`: Edit existing files
- `glob`: Pattern matching for files
- `grep`: Search within files

Documentation: https://docs.langchain.com/oss/python/deepagents/backends

## Testing & Validation

### Testing Workflow

For effective AI-assisted data analysis, follow a structured verification process to catch potential hallucinations or logic errors:

#### 1. Sanity Check
Compare AI-generated outputs against known totals or historical reports.

```bash
# Run sanity check tests
python tests/test_data_agent.py --sanity-check
```

#### 2. Inspect Generated Code
Review AI-generated Python or SQL for correct table references, filters, and calculation logic.

Enable verbose mode to see generated code:
```python
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4"), 
    df, 
    verbose=True  # Shows generated code
)
```

#### 3. Manual Reproduction
Attempt to reproduce a small sample of results manually to verify the AI's logic is stable.

#### 4. Spot-check Samples
Pull random underlying records and verify the AI's interpretation matches reality.

```python
# Verify sample records
sample = df.sample(10)
print(sample)
```

#### 5. Automate Validation
Use frameworks like Pydantic to enforce structured data types and schemas, ensuring AI outputs conform to expected formats for programmatic use.

```python
from pydantic import BaseModel, validator

class AnalysisResult(BaseModel):
    metric: str
    value: float
    confidence: float
    
    @validator('confidence')
    def confidence_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v
```

### AI Agents for Automated Code Testing

These agents integrate with your development workflow to ensure the Python code you write for data analysis is robust and accurate:

**Qodo (formerly Codium)**: An IDE extension that analyzes code and generates tailored unit tests and edge-case scenarios specifically for Python and other languages.

**Tusk**: A Y Combinator-backed AI agent that integrates with pull requests to generate unit and integration tests. It "self-heals" by running generated tests and iterating if they fail until they are executable.

**Scenario**: An open-source Python library that uses an AI agent to test other AI agents or complex application logic. It allows you to write tests in plain English which the testing agent then uses as a reference.

### Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_data_agent.py -v

# Run with coverage
pytest tests/ --cov=use_case_1 --cov=use_case_2
```

### Test Examples

```python
def test_data_loading():
    """Test that CSV data loads correctly"""
    df = pd.read_csv("use_case_1/sales_data.csv")
    assert not df.empty
    assert 'product_category' in df.columns

def test_agent_response():
    """Test that agent provides valid response"""
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4"), 
        df, 
        verbose=True
    )
    response = agent.run("What is the total revenue?")
    assert response is not None
    assert isinstance(response, str)
```

## AI Tools for Data Analysis

### Tool Categories

#### Data Exploration & Visualization
- **Quadratic**: Spreadsheet-AI hybrid
- **Tableau Pulse**: AI-powered business intelligence
- **Power BI**: Microsoft's analytics platform

#### Automation & Querying
- **SheetAI**: AI for spreadsheet automation
- **MonkeyLearn**: Text analysis and NLP
- **Ajelix**: Excel and data automation

#### Predictive/Advanced Analytics
- **DataRobot**: Automated machine learning (AutoML)

### How to Use AI for Data Analysis (Step-by-Step)

#### 1. Define Goals
Identify what you need—demand forecasting, churn prediction, or quick reporting.

#### 2. Prep Data
Ensure data is clean, formatted, and connected (e.g., uploading CSVs or linking SQL databases).

```python
# Data cleaning example
df = df.dropna()  # Remove missing values
df = df.drop_duplicates()  # Remove duplicates
df['date'] = pd.to_datetime(df['date'])  # Ensure proper date format
```

#### 3. Use Natural Language Querying
Ask tools questions like:
- "What was the sales trend in Q3?"
- "Which factors influence customer churn?"

#### 4. Iterative Prompting
Refine AI results by:
- Asking follow-up questions
- Requesting different chart types
- Asking for the underlying logic

#### 5. Validate Insights
Evaluate AI-generated trends for accuracy and business context.

### Why Use AI-Based Tools?

**Accessibility**: Designed with non-technical users in mind, featuring intuitive interfaces and natural language processing capabilities.

**Versatility**: Handle various data formats, from Excel spreadsheets to PDF reports, making them adaptable to existing workflows.

**Speed**: AI algorithms process and analyze large datasets much faster than traditional methods, providing quick insights.

**Cost-effectiveness**: Many tools offer affordable pricing plans, making advanced data analysis accessible to businesses of all sizes.

### Best Practices

Before diving in, ensure your data is organized and clean:
- For Excel sheets, use clear headers and consistent formatting
- When working with PDFs, focus on documents with structured data like reports or invoices
- Once you've selected a tool, upload your data and start exploring

The AI agent needs context and specific instructions. The better you are with your instructions, the stronger the output.

## How AI Agents Work

AI agents act as autonomous "digital analysts" that monitor, clean, and analyze complex datasets 24/7 using natural language, significantly reducing reliance on manual SQL or spreadsheet work.

### Key Use Cases

- **Automated Sales Reporting**: Generate daily/weekly/monthly sales reports automatically
- **Predictive Maintenance**: In manufacturing, predict equipment failures before they occur
- **Instant, Conversational Insights**: Query massive datasets using natural language

### Multi-Step Workflow

#### Step 1: Understand the Business Question
The agent receives input through natural language (e.g., "Why did churn increase last week?") or a scheduled trigger. It interprets intent using natural language processing.

#### Step 2: Plan the Investigation
The agent breaks the problem into sub-tasks:
- Identify relevant data sources
- Determine which metrics to compare
- Decide on time periods and segments to analyze

#### Step 3: Connect to Data Sources
The agent queries databases, data warehouses, and business tools directly:
- Generates SQL for structured databases
- Calls APIs for external data
- Retrieves information from connected systems (Snowflake, Salesforce, Google Sheets)

#### Step 4: Execute Analysis
The agent:
- Runs queries
- Aggregates results
- Compares segments
- Checks for statistical anomalies
- If initial results are inconclusive, adjusts approach and queries alternative sources

#### Step 5: Deliver Findings
The agent assembles results into a readable format:
- Charts and visualizations
- Summary insights
- Confidence levels
- Recommended next steps
- Source attribution for verification

### How AI Agents Generate SQL Queries

When a user asks a question in natural language:

1. **Schema Access**: Agent accesses table schemas (column names, data types, relationships)
2. **Query Generation**: Generates SQL query matching the intent
3. **Validation**: Validates query structure before execution
4. **Execution**: Runs query against database with appropriate limits
5. **Formatting**: Formats results and provides context

Example:
```
User: "What were the top 5 products by revenue last month?"

Agent generates:
SELECT product_name, SUM(revenue) as total_revenue
FROM sales
WHERE date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)
GROUP BY product_name
ORDER BY total_revenue DESC
LIMIT 5;
```

### Common Structured Data Sources

- **Relational database systems**: MySQL, PostgreSQL, etc., which store data in table form
- **Spreadsheet files**: Excel, Google Sheets and other office documents
- **Knowledge graph systems**: Wikidata and other semantic network databases

## Frameworks & Technologies

### How Generative AI Accelerates Development

These frameworks help you build, deploy, and manage Generative AI applications by providing structure, automation, and integration capabilities.

#### LangChain/LangGraph
Open-source frameworks for building LLM-powered applications. LangChain lets you integrate external components and data sources, while LangGraph extends these capabilities by allowing developers to create structured, state-aware agent workflows using graph-based architectures.

- Industry standard for building custom agentic workflows
- Documentation: https://docs.langchain.com/

#### Pydantic AI
Provides a structured framework for building type-safe LLM applications, ensuring data validation and consistent output formatting. Build reliable AI applications with predictable response structures and error handling.

- Documentation: https://pydantic.dev/docs/ai/overview/

#### AutoGen
Helps you build multi-agent systems that can collaborate to solve complex tasks. Develop conversational agents that work together, share context, and execute multi-step workflows autonomously.

- Allows multiple agents (e.g., "Coder" and "Reviewer") to collaborate

#### CrewAI
Enables you to orchestrate multiple AI agents to handle complex tasks. Create agent-based workflows where specialized agents collaborate, delegate tasks, and share information to achieve specific goals.

- Orchestrates role-playing agents to complete complex, multi-step data pipelines

### Tools in Deep Agents

Tools extend what agents can do, such as:
- Fetch real-time data
- Execute code
- Query external databases
- Take actions in the world

#### Tool Documentation
- Tools Overview: https://docs.langchain.com/oss/python/langchain/tools
- Tool Calling: https://docs.langchain.com/oss/python/langchain/models#tool-calling

#### Tool Calling Flow
Models can request to call tools that perform tasks such as:
- Fetching data from a database
- Searching the web
- Running code

Explore the tool calling flow: https://docs.langchain.com/oss/python/langchain/models#tool-calling

### Google Cloud Integration

Google Cloud's agent development tools reduce the need for developers to build custom database connectors through ADK and MCP integration methods.

- Documentation: https://cloud.google.com/use-cases/data-analytics-agents#agent-development-tools

### Use Cases of Generative AI

#### Writing Code
LLM-powered tools generate the first version of code, allowing developers to reduce time to market and finalize projects faster. AI tools are invaluable for teams that need to quickly access code snippets for simple use cases and adapt legacy code for modern websites.

#### Chatbots
Virtual assistants help entities streamline daily interactions with customers, win their support, and build brand loyalty.

#### Data Governance
AI tools simplify the process of handling documentation and allow one to create metadata and track lineage automatically.

#### Visualizations
Modern web-based services with built-in AI tools facilitate creating visualizations using chatbots. Write a conversational prompt, and an AI app will generate infographics and other types of images.

#### AI Agents
These tools have advanced reasoning skills, which make it easier to handle challenging analytical issues. They adapt to a changing environment and fine-tune their functioning based on previous tasks and projects.

## DevOps Setup Guide

### For DevOps Engineers

This section provides step-by-step instructions for setting up the development environment and deploying the AI data analysis applications.

#### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, Debian, RHEL, CentOS), macOS 10.15+, Windows 10+ with WSL2
- **Python**: Version 3.8, 3.9, 3.10, or 3.11
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Disk Space**: At least 2GB free space
- **Network**: Internet connection for package downloads and API calls

#### Initial Setup

```bash
# 1. Clone or navigate to the repository
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Introduction

# 2. Verify Python installation
python3 --version  # Should be 3.8 or higher

# 3. Create virtual environment
python3 -m venv venv

# 4. Activate virtual environment
source venv/bin/activate

# 5. Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# 6. Install dependencies
pip install -r requirements.txt

# 7. Verify installation
python -c "import langchain; import pandas; import streamlit; print('SUCCESS')"
```

#### Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
cat > .env << 'EOF'
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4

# Anthropic Configuration (Optional)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# LangSmith Configuration (Optional - for tracing and debugging)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key-here
LANGSMITH_PROJECT=data-analysis-agents

# Application Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
EOF

# Load environment variables
source .env
```

#### Running the Applications

**Use Case 1 - Data Agent (Streamlit)**:

```bash
# Activate environment
source venv/bin/activate

# Navigate to use case 1
cd use_case_1

# Run Streamlit app
streamlit run data_agent.py --server.port 8501 --server.address localhost

# Access at: http://localhost:8501
```

**Use Case 2 - Deep Agent**:

```bash
# Activate environment
source venv/bin/activate

# Navigate to use case 2
cd use_case_2

# Run deep agent
python deep_agent.py

# Or run CSV analyzer
python csv_analyzer.py --input sample_data.csv
```

#### Testing

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=use_case_1 --cov=use_case_2 --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

#### Continuous Integration

Example GitHub Actions workflow (`.github/workflows/ci.yml`):

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

#### Troubleshooting

**Issue: ModuleNotFoundError**
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
source venv/bin/activate
pip install -r requirements.txt
```

**Issue: API Key Errors**
```bash
# Solution: Verify environment variables are set
echo $OPENAI_API_KEY
# If empty, export the key:
export OPENAI_API_KEY='your-key-here'
```

**Issue: Streamlit Port Already in Use**
```bash
# Solution: Use a different port
streamlit run data_agent.py --server.port 8502
```

**Issue: Permission Denied on Scripts**
```bash
# Solution: Make scripts executable
chmod +x setup_env.sh activate_env.sh
```

#### Deployment

**Docker Deployment**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "use_case_1/data_agent.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Production Best Practices**:
- Use environment-specific configuration files
- Implement proper logging (not just verbose mode)
- Set up monitoring and alerting
- Use secret management tools (AWS Secrets Manager, Azure Key Vault)
- Implement rate limiting for API calls
- Cache frequently accessed data
- Use connection pooling for database connections

#### Monitoring

```bash
# Monitor Streamlit app logs
streamlit run data_agent.py --logger.level debug

# Monitor Python application
python deep_agent.py --log-level DEBUG

# System resource monitoring
htop
# or
top
```

#### Maintenance

```bash
# Update dependencies
pip list --outdated
pip install --upgrade package-name

# Regenerate requirements.txt
pip freeze > requirements.txt

# Clean up
# Remove cache files
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Deactivate virtual environment when done
deactivate
```

## Additional Resources

### Documentation Links

- **LangChain Documentation**: https://docs.langchain.com/
- **DeepAgents Overview**: https://docs.langchain.com/oss/python/deepagents/overview
- **DeepAgents PyPI**: https://pypi.org/project/deepagents/
- **Pydantic AI**: https://pydantic.dev/docs/ai/overview/
- **LangSmith**: https://smith.langchain.com/
- **Ollama Integrations**: https://docs.langchain.com/oss/python/integrations/providers/ollama
- **Google Cloud Agent Tools**: https://cloud.google.com/use-cases/data-analytics-agents#agent-development-tools

### Learning Resources

- **Build Data Analysis Agent**: https://docs.langchain.com/oss/python/deepagents/data-analysis
- **Deep Agents Models**: https://docs.langchain.com/oss/python/deepagents/models
- **Tools Documentation**: https://docs.langchain.com/oss/python/langchain/tools
- **Tool Calling Flow**: https://docs.langchain.com/oss/python/langchain/models#tool-calling
- **Backends Configuration**: https://docs.langchain.com/oss/python/deepagents/backends

## License

Please comply with the terms of service for any third-party APIs and services used.

---

**Note**: This project requires API keys from OpenAI or other LLM providers.
