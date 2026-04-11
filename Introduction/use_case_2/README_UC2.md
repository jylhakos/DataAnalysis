# Use Case 2: Deep Agent for CSV Analysis

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [What are Deep Agents?](#what-are-deep-agents)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Output Structure](#output-structure)
- [Sample Data](#sample-data)
- [Deep Agent Features](#deep-agent-features)
- [Interactive Mode](#interactive-mode)
- [Examples](#examples)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Performance Considerations](#performance-considerations)
- [Testing](#testing)
- [Next Steps](#next-steps)
- [Resources](#resources)
- [Key Differences from Use Case 1](#key-differences-from-use-case-1)
- [Contributing](#contributing)

## Overview

This use case demonstrates building a sophisticated deep agent using the DeepAgents library that can autonomously analyze CSV files, perform exploratory data analysis, and generate visualizations.

## Features

- Deep agent implementation with planning capabilities
- Exploratory data analysis (EDA)
- Automated statistical analysis
- Data quality assessment
- Outlier detection
- Correlation analysis
- Multiple visualization types (distributions, heatmaps, box plots)
- Detailed analysis reports
- Command-line interface for easy usage
- Interactive agent mode

## Architecture

```
CSV File
    ↓
Deep Agent (LangGraph Runtime)
    ↓
┌─────────────┬──────────────┬───────────────┐
│  Read File  │  Analyze     │  Visualize    │
│  Tool       │  Tool        │  Tool         │
└─────────────┴──────────────┴───────────────┘
    ↓              ↓              ↓
Planning → Code Execution → Artifact Creation
    ↓
Results & Insights
```

## What are Deep Agents?

Using an LLM to call tools in a loop is the simplest form of an agent. This architecture, however, can yield agents that are "shallow".

Deep agents provide:
- **Planning capabilities**: Break down complex tasks into steps
- **Code execution**: Generate and run analysis code
- **Artifact management**: Work with scripts, reports, and plots
- **Multi-step reasoning**: Handle complex analytical workflows
- **State management**: Maintain context across operations
- **Durable execution**: Resume from failures
- **Streaming**: Real-time output
- **Human-in-the-loop**: Allow for manual intervention

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment activated
- OpenAI API key

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

```bash
# Required: OpenAI API key
export OPENAI_API_KEY='your-openai-api-key-here'

# Optional: LangSmith for tracing
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY='your-langsmith-api-key-here'
```

## Usage

### Running the Deep Agent

```bash
# Navigate to use case 2 directory
cd use_case_2

# Run the deep agent (interactive mode)
python deep_agent.py
```

The agent will:
1. Initialize with custom tools
2. Run example scenarios
3. Enter interactive mode for questions

### Running the CSV Analyzer

```bash
# Analyze the sample data
python csv_analyzer.py --input sample_data.csv --output output

# Analyze your own data
python csv_analyzer.py --input /path/to/your/data.csv --output results

# Use default settings (analyzes sample_data.csv)
python csv_analyzer.py
```

### Command-Line Options

```bash
# Show help
python csv_analyzer.py --help

# Specify input file
python csv_analyzer.py -i data.csv

# Specify output directory
python csv_analyzer.py -o my_results

# Full example
python csv_analyzer.py --input customer_data.csv --output analysis_results
```

## Components

### 1. Deep Agent (deep_agent.py)

The main deep agent implementation with custom tools:

**Tools available**:
- `analyze_csv`: Perform statistical analysis on CSV files
- `create_visualization`: Generate charts and graphs
- `read_csv_file`: Read CSV file contents
- `get_data_summary`: Get data summaries

**Example usage**:

```python
from deepagents import create_deep_agent

# Define tools
tools = [analyze_csv, create_visualization, read_csv_file]

# Create agent
agent = create_deep_agent(
    tools=tools,
    system_prompt="You are a helpful data analysis assistant..."
)

# Run agent
response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Please analyze the CSV file 'sample_data.csv'"
    }]
})
```

### 2. CSV Analyzer (csv_analyzer.py)

Advanced data analysis tool with EDA capabilities:

**Features**:
- Automatic column type detection (numeric, categorical, datetime)
- Data quality checks (missing values, duplicates)
- Statistical summaries
- Correlation analysis
- Outlier detection (IQR and Z-score methods)
- Multiple visualization types
- Detailed text reports

**Example usage**:

```python
from csv_analyzer import CSVAnalyzer

# Create analyzer
analyzer = CSVAnalyzer('sample_data.csv')

# Run full analysis
analyzer.run_full_analysis(output_dir='output')
```

## Output Structure

After running the CSV analyzer, you'll get:

```
output/
├── analysis_report.txt           # Text report
└── visualizations/
    ├── distributions.png          # Distribution histograms
    ├── correlation_heatmap.png    # Correlation matrix
    ├── boxplots.png              # Box plots for outliers
    └── categorical_distributions.png  # Category distributions
```

## Sample Data

The included `sample_data.csv` contains 500 rows with:

- `customer_id`: Unique customer identifier
- `age`: Customer age (18-80)
- `income`: Annual income
- `purchase_amount`: Purchase amount
- `satisfaction_score`: Satisfaction rating (1-10)
- `region`: Geographic region (North, South, East, West)
- `customer_type`: Customer category (New, Returning, VIP)
- `product_category`: Product category (Electronics, Clothing, Food, Books)

## Deep Agent Features

### Filesystem Tools

Deep Agents expose filesystem operations via tools:

- `ls`: List directory contents
- `read_file`: Read file contents
- `write_file`: Write to files
- `edit_file`: Edit existing files
- `glob`: Pattern matching for files
- `grep`: Search within files

Documentation: https://docs.langchain.com/oss/python/deepagents/backends

### Model Configuration

You can specify different models:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from deepagents import create_deep_agent

# Use OpenAI GPT-4
agent = create_deep_agent(
    tools=tools,
    model=ChatOpenAI(model="gpt-4", temperature=0)
)

# Use Anthropic Claude
agent = create_deep_agent(
    tools=tools,
    model=ChatAnthropic(model="claude-3-sonnet-20240229")
)

# Use Ollama (local model)
from langchain_ollama import ChatOllama

agent = create_deep_agent(
    tools=tools,
    model=ChatOllama(model="llama2")
)
```

Documentation: https://docs.langchain.com/oss/python/deepagents/models

### LangSmith Tracing

Enable tracing for debugging and evaluation:

```bash
# Set environment variables
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY='your-api-key'
export LANGSMITH_PROJECT='csv-analysis-agents'

# Run your agent
python deep_agent.py
```

Visit https://smith.langchain.com/ to:
- Trace requests
- Debug agent behavior
- Evaluate outputs
- Monitor performance

## Interactive Mode

The deep agent supports interactive mode:

```bash
python deep_agent.py
```

Then ask questions:

```
You: Analyze the sample_data.csv file and tell me about customer age distribution

Agent: [Provides detailed analysis]

You: What is the correlation between age and income?

Agent: [Generates correlation analysis]

You: Create a visualization showing satisfaction scores by region

Agent: [Creates visualization]
```

Type `exit`, `quit`, or press Ctrl+C to exit.

## Examples

### Example 1: Basic Analysis

```bash
python csv_analyzer.py --input sample_data.csv
```

Output:
- Statistical summary
- Missing value report
- Correlation matrix
- Outlier detection
- Visualizations

### Example 2: Custom Analysis with Deep Agent

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    tools=[analyze_csv, create_visualization],
    system_prompt="You are a data scientist."
)

# Ask for specific analysis
response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": """
        Analyze sample_data.csv and:
        1. Identify which region has the highest average income
        2. Find correlations between satisfaction score and other variables
        3. Detect any outliers in purchase amounts
        4. Create a visualization of age distribution by customer type
        """
    }]
})
```

### Example 3: Programmatic Analysis

```python
from csv_analyzer import CSVAnalyzer

# Initialize analyzer
analyzer = CSVAnalyzer('sample_data.csv')
analyzer.load_data()

# Get basic info
info = analyzer.basic_info()
print(f"Dataset has {info['shape'][0]} rows and {info['shape'][1]} columns")

# Check data quality
quality = analyzer.data_quality_check()
print(f"Missing values: {sum(quality['missing_values'].values())}")
print(f"Duplicate rows: {quality['duplicate_rows']}")

# Get statistics
stats = analyzer.statistical_summary()
print(stats)

# Detect outliers
outliers = analyzer.detect_outliers(method='iqr')
for col, indices in outliers.items():
    print(f"{col}: {len(indices)} outliers")

# Generate visualizations
analyzer.generate_visualizations(output_dir='my_plots')

# Create report
analyzer.generate_report(output_file='my_report.txt')
```

## Advanced Features

### Custom Tools

Create custom tools for your specific analysis needs:

```python
def calculate_customer_lifetime_value(data: str) -> str:
    """Calculate customer lifetime value."""
    import pandas as pd
    from io import StringIO
    
    df = pd.read_csv(StringIO(data))
    
    # Your custom calculation
    clv = df.groupby('customer_id')['purchase_amount'].sum().mean()
    
    return f"Average customer lifetime value: ${clv:.2f}"

# Add to agent
agent = create_deep_agent(
    tools=[analyze_csv, calculate_customer_lifetime_value],
    system_prompt="You are a business analyst."
)
```

### Outlier Detection Methods

```python
# IQR method (default)
outliers_iqr = analyzer.detect_outliers(method='iqr')

# Z-score method
outliers_zscore = analyzer.detect_outliers(method='zscore')
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
analyzer = CSVAnalyzer('sample_data.csv')
analyzer.load_data()

# Create custom plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=analyzer.df, x='region', y='income', hue='customer_type')
plt.title('Income Distribution by Region and Customer Type')
plt.savefig('custom_plot.png')
```

## Troubleshooting

### Issue: deepagents not found

```bash
# Error: ModuleNotFoundError: No module named 'deepagents'

# Solution: Install the package
pip install deepagents
# or
pip install -r requirements.txt
```

### Issue: Visualization not displaying

```bash
# If running on a server without display

# Solution: Use non-interactive backend
export MPLBACKEND=Agg

# Or in code:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### Issue: Memory error with large files

```python
# For large CSV files, use chunking
import pandas as pd

chunks = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunks:
    # Process each chunk
    print(chunk.describe())
```

## Performance Considerations

1. **Large Datasets**: For files > 100MB, consider sampling or chunking
2. **API Costs**: Deep agents make multiple LLM calls; monitor usage
3. **Visualization Memory**: Large plots consume memory; limit resolution if needed
4. **Caching**: Cache analysis results for repeated queries

## Testing

Run the included tests:

```bash
# From project root
pytest tests/test_deep_agent.py -v
```

## Next Steps

- Integrate with databases (PostgreSQL, MySQL)
- Add time series analysis capabilities
- Implement predictive modeling
- Create custom visualization templates
- Build a web interface
- Deploy as an API service
- Add support for multiple file formats (Excel, JSON, Parquet)

## Resources

- **DeepAgents Documentation**: https://docs.langchain.com/oss/python/deepagents/overview
- **DeepAgents PyPI**: https://pypi.org/project/deepagents/
- **LangChain Tools**: https://docs.langchain.com/oss/python/langchain/tools
- **LangSmith**: https://smith.langchain.com/
- **Build Data Analysis Agent**: https://docs.langchain.com/oss/python/deepagents/data-analysis

## Key Differences from Use Case 1

| Feature | Use Case 1 (Data Agent) | Use Case 2 (Deep Agent) |
|---------|------------------------|------------------------|
| Interface | Streamlit web app | CLI / Interactive |
| Agent Type | Shallow (single loop) | Deep (planning + execution) |
| Analysis | On-demand queries | EDA |
| Visualizations | None (text only) | Multiple charts |
| Reports | Chat responses | Detailed text reports |
| Tools | Built-in pandas | Custom tools |
| State Management | Session-based | Durable execution |
| Use Case | Interactive Q&A | Automated analysis |

## Contributing

Suggestions for improvements:
- Add more visualization types
- Support for multi-file analysis
- Integration with cloud storage
- Real-time data streaming
- Advanced ML models
- Custom report templates

---

For questions or issues, refer to the main README.md or the official DeepAgents documentation.
