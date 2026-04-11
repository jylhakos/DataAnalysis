"""
Deep Agent - Use Case 2
A deep agent implementation using DeepAgents library for CSV analysis.
This agent performs exploratory data analysis and generates visualizations.
"""

import os
from typing import List, Dict, Any
from deepagents import create_deep_agent


def analyze_csv(file_path: str) -> str:
    """
    Analyze a CSV file and return statistical insights.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Analysis results as a string
    """
    import pandas as pd
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Basic statistics
        results = []
        results.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        results.append(f"\nColumns: {', '.join(df.columns.tolist())}")
        results.append(f"\nData Types:\n{df.dtypes.to_string()}")
        results.append(f"\nMissing Values:\n{df.isnull().sum().to_string()}")
        results.append(f"\nBasic Statistics:\n{df.describe().to_string()}")
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"


def create_visualization(df_data: str, chart_type: str = "histogram") -> str:
    """
    Create a visualization from dataframe data.
    
    Args:
        df_data: CSV data as string
        chart_type: Type of chart to create
        
    Returns:
        Message indicating visualization was created
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import StringIO
    
    try:
        # Load data
        df = pd.read_csv(StringIO(df_data))
        
        # Create visualization based on type
        plt.figure(figsize=(10, 6))
        
        if chart_type == "histogram":
            # Plot numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df[numeric_cols].hist(figsize=(12, 8), bins=20)
                plt.suptitle("Distribution of Numeric Variables")
        
        elif chart_type == "correlation":
            # Correlation heatmap
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                import seaborn as sns
                plt.figure(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
                plt.title("Correlation Matrix")
        
        elif chart_type == "bar":
            # Bar chart of first categorical column
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                df[cat_cols[0]].value_counts().plot(kind='bar')
                plt.title(f"Distribution of {cat_cols[0]}")
                plt.xlabel(cat_cols[0])
                plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig("visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Visualization created and saved as 'visualization.png'"
    
    except Exception as e:
        return f"Error creating visualization: {str(e)}"


def get_data_summary(city: str) -> str:
    """
    Example tool: Get weather data for a city.
    
    Args:
        city: Name of the city
        
    Returns:
        Weather information
    """
    return f"It's always sunny in {city}!"


def read_csv_file(file_path: str) -> str:
    """
    Read and return CSV file contents.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        CSV contents as string
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


def main():
    """
    Main function to demonstrate deep agent usage.
    """
    print("=== Deep Agent for CSV Analysis ===\n")
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("Please set your API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Define tools for the agent
    tools = [
        analyze_csv,
        create_visualization,
        read_csv_file,
        get_data_summary
    ]
    
    # Create the deep agent
    print("Initializing deep agent...")
    agent = create_deep_agent(
        tools=tools,
        system_prompt="""You are a helpful data analysis assistant. 
        You can analyze CSV files, generate statistical insights, and create visualizations.
        When asked to analyze data, use the available tools to:
        1. Read the CSV file
        2. Perform statistical analysis
        3. Create appropriate visualizations
        4. Provide actionable insights
        Always be thorough and explain your findings clearly.""",
    )
    
    print("Agent initialized successfully!\n")
    
    # Example usage scenarios
    
    # Scenario 1: Basic agent interaction
    print("=== Scenario 1: Basic Interaction ===")
    response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "what is the weather in San Francisco?"
        }]
    })
    print(f"Response: {response}\n")
    
    # Scenario 2: CSV Analysis
    print("=== Scenario 2: CSV Analysis ===")
    csv_file = "sample_data.csv"
    
    if os.path.exists(csv_file):
        print(f"Analyzing {csv_file}...")
        response = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Please analyze the CSV file '{csv_file}' and provide insights about the data."
            }]
        })
        print(f"Analysis: {response}\n")
    else:
        print(f"Warning: {csv_file} not found. Creating sample data...")
        # The csv_analyzer.py will create sample data
    
    # Interactive mode
    print("=== Interactive Mode ===")
    print("You can now ask questions about your data.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Get agent response
            response = agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": user_input
                }]
            })
            
            print(f"\nAgent: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()
