"""
Data Agent - Use Case 1
A Streamlit application that uses LangChain to create an AI agent
that can chat with your data using natural language queries.
"""

import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data from file path.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def initialize_agent(df: pd.DataFrame, model_choice: str = "openai", temperature: float = 0):
    """
    Initialize the LangChain pandas dataframe agent.
    
    Args:
        df: pandas DataFrame to analyze
        model_choice: Choice of LLM provider ('openai' or 'anthropic')
        temperature: Model temperature (0 = deterministic, 1 = creative)
        
    Returns:
        LangChain agent
    """
    try:
        if model_choice == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY not found in environment variables")
                return None
            
            llm = ChatOpenAI(
                temperature=temperature,
                model="gpt-4",
                api_key=api_key
            )
        elif model_choice == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("ANTHROPIC_API_KEY not found in environment variables")
                return None
            
            llm = ChatAnthropic(
                temperature=temperature,
                model="claude-3-sonnet-20240229",
                api_key=api_key
            )
        else:
            st.error("Invalid model choice. Choose 'openai' or 'anthropic'")
            return None
        
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True  # Required for code execution
        )
        
        return agent
    
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None


def main():
    """
    Main Streamlit application.
    """
    st.set_page_config(
        page_title="AI Data Analysis Agent",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 AI Data Analysis Agent")
    st.markdown("""
    Ask questions about your data in plain English and get intelligent answers.
    This agent uses LangChain and Large Language Models to analyze your CSV data.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select LLM Provider",
        ["openai", "anthropic"],
        help="Choose between OpenAI (GPT-4) or Anthropic (Claude)"
    )
    
    # Temperature setting
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 = deterministic, 1 = creative"
    )
    
    # File upload or default data
    st.sidebar.header("Data Source")
    use_default = st.sidebar.checkbox("Use default sales data", value=True)
    
    df = None
    
    if use_default:
        # Use default sales_data.csv
        default_path = os.path.join(os.path.dirname(__file__), "sales_data.csv")
        if os.path.exists(default_path):
            df = load_data(default_path)
            st.sidebar.success("Using default sales data")
        else:
            st.sidebar.warning("Default data not found. Please upload a CSV file.")
    else:
        # File uploader
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload your CSV data file"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
    
    # Display data preview
    if df is not None:
        st.subheader("Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Show first few rows
        with st.expander("View Data Sample", expanded=False):
            st.dataframe(df.head(10))
        
        # Show column info
        with st.expander("Column Information", expanded=False):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info)
        
        # Initialize agent
        if 'agent' not in st.session_state or st.sidebar.button("Reinitialize Agent"):
            with st.spinner("Initializing AI agent..."):
                agent = initialize_agent(df, model_choice, temperature)
                if agent:
                    st.session_state.agent = agent
                    st.success("Agent initialized successfully!")
        
        # Query interface
        st.subheader("Ask Questions About Your Data")
        
        # Example questions
        with st.expander("Example Questions", expanded=False):
            st.markdown("""
            - What was the average profit margin per product category?
            - Show me the top 5 products by revenue
            - What is the trend in sales over the last quarter?
            - Which customer segment has the highest lifetime value?
            - Calculate the correlation between price and sales volume
            - What is the distribution of sales by region?
            - Identify any outliers in the revenue data
            """)
        
        # Chat interface
        if 'agent' in st.session_state:
            # Initialize chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your data..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get agent response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        try:
                            response = st.session_state.agent.run(prompt)
                            st.markdown(response)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response
                            })
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_msg
                            })
            
            # Clear chat button
            if st.sidebar.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        
        else:
            st.warning("Please initialize the agent first using the sidebar settings.")
    
    else:
        st.info("Please load data using the sidebar options.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This application uses:
    - **LangChain** for agent orchestration
    - **OpenAI GPT-4** or **Anthropic Claude** for reasoning
    - **Streamlit** for the user interface
    
    The agent can generate and execute Python code to answer your questions.
    """)


if __name__ == "__main__":
    main()
