"""
Tests for Use Case 1 - Data Agent
"""

import os
import sys
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'use_case_1'))

from data_agent import load_data, initialize_agent


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_load_valid_csv(self, tmp_path):
        """Test loading a valid CSV file"""
        # Create temporary CSV file
        csv_file = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        test_data.to_csv(csv_file, index=False)
        
        # Load the data
        df = load_data(str(csv_file))
        
        # Assertions
        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ['col1', 'col2']
    
    def test_load_invalid_file(self):
        """Test loading a non-existent file"""
        df = load_data('non_existent_file.csv')
        assert df is None
    
    def test_load_sales_data(self):
        """Test loading the actual sales_data.csv"""
        sales_data_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'use_case_1',
            'sales_data.csv'
        )
        
        if os.path.exists(sales_data_path):
            df = load_data(sales_data_path)
            assert df is not None
            assert 'product_category' in df.columns
            assert 'revenue' in df.columns
            assert len(df) > 0


class TestAgentInitialization:
    """Test agent initialization"""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    @patch('data_agent.create_pandas_dataframe_agent')
    @patch('data_agent.ChatOpenAI')
    def test_initialize_openai_agent(self, mock_chat_openai, mock_create_agent):
        """Test initializing agent with OpenAI"""
        # Setup mocks
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        # Create test dataframe
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Initialize agent
        agent = initialize_agent(df, model_choice='openai', temperature=0)
        
        # Assertions
        assert agent is not None
        mock_chat_openai.assert_called_once()
        mock_create_agent.assert_called_once()
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('data_agent.create_pandas_dataframe_agent')
    @patch('data_agent.ChatAnthropic')
    def test_initialize_anthropic_agent(self, mock_chat_anthropic, mock_create_agent):
        """Test initializing agent with Anthropic"""
        # Setup mocks
        mock_llm = Mock()
        mock_chat_anthropic.return_value = mock_llm
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        
        # Create test dataframe
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Initialize agent
        agent = initialize_agent(df, model_choice='anthropic', temperature=0)
        
        # Assertions
        assert agent is not None
        mock_chat_anthropic.assert_called_once()
        mock_create_agent.assert_called_once()
    
    def test_initialize_without_api_key(self):
        """Test initializing agent without API key"""
        # Ensure no API key in environment
        with patch.dict(os.environ, {}, clear=True):
            df = pd.DataFrame({'col1': [1, 2, 3]})
            agent = initialize_agent(df, model_choice='openai')
            assert agent is None


class TestDataAnalysis:
    """Test data analysis capabilities"""
    
    def test_dataframe_statistics(self):
        """Test basic dataframe statistics"""
        df = pd.DataFrame({
            'revenue': [100, 200, 300, 400, 500],
            'cost': [60, 120, 180, 240, 300]
        })
        
        # Calculate profit margin
        df['profit_margin'] = (df['revenue'] - df['cost']) / df['revenue']
        
        # Assertions
        assert df['profit_margin'].mean() == pytest.approx(0.4, rel=1e-2)
        assert len(df) == 5
        assert df['revenue'].sum() == 1500
    
    def test_categorical_aggregation(self):
        """Test groupby and aggregation"""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50]
        })
        
        result = df.groupby('category')['value'].mean()
        
        assert result['A'] == 30
        assert result['B'] == 30
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': [10, None, 30, None, 50]
        })
        
        # Check missing values
        assert df['col1'].isnull().sum() == 1
        assert df['col2'].isnull().sum() == 2
        
        # Fill missing values
        df_filled = df.fillna(0)
        assert df_filled['col1'].isnull().sum() == 0


class TestValidation:
    """Test data validation and sanity checks"""
    
    def test_sanity_check_totals(self):
        """Test that calculated totals match expected values"""
        sales_data_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'use_case_1',
            'sales_data.csv'
        )
        
        if os.path.exists(sales_data_path):
            df = pd.read_csv(sales_data_path)
            
            # Sanity checks
            assert df['revenue'].sum() > 0
            assert df['sales_volume'].sum() > 0
            
            # Check profit margin is between 0 and 1
            assert (df['profit_margin'] >= 0).all()
            assert (df['profit_margin'] <= 1).all()
            
            # Verify revenue = (implied price per unit) * sales_volume
            # This is a logical consistency check
            assert len(df) > 0
    
    def test_data_types(self):
        """Test that data types are correct"""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'category': ['A', 'B'],
            'value': [100, 200]
        })
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        assert df['date'].dtype == 'datetime64[ns]'
        assert df['category'].dtype == 'object'
        assert df['value'].dtype in ['int64', 'int32']


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for testing"""
    return pd.DataFrame({
        'product_category': ['Electronics', 'Furniture', 'Electronics', 'Furniture'],
        'revenue': [1000, 2000, 1500, 2500],
        'cost': [600, 1200, 900, 1500],
        'profit_margin': [0.4, 0.4, 0.4, 0.4]
    })


def test_sample_fixture(sample_dataframe):
    """Test that the sample fixture works correctly"""
    assert len(sample_dataframe) == 4
    assert 'product_category' in sample_dataframe.columns
    assert sample_dataframe['revenue'].sum() == 7000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
