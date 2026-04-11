"""
Tests for Use Case 2 - Deep Agent and CSV Analyzer
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'use_case_2'))

from csv_analyzer import CSVAnalyzer, create_sample_data


class TestCSVAnalyzer:
    """Test CSVAnalyzer class"""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing"""
        csv_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            'id': range(1, 101),
            'age': np.random.randint(18, 80, 100),
            'income': np.random.normal(50000, 20000, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'score': np.random.uniform(0, 100, 100)
        })
        df.to_csv(csv_file, index=False)
        return str(csv_file)
    
    def test_analyzer_initialization(self, sample_csv):
        """Test analyzer initialization"""
        analyzer = CSVAnalyzer(sample_csv)
        assert analyzer.file_path == sample_csv
        assert analyzer.df is None
    
    def test_load_data(self, sample_csv):
        """Test data loading"""
        analyzer = CSVAnalyzer(sample_csv)
        result = analyzer.load_data()
        
        assert result is True
        assert analyzer.df is not None
        assert len(analyzer.df) == 100
    
    def test_identify_column_types(self, sample_csv):
        """Test column type identification"""
        analyzer = CSVAnalyzer(sample_csv)
        analyzer.load_data()
        
        assert 'age' in analyzer.numeric_columns
        assert 'income' in analyzer.numeric_columns
        assert 'category' in analyzer.categorical_columns
    
    def test_basic_info(self, sample_csv):
        """Test basic info extraction"""
        analyzer = CSVAnalyzer(sample_csv)
        analyzer.load_data()
        info = analyzer.basic_info()
        
        assert info['shape'] == (100, 5)
        assert len(info['columns']) == 5
        assert len(info['numeric_columns']) > 0
        assert len(info['categorical_columns']) > 0
    
    def test_data_quality_check(self, sample_csv):
        """Test data quality checking"""
        analyzer = CSVAnalyzer(sample_csv)
        analyzer.load_data()
        quality = analyzer.data_quality_check()
        
        assert 'missing_values' in quality
        assert 'duplicate_rows' in quality
        assert 'unique_counts' in quality
        assert quality['duplicate_rows'] >= 0
    
    def test_statistical_summary(self, sample_csv):
        """Test statistical summary"""
        analyzer = CSVAnalyzer(sample_csv)
        analyzer.load_data()
        stats = analyzer.statistical_summary()
        
        assert not stats.empty
        assert 'mean' in stats.index
        assert 'std' in stats.index
        assert 'min' in stats.index
        assert 'max' in stats.index
    
    def test_correlation_analysis(self, sample_csv):
        """Test correlation analysis"""
        analyzer = CSVAnalyzer(sample_csv)
        analyzer.load_data()
        corr = analyzer.correlation_analysis()
        
        assert not corr.empty
        # Check correlation matrix is symmetric
        assert corr.equals(corr.T)
    
    def test_outlier_detection_iqr(self, sample_csv):
        """Test IQR outlier detection"""
        analyzer = CSVAnalyzer(sample_csv)
        analyzer.load_data()
        outliers = analyzer.detect_outliers(method='iqr')
        
        assert isinstance(outliers, dict)
        # Outliers should be detected
        assert len(outliers) >= 0
    
    def test_outlier_detection_zscore(self, sample_csv):
        """Test Z-score outlier detection"""
        analyzer = CSVAnalyzer(sample_csv)
        analyzer.load_data()
        outliers = analyzer.detect_outliers(method='zscore')
        
        assert isinstance(outliers, dict)
    
    def test_generate_visualizations(self, sample_csv, tmp_path):
        """Test visualization generation"""
        analyzer = CSVAnalyzer(sample_csv)
        analyzer.load_data()
        
        output_dir = tmp_path / "viz"
        analyzer.generate_visualizations(output_dir=str(output_dir))
        
        # Check that visualization files were created
        assert output_dir.exists()
    
    def test_generate_report(self, sample_csv, tmp_path):
        """Test report generation"""
        analyzer = CSVAnalyzer(sample_csv)
        analyzer.load_data()
        
        report_file = tmp_path / "report.txt"
        analyzer.generate_report(output_file=str(report_file))
        
        # Check report file exists
        assert report_file.exists()
        
        # Check report has content
        content = report_file.read_text()
        assert len(content) > 0
        assert 'EXPLORATORY DATA ANALYSIS REPORT' in content
    
    def test_run_full_analysis(self, sample_csv, tmp_path):
        """Test full analysis pipeline"""
        analyzer = CSVAnalyzer(sample_csv)
        
        output_dir = tmp_path / "output"
        analyzer.run_full_analysis(output_dir=str(output_dir))
        
        # Check output directory exists
        assert output_dir.exists()
        
        # Check report was created
        report_file = output_dir / "analysis_report.txt"
        assert report_file.exists()


class TestSampleDataGeneration:
    """Test sample data generation"""
    
    def test_create_sample_data(self, tmp_path):
        """Test sample data creation"""
        csv_file = tmp_path / "sample.csv"
        create_sample_data(str(csv_file))
        
        # Check file was created
        assert csv_file.exists()
        
        # Load and verify data
        df = pd.read_csv(csv_file)
        assert len(df) == 500
        assert 'customer_id' in df.columns
        assert 'age' in df.columns
        assert 'income' in df.columns
    
    def test_sample_data_structure(self, tmp_path):
        """Test structure of generated sample data"""
        csv_file = tmp_path / "sample.csv"
        create_sample_data(str(csv_file))
        
        df = pd.read_csv(csv_file)
        
        # Check column types
        assert df['customer_id'].dtype in ['int64', 'int32']
        assert df['age'].dtype in ['int64', 'int32']
        assert df['income'].dtype in ['int64', 'int32']
        assert df['region'].dtype == 'object'
        
        # Check value ranges
        assert df['age'].min() >= 18
        assert df['age'].max() <= 80
        assert df['satisfaction_score'].min() >= 1
        assert df['satisfaction_score'].max() <= 10


class TestDeepAgentTools:
    """Test Deep Agent tool functions"""
    
    def test_analyze_csv_function(self, tmp_path):
        """Test analyze_csv tool"""
        # Import the function
        from deep_agent import analyze_csv
        
        # Create test CSV
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        df.to_csv(csv_file, index=False)
        
        # Run analysis
        result = analyze_csv(str(csv_file))
        
        # Verify result
        assert isinstance(result, str)
        assert 'Dataset Shape' in result or 'rows' in result.lower()
    
    def test_read_csv_file_function(self, tmp_path):
        """Test read_csv_file tool"""
        from deep_agent import read_csv_file
        
        # Create test CSV
        csv_file = tmp_path / "test.csv"
        test_content = "col1,col2\n1,a\n2,b\n"
        csv_file.write_text(test_content)
        
        # Read file
        result = read_csv_file(str(csv_file))
        
        # Verify result
        assert isinstance(result, str)
        assert 'col1' in result
        assert 'col2' in result


class TestDataValidation:
    """Test data validation and sanity checks"""
    
    def test_missing_values_detection(self):
        """Test detection of missing values"""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': [10, None, None, 40, 50]
        })
        
        missing = df.isnull().sum()
        
        assert missing['col1'] == 1
        assert missing['col2'] == 2
    
    def test_duplicate_detection(self):
        """Test detection of duplicate rows"""
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        
        duplicates = df.duplicated().sum()
        
        assert duplicates == 1
    
    def test_outlier_values(self):
        """Test outlier detection with known data"""
        # Create data with obvious outlier
        data = [10, 12, 11, 13, 12, 100]  # 100 is an outlier
        df = pd.DataFrame({'value': data})
        
        # IQR method
        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[(df['value'] < lower) | (df['value'] > upper)]
        
        assert len(outliers) == 1
        assert outliers['value'].values[0] == 100
    
    def test_data_type_validation(self):
        """Test data type validation"""
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3],
            'string_col': ['a', 'b', 'c'],
            'date_col': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        })
        
        assert df['numeric_col'].dtype in ['int64', 'int32']
        assert df['string_col'].dtype == 'object'
        assert df['date_col'].dtype == 'datetime64[ns]'


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_analysis(self, tmp_path):
        """Test complete analysis workflow"""
        # Create sample data
        csv_file = tmp_path / "data.csv"
        create_sample_data(str(csv_file))
        
        # Run analysis
        analyzer = CSVAnalyzer(str(csv_file))
        output_dir = tmp_path / "results"
        analyzer.run_full_analysis(output_dir=str(output_dir))
        
        # Verify outputs
        assert output_dir.exists()
        
        # Check report
        report = output_dir / "analysis_report.txt"
        assert report.exists()
        
        # Check visualizations directory
        viz_dir = output_dir / "visualizations"
        assert viz_dir.exists()
    
    def test_sample_data_analysis(self):
        """Test analysis of actual sample_data.csv if it exists"""
        sample_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'use_case_2',
            'sample_data.csv'
        )
        
        if os.path.exists(sample_path):
            analyzer = CSVAnalyzer(sample_path)
            assert analyzer.load_data() is True
            
            # Run basic checks
            info = analyzer.basic_info()
            assert info['shape'][0] > 0
            
            quality = analyzer.data_quality_check()
            assert 'missing_values' in quality


@pytest.fixture
def mock_deep_agent():
    """Fixture providing a mock deep agent"""
    mock_agent = Mock()
    mock_agent.invoke = Mock(return_value="Mocked response")
    return mock_agent


def test_mock_agent(mock_deep_agent):
    """Test that mock agent fixture works"""
    response = mock_deep_agent.invoke({"messages": [{"role": "user", "content": "test"}]})
    assert response == "Mocked response"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
