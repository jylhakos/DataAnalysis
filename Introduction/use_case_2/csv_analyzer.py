"""
CSV Analyzer - Advanced data analysis tool for Use Case 2
Performs comprehensive exploratory data analysis and generates visualizations.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from pathlib import Path


class CSVAnalyzer:
    """
    Advanced CSV data analyzer with EDA capabilities.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the analyzer with a CSV file.
        
        Args:
            file_path: Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
    def load_data(self) -> bool:
        """
        Load the CSV file into a pandas DataFrame.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            self._identify_column_types()
            return True
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return False
    
    def _identify_column_types(self):
        """
        Identify and categorize column types.
        """
        self.numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Try to identify datetime columns
        for col in self.categorical_columns:
            try:
                pd.to_datetime(self.df[col])
                self.datetime_columns.append(col)
            except:
                pass
        
        # Remove datetime columns from categorical
        self.categorical_columns = [col for col in self.categorical_columns 
                                   if col not in self.datetime_columns]
    
    def basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Returns:
            Dictionary with basic dataset information
        """
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'datetime_columns': self.datetime_columns
        }
        return info
    
    def data_quality_check(self) -> Dict[str, Any]:
        """
        Check data quality: missing values, duplicates, etc.
        
        Returns:
            Dictionary with data quality metrics
        """
        quality = {
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicate_rows': self.df.duplicated().sum(),
            'unique_counts': {col: self.df[col].nunique() for col in self.df.columns}
        }
        return quality
    
    def statistical_summary(self) -> pd.DataFrame:
        """
        Get statistical summary of numeric columns.
        
        Returns:
            DataFrame with statistical summary
        """
        if len(self.numeric_columns) > 0:
            return self.df[self.numeric_columns].describe()
        return pd.DataFrame()
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns.
        
        Returns:
            Correlation matrix
        """
        if len(self.numeric_columns) > 1:
            return self.df[self.numeric_columns].corr()
        return pd.DataFrame()
    
    def detect_outliers(self, method: str = 'iqr') -> Dict[str, List]:
        """
        Detect outliers in numeric columns.
        
        Args:
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            Dictionary with outlier indices for each column
        """
        outliers = {}
        
        for col in self.numeric_columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = self.df[(self.df[col] < lower_bound) | 
                                         (self.df[col] > upper_bound)].index.tolist()
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outlier_indices = self.df[z_scores > 3].index.tolist()
            else:
                outlier_indices = []
            
            if outlier_indices:
                outliers[col] = outlier_indices
        
        return outliers
    
    def generate_visualizations(self, output_dir: str = 'visualizations'):
        """
        Generate comprehensive visualizations.
        
        Args:
            output_dir: Directory to save visualizations
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\nGenerating visualizations in '{output_dir}/'...")
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Distribution plots for numeric columns
        if len(self.numeric_columns) > 0:
            fig, axes = plt.subplots(
                nrows=(len(self.numeric_columns) + 2) // 3,
                ncols=3,
                figsize=(15, 5 * ((len(self.numeric_columns) + 2) // 3))
            )
            axes = axes.flatten() if len(self.numeric_columns) > 1 else [axes]
            
            for idx, col in enumerate(self.numeric_columns):
                self.df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            
            # Hide empty subplots
            for idx in range(len(self.numeric_columns), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Created: distributions.png")
        
        # 2. Correlation heatmap
        if len(self.numeric_columns) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = self.df[self.numeric_columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Created: correlation_heatmap.png")
        
        # 3. Box plots for outlier detection
        if len(self.numeric_columns) > 0:
            fig, axes = plt.subplots(
                nrows=(len(self.numeric_columns) + 2) // 3,
                ncols=3,
                figsize=(15, 5 * ((len(self.numeric_columns) + 2) // 3))
            )
            axes = axes.flatten() if len(self.numeric_columns) > 1 else [axes]
            
            for idx, col in enumerate(self.numeric_columns):
                self.df.boxplot(column=col, ax=axes[idx])
                axes[idx].set_title(f'Box Plot: {col}')
                axes[idx].set_ylabel(col)
            
            # Hide empty subplots
            for idx in range(len(self.numeric_columns), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/boxplots.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Created: boxplots.png")
        
        # 4. Categorical variable distributions
        if len(self.categorical_columns) > 0:
            fig, axes = plt.subplots(
                nrows=(len(self.categorical_columns) + 1) // 2,
                ncols=2,
                figsize=(15, 5 * ((len(self.categorical_columns) + 1) // 2))
            )
            axes = axes.flatten() if len(self.categorical_columns) > 1 else [axes]
            
            for idx, col in enumerate(self.categorical_columns):
                value_counts = self.df[col].value_counts()
                value_counts.plot(kind='bar', ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Count')
                axes[idx].tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for idx in range(len(self.categorical_columns), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/categorical_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Created: categorical_distributions.png")
        
        print(f"\n✓ All visualizations saved to '{output_dir}/'")
    
    def generate_report(self, output_file: str = 'analysis_report.txt'):
        """
        Generate a comprehensive text report.
        
        Args:
            output_file: Path to save the report
        """
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic Info
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"File: {self.file_path}\n")
            f.write(f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n")
            f.write(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB\n\n")
            
            # Column Information
            f.write("2. COLUMN INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Numeric Columns ({len(self.numeric_columns)}): {', '.join(self.numeric_columns)}\n")
            f.write(f"Categorical Columns ({len(self.categorical_columns)}): {', '.join(self.categorical_columns)}\n")
            f.write(f"Datetime Columns ({len(self.datetime_columns)}): {', '.join(self.datetime_columns)}\n\n")
            
            # Data Quality
            f.write("3. DATA QUALITY CHECK\n")
            f.write("-" * 80 + "\n")
            quality = self.data_quality_check()
            f.write(f"Duplicate Rows: {quality['duplicate_rows']}\n\n")
            
            f.write("Missing Values:\n")
            for col, count in quality['missing_values'].items():
                if count > 0:
                    pct = quality['missing_percentage'][col]
                    f.write(f"  - {col}: {count} ({pct:.2f}%)\n")
            if sum(quality['missing_values'].values()) == 0:
                f.write("  No missing values detected.\n")
            f.write("\n")
            
            # Statistical Summary
            if len(self.numeric_columns) > 0:
                f.write("4. STATISTICAL SUMMARY\n")
                f.write("-" * 80 + "\n")
                f.write(self.statistical_summary().to_string())
                f.write("\n\n")
            
            # Correlation Analysis
            if len(self.numeric_columns) > 1:
                f.write("5. CORRELATION ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(self.correlation_analysis().to_string())
                f.write("\n\n")
            
            # Outliers
            outliers = self.detect_outliers()
            if outliers:
                f.write("6. OUTLIER DETECTION\n")
                f.write("-" * 80 + "\n")
                for col, indices in outliers.items():
                    f.write(f"{col}: {len(indices)} outliers detected\n")
                f.write("\n")
            
            # Unique Values
            f.write("7. UNIQUE VALUE COUNTS\n")
            f.write("-" * 80 + "\n")
            for col in self.df.columns:
                unique_count = self.df[col].nunique()
                total_count = len(self.df)
                f.write(f"{col}: {unique_count} unique values ({unique_count/total_count*100:.2f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✓ Analysis report saved to '{output_file}'")
    
    def run_full_analysis(self, output_dir: str = 'output'):
        """
        Run complete analysis pipeline.
        
        Args:
            output_dir: Directory to save all outputs
        """
        print("\n" + "=" * 80)
        print("RUNNING FULL EXPLORATORY DATA ANALYSIS")
        print("=" * 80 + "\n")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load data
        if not self.load_data():
            return
        
        # Display basic info
        print("\n--- Basic Information ---")
        info = self.basic_info()
        print(f"Shape: {info['shape']}")
        print(f"Numeric columns: {len(info['numeric_columns'])}")
        print(f"Categorical columns: {len(info['categorical_columns'])}")
        
        # Data quality
        print("\n--- Data Quality ---")
        quality = self.data_quality_check()
        print(f"Duplicate rows: {quality['duplicate_rows']}")
        total_missing = sum(quality['missing_values'].values())
        print(f"Total missing values: {total_missing}")
        
        # Generate visualizations
        self.generate_visualizations(output_dir=f'{output_dir}/visualizations')
        
        # Generate report
        self.generate_report(output_file=f'{output_dir}/analysis_report.txt')
        
        print("\n" + "=" * 80)
        print(f"✓ ANALYSIS COMPLETE - Results saved to '{output_dir}/'")
        print("=" * 80 + "\n")


def main():
    """
    Main function with CLI interface.
    """
    parser = argparse.ArgumentParser(
        description='CSV Analyzer - Perform exploratory data analysis on CSV files'
    )
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        default='sample_data.csv',
        help='Path to input CSV file (default: sample_data.csv)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found")
        print(f"\nCreating sample data file...")
        create_sample_data(args.input)
        print(f"✓ Sample data created at '{args.input}'")
    
    # Run analysis
    analyzer = CSVAnalyzer(args.input)
    analyzer.run_full_analysis(output_dir=args.output)


def create_sample_data(file_path: str = 'sample_data.csv'):
    """
    Create sample data for demonstration.
    
    Args:
        file_path: Path to save the sample CSV file
    """
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 500
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples).astype(int),
        'purchase_amount': np.random.exponential(100, n_samples),
        'satisfaction_score': np.random.randint(1, 11, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'customer_type': np.random.choice(['New', 'Returning', 'VIP'], n_samples, p=[0.3, 0.5, 0.2]),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_samples)
    }
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()
