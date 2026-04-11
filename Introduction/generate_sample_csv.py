"""
Generate sample CSV files for use cases
This script creates sample data files for both use_case_1 and use_case_2
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sales_data(output_path='use_case_1/sales_data.csv'):
    """Generate sample sales data for use case 1"""
    print(f"Generating sales data...")
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = {
        'Date': dates,
        'Product': np.random.choice(['Widget A', 'Widget B', 'Widget C'], 100),
        'Quantity': np.random.randint(1, 50, 100),
        'Price': np.random.uniform(10, 100, 100).round(2),
    }
    df = pd.DataFrame(data)
    df['Total'] = df['Quantity'] * df['Price']
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path} ({len(df)} rows)")
    return df


def generate_sample_data(output_path='use_case_2/sample_data.csv'):
    """Generate sample data for use case 2"""
    print(f"Generating sample data for analysis...")
    
    data = {
        'Name': [f'Customer {i}' for i in range(200)],
        'Age': np.random.randint(18, 80, 200),
        'Score': np.random.uniform(0, 100, 200).round(2),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 200),
    }
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path} ({len(df)} rows)")
    return df


def main():
    """Main function to generate all sample CSV files"""
    print("=" * 60)
    print("Sample CSV File Generator")
    print("=" * 60)
    
    try:
        # Generate both files
        generate_sales_data()
        generate_sample_data()
        
        print("=" * 60)
        print("Success! All sample CSV files have been generated.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
