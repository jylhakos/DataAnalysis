"""
Data loader utility for Machine Learning examples
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_iris_dataset(filepath='data/raw/iris.csv'):
    """
    Load the Iris dataset from CSV file
    
    Args:
        filepath (str): Path to the iris.csv file
        
    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    full_path = project_root / filepath
    
    # Load CSV
    df = pd.read_csv(full_path)
    
    # Separate features (X) and target (y)
    X = df.drop('species', axis=1).values
    y = df['species'].values
    
    print(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets
    
    Args:
        X (array): Features
        y (array): Labels
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the data loader
    X, y = load_iris_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
