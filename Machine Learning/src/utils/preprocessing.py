"""
Data preprocessing utilities for Machine Learning
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def standardize_features(X_train, X_test=None):
    """
    Standardize features by removing the mean and scaling to unit variance
    
    Formula: z = (x - μ) / σ
    
    Args:
        X_train (array): Training features
        X_test (array, optional): Test features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler) or (X_train_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


def normalize_features(X_train, X_test=None):
    """
    Normalize features to [0, 1] range
    
    Formula: x_norm = (x - x_min) / (x_max - x_min)
    
    Args:
        X_train (array): Training features
        X_test (array, optional): Test features
        
    Returns:
        tuple: (X_train_norm, X_test_norm, scaler) or (X_train_norm, scaler)
    """
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)
        return X_train_norm, X_test_norm, scaler
    
    return X_train_norm, scaler


def encode_labels(y_train, y_test=None):
    """
    Encode categorical labels to numeric values
    
    Args:
        y_train (array): Training labels
        y_test (array, optional): Test labels
        
    Returns:
        tuple: (y_train_encoded, y_test_encoded, encoder) or (y_train_encoded, encoder)
    """
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    
    if y_test is not None:
        y_test_encoded = encoder.transform(y_test)
        return y_train_encoded, y_test_encoded, encoder
    
    return y_train_encoded, encoder


def add_polynomial_features(X, degree=2):
    """
    Generate polynomial features
    
    Example: [a, b] with degree=2 -> [1, a, b, a², ab, b²]
    
    Args:
        X (array): Input features
        degree (int): Polynomial degree
        
    Returns:
        array: Polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    print(f"Original features: {X.shape[1]}")
    print(f"Polynomial features (degree={degree}): {X_poly.shape[1]}")
    
    return X_poly, poly


if __name__ == "__main__":
    # Test preprocessing functions
    from data_loader import load_iris_dataset, split_data
    
    X, y = load_iris_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Test standardization
    X_train_std, X_test_std, scaler = standardize_features(X_train, X_test)
    print(f"\nStandardized features - Mean: {X_train_std.mean():.4f}, Std: {X_train_std.std():.4f}")
    
    # Test normalization
    X_train_norm, X_test_norm, norm_scaler = normalize_features(X_train, X_test)
    print(f"Normalized features - Min: {X_train_norm.min():.4f}, Max: {X_train_norm.max():.4f}")
    
    # Test label encoding
    y_train_enc, y_test_enc, encoder = encode_labels(y_train, y_test)
    print(f"\nOriginal labels: {y_train[:5]}")
    print(f"Encoded labels: {y_train_enc[:5]}")
    print(f"Classes: {encoder.classes_}")
