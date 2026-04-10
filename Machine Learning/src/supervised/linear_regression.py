"""
Linear Regression Example

Linear regression models the relationship between a dependent variable and
one or more independent variables using a linear equation.

Formula: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Where:
- y = predicted value
- β₀ = intercept
- βᵢ = coefficients
- xᵢ = input features
- ε = error term
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import load_iris_dataset, split_data
from utils.preprocessing import standardize_features


def train_linear_regression(X_train, y_train):
    """
    Train linear regression model
    
    Args:
        X_train (array): Training features
        y_train (array): Training targets
        
    Returns:
        LinearRegression: Trained model
    """
    print("\nTraining Linear Regression model...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Model trained successfully!")
    print(f"  Coefficients: {model.coef_}")
    print(f"  Intercept: {model.intercept_:.4f}")
    
    return model


def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate regression model
    
    Args:
        model: Trained model
        X_test (array): Test features
        y_test (array): Test targets
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"  (Proportion of variance explained: {r2*100:.2f}%)")
    
    return y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def plot_predictions(y_test, y_pred):
    """
    Plot actual vs predicted values
    
    Args:
        y_test (array): Actual values
        y_pred (array): Predicted values
    """
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals plot
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to demonstrate linear regression
    """
    print("="*50)
    print(" LINEAR REGRESSION EXAMPLE ")
    print("="*50)
    
    # Load data
    X, y = load_iris_dataset()
    
    # For regression, we'll predict petal_length from other features
    # Remove petal_length (column 2) from features and use it as target
    X_features = np.delete(X, 2, axis=1)
    y_target = X[:, 2]  # petal_length
    
    print("\nRegression Task: Predicting Petal Length")
    print(f"  Features: sepal_length, sepal_width, petal_width")
    print(f"  Target: petal_length")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X_features, y_target, test_size=0.3, random_state=42
    )
    
    # Standardize features
    print("\nStandardizing features...")
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    
    # Train model
    model = train_linear_regression(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred, metrics = evaluate_regression_model(model, X_test_scaled, y_test)
    
    # Plot results
    plot_predictions(y_test, y_pred)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"✓ Algorithm: Linear Regression")
    print(f"✓ Dataset: Iris")
    print(f"✓ Task: Predicting Petal Length")
    print(f"✓ R² Score: {metrics['r2']:.4f}")
    print(f"✓ RMSE: {metrics['rmse']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
