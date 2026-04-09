"""
Multiple Linear Regression Example - Machine Learning Approach

This script demonstrates multiple linear regression using scikit-learn for 
predicting RAM performance based on multiple features.

Mathematical Formula:
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Where:
- y is the predicted value (RAM performance score)
- x₁, x₂, ..., xₙ are the features
- β₀ is the intercept
- β₁, β₂, ..., βₙ are the coefficients for each feature
- ε is the error term
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def main():
    # Generate sample data: RAM performance based on multiple characteristics
    np.random.seed(42)
    
    # Features: Size (GB), Frequency (MHz), Bandwidth (GB/s), Voltage (V), Latency (CL)
    data = {
        'Size_GB': [8, 16, 32, 16, 32, 64, 8, 16, 32, 64, 16, 32, 64, 128],
        'Frequency_MHz': [2400, 2666, 3200, 2933, 3600, 3200, 2133, 2400, 2933, 3200, 3000, 3600, 4000, 3200],
        'Bandwidth_GBs': [19.2, 21.3, 25.6, 23.5, 28.8, 25.6, 17.0, 19.2, 23.5, 25.6, 24.0, 28.8, 32.0, 25.6],
        'Voltage_V': [1.2, 1.2, 1.35, 1.2, 1.35, 1.2, 1.2, 1.2, 1.2, 1.35, 1.2, 1.35, 1.35, 1.2],
        'Latency_CL': [16, 16, 16, 14, 18, 14, 15, 16, 15, 16, 15, 16, 19, 14],
        # Performance score (synthetic metric: higher is better)
        'Performance_Score': [65, 78, 92, 85, 98, 105, 55, 70, 88, 108, 82, 102, 115, 112]
    }
    
    df = pd.DataFrame(data)
    
    # Separate features and target
    X = df.drop('Performance_Score', axis=1)
    y = df['Performance_Score']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling (optional for linear regression, but good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the multiple linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=" * 70)
    print("Multiple Linear Regression Model Results")
    print("=" * 70)
    print(f"Intercept (β₀): {model.intercept_:.4f}")
    print("\nCoefficients (β₁, β₂, ..., βₙ):")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature:20s}: {coef:8.4f}")
    
    print("\nModel Performance Metrics:")
    print(f"  Mean Squared Error:  {mse:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  R² Score:            {r2:.4f}")
    print("=" * 70)
    
    # Feature importance (absolute value of standardized coefficients)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nFeature Importance (sorted by absolute coefficient):")
    print(feature_importance.to_string(index=False))
    
    # Make a prediction for new RAM configuration
    new_ram = pd.DataFrame({
        'Size_GB': [32],
        'Frequency_MHz': [3200],
        'Bandwidth_GBs': [25.6],
        'Voltage_V': [1.2],
        'Latency_CL': [15]
    })
    
    new_ram_scaled = scaler.transform(new_ram)
    predicted_performance = model.predict(new_ram_scaled)
    
    print("\n" + "=" * 70)
    print("Prediction for New RAM Configuration:")
    print("=" * 70)
    for col in new_ram.columns:
        print(f"  {col:20s}: {new_ram[col].values[0]}")
    print(f"\n  Predicted Performance Score: {predicted_performance[0]:.2f}")
    print("=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual Performance Score')
    axes[0, 0].set_ylabel('Predicted Performance Score')
    axes[0, 0].set_title('Actual vs Predicted Performance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Performance Score')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Feature Coefficients
    axes[1, 0].barh(feature_importance['Feature'], 
                     feature_importance['Coefficient'])
    axes[1, 0].set_xlabel('Coefficient Value')
    axes[1, 0].set_title('Feature Coefficients')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance
    axes[1, 1].barh(feature_importance['Feature'], 
                     feature_importance['Abs_Coefficient'],
                     color='orange')
    axes[1, 1].set_xlabel('Absolute Coefficient Value')
    axes[1, 1].set_title('Feature Importance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiple_regression_plot.png', dpi=300)
    print("\nPlot saved as 'multiple_regression_plot.png'")


if __name__ == "__main__":
    main()
