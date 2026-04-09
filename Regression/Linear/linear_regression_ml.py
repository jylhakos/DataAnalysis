"""
Linear Regression Example - Machine Learning Approach

This script demonstrates linear regression using scikit-learn for predicting
house prices based on square footage.

Mathematical Formula:
y = β₀ + β₁x + ε

Where:
- y is the predicted value (house price)
- x is the feature (square footage)
- β₀ is the intercept
- β₁ is the slope coefficient
- ε is the error term
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def main():
    # Generate sample data: House prices based on square footage
    np.random.seed(42)
    
    # Square footage (independent variable)
    X = np.array([[1000], [1200], [1500], [1800], [2000], 
                  [2200], [2500], [2800], [3000], [3200]])
    
    # House prices in dollars (dependent variable)
    # Price roughly follows: 100 * sqft + 100000 with some noise
    y = np.array([200000, 220000, 250000, 280000, 300000,
                  320000, 350000, 380000, 400000, 420000])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=" * 50)
    print("Linear Regression Model Results")
    print("=" * 50)
    print(f"Intercept (β₀): ${model.intercept_:,.2f}")
    print(f"Coefficient (β₁): ${model.coef_[0]:,.2f} per sqft")
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print("=" * 50)
    
    # Make a prediction for a new house
    new_sqft = np.array([[1800]])
    predicted_price = model.predict(new_sqft)
    print(f"\nPrediction for {new_sqft[0][0]} sqft house:")
    print(f"Predicted Price: ${predicted_price[0]:,.2f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.6)
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Square Footage')
    plt.ylabel('Price ($)')
    plt.title('Linear Regression: House Price vs Square Footage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('linear_regression_plot.png', dpi=300)
    print("\nPlot saved as 'linear_regression_plot.png'")


if __name__ == "__main__":
    main()
