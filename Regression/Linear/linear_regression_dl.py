"""
Linear Regression Example - Deep Learning Approach

This script demonstrates linear regression using TensorFlow/Keras,
showing how it can be viewed as a single-layer neural network.

Mathematical Formula:
y = Wx + b

Where:
- W is the weight matrix (equivalent to β₁ in classical linear regression)
- b is the bias (equivalent to β₀ in classical linear regression)
- Optimized using Mean Squared Error (MSE) loss
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def main():
    # Generate sample data
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Square footage (independent variable)
    X = np.array([[1000], [1200], [1500], [1800], [2000], 
                  [2200], [2500], [2800], [3000], [3200]], dtype=np.float32)
    
    # House prices in dollars (dependent variable)
    y = np.array([[200000], [220000], [250000], [280000], [300000],
                  [320000], [350000], [380000], [400000], [420000]], dtype=np.float32)
    
    # Feature scaling (important for neural networks)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Build the model: Single Dense layer (no activation = linear activation)
    model = keras.Sequential([
        keras.layers.Dense(1, input_shape=(1,), activation=None, name='linear_layer')
    ])
    
    # Compile with MSE loss (standard for regression)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae']
    )
    
    print("=" * 50)
    print("Linear Regression - Deep Learning Model")
    print("=" * 50)
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=2,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest MSE Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Get the learned weights and bias
    weights = model.layers[0].get_weights()
    W = weights[0][0][0]
    b = weights[1][0]
    
    print(f"\nLearned Parameters:")
    print(f"Weight (W): {W:.4f}")
    print(f"Bias (b): {b:.4f}")
    
    # Make a prediction
    new_sqft = np.array([[1800]], dtype=np.float32)
    new_sqft_scaled = scaler_X.transform(new_sqft)
    prediction_scaled = model.predict(new_sqft_scaled, verbose=0)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    print(f"\nPrediction for {new_sqft[0][0]:.0f} sqft house:")
    print(f"Predicted Price: ${prediction[0][0]:,.2f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.6)
    plt.plot(X, y_pred, color='red', linewidth=2, label='Model Prediction')
    plt.xlabel('Square Footage')
    plt.ylabel('Price ($)')
    plt.title('Deep Learning Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_dl_plot.png', dpi=300)
    print("\nPlot saved as 'linear_regression_dl_plot.png'")


if __name__ == "__main__":
    main()
