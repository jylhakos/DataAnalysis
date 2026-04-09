"""
Multiple Linear Regression Example - Deep Learning Approach

This script demonstrates multiple linear regression using TensorFlow/Keras,
showing how it maps to a single-layer neural network with multiple inputs.

Mathematical Formula:
y = W₁x₁ + W₂x₂ + ... + Wₙxₙ + b

Matrix form:
y = XW + b

Where:
- X is the input feature matrix (n_samples × n_features)
- W is the weight vector (n_features × 1)
- b is the bias term
- Optimized using Mean Squared Error (MSE) loss
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Generate sample data: RAM performance
    data = {
        'Size_GB': [8, 16, 32, 16, 32, 64, 8, 16, 32, 64, 16, 32, 64, 128],
        'Frequency_MHz': [2400, 2666, 3200, 2933, 3600, 3200, 2133, 2400, 2933, 3200, 3000, 3600, 4000, 3200],
        'Bandwidth_GBs': [19.2, 21.3, 25.6, 23.5, 28.8, 25.6, 17.0, 19.2, 23.5, 25.6, 24.0, 28.8, 32.0, 25.6],
        'Voltage_V': [1.2, 1.2, 1.35, 1.2, 1.35, 1.2, 1.2, 1.2, 1.2, 1.35, 1.2, 1.35, 1.35, 1.2],
        'Latency_CL': [16, 16, 16, 14, 18, 14, 15, 16, 15, 16, 15, 16, 19, 14],
        'Performance_Score': [65, 78, 92, 85, 98, 105, 55, 70, 88, 108, 82, 102, 115, 112]
    }
    
    df = pd.DataFrame(data)
    
    # Separate features and target
    X = df.drop('Performance_Score', axis=1).values.astype(np.float32)
    y = df['Performance_Score'].values.astype(np.float32).reshape(-1, 1)
    
    # Feature names for later use
    feature_names = df.drop('Performance_Score', axis=1).columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling (critical for neural networks)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Build the model: Single Dense layer with 5 inputs, 1 output, no activation
    model = keras.Sequential([
        keras.layers.Dense(
            1, 
            input_shape=(5,),  # 5 features
            activation=None,   # Linear activation
            name='multiple_linear_layer'
        )
    ])
    
    # Compile with MSE loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae']
    )
    
    print("=" * 70)
    print("Multiple Linear Regression - Deep Learning Model")
    print("=" * 70)
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=200,
        batch_size=4,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
    
    # Make predictions (scaled)
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    
    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics on original scale
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  Test MSE Loss (scaled): {test_loss:.4f}")
    print(f"  Test MAE (scaled):      {test_mae:.4f}")
    print(f"  MSE (original scale):   {mse:.4f}")
    print(f"  R² Score:               {r2:.4f}")
    
    # Get the learned weights and bias
    weights, bias = model.layers[0].get_weights()
    
    print(f"\nLearned Parameters:")
    print(f"  Bias (b): {bias[0]:.4f}")
    print("\n  Weights (W₁, W₂, ..., Wₙ):")
    for feature, weight in zip(feature_names, weights.flatten()):
        print(f"    {feature:20s}: {weight:8.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Weight': weights.flatten(),
        'Abs_Weight': np.abs(weights.flatten())
    }).sort_values('Abs_Weight', ascending=False)
    
    print("\n  Feature Importance (sorted by absolute weight):")
    for _, row in feature_importance.iterrows():
        print(f"    {row['Feature']:20s}: {row['Abs_Weight']:8.4f}")
    
    # Make a prediction for new RAM configuration
    new_ram = np.array([[32, 3200, 25.6, 1.2, 15]], dtype=np.float32)
    new_ram_scaled = scaler_X.transform(new_ram)
    prediction_scaled = model.predict(new_ram_scaled, verbose=0)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    print("\n" + "=" * 70)
    print("Prediction for New RAM Configuration:")
    print("=" * 70)
    for feature, value in zip(feature_names, new_ram[0]):
        print(f"  {feature:20s}: {value}")
    print(f"\n  Predicted Performance Score: {prediction[0][0]:.2f}")
    print("=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training History
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted
    axes[0, 1].scatter(y_test, y_pred, alpha=0.6, color='blue')
    axes[0, 1].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', linewidth=2)
    axes[0, 1].set_xlabel('Actual Performance Score')
    axes[0, 1].set_ylabel('Predicted Performance Score')
    axes[0, 1].set_title('Actual vs Predicted Performance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Weights
    axes[1, 0].barh(feature_importance['Feature'], 
                     feature_importance['Weight'])
    axes[1, 0].set_xlabel('Weight Value')
    axes[1, 0].set_title('Learned Weights for Each Feature')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance
    axes[1, 1].barh(feature_importance['Feature'], 
                     feature_importance['Abs_Weight'],
                     color='orange')
    axes[1, 1].set_xlabel('Absolute Weight Value')
    axes[1, 1].set_title('Feature Importance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiple_regression_dl_plot.png', dpi=300)
    print("\nPlot saved as 'multiple_regression_dl_plot.png'")


if __name__ == "__main__":
    main()
