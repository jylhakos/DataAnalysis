"""
Logistic Regression Example - Deep Learning Approach

This script demonstrates logistic regression using TensorFlow/Keras,
showing how it can be viewed as a single-layer neural network with 
sigmoid activation.

Mathematical Formula:
P(y=1|x) = σ(Wx + b)

Where:
- σ is the sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
- W is the weight matrix
- b is the bias term
- Optimized using Binary Cross-Entropy (BCE) loss
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Generate sample data: Tumor diagnosis based on size
    X = np.array([3.78, 2.44, 2.09, 0.14, 4.92, 5.88, 1.50, 3.20,
                  4.10, 5.20, 2.80, 1.90, 3.50, 4.50, 2.30, 5.60,
                  1.20, 3.90, 4.80, 2.60], dtype=np.float32).reshape(-1, 1)
    
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0], 
                 dtype=np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build the model: Single Dense layer with sigmoid activation
    model = keras.Sequential([
        keras.layers.Dense(
            1,
            input_shape=(1,),
            activation='sigmoid',  # Sigmoid activation for binary classification
            name='logistic_layer'
        )
    ])
    
    # Compile with Binary Cross-Entropy loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',  # Standard loss for binary classification
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    print("=" * 70)
    print("Logistic Regression - Deep Learning Model")
    print("=" * 70)
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=200,
        batch_size=4,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate the model
    test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"\nTest Results:")
    print(f"  Test Loss:      {test_results[0]:.4f}")
    print(f"  Test Accuracy:  {test_results[1]:.4f}")
    print(f"  Test Precision: {test_results[2]:.4f}")
    print(f"  Test Recall:    {test_results[3]:.4f}")
    
    # Get the learned weights and bias
    weights, bias = model.layers[0].get_weights()
    
    print(f"\nLearned Parameters:")
    print(f"  Weight (W): {weights[0][0]:.4f}")
    print(f"  Bias (b):   {bias[0]:.4f}")
    
    # Make predictions
    y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\nDetailed Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Benign', 'Malignant'],
                                zero_division=0))
    
    # Make a prediction for a new tumor
    new_tumor_size = np.array([[3.46]], dtype=np.float32)
    new_tumor_scaled = scaler.transform(new_tumor_size)
    predicted_proba = model.predict(new_tumor_scaled, verbose=0)[0][0]
    predicted_class = int(predicted_proba > 0.5)
    
    print("=" * 70)
    print("Prediction for New Tumor (size: 3.46 cm):")
    print("=" * 70)
    print(f"  Predicted Class: {'Malignant' if predicted_class == 1 else 'Benign'}")
    print(f"  Probability of Malignancy: {predicted_proba:.4f}")
    print(f"  Probability of Benign:     {1 - predicted_proba:.4f}")
    print("=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training History - Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Binary Cross-Entropy Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Decision Boundary
    X_range = np.linspace(X.min() - 1, X.max() + 1, 300).reshape(-1, 1)
    X_range_scaled = scaler.transform(X_range)
    y_prob = model.predict(X_range_scaled, verbose=0).flatten()
    
    axes[0, 1].scatter(X_train, y_train, color='blue', alpha=0.6, 
                       label='Training Data', s=100, edgecolors='k')
    axes[0, 1].scatter(X_test, y_test, color='red', alpha=0.6, 
                       label='Test Data', s=100, marker='^', edgecolors='k')
    axes[0, 1].plot(X_range, y_prob, color='green', linewidth=2, 
                    label='Sigmoid Curve (DL)')
    axes[0, 1].axhline(y=0.5, color='black', linestyle='--', linewidth=1, 
                       label='Decision Boundary')
    axes[0, 1].set_xlabel('Tumor Size (cm)')
    axes[0, 1].set_ylabel('Probability of Malignancy')
    axes[0, 1].set_title('Deep Learning Logistic Regression')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training History - Accuracy
    axes[1, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Training and Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix Heatmap
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix')
    plt.colorbar(im, ax=axes[1, 1])
    tick_marks = np.arange(2)
    axes[1, 1].set_xticks(tick_marks)
    axes[1, 1].set_yticks(tick_marks)
    axes[1, 1].set_xticklabels(['Benign', 'Malignant'])
    axes[1, 1].set_yticklabels(['Benign', 'Malignant'])
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, str(cm[i, j]),
                           ha="center", va="center", color="black", fontsize=20)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_dl_plot.png', dpi=300)
    print("\nPlot saved as 'logistic_regression_dl_plot.png'")


if __name__ == "__main__":
    main()
