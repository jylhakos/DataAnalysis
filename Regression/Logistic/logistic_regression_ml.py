"""
Logistic Regression Example - Machine Learning Approach

This script demonstrates logistic regression using scikit-learn for binary
classification (e.g., tumor diagnosis based on size).

Mathematical Formula:
P(y=1|x) = σ(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)

Where:
- σ(z) = 1 / (1 + e^(-z)) is the sigmoid function
- P(y=1|x) is the probability of the positive class
- β₀, β₁, ..., βₙ are the model parameters
- x₁, x₂, ..., xₙ are the input features
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt


def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))


def main():
    # Generate sample data: Tumor diagnosis based on size
    np.random.seed(42)
    
    # Tumor sizes (in cm)
    X = np.array([3.78, 2.44, 2.09, 0.14, 4.92, 5.88, 1.50, 3.20,
                  4.10, 5.20, 2.80, 1.90, 3.50, 4.50, 2.30, 5.60,
                  1.20, 3.90, 4.80, 2.60]).reshape(-1, 1)
    
    # Binary target: 0 = Benign, 1 = Malignant
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Feature scaling (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the logistic regression model
    # penalty='l2': Ridge regularization (default)
    # solver='lbfgs': Optimizer suitable for small to medium datasets
    # max_iter: Maximum iterations for convergence
    model = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("=" * 70)
    print("Logistic Regression Model Results")
    print("=" * 70)
    print(f"Intercept (β₀): {model.intercept_[0]:.4f}")
    print(f"Coefficient (β₁): {model.coef_[0][0]:.4f}")
    
    print("\nModel Performance Metrics:")
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
    new_tumor_size = np.array([[3.46]])
    new_tumor_scaled = scaler.transform(new_tumor_size)
    predicted_class = model.predict(new_tumor_scaled)[0]
    predicted_proba = model.predict_proba(new_tumor_scaled)[0]
    
    print("=" * 70)
    print("Prediction for New Tumor (size: 3.46 cm):")
    print("=" * 70)
    print(f"  Predicted Class: {'Malignant' if predicted_class == 1 else 'Benign'}")
    print(f"  Probability [Benign, Malignant]: [{predicted_proba[0]:.4f}, {predicted_proba[1]:.4f}]")
    print("=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Decision Boundary
    X_range = np.linspace(X.min() - 1, X.max() + 1, 300).reshape(-1, 1)
    X_range_scaled = scaler.transform(X_range)
    y_prob = model.predict_proba(X_range_scaled)[:, 1]
    
    axes[0, 0].scatter(X_train, y_train, color='blue', alpha=0.6, 
                       label='Training Data', s=100, edgecolors='k')
    axes[0, 0].scatter(X_test, y_test, color='red', alpha=0.6, 
                       label='Test Data', s=100, marker='^', edgecolors='k')
    axes[0, 0].plot(X_range, y_prob, color='green', linewidth=2, 
                    label='Sigmoid Curve')
    axes[0, 0].axhline(y=0.5, color='black', linestyle='--', linewidth=1, 
                       label='Decision Boundary (p=0.5)')
    axes[0, 0].set_xlabel('Tumor Size (cm)')
    axes[0, 0].set_ylabel('Probability of Malignancy')
    axes[0, 0].set_title('Logistic Regression: Tumor Classification')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: ROC Curve
    if len(np.unique(y_test)) > 1:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color='darkorange', linewidth=2,
                        label=f'ROC Curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', linewidth=2, 
                        linestyle='--', label='Random Classifier')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('Receiver Operating Characteristic (ROC) Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix Heatmap
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('Confusion Matrix')
    plt.colorbar(im, ax=axes[1, 0])
    tick_marks = np.arange(2)
    axes[1, 0].set_xticks(tick_marks)
    axes[1, 0].set_yticks(tick_marks)
    axes[1, 0].set_xticklabels(['Benign', 'Malignant'])
    axes[1, 0].set_yticklabels(['Benign', 'Malignant'])
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, str(cm[i, j]),
                           ha="center", va="center", color="black", fontsize=20)
    
    # Plot 4: Sigmoid Function Illustration
    z = np.linspace(-6, 6, 100)
    sigmoid_values = sigmoid(z)
    
    axes[1, 1].plot(z, sigmoid_values, color='purple', linewidth=2)
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', linewidth=1)
    axes[1, 1].axvline(x=0, color='gray', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('z = β₀ + β₁x')
    axes[1, 1].set_ylabel('σ(z) = 1 / (1 + e^(-z))')
    axes[1, 1].set_title('Sigmoid Activation Function')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0, 0.5, '  (0, 0.5)', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('logistic_regression_plot.png', dpi=300)
    print("\nPlot saved as 'logistic_regression_plot.png'")


if __name__ == "__main__":
    main()
