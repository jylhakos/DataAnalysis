"""
Logistic Regression for Classification

Logistic regression models the probability of a class using the logistic function:
P(y=1|x) = 1 / (1 + e^(-z)) where z = w^T x + b

For multi-class: Uses one-vs-rest (OvR) or softmax (multinomial)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import load_iris_dataset, split_data
from utils.visualization import plot_confusion_matrix


def train_logistic_regression(X_train, y_train, max_iter=1000, random_state=42):
    """
    Train logistic regression classifier
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        max_iter (int): Maximum iterations
        random_state (int): Random seed
        
    Returns:
        LogisticRegression: Trained model
    """
    print("\nTraining Logistic Regression...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    # Train model
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    
    print("✓ Training complete")
    
    return model


def evaluate_logistic_regression(model, X_test, y_test, class_names=None):
    """
    Evaluate logistic regression model
    
    Args:
        model (LogisticRegression): Trained model
        X_test (array): Test features
        y_test (array): True labels
        class_names (list): Names of classes
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_proba,
        'confusion_matrix': cm
    }


def plot_decision_probabilities(model, X_test, y_test, class_names):
    """
    Plot prediction probabilities
    
    Args:
        model (LogisticRegression): Trained model
        X_test (array): Test features
        y_test (array): True labels
        class_names (list): Names of classes
    """
    y_proba = model.predict_proba(X_test)
    
    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 4))
    
    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Get probabilities for this class
        probs = y_proba[:, i]
        
        # Separate by true class
        for j, true_class in enumerate(class_names):
            mask = (y_test == true_class)
            ax.hist(probs[mask], bins=20, alpha=0.5, label=f'True: {true_class}')
        
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'P({class_name}|X)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_coefficients(model, feature_names, class_names):
    """
    Plot model coefficients
    
    Args:
        model (LogisticRegression): Trained model
        feature_names (list): Names of features
        class_names (list): Names of classes
    """
    coefficients = model.coef_
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(feature_names))
    width = 0.25
    
    for i, class_name in enumerate(class_names):
        offset = (i - len(class_names)/2 + 0.5) * width
        ax.bar(x + offset, coefficients[i], width, label=class_name, alpha=0.8)
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Logistic Regression Coefficients', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_regularization(X_train, y_train, X_test, y_test):
    """
    Compare different regularization strengths
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_test (array): Test features
        y_test (array): Test labels
    """
    print("\n" + "="*50)
    print("REGULARIZATION COMPARISON")
    print("="*50)
    
    # Different C values (inverse of regularization strength)
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    train_scores = []
    test_scores = []
    
    for C in C_values:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        train_scores.append(train_acc)
        test_scores.append(test_acc)
        
        print(f"C={C:7.3f} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, train_scores, 'o-', label='Training Accuracy', linewidth=2, markersize=8)
    plt.semilogx(C_values, test_scores, 's-', label='Test Accuracy', linewidth=2, markersize=8)
    plt.xlabel('C (Inverse Regularization Strength)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Effect of Regularization on Model Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Find optimal C
    optimal_idx = np.argmax(test_scores)
    optimal_C = C_values[optimal_idx]
    print(f"\nOptimal C: {optimal_C} (Test Accuracy: {test_scores[optimal_idx]:.4f})")


def demonstrate_multiclass_strategies(X_train, y_train, X_test, y_test):
    """
    Compare one-vs-rest and multinomial strategies
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_test (array): Test features
        y_test (array): Test labels
    """
    print("\n" + "="*50)
    print("MULTICLASS STRATEGY COMPARISON")
    print("="*50)
    
    strategies = {
        'One-vs-Rest (OvR)': 'ovr',
        'Multinomial (Softmax)': 'multinomial'
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\n{name}:")
        model = LogisticRegression(multi_class=strategy, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        results[name] = {'train': train_acc, 'test': test_acc}
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies_list = list(results.keys())
    x = np.arange(len(strategies_list))
    width = 0.35
    
    train_accs = [results[s]['train'] for s in strategies_list]
    test_accs = [results[s]['test'] for s in strategies_list]
    
    ax.bar(x - width/2, train_accs, width, label='Training', alpha=0.8)
    ax.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Multiclass Strategy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies_list)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to demonstrate logistic regression
    """
    print("="*50)
    print(" LOGISTIC REGRESSION CLASSIFICATION ")
    print("="*50)
    
    # Load dataset
    X, y = load_iris_dataset()
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    class_names = ['setosa', 'versicolor', 'virginica']
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.3, random_state=42)
    
    # Train model
    model = train_logistic_regression(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_logistic_regression(model, X_test, y_test, class_names)
    
    # Plot decision probabilities
    plot_decision_probabilities(model, X_test, y_test, class_names)
    
    # Plot coefficients
    plot_coefficients(model, feature_names, class_names)
    
    # Compare regularization strengths
    compare_regularization(X_train, y_train, X_test, y_test)
    
    # Compare multiclass strategies
    demonstrate_multiclass_strategies(X_train, y_train, X_test, y_test)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"✓ Algorithm: Logistic Regression")
    print(f"✓ Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"✓ Classes: {len(class_names)}")
    print(f"✓ Multiclass strategy: One-vs-Rest (OvR)")
    print(f"✓ Model successfully trained and evaluated")
    print("="*50)


if __name__ == "__main__":
    main()
