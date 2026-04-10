"""
Visualization utilities for Machine Learning
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_dataset_distribution(X, y, feature_names=None):
    """
    Visualize dataset distribution with pair plots
    
    Args:
        X (array): Features
        y (array): Labels
        feature_names (list): Names of features
    """
    import pandas as pd
    
    if feature_names is None:
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    
    # Pair plot
    sns.pairplot(df, hue='Target', diag_kind='hist', height=2.5)
    plt.suptitle('Dataset Distribution', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): Names of classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = np.unique(y_true)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_dict):
    """
    Compare multiple models' performance
    
    Args:
        results_dict (dict): Dictionary with model names as keys and scores as values
    """
    models = list(results_dict.keys())
    scores = list(results_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(train_scores, val_scores, title='Learning Curve'):
    """
    Plot learning curves showing training and validation performance
    
    Args:
        train_scores (array): Training scores over epochs/iterations
        val_scores (array): Validation scores over epochs/iterations
        title (str): Plot title
    """
    epochs = range(1, len(train_scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, 'b-o', label='Training Score', linewidth=2)
    plt.plot(epochs, val_scores, 'r-s', label='Validation Score', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances):
    """
    Plot feature importance
    
    Args:
        feature_names (list): Names of features
        importances (array): Importance scores
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], color='teal', alpha=0.7)
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('Feature Importance', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary_2d(X, y, model, feature_names=None, resolution=0.02):
    """
    Plot decision boundary for 2D features
    
    Args:
        X (array): 2D features
        y (array): Labels
        model: Trained classifier
        feature_names (list): Names of two features
        resolution (float): Grid resolution
    """
    if X.shape[1] != 2:
        raise ValueError("This function only works with 2D features")
    
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.cm.RdYlBu
    
    # Plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=colors[idx],
                   marker=markers[idx], label=cl, edgecolor='black')
    
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.title('Decision Boundary', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test visualization functions
    from data_loader import load_iris_dataset
    
    X, y = load_iris_dataset()
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Plot dataset distribution
    print("Plotting dataset distribution...")
    plot_dataset_distribution(X, y, feature_names)
