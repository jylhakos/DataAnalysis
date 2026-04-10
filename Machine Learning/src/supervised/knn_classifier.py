"""
k-Nearest Neighbors (k-NN) Classifier Example

k-NN is a simple, non-parametric classification algorithm that classifies
new instances based on the majority class among the k closest training examples.

Algorithm:
1. Calculate distance between new point and all training points
2. Select k nearest neighbors
3. Assign the most common class among those neighbors

Distance Metric (Euclidean): d(x, x') = sqrt(Σ(xi - xi')²)
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import load_iris_dataset, split_data
from utils.preprocessing import standardize_features, encode_labels
from utils.visualization import plot_confusion_matrix, plot_decision_boundary_2d


def train_knn_classifier(X_train, y_train, n_neighbors=5):
    """
    Train k-NN classifier
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        n_neighbors (int): Number of neighbors to use
        
    Returns:
        KNeighborsClassifier: Trained model
    """
    print(f"\nTraining k-NN classifier with k={n_neighbors}...")
    
    # Create and train the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X_train, y_train)
    
    print(f"Model trained successfully!")
    
    return knn


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained classifier
        X_test (array): Test features
        y_test (array): Test labels
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\nClassification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    return accuracy, y_pred


def test_different_k_values(X_train, X_test, y_train, y_test):
    """
    Test different values of k and compare results
    
    Args:
        X_train, X_test, y_train, y_test: Train/test split data
    """
    print("\n" + "="*50)
    print("TESTING DIFFERENT K VALUES")
    print("="*50)
    
    k_values = [1, 3, 5, 7, 9, 11, 15]
    results = {}
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[f'k={k}'] = accuracy
        print(f"k={k:2d} -> Accuracy: {accuracy:.4f}")
    
    # Find best k
    best_k = max(results, key=results.get)
    print(f"\nBest k: {best_k} with accuracy: {results[best_k]:.4f}")
    
    return results


def demonstrate_decision_boundary(X_train, y_train):
    """
    Demonstrate decision boundary using only 2 features
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
    """
    print("\n" + "="*50)
    print("DECISION BOUNDARY VISUALIZATION")
    print("="*50)
    print("Using only first 2 features (sepal length and width)")
    
    # Use only first 2 features for visualization
    X_train_2d = X_train[:, :2]
    
    # Train model on 2D data
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_2d, y_train)
    
    # Plot decision boundary
    plot_decision_boundary_2d(
        X_train_2d, y_train, knn,
        feature_names=['Sepal Length', 'Sepal Width']
    )


def main():
    """
    Main function to demonstrate k-NN classifier
    """
    print("="*50)
    print(" k-NEAREST NEIGHBORS (k-NN) CLASSIFIER ")
    print("="*50)
    
    # Load the Iris dataset
    X, y = load_iris_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)
    
    # Standardize features (important for distance-based algorithms)
    print("\nStandardizing features...")
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    
    # Train k-NN classifier with k=5
    knn_model = train_knn_classifier(X_train_scaled, y_train, n_neighbors=5)
    
    # Evaluate the model
    accuracy, predictions = evaluate_model(knn_model, X_test_scaled, y_test)
    
    # Test different k values
    results = test_different_k_values(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Demonstrate decision boundary with 2D data
    demonstrate_decision_boundary(X_train, y_train)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"✓ Dataset: Iris (150 samples, 4 features, 3 classes)")
    print(f"✓ Algorithm: k-Nearest Neighbors")
    print(f"✓ Best Model Accuracy: {accuracy:.4f}")
    print(f"✓ Features were standardized for better distance calculations")
    print("="*50)


if __name__ == "__main__":
    main()
