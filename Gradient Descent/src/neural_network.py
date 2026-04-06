"""
Neural Network with Gradient Descent

This module implements a simple neural network trained using gradient descent
and backpropagation, demonstrating how gradient descent works in deep learning.

Mathematical concepts:
- Forward propagation: z[l] = W[l]·a[l-1] + b[l], a[l] = g(z[l])
- Backpropagation: compute gradients using chain rule
- Parameter update: W ← W - η∇W, b ← b - η∇b
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable


class Activation:
    """Activation functions and their derivatives."""
    
    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation: σ(z) = 1/(1+e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid: σ'(z) = σ(z)(1-σ(z))"""
        s = Activation.sigmoid(z)
        return s * (1 - s)
    
    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """Tanh activation"""
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative of tanh: tanh'(z) = 1 - tanh²(z)"""
        return 1 - np.tanh(z)**2
    
    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Softmax activation for multi-class classification"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class NeuralNetwork:
    """
    Multi-layer neural network trained with gradient descent.
    
    Architecture: Input → Hidden Layers → Output
    Training: Mini-batch gradient descent with backpropagation
    """
    
    def __init__(self, layer_sizes: List[int], activation: str = 'relu',
                 learning_rate: float = 0.01, random_seed: int = 42):
        """
        Initialize neural network.
        
        Args:
            layer_sizes: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            learning_rate: Learning rate for gradient descent
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        
        # Set activation function
        if activation == 'relu':
            self.activation = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize parameters using He initialization for ReLU
        self.parameters = {}
        for l in range(1, self.num_layers):
            # He initialization: scale by sqrt(2/n_in) for ReLU
            scale = np.sqrt(2.0 / layer_sizes[l-1]) if activation == 'relu' else 0.01
            self.parameters[f'W{l}'] = np.random.randn(layer_sizes[l-1], layer_sizes[l]) * scale
            self.parameters[f'b{l}'] = np.zeros((1, layer_sizes[l]))
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def forward_propagation(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward propagation through the network.
        
        For each layer l:
            Z[l] = A[l-1]·W[l] + b[l]
            A[l] = g(Z[l])
        
        Args:
            X: Input data (batch_size, input_dim)
            
        Returns:
            Output activations and cache of intermediate values
        """
        cache = {'A0': X}
        A = X
        
        # Forward through hidden layers
        for l in range(1, self.num_layers - 1):
            Z = A @ self.parameters[f'W{l}'] + self.parameters[f'b{l}']
            A = self.activation(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        # Output layer (linear for regression, softmax for classification)
        l = self.num_layers - 1
        Z = A @ self.parameters[f'W{l}'] + self.parameters[f'b{l}']
        
        # Use softmax for multi-class, sigmoid for binary, linear for regression
        if self.layer_sizes[-1] > 1:
            A = Activation.softmax(Z)
        else:
            A = Z  # Linear output for regression
        
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
        
        return A, cache
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute loss function.
        
        For classification: Cross-entropy loss
        For regression: Mean squared error
        
        Args:
            y_pred: Predicted values
            y_true: True labels
            
        Returns:
            Loss value
        """
        m = y_true.shape[0]
        
        if self.layer_sizes[-1] > 1:
            # Cross-entropy loss for classification
            # L = -1/m * Σ y·log(ŷ)
            epsilon = 1e-8  # Prevent log(0)
            loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        else:
            # MSE for regression
            # L = 1/2m * Σ(y - ŷ)²
            loss = np.mean((y_pred - y_true) ** 2) / 2
        
        return loss
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, cache: dict) -> dict:
        """
        Backward propagation using chain rule.
        
        Compute gradients:
            dL/dW[l] = A[l-1]^T · dZ[l]
            dL/db[l] = sum(dZ[l])
            dZ[l] = dA[l] ⊙ g'(Z[l])
        
        Args:
            X: Input data
            y: True labels
            cache: Cache from forward propagation
            
        Returns:
            Dictionary of gradients
        """
        m = X.shape[0]
        gradients = {}
        
        # Output layer gradient
        l = self.num_layers - 1
        dZ = cache[f'A{l}'] - y  # For softmax + cross-entropy or MSE
        
        # Backpropagate through layers
        for l in range(self.num_layers - 1, 0, -1):
            A_prev = cache[f'A{l-1}']
            
            # Compute gradients for current layer
            gradients[f'dW{l}'] = (A_prev.T @ dZ) / m
            gradients[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Compute gradient for previous layer (if not input layer)
            if l > 1:
                dA_prev = dZ @ self.parameters[f'W{l}'].T
                Z_prev = cache[f'Z{l-1}']
                dZ = dA_prev * self.activation_derivative(Z_prev)
        
        return gradients
    
    def update_parameters(self, gradients: dict, learning_rate: float = None):
        """
        Update parameters using gradient descent.
        
        W[l] ← W[l] - η·dW[l]
        b[l] ← b[l] - η·db[l]
        
        Args:
            gradients: Dictionary of gradients
            learning_rate: Learning rate (optional, uses self.learning_rate if None)
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        for l in range(1, self.num_layers):
            self.parameters[f'W{l}'] -= lr * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= lr * gradients[f'db{l}']
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32, verbose: bool = True):
        """
        Train neural network using mini-batch gradient descent.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Print training progress
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch gradient descent
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Forward propagation
                y_pred, cache = self.forward_propagation(X_batch)
                
                # Compute loss
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += batch_loss * len(X_batch)
                
                # Backward propagation
                gradients = self.backward_propagation(X_batch, y_batch, cache)
                
                # Update parameters
                self.update_parameters(gradients)
            
            # Compute average epoch loss
            avg_train_loss = epoch_loss / n_samples
            self.history['train_loss'].append(avg_train_loss)
            
            # Compute training accuracy
            train_acc = self.accuracy(X_train, y_train)
            self.history['train_acc'].append(train_acc)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                y_val_pred, _ = self.forward_propagation(X_val)
                val_loss = self.compute_loss(y_val_pred, y_val)
                val_acc = self.accuracy(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch:3d}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch:3d}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        y_pred, _ = self.forward_propagation(X)
        
        if self.layer_sizes[-1] > 1:
            # Return class labels for classification
            return np.argmax(y_pred, axis=1)
        else:
            # Return continuous values for regression
            return y_pred
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy for classification tasks.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        
        if self.layer_sizes[-1] > 1:
            y_true = np.argmax(y, axis=1)
            return np.mean(y_pred == y_true)
        else:
            # For regression, return R² score
            y_pred_prob, _ = self.forward_propagation(X)
            ss_res = np.sum((y - y_pred_prob) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def plot_training_history(self, save_path: str = 'visualizations/training_history.png'):
        """
        Plot training and validation loss/accuracy curves.
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if self.history['val_loss']:
            ax1.plot(self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        if self.history['val_acc']:
            ax2.plot(self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """
    Demonstrate neural network training with gradient descent.
    """
    print("=" * 70)
    print("Neural Network with Gradient Descent Demo")
    print("=" * 70)
    
    # Generate synthetic classification dataset
    from sklearn.datasets import make_moons, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Example 1: Binary classification (moons dataset)
    print("\n1. Binary Classification (Moons Dataset)")
    print("-" * 70)
    
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert labels to one-hot encoding
    y_train_onehot = np.eye(2)[y_train]
    y_test_onehot = np.eye(2)[y_test]
    
    # Create and train neural network
    nn = NeuralNetwork(
        layer_sizes=[2, 16, 16, 2],  # Input(2) → Hidden(16) → Hidden(16) → Output(2)
        activation='relu',
        learning_rate=0.01
    )
    
    nn.train(X_train, y_train_onehot, X_test, y_test_onehot, 
             epochs=200, batch_size=32, verbose=True)
    
    # Evaluate
    test_acc = nn.accuracy(X_test, y_test_onehot)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # Plot training history
    nn.plot_training_history('visualizations/nn_training_history.png')
    
    # Visualize decision boundary
    visualize_decision_boundary(nn, X_test, y_test, 'Neural Network Decision Boundary')
    
    print("\n" + "=" * 70)
    print("Demo Complete! Visualizations saved to visualizations/")
    print("=" * 70)


def visualize_decision_boundary(model: NeuralNetwork, X: np.ndarray, y: np.ndarray, title: str):
    """
    Visualize decision boundary for 2D classification.
    
    Args:
        model: Trained neural network
        X: Test data
        y: Test labels
        title: Plot title
    """
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.colorbar()
    plt.savefig(f'visualizations/{title.replace(" ", "_").lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    main()
