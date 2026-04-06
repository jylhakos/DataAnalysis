"""
Gradient Descent Implementations

This module implements various gradient descent algorithms:
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-batch Gradient Descent

Mathematical foundation:
    For function f(x), update rule: x ← x - η∇f(x)
    where η is the learning rate
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional


class GradientDescent:
    """Base class for gradient descent optimization algorithms."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-6):
        """
        Initialize gradient descent optimizer.
        
        Args:
            learning_rate: Step size η for parameter updates
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = {'params': [], 'loss': []}
    
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray) -> np.ndarray:
        """
        Optimize function f starting from x0.
        
        Args:
            f: Objective function to minimize
            grad_f: Gradient of objective function
            x0: Initial parameters
            
        Returns:
            Optimized parameters
        """
        raise NotImplementedError("Subclasses must implement optimize()")


class BatchGradientDescent(GradientDescent):
    """
    Batch Gradient Descent
    
    Computes gradient using entire dataset:
        w ← w - η * (1/n) * Σ∇L(f(x_i; w), y_i)
    """
    
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray, 
                 X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform batch gradient descent optimization.
        
        Args:
            f: Loss function
            grad_f: Gradient function
            x0: Initial parameters
            X: Input data (optional, for visualization)
            y: Target values (optional, for visualization)
            
        Returns:
            Optimized parameters
        """
        x = x0.copy()
        self.history = {'params': [x.copy()], 'loss': [f(x)]}
        
        for iteration in range(self.max_iterations):
            # Compute gradient
            gradient = grad_f(x)
            
            # Update parameters
            x_new = x - self.learning_rate * gradient
            
            # Store history
            loss = f(x_new)
            self.history['params'].append(x_new.copy())
            self.history['loss'].append(loss)
            
            # Check convergence
            if np.linalg.norm(x_new - x) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break
            
            x = x_new
        
        return x


class StochasticGradientDescent(GradientDescent):
    """
    Stochastic Gradient Descent (SGD)
    
    Updates parameters using one sample at a time:
        w ← w - η * ∇L(f(x_i; w), y_i)
    """
    
    def optimize(self, loss_fn: Callable, grad_fn: Callable, x0: np.ndarray,
                 X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Perform stochastic gradient descent.
        
        Args:
            loss_fn: Loss function that takes parameters and single sample
            grad_fn: Gradient function that takes parameters and single sample
            x0: Initial parameters
            X: Input data
            y: Target values
            
        Returns:
            Optimized parameters
        """
        x = x0.copy()
        n_samples = len(X)
        self.history = {'params': [x.copy()], 'loss': []}
        
        for iteration in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0
            for idx in indices:
                # Compute gradient for single sample
                gradient = grad_fn(x, X[idx:idx+1], y[idx:idx+1])
                
                # Update parameters
                x = x - self.learning_rate * gradient
                
                # Accumulate loss for monitoring
                epoch_loss += loss_fn(x, X[idx:idx+1], y[idx:idx+1])
            
            # Store history (once per epoch)
            avg_loss = epoch_loss / n_samples
            self.history['params'].append(x.copy())
            self.history['loss'].append(avg_loss)
            
            # Check convergence (less strict for SGD due to noise)
            if len(self.history['loss']) > 1:
                if abs(self.history['loss'][-1] - self.history['loss'][-2]) < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return x


class MiniBatchGradientDescent(GradientDescent):
    """
    Mini-batch Gradient Descent
    
    Balance between batch and stochastic:
        w ← w - η * (1/m) * Σ∇L(f(x_i; w), y_i) for mini-batch of size m
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, batch_size: int = 32):
        """
        Initialize mini-batch gradient descent.
        
        Args:
            learning_rate: Step size
            max_iterations: Maximum epochs
            tolerance: Convergence threshold
            batch_size: Size of mini-batches
        """
        super().__init__(learning_rate, max_iterations, tolerance)
        self.batch_size = batch_size
    
    def optimize(self, loss_fn: Callable, grad_fn: Callable, x0: np.ndarray,
                 X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Perform mini-batch gradient descent.
        
        Args:
            loss_fn: Loss function
            grad_fn: Gradient function
            x0: Initial parameters
            X: Input data
            y: Target values
            
        Returns:
            Optimized parameters
        """
        x = x0.copy()
        n_samples = len(X)
        self.history = {'params': [x.copy()], 'loss': []}
        
        for iteration in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                batch_end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Compute gradient for batch
                gradient = grad_fn(x, X_batch, y_batch)
                
                # Update parameters
                x = x - self.learning_rate * gradient
                
                # Accumulate loss
                epoch_loss += loss_fn(x, X_batch, y_batch) * len(X_batch)
                n_batches += 1
            
            # Store history
            avg_loss = epoch_loss / n_samples
            self.history['params'].append(x.copy())
            self.history['loss'].append(avg_loss)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {avg_loss:.6f}")
            
            # Check convergence
            if len(self.history['loss']) > 1:
                if abs(self.history['loss'][-1] - self.history['loss'][-2]) < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return x


# Example functions for testing
def quadratic_function(x: np.ndarray) -> float:
    """
    Simple quadratic function: f(x) = x^T x
    Minimum at x = 0
    """
    return np.sum(x ** 2)


def quadratic_gradient(x: np.ndarray) -> np.ndarray:
    """
    Gradient of quadratic function: ∇f(x) = 2x
    """
    return 2 * x


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock function (challenging optimization problem)
    f(x,y) = (a-x)^2 + b(y-x^2)^2
    Global minimum at (a,a^2), typically a=1, b=100
    """
    a, b = 1, 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """
    Gradient of Rosenbrock function
    """
    a, b = 1, 100
    df_dx = -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2)
    df_dy = 2*b*(x[1] - x[0]**2)
    return np.array([df_dx, df_dy])


def visualize_optimization(optimizer: GradientDescent, title: str):
    """
    Visualize optimization trajectory and loss convergence.
    
    Args:
        optimizer: Fitted gradient descent optimizer with history
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss convergence
    ax1.plot(optimizer.history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{title}: Loss Convergence', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Parameter trajectory (for 2D problems)
    params = np.array(optimizer.history['params'])
    if params.shape[1] == 2:
        # Create contour plot for Rosenbrock function
        x_range = np.linspace(-2, 2, 100)
        y_range = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = rosenbrock_function(np.array([X[i, j], Y[i, j]]))
        
        contour = ax2.contour(X, Y, Z, levels=np.logspace(-1, 3.5, 20), cmap='viridis', alpha=0.6)
        ax2.clabel(contour, inline=True, fontsize=8)
        
        # Plot optimization trajectory
        ax2.plot(params[:, 0], params[:, 1], 'r.-', linewidth=2, markersize=8, label='Trajectory')
        ax2.plot(params[0, 0], params[0, 1], 'go', markersize=12, label='Start')
        ax2.plot(params[-1, 0], params[-1, 1], 'r*', markersize=15, label='End')
        ax2.plot(1, 1, 'b*', markersize=15, label='Global Minimum')
        
        ax2.set_xlabel('x₁', fontsize=12)
        ax2.set_ylabel('x₂', fontsize=12)
        ax2.set_title(f'{title}: Optimization Trajectory', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{title.replace(" ", "_").lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """
    Demonstrate different gradient descent algorithms.
    """
    print("=" * 70)
    print("Gradient Descent Implementations Demo")
    print("=" * 70)
    
    # Example 1: Batch Gradient Descent on quadratic function
    print("\n1. Batch Gradient Descent (Quadratic Function)")
    print("-" * 70)
    x0 = np.array([10.0, 10.0])
    bgd = BatchGradientDescent(learning_rate=0.1, max_iterations=100)
    x_opt = bgd.optimize(quadratic_function, quadratic_gradient, x0)
    print(f"Starting point: {x0}")
    print(f"Optimized parameters: {x_opt}")
    print(f"Final loss: {quadratic_function(x_opt):.8f}")
    visualize_optimization(bgd, "Batch Gradient Descent")
    
    # Example 2: Batch Gradient Descent on Rosenbrock function
    print("\n2. Batch Gradient Descent (Rosenbrock Function)")
    print("-" * 70)
    x0 = np.array([-1.0, -1.0])
    bgd_rosenbrock = BatchGradientDescent(learning_rate=0.001, max_iterations=5000, tolerance=1e-8)
    x_opt = bgd_rosenbrock.optimize(rosenbrock_function, rosenbrock_gradient, x0)
    print(f"Starting point: {x0}")
    print(f"Optimized parameters: {x_opt}")
    print(f"Final loss: {rosenbrock_function(x_opt):.8f}")
    print(f"True minimum: [1.0, 1.0], Loss: 0.0")
    visualize_optimization(bgd_rosenbrock, "Batch GD Rosenbrock")
    
    # Example 3: Compare learning rates
    print("\n3. Impact of Learning Rate")
    print("-" * 70)
    learning_rates = [0.0001, 0.001, 0.01]
    x0 = np.array([-1.0, -1.0])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, lr in enumerate(learning_rates):
        optimizer = BatchGradientDescent(learning_rate=lr, max_iterations=1000, tolerance=1e-8)
        x_opt = optimizer.optimize(rosenbrock_function, rosenbrock_gradient, x0)
        
        axes[i].plot(optimizer.history['loss'])
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(f'Learning Rate: {lr}')
        axes[i].set_yscale('log')
        axes[i].grid(True, alpha=0.3)
        
        print(f"LR={lr}: Final loss={rosenbrock_function(x_opt):.6f}, Final params={x_opt}")
    
    plt.tight_layout()
    plt.savefig('visualizations/learning_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("Demo Complete! Visualizations saved to visualizations/")
    print("=" * 70)


if __name__ == "__main__":
    # Create visualizations directory if it doesn't exist
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    main()
