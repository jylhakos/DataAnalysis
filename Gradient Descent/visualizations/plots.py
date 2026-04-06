"""
Visualization Utilities for Gradient Descent

This module provides visualization tools for understanding gradient descent
behavior, loss landscapes, and optimization trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Optional, List, Tuple


class LossLandscapeVisualizer:
    """Visualize loss landscapes and optimization trajectories."""
    
    @staticmethod
    def plot_1d_loss_landscape(loss_fn: Callable, x_range: Tuple[float, float],
                               optim_history: Optional[List[float]] = None,
                               title: str = "1D Loss Landscape"):
        """
        Plot 1D loss landscape and optimization trajectory.
        
        Args:
            loss_fn: Loss function f(x) → loss
            x_range: Range of x values (x_min, x_max)
            optim_history: List of x values from optimization
            title: Plot title
        """
        # Create x values
        x = np.linspace(x_range[0], x_range[1], 1000)
        y = np.array([loss_fn(np.array([xi])) for xi in x])
        
        # Plot loss landscape
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, 'b-', linewidth=2, label='Loss Function')
        
        # Plot optimization trajectory
        if optim_history is not None:
            optim_x = [h[0] if isinstance(h, np.ndarray) else h for h in optim_history]
            optim_y = [loss_fn(np.array([xi])) for xi in optim_x]
            
            plt.plot(optim_x, optim_y, 'ro-', markersize=6, linewidth=1.5,
                    label='Optimization Path', alpha=0.7)
            plt.plot(optim_x[0], optim_y[0], 'go', markersize=12, label='Start')
            plt.plot(optim_x[-1], optim_y[-1], 'r*', markersize=15, label='End')
        
        plt.xlabel('Parameter x', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_2d_contour(loss_fn: Callable,
                       x_range: Tuple[float, float],
                       y_range: Tuple[float, float],
                       optim_history: Optional[List[np.ndarray]] = None,
                       title: str = "2D Loss Landscape",
                       levels: int = 30):
        """
        Plot 2D loss landscape as contour plot.
        
        Args:
            loss_fn: Loss function f([x, y]) → loss
            x_range: Range of x values
            y_range: Range of y values
            optim_history: List of [x, y] points from optimization
            title: Plot title
            levels: Number of contour levels
        """
        # Create meshgrid
        x = np.linspace(x_range[0], x_range[1], 200)
        y = np.linspace(y_range[0], y_range[1], 200)
        X, Y = np.meshgrid(x, y)
        
        # Compute loss for each point
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = loss_fn(np.array([X[i, j], Y[i, j]]))
        
        # Plot contours
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use log scale for better visualization of steep landscapes
        contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', linewidths=1.5)
        contourf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        
        plt.colorbar(contourf, ax=ax, label='Loss')
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Plot optimization trajectory
        if optim_history is not None:
            path = np.array(optim_history)
            ax.plot(path[:, 0], path[:, 1], 'r.-', linewidth=2, markersize=8,
                   label='Optimization Path')
            ax.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start', zorder=5)
            ax.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='End', zorder=5)
        
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_3d_surface(loss_fn: Callable,
                       x_range: Tuple[float, float],
                       y_range: Tuple[float, float],
                       optim_history: Optional[List[np.ndarray]] = None,
                       title: str = "3D Loss Surface"):
        """
        Plot 3D loss surface.
        
        Args:
            loss_fn: Loss function f([x, y]) → loss
            x_range: Range of x values
            y_range: Range of y values
            optim_history: List of [x, y] points from optimization
            title: Plot title
        """
        # Create meshgrid
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute loss
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = loss_fn(np.array([X[i, j], Y[i, j]]))
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7,
                              linewidth=0, antialiased=True)
        
        # Plot optimization trajectory
        if optim_history is not None:
            path = np.array(optim_history)
            path_z = np.array([loss_fn(p) for p in path])
            ax.plot(path[:, 0], path[:, 1], path_z, 'r.-', linewidth=2,
                   markersize=8, label='Optimization Path')
            ax.scatter(path[0, 0], path[0, 1], path_z[0], c='g', s=100,
                      marker='o', label='Start')
            ax.scatter(path[-1, 0], path[-1, 1], path_z[-1], c='r', s=150,
                      marker='*', label='End')
        
        ax.set_xlabel('x₁', fontsize=11)
        ax.set_ylabel('x₂', fontsize=11)
        ax.set_zlabel('Loss', fontsize=11)
        ax.set_title(title, fontsize=14)
        ax.legend()
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_gradient_field(loss_fn: Callable, grad_fn: Callable,
                           x_range: Tuple[float, float],
                           y_range: Tuple[float, float],
                           title: str = "Gradient Field",
                           density: int = 20):
        """
        Plot gradient vector field.
        
        Args:
            loss_fn: Loss function
            grad_fn: Gradient function
            x_range: Range of x values
            y_range: Range of y values
            title: Plot title
            density: Density of gradient arrows
        """
        # Create grid
        x = np.linspace(x_range[0], x_range[1], density)
        y = np.linspace(y_range[0], y_range[1], density)
        X, Y = np.meshgrid(x, y)
        
        # Compute gradients
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                grad = grad_fn(point)
                U[i, j] = -grad[0]  # Negative gradient (descent direction)
                V[i, j] = -grad[1]
                Z[i, j] = loss_fn(point)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Background: loss contours
        contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
        plt.colorbar(contourf, ax=ax, label='Loss')
        
        # Gradient arrows
        quiver = ax.quiver(X, Y, U, V, color='red', alpha=0.7, width=0.003,
                          label='Gradient Descent Direction')
        
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig


class TrainingVisualizer:
    """Visualize training dynamics and convergence."""
    
    @staticmethod
    def plot_loss_convergence(train_losses: List[float],
                             val_losses: Optional[List[float]] = None,
                             title: str = "Loss Convergence",
                             log_scale: bool = False):
        """
        Plot training and validation loss over time.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses (optional)
            title: Plot title
            log_scale: Use logarithmic scale for y-axis
        """
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        
        if val_losses is not None:
            plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if log_scale:
            plt.yscale('log')
        
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_learning_rate_comparison(results: dict, metric: str = 'loss'):
        """
        Compare different learning rates.
        
        Args:
            results: Dict mapping learning_rate → metric_values
            metric: Metric name ('loss' or 'accuracy')
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for lr, values in results.items():
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, linewidth=2, label=f'LR = {lr}')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} vs Learning Rate', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if metric == 'loss':
            ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_gradient_norms(gradient_norms: List[float],
                           clip_threshold: Optional[float] = None,
                           title: str = "Gradient Norms During Training"):
        """
        Plot gradient norms to visualize exploding/vanishing gradients.
        
        Args:
            gradient_norms: List of gradient norm values
            clip_threshold: Gradient clipping threshold (if used)
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        iterations = range(1, len(gradient_norms) + 1)
        plt.plot(iterations, gradient_norms, 'b-', linewidth=1, alpha=0.7,
                label='Gradient Norm')
        
        if clip_threshold is not None:
            plt.axhline(y=clip_threshold, color='r', linestyle='--', linewidth=2,
                       label=f'Clipping Threshold ({clip_threshold})')
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Gradient Norm ||∇L||', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        
        return plt.gcf()


def demo_visualizations():
    """Demonstrate all visualization capabilities."""
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    print("=" * 70)
    print("Gradient Descent Visualization Demo")
    print("=" * 70)
    
    # Define test functions
    def quadratic(x):
        return np.sum(x**2)
    
    def quadratic_grad(x):
        return 2*x
    
    def rosenbrock(x):
        return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dy = 200*(x[1] - x[0]**2)
        return np.array([dx, dy])
    
    # Simulate optimization trajectory for Rosenbrock
    x = np.array([-1.0, -1.0])
    history = [x.copy()]
    lr = 0.001
    
    for _ in range(500):
        grad = rosenbrock_grad(x)
        x = x - lr * grad
        history.append(x.copy())
    
    print("\n1. Plotting 2D Contour Map...")
    viz = LossLandscapeVisualizer()
    fig = viz.plot_2d_contour(rosenbrock, (-2, 2), (-1, 3), history,
                             "Rosenbrock Function: 2D Contour")
    plt.savefig('visualizations/rosenbrock_contour.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("2. Plotting 3D Surface...")
    fig = viz.plot_3d_surface(rosenbrock, (-2, 2), (-1, 3), history,
                             "Rosenbrock Function: 3D Surface")
    plt.savefig('visualizations/rosenbrock_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("3. Plotting Gradient Field...")
    fig = viz.plot_gradient_field(rosenbrock, rosenbrock_grad, (-2, 2), (-1, 3),
                                 "Gradient Field", density=15)
    plt.savefig('visualizations/gradient_field.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("4. Plotting Loss Convergence...")
    train_losses = [rosenbrock(h) for h in history]
    train_viz = TrainingVisualizer()
    fig = train_viz.plot_loss_convergence(train_losses, title="Optimization Progress",
                                         log_scale=True)
    plt.savefig('visualizations/loss_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("5. Plotting Gradient Norms...")
    grad_norms = [np.linalg.norm(rosenbrock_grad(h)) for h in history]
    fig = train_viz.plot_gradient_norms(grad_norms, clip_threshold=1.0)
    plt.savefig('visualizations/gradient_norms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 70)
    print("Visualization demo complete! Check visualizations/ folder")
    print("=" * 70)


if __name__ == "__main__":
    demo_visualizations()
