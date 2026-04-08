"""
Visualization utilities for Monte Carlo simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Tuple, Optional


def plot_monte_carlo_points(n_points: int = 5000, save_path: str = "results/monte_carlo_points.png"):
    """
    Visualize Monte Carlo point distribution for pi estimation.
    
    Args:
        n_points: Number of random points to plot
        save_path: Path to save figure
    """
    # Generate random points
    np.random.seed(42)
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    
    # Determine which points are inside the circle
    inside = x**2 + y**2 <= 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot points
    ax.scatter(x[inside], y[inside], c='blue', s=1, alpha=0.5, label='Inside circle')
    ax.scatter(x[~inside], y[~inside], c='red', s=1, alpha=0.5, label='Outside circle')
    
    # Draw circle
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
    ax.add_patch(circle)
    
    # Draw square
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', linewidth=2)
    
    # Estimate pi
    pi_estimate = 4 * np.sum(inside) / n_points
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Monte Carlo Estimation of π\n{n_points} points, Estimate: {pi_estimate:.5f}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.close()


def plot_score_distribution(scores: List[float], title: str = "Response Quality Distribution",
                           save_path: str = "results/score_distribution.png"):
    """
    Plot distribution of Monte Carlo sample scores.
    
    Args:
        scores: List of quality scores
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(scores, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(scores):.2f}')
    ax1.axvline(np.median(scores), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(scores):.2f}')
    ax1.set_xlabel('Quality Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'{title}\nHistogram', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(scores, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Quality Score', fontsize=12)
    ax2.set_title(f'{title}\nBox Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(scores):.2f}\n'
    stats_text += f'Std: {np.std(scores):.2f}\n'
    stats_text += f'Min: {np.min(scores):.2f}\n'
    stats_text += f'Max: {np.max(scores):.2f}'
    ax2.text(1.15, np.mean(scores), stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.close()


def plot_mcts_tree_stats(results_path: str = "results/mcts_results.json",
                        save_path: str = "results/mcts_stats.png"):
    """
    Visualize MCTS statistics from saved results.
    
    Args:
        results_path: Path to MCTS results JSON
        save_path: Path to save figure
    """
    # Load results
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {results_path}")
        print("Run mcts_llm.py first to generate results.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create simple visualization showing the path
    path = results.get('path', [])
    if not path:
        print("No path found in results")
        return
    
    # Show improvement through path
    ax.barh(range(len(path)), [len(str(s)) for s in path], color='steelblue')
    ax.set_ylabel('Step in Reasoning Path', fontsize=12)
    ax.set_xlabel('Response Length (characters)', fontsize=12)
    ax.set_title(f'MCTS Reasoning Path\nBest Reward: {results.get("best_reward", 0):.2f}',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.close()


def generate_all_visualizations():
    """Generate all available visualizations."""
    print("="*60)
    print("Generating Visualizations")
    print("="*60)
    print()
    
    # 1. Monte Carlo points visualization
    print("1. Generating Monte Carlo point distribution...")
    plot_monte_carlo_points(n_points=5000)
    print()
    
    # 2. Sample score distribution (synthetic data for demonstration)
    print("2. Generating score distribution example...")
    np.random.seed(42)
    synthetic_scores = np.random.beta(8, 2, 100) * 100  # Beta distribution scaled to 0-100
    plot_score_distribution(synthetic_scores, title="Example: Monte Carlo Sample Scores")
    print()
    
    # 3. MCTS stats (if available)
    print("3. Generating MCTS statistics...")
    plot_mcts_tree_stats()
    print()
    
    print("="*60)
    print("Visualization generation complete!")
    print("Check the results/ directory for output files.")
    print("="*60)


if __name__ == "__main__":
    generate_all_visualizations()
