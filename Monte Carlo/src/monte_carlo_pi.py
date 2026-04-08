"""
Basic Monte Carlo Simulation: Estimating Pi

This script demonstrates the fundamental Monte Carlo method by estimating
the value of π through random sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def estimate_pi(n_simulations):
    """
    Estimate the value of pi using Monte Carlo simulation.
    
    The method randomly places points in a 2x2 square centered at origin.
    Points are checked if they fall within the unit circle (radius = 1).
    The ratio of points inside the circle to total points approximates π/4.
    
    Args:
        n_simulations (int): Number of random points to generate
        
    Returns:
        float: Estimated value of pi
    """
    n_points_circle = 0
    
    # Generate random x and y coordinates between -1 and 1
    l_xs = np.random.uniform(-1, 1, n_simulations)
    l_ys = np.random.uniform(-1, 1, n_simulations)
    
    for i in range(n_simulations):
        # Check if point is inside the circle (radius = 1)
        if l_xs[i]**2 + l_ys[i]**2 <= 1:
            n_points_circle += 1
    
    # Ratio of circle area to square area is pi/4
    return 4 * n_points_circle / n_simulations


def estimate_pi_with_convergence(max_simulations=100000, step=1000):
    """
    Estimate pi with increasing number of simulations to show convergence.
    
    Args:
        max_simulations (int): Maximum number of simulations
        step (int): Step size for increasing simulations
        
    Returns:
        tuple: (simulation_counts, pi_estimates, errors)
    """
    simulation_counts = []
    pi_estimates = []
    errors = []
    
    for n in range(step, max_simulations + 1, step):
        pi_est = estimate_pi(n)
        error = abs(pi_est - np.pi)
        
        simulation_counts.append(n)
        pi_estimates.append(pi_est)
        errors.append(error)
        
        if n % 10000 == 0:
            print(f"Simulations: {n:6d} | Estimated π: {pi_est:.6f} | Error: {error:.6f}")
    
    return simulation_counts, pi_estimates, errors


def visualize_convergence(simulation_counts, pi_estimates, errors, save_path="results/pi_estimation.png"):
    """
    Create visualization of pi estimation convergence.
    
    Args:
        simulation_counts (list): Number of simulations at each step
        pi_estimates (list): Estimated pi values
        errors (list): Absolute errors
        save_path (str): Path to save the figure
    """
    # Create results directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Pi estimation convergence
    ax1.plot(simulation_counts, pi_estimates, 'b-', label='Estimated π', linewidth=2)
    ax1.axhline(y=np.pi, color='r', linestyle='--', label='True π', linewidth=2)
    ax1.set_xlabel('Number of Simulations', fontsize=12)
    ax1.set_ylabel('Estimated π', fontsize=12)
    ax1.set_title('Monte Carlo Estimation of π - Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error over iterations
    ax2.plot(simulation_counts, errors, 'g-', linewidth=2)
    ax2.set_xlabel('Number of Simulations', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Estimation Error vs Number of Simulations', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.close()


def main():
    """Main execution function."""
    print("="*60)
    print("Monte Carlo Simulation: Estimating π")
    print("="*60)
    print()
    
    # Quick estimation
    print("Quick estimation with 100,000 simulations:")
    pi_est = estimate_pi(100000)
    print(f"Estimated π: {pi_est:.6f}")
    print(f"True π:      {np.pi:.6f}")
    print(f"Error:       {abs(pi_est - np.pi):.6f}")
    print()
    
    # Convergence analysis
    print("Running convergence analysis...")
    print()
    sim_counts, pi_ests, errors = estimate_pi_with_convergence(
        max_simulations=100000, 
        step=1000
    )
    
    # Visualize results
    print()
    print("Creating visualization...")
    visualize_convergence(sim_counts, pi_ests, errors)
    
    print()
    print("="*60)
    print("Simulation complete!")
    print("="*60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
