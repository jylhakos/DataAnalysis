"""
Main script to run all probability examples
Execute this file to generate all visualizations and demonstrations
"""

import sys
import os

# Add examples directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))

def main():
    """Run all example scripts"""
    
    print("\n" + "=" * 70)
    print(" " * 15 + "PROBABILITY IN DATA ANALYSIS")
    print(" " * 20 + "Example Demonstrations")
    print("=" * 70 + "\n")
    
    print("This script will run all probability examples and generate visualizations.")
    print("Please ensure you have activated the virtual environment and installed")
    print("all required dependencies.\n")
    
    input("Press Enter to continue...")
    
    # Import and run examples
    print("\n\n[1/3] Running Distributions Examples...")
    print("-" * 70)
    from examples import distributions
    
    print("\n\n[2/3] Running Conditional Probability Examples...")
    print("-" * 70)
    from examples import conditional_probability
    
    print("\n\n[3/3] Running Bias-Variance Tradeoff Examples...")
    print("-" * 70)
    from examples import bias_variance
    
    print("\n\n" + "=" * 70)
    print(" " * 20 + "ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nGenerated files are saved in the 'examples/' directory:")
    print("  • normal_distribution.png")
    print("  • binomial_distribution.png")
    print("  • poisson_distribution.png")
    print("  • exponential_distribution.png")
    print("  • uniform_distribution.png")
    print("  • fitted_distribution.png")
    print("  • conditional_probability.png")
    print("  • bayes_theorem.png")
    print("  • bayesian_updating.png")
    print("  • bias_variance_models.png")
    print("  • bias_variance_tradeoff.png")
    print("\nYou can view these images to see the results of the demonstrations.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
