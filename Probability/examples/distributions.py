"""
Probability Distributions Examples
Demonstrates various probability distributions and their properties
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson, expon, uniform, chi2
from scipy import stats


def demonstrate_normal_distribution():
    """Demonstrate normal distribution properties"""
    print("=" * 60)
    print("NORMAL DISTRIBUTION")
    print("=" * 60)
    
    # Generate x-axis values
    x = np.linspace(-4, 4, 100)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. PDF of standard normal distribution
    axes[0, 0].plot(x, norm.pdf(x), 'b-', linewidth=2)
    axes[0, 0].set_title('Standard Normal Distribution PDF')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. CDF of standard normal distribution
    axes[0, 1].plot(x, norm.cdf(x), 'r-', linewidth=2)
    axes[0, 1].set_title('Standard Normal Distribution CDF')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Multiple normal distributions with different parameters
    for mu, sigma in [(0, 1), (0, 2), (2, 1)]:
        axes[1, 0].plot(x, norm.pdf(x, mu, sigma), 
                       label=f'μ={mu}, σ={sigma}', linewidth=2)
    axes[1, 0].set_title('Normal Distributions with Different Parameters')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Histogram of random samples
    samples = np.random.randn(1000)
    axes[1, 1].hist(samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].plot(x, norm.pdf(x), 'r-', linewidth=2, label='Theoretical PDF')
    axes[1, 1].set_title('Random Samples from Normal Distribution')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/normal_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: examples/normal_distribution.png")
    
    # Calculate probabilities
    print("\nProbability Calculations:")
    print(f"P(Z < 1.96) = {norm.cdf(1.96):.4f}")
    print(f"P(Z > 1.96) = {1 - norm.cdf(1.96):.4f}")
    print(f"P(-1 < Z < 1) = {norm.cdf(1) - norm.cdf(-1):.4f}")
    print(f"95th percentile = {norm.ppf(0.95):.4f}")
    

def demonstrate_binomial_distribution():
    """Demonstrate binomial distribution"""
    print("\n" + "=" * 60)
    print("BINOMIAL DISTRIBUTION")
    print("=" * 60)
    
    n, p = 10, 0.5  # 10 trials, 50% success probability
    x = np.arange(0, n + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. PMF
    pmf = binom.pmf(x, n, p)
    axes[0].bar(x, pmf, alpha=0.7, color='green', edgecolor='black')
    axes[0].set_title(f'Binomial Distribution PMF (n={n}, p={p})')
    axes[0].set_xlabel('Number of Successes')
    axes[0].set_ylabel('Probability')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Different probabilities
    for prob in [0.3, 0.5, 0.7]:
        axes[1].plot(x, binom.pmf(x, n, prob), marker='o', label=f'p={prob}', linewidth=2)
    axes[1].set_title(f'Binomial Distribution with Different p (n={n})')
    axes[1].set_xlabel('Number of Successes')
    axes[1].set_ylabel('Probability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/binomial_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: examples/binomial_distribution.png")
    
    print(f"\nExpected value: {binom.mean(n, p):.2f}")
    print(f"Variance: {binom.var(n, p):.2f}")
    print(f"P(X = 5) = {binom.pmf(5, n, p):.4f}")
    print(f"P(X <= 5) = {binom.cdf(5, n, p):.4f}")


def demonstrate_poisson_distribution():
    """Demonstrate Poisson distribution"""
    print("\n" + "=" * 60)
    print("POISSON DISTRIBUTION")
    print("=" * 60)
    
    x = np.arange(0, 20)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Single lambda
    lambda_val = 5
    pmf = poisson.pmf(x, lambda_val)
    axes[0].bar(x, pmf, alpha=0.7, color='orange', edgecolor='black')
    axes[0].set_title(f'Poisson Distribution PMF (λ={lambda_val})')
    axes[0].set_xlabel('Number of Events')
    axes[0].set_ylabel('Probability')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Different lambdas
    for lam in [2, 5, 10]:
        axes[1].plot(x, poisson.pmf(x, lam), marker='o', label=f'λ={lam}', linewidth=2)
    axes[1].set_title('Poisson Distribution with Different λ')
    axes[1].set_xlabel('Number of Events')
    axes[1].set_ylabel('Probability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/poisson_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: examples/poisson_distribution.png")
    
    print(f"\nExpected value: {poisson.mean(lambda_val):.2f}")
    print(f"Variance: {poisson.var(lambda_val):.2f}")
    print(f"P(X = 5) = {poisson.pmf(5, lambda_val):.4f}")


def demonstrate_exponential_distribution():
    """Demonstrate exponential distribution"""
    print("\n" + "=" * 60)
    print("EXPONENTIAL DISTRIBUTION")
    print("=" * 60)
    
    x = np.linspace(0, 5, 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. PDF
    scale = 1.0  # scale = 1/lambda
    axes[0].plot(x, expon.pdf(x, scale=scale), 'b-', linewidth=2)
    axes[0].fill_between(x, expon.pdf(x, scale=scale), alpha=0.3)
    axes[0].set_title(f'Exponential Distribution PDF (λ=1)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Probability Density')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Different rates
    for rate in [0.5, 1.0, 2.0]:
        axes[1].plot(x, expon.pdf(x, scale=1/rate), 
                    label=f'λ={rate}', linewidth=2)
    axes[1].set_title('Exponential Distribution with Different λ')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Probability Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/exponential_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: examples/exponential_distribution.png")
    
    print(f"\nMean: {expon.mean(scale=scale):.2f}")
    print(f"P(X < 2) = {expon.cdf(2, scale=scale):.4f}")


def demonstrate_uniform_distribution():
    """Demonstrate uniform distribution"""
    print("\n" + "=" * 60)
    print("UNIFORM DISTRIBUTION")
    print("=" * 60)
    
    a, b = 0, 10
    x = np.linspace(-2, 12, 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. PDF
    axes[0].plot(x, uniform.pdf(x, a, b-a), 'b-', linewidth=2)
    axes[0].fill_between(x, uniform.pdf(x, a, b-a), alpha=0.3)
    axes[0].set_title(f'Uniform Distribution PDF [a={a}, b={b}]')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Probability Density')
    axes[0].grid(True, alpha=0.3)
    
    # 2. CDF
    axes[1].plot(x, uniform.cdf(x, a, b-a), 'r-', linewidth=2)
    axes[1].set_title(f'Uniform Distribution CDF [a={a}, b={b}]')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/uniform_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: examples/uniform_distribution.png")
    
    print(f"\nMean: {uniform.mean(a, b-a):.2f}")
    print(f"Variance: {uniform.var(a, b-a):.2f}")


def fit_distribution_example():
    """Demonstrate fitting a distribution to data"""
    print("\n" + "=" * 60)
    print("FITTING DISTRIBUTIONS TO DATA")
    print("=" * 60)
    
    # Generate sample data from a normal distribution
    np.random.seed(42)
    true_mean, true_std = 5, 2
    data = np.random.normal(true_mean, true_std, 1000)
    
    # Fit the distribution
    params = norm.fit(data)
    fitted_mean, fitted_std = params
    
    print(f"\nTrue parameters: μ={true_mean}, σ={true_std}")
    print(f"Estimated parameters: μ={fitted_mean:.4f}, σ={fitted_std:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=40, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black', label='Sample Data')
    
    x = np.linspace(data.min(), data.max(), 100)
    plt.plot(x, norm.pdf(x, fitted_mean, fitted_std), 
             'r-', linewidth=2, label=f'Fitted Normal(μ={fitted_mean:.2f}, σ={fitted_std:.2f})')
    plt.plot(x, norm.pdf(x, true_mean, true_std), 
             'g--', linewidth=2, label=f'True Normal(μ={true_mean}, σ={true_std})')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Fitting Normal Distribution to Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('examples/fitted_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: examples/fitted_distribution.png")


if __name__ == "__main__":
    print("Probability Distributions Demonstration")
    print("=" * 60)
    
    demonstrate_normal_distribution()
    demonstrate_binomial_distribution()
    demonstrate_poisson_distribution()
    demonstrate_exponential_distribution()
    demonstrate_uniform_distribution()
    fit_distribution_example()
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
