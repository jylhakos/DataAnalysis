"""
Bias-Variance Tradeoff Demonstration
Illustrates the relationship between bias, variance, and model complexity
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def generate_data(n_samples=100, noise=0.5, random_state=42):
    """Generate synthetic data with true underlying function"""
    np.random.seed(random_state)
    X = np.sort(np.random.rand(n_samples) * 10)
    # True function: quadratic
    y_true = 2 + 0.5 * X + 0.3 * X**2
    # Add noise
    y = y_true + np.random.randn(n_samples) * noise * (1 + 0.3 * X)
    return X, y, y_true


def fit_polynomial_models(X_train, y_train, X_test, y_test, degrees):
    """Fit polynomial models of various degrees"""
    results = []
    
    for degree in degrees:
        # Transform features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
        X_test_poly = poly.transform(X_test.reshape(-1, 1))
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Calculate errors
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        results.append({
            'degree': degree,
            'model': model,
            'poly': poly,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        })
    
    return results


def demonstrate_bias_variance_tradeoff():
    """Main demonstration of bias-variance tradeoff"""
    print("=" * 60)
    print("BIAS-VARIANCE TRADEOFF DEMONSTRATION")
    print("=" * 60)
    
    # Generate data
    X, y, y_true = generate_data(n_samples=100, noise=3.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Fit models with different complexities
    degrees = [1, 2, 3, 5, 10, 15]
    results = fit_polynomial_models(X_train, y_train, X_test, y_test, degrees)
    
    print("\nModel Performance:")
    print("-" * 60)
    print(f"{'Degree':<10} {'Train MSE':<15} {'Test MSE':<15} {'Interpretation'}")
    print("-" * 60)
    
    for r in results:
        interpretation = ""
        if r['train_mse'] > 50:
            interpretation = "High Bias (Underfitting)"
        elif r['test_mse'] > r['train_mse'] * 2:
            interpretation = "High Variance (Overfitting)"
        else:
            interpretation = "Good Balance"
        
        print(f"{r['degree']:<10} {r['train_mse']:<15.4f} {r['test_mse']:<15.4f} {interpretation}")
    
    # Visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Plot individual models
    X_plot = np.linspace(X.min(), X.max(), 300)
    
    for idx, r in enumerate(results[:6]):
        ax = plt.subplot(2, 3, idx + 1)
        
        # Plot data
        ax.scatter(X_train, y_train, alpha=0.4, s=20, label='Train Data', color='blue')
        ax.scatter(X_test, y_test, alpha=0.4, s=20, label='Test Data', color='red')
        
        # Plot true function
        y_true_plot = 2 + 0.5 * X_plot + 0.3 * X_plot**2
        ax.plot(X_plot, y_true_plot, 'g--', linewidth=2, 
                label='True Function', alpha=0.7)
        
        # Plot model predictions
        X_plot_poly = r['poly'].transform(X_plot.reshape(-1, 1))
        y_plot_pred = r['model'].predict(X_plot_poly)
        ax.plot(X_plot, y_plot_pred, 'purple', linewidth=2.5, 
                label=f'Degree {r["degree"]} Model')
        
        ax.set_title(f'Polynomial Degree {r["degree"]}\n'
                    f'Train MSE: {r["train_mse"]:.2f}, Test MSE: {r["test_mse"]:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([y.min() - 5, y.max() + 5])
    
    plt.tight_layout()
    plt.savefig('examples/bias_variance_models.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: examples/bias_variance_models.png")
    
    # Plot learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE vs Model Complexity
    train_mses = [r['train_mse'] for r in results]
    test_mses = [r['test_mse'] for r in results]
    degrees_list = [r['degree'] for r in results]
    
    axes[0].plot(degrees_list, train_mses, 'o-', linewidth=2, 
                markersize=8, label='Training MSE', color='blue')
    axes[0].plot(degrees_list, test_mses, 's-', linewidth=2, 
                markersize=8, label='Test MSE', color='red')
    axes[0].set_xlabel('Model Complexity (Polynomial Degree)')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Bias-Variance Tradeoff')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Annotate regions
    axes[0].axvspan(0, 2.5, alpha=0.2, color='orange', label='High Bias')
    axes[0].axvspan(7, 16, alpha=0.2, color='purple', label='High Variance')
    axes[0].text(1.5, max(test_mses) * 0.9, 'Underfitting\n(High Bias)', 
                ha='center', fontsize=10, fontweight='bold')
    axes[0].text(12, max(test_mses) * 0.9, 'Overfitting\n(High Variance)', 
                ha='center', fontsize=10, fontweight='bold')
    
    # Decomposition illustration
    categories = ['Squared Bias', 'Variance', 'Noise']
    simple_model = [40, 5, 10]  # High bias, low variance
    complex_model = [5, 50, 10]  # Low bias, high variance
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, simple_model, width, 
                       label='Simple Model (Degree 1)', color='#2E86AB', alpha=0.7)
    bars2 = axes[1].bar(x + width/2, complex_model, width, 
                       label='Complex Model (Degree 15)', color='#A23B72', alpha=0.7)
    
    axes[1].set_xlabel('Error Component')
    axes[1].set_ylabel('Contribution to Total Error')
    axes[1].set_title('Error Decomposition: Bias² + Variance + Noise')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('examples/bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: examples/bias_variance_tradeoff.png")


def demonstrate_error_decomposition():
    """Mathematical demonstration of error decomposition"""
    print("\n" + "=" * 60)
    print("ERROR DECOMPOSITION FORMULA")
    print("=" * 60)
    
    print("""
The expected prediction error can be decomposed as:

E[(y - f̂(x))²] = Bias²[f̂(x)] + Var[f̂(x)] + σ²

Where:
  • Bias²[f̂(x)] = (E[f̂(x)] - f(x))²
    Measures how far the average prediction is from the true value
    
  • Var[f̂(x)] = E[(f̂(x) - E[f̂(x)])²]
    Measures how much predictions vary for different training sets
    
  • σ² = Irreducible error (noise in the data)

Key Insights:
  • High Bias → Underfitting → Model too simple
  • High Variance → Overfitting → Model too complex
  • Goal: Find the sweet spot that minimizes total error
    """)


if __name__ == "__main__":
    print("Bias-Variance Tradeoff Analysis")
    print("=" * 60)
    
    demonstrate_bias_variance_tradeoff()
    demonstrate_error_decomposition()
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
