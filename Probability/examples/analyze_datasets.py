"""
Example: Using Generated Datasets for Probability Analysis
This script demonstrates how to load and analyze the generated datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def analyze_customer_purchases():
    """Analyze customer purchase data for conditional probability"""
    print("=" * 70)
    print("CONDITIONAL PROBABILITY: Customer Purchase Analysis")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv('data/sample_datasets/customer_purchases.csv')
    
    print(f"\nDataset loaded: {len(df)} customers")
    print(f"Columns: {list(df.columns)}")
    
    # Calculate probabilities
    p_purchased_before = (df['purchased_before'] == 'Yes').mean()
    p_received_email = (df['received_email'] == 'Yes').mean()
    p_made_purchase = (df['made_purchase'] == 'Yes').mean()
    
    print("\n" + "-" * 70)
    print("MARGINAL PROBABILITIES:")
    print("-" * 70)
    print(f"P(Purchased Before) = {p_purchased_before:.4f}")
    print(f"P(Received Email) = {p_received_email:.4f}")
    print(f"P(Made Purchase) = {p_made_purchase:.4f}")
    
    # Conditional probabilities
    email_customers = df[df['received_email'] == 'Yes']
    no_email_customers = df[df['received_email'] == 'No']
    
    p_purchase_given_email = (email_customers['made_purchase'] == 'Yes').mean()
    p_purchase_given_no_email = (no_email_customers['made_purchase'] == 'Yes').mean()
    
    print("\n" + "-" * 70)
    print("CONDITIONAL PROBABILITIES:")
    print("-" * 70)
    print(f"P(Purchase | Received Email) = {p_purchase_given_email:.4f}")
    print(f"P(Purchase | No Email) = {p_purchase_given_no_email:.4f}")
    print(f"Lift from email marketing: {p_purchase_given_email/p_purchase_given_no_email:.2f}x")
    
    # Create contingency table
    contingency = pd.crosstab(df['received_email'], df['made_purchase'], margins=True)
    print("\n" + "-" * 70)
    print("CONTINGENCY TABLE:")
    print("-" * 70)
    print(contingency)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Purchase rate by email status
    purchase_rates = df.groupby('received_email')['made_purchase'].apply(
        lambda x: (x == 'Yes').mean()
    )
    purchase_rates.plot(kind='bar', ax=axes[0], color=['#E63946', '#06FFA5'], 
                        edgecolor='black', alpha=0.7)
    axes[0].set_title('Purchase Rate by Email Status', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Received Email')
    axes[0].set_ylabel('Purchase Rate')
    axes[0].set_ylim([0, purchase_rates.max() * 1.2])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(purchase_rates):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Purchase amount distribution
    purchase_amounts = df[df['made_purchase'] == 'Yes']['purchase_amount']
    axes[1].hist(purchase_amounts, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(purchase_amounts.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: ${purchase_amounts.mean():.2f}')
    axes[1].set_title('Distribution of Purchase Amounts', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Purchase Amount ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('examples/customer_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: examples/customer_analysis.png")


def analyze_test_scores():
    """Analyze test scores for normal distribution"""
    print("\n\n" + "=" * 70)
    print("NORMAL DISTRIBUTION: Test Score Analysis")
    print("=" * 70)
    
    df = pd.read_csv('data/sample_datasets/student_test_scores.csv')
    
    scores = df['test_score']
    
    # Calculate statistics
    mean = scores.mean()
    std = scores.std()
    median = scores.median()
    
    print(f"\nDataset loaded: {len(df)} students")
    print("\n" + "-" * 70)
    print("DESCRIPTIVE STATISTICS:")
    print("-" * 70)
    print(f"Mean: {mean:.2f}")
    print(f"Standard Deviation: {std:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Pass Rate (≥60): {(scores >= 60).mean():.2%}")
    
    # Test for normality
    statistic, p_value = stats.normaltest(scores)
    print(f"\nNormality Test (D'Agostino-Pearson):")
    print(f"  Statistic: {statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Conclusion: {'Normally distributed' if p_value > 0.05 else 'Not normally distributed'}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with normal curve
    axes[0].hist(scores, bins=30, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black', label='Observed')
    x = np.linspace(scores.min(), scores.max(), 100)
    axes[0].plot(x, stats.norm.pdf(x, mean, std), 'r-', 
                linewidth=2, label=f'Normal(μ={mean:.1f}, σ={std:.1f})')
    axes[0].axvline(mean, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}')
    axes[0].set_title('Test Score Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Test Score')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(scores, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Test for Normality)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/test_scores_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: examples/test_scores_analysis.png")


def analyze_website_traffic():
    """Analyze website traffic for Poisson distribution"""
    print("\n\n" + "=" * 70)
    print("POISSON DISTRIBUTION: Website Traffic Analysis")
    print("=" * 70)
    
    df = pd.read_csv('data/sample_datasets/website_traffic.csv')
    
    visitors = df['visitors']
    hourly_mean = visitors.mean()
    
    print(f"\nDataset loaded: {len(df)} observations (hourly data)")
    print(f"Average visitors per hour: {hourly_mean:.2f}")
    
    # Test if it follows Poisson
    print("\n" + "-" * 70)
    print("POISSON DISTRIBUTION TEST:")
    print("-" * 70)
    print(f"Mean: {visitors.mean():.2f}")
    print(f"Variance: {visitors.var():.2f}")
    print(f"Variance/Mean ratio: {visitors.var()/visitors.mean():.2f}")
    print("(For Poisson, variance should equal mean)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution
    counts, bins, _ = axes[0].hist(visitors, bins=40, density=True, 
                                   alpha=0.7, color='orange', edgecolor='black')
    
    # Overlay Poisson PMF
    x_poisson = np.arange(0, visitors.max())
    axes[0].plot(x_poisson, stats.poisson.pmf(x_poisson, hourly_mean), 
                'ro-', linewidth=2, markersize=6, label=f'Poisson(λ={hourly_mean:.1f})')
    axes[0].set_title('Hourly Visitor Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Visitors')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Traffic by hour
    hourly_avg = df.groupby('hour')['visitors'].mean()
    axes[1].plot(hourly_avg.index, hourly_avg.values, marker='o', 
                linewidth=2, markersize=8, color='purple')
    axes[1].fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3)
    axes[1].set_title('Average Traffic by Hour of Day', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Average Visitors')
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/website_traffic_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: examples/website_traffic_analysis.png")


if __name__ == "__main__":
    print("Dataset Analysis Examples")
    print("=" * 70)
    
    analyze_customer_purchases()
    analyze_test_scores()
    analyze_website_traffic()
    
    print("\n\n" + "=" * 70)
    print("ALL ANALYSES COMPLETED!")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  • examples/customer_analysis.png")
    print("  • examples/test_scores_analysis.png")
    print("  • examples/website_traffic_analysis.png")
    print("=" * 70)
