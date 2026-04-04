"""
Conditional Probability and Bayes' Theorem Examples
Demonstrates joint probabilities, conditional probabilities, and Bayesian inference
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def joint_and_conditional_probabilities():
    """Demonstrate joint and conditional probability calculations"""
    print("=" * 60)
    print("JOINT AND CONDITIONAL PROBABILITIES")
    print("=" * 60)
    
    # Example: Boolean arrays for two events
    np.random.seed(42)
    event_A = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    event_B = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    
    print("\nEvent A:", event_A)
    print("Event B:", event_B)
    
    # Calculate marginal probabilities
    p_A = np.mean(event_A == 1)
    p_B = np.mean(event_B == 1)
    
    # Calculate joint probability P(A and B)
    joint_prob = np.mean((event_A == 1) & (event_B == 1))
    
    # Calculate conditional probabilities
    # P(A|B) = P(A and B) / P(B)
    conditional_A_given_B = joint_prob / p_B if p_B > 0 else 0
    
    # Alternative method: filter where B is true, then find mean of A
    conditional_A_given_B_alt = event_A[event_B == 1].mean()
    
    # P(B|A) = P(A and B) / P(A)
    conditional_B_given_A = joint_prob / p_A if p_A > 0 else 0
    
    print("\n" + "-" * 60)
    print("PROBABILITY CALCULATIONS:")
    print("-" * 60)
    print(f"P(A) = {p_A:.4f}")
    print(f"P(B) = {p_B:.4f}")
    print(f"P(A and B) = {joint_prob:.4f}")
    print(f"P(A|B) = {conditional_A_given_B:.4f}")
    print(f"P(A|B) [alternative method] = {conditional_A_given_B_alt:.4f}")
    print(f"P(B|A) = {conditional_B_given_A:.4f}")
    
    # Check independence: A and B are independent if P(A and B) = P(A) * P(B)
    expected_if_independent = p_A * p_B
    print(f"\nIf independent, P(A and B) would be: {expected_if_independent:.4f}")
    print(f"Actual P(A and B): {joint_prob:.4f}")
    print(f"Events are {'independent' if abs(joint_prob - expected_if_independent) < 0.01 else 'dependent'}")


def contingency_table_example():
    """Demonstrate probability calculations with a contingency table"""
    print("\n" + "=" * 60)
    print("CONTINGENCY TABLE EXAMPLE")
    print("=" * 60)
    
    # Example: Customer purchase data
    # Rows: Previously purchased (Yes/No)
    # Columns: Received email (Yes/No)
    data = {
        'Email_Yes': [120, 80],  # Purchased before: Yes, No
        'Email_No': [30, 170]
    }
    
    df = pd.DataFrame(data, index=['Purchased_Before', 'Not_Purchased_Before'])
    print("\nContingency Table:")
    print(df)
    
    total = df.sum().sum()
    print(f"\nTotal customers: {total}")
    
    # Calculate probabilities
    p_email = df['Email_Yes'].sum() / total
    p_no_email = df['Email_No'].sum() / total
    p_purchased_before = df.loc['Purchased_Before'].sum() / total
    p_not_purchased = df.loc['Not_Purchased_Before'].sum() / total
    
    # Joint probabilities
    p_email_and_purchased = df.loc['Purchased_Before', 'Email_Yes'] / total
    
    # Conditional probability: P(Purchased Before | Received Email)
    p_purchased_given_email = df.loc['Purchased_Before', 'Email_Yes'] / df['Email_Yes'].sum()
    
    # Conditional probability: P(Purchased Before | No Email)
    p_purchased_given_no_email = df.loc['Purchased_Before', 'Email_No'] / df['Email_No'].sum()
    
    print("\n" + "-" * 60)
    print("PROBABILITY CALCULATIONS:")
    print("-" * 60)
    print(f"P(Received Email) = {p_email:.4f}")
    print(f"P(Purchased Before) = {p_purchased_before:.4f}")
    print(f"P(Email and Purchased) = {p_email_and_purchased:.4f}")
    print(f"P(Purchased | Email) = {p_purchased_given_email:.4f}")
    print(f"P(Purchased | No Email) = {p_purchased_given_no_email:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Stacked bar chart
    df.T.plot(kind='bar', stacked=True, ax=axes[0], 
              color=['#2E86AB', '#A23B72'], edgecolor='black')
    axes[0].set_title('Customer Purchase History by Email Status')
    axes[0].set_xlabel('Email Received')
    axes[0].set_ylabel('Number of Customers')
    axes[0].legend(title='Purchase History')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Conditional probabilities
    categories = ['Received Email', 'No Email']
    purchased_rates = [p_purchased_given_email, p_purchased_given_no_email]
    
    axes[1].bar(categories, purchased_rates, color=['#2E86AB', '#A23B72'], 
                edgecolor='black', alpha=0.7)
    axes[1].set_title('P(Purchased Before | Email Status)')
    axes[1].set_ylabel('Probability')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(purchased_rates):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('examples/conditional_probability.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: examples/conditional_probability.png")


def bayes_theorem_example():
    """Demonstrate Bayes' Theorem with a medical test example"""
    print("\n" + "=" * 60)
    print("BAYES' THEOREM: MEDICAL TEST EXAMPLE")
    print("=" * 60)
    
    # Problem: A disease affects 1% of the population
    # Test accuracy: 95% true positive rate, 90% true negative rate
    
    p_disease = 0.01  # Prior: P(Disease)
    p_no_disease = 1 - p_disease  # P(No Disease)
    
    p_positive_given_disease = 0.95  # Sensitivity: P(Positive | Disease)
    p_negative_given_no_disease = 0.90  # Specificity: P(Negative | No Disease)
    p_positive_given_no_disease = 1 - p_negative_given_no_disease  # False positive rate
    
    # Apply Bayes' Theorem: P(Disease | Positive Test)
    # P(Disease | Positive) = P(Positive | Disease) * P(Disease) / P(Positive)
    
    # Calculate P(Positive) using law of total probability
    p_positive = (p_positive_given_disease * p_disease + 
                  p_positive_given_no_disease * p_no_disease)
    
    # Posterior probability
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
    
    print("\nGiven Information:")
    print(f"  P(Disease) = {p_disease:.4f} (1% of population)")
    print(f"  P(Positive | Disease) = {p_positive_given_disease:.4f} (Sensitivity)")
    print(f"  P(Negative | No Disease) = {p_negative_given_no_disease:.4f} (Specificity)")
    
    print("\nCalculations:")
    print(f"  P(Positive) = {p_positive:.4f}")
    print(f"  P(Disease | Positive Test) = {p_disease_given_positive:.4f}")
    
    print(f"\nInterpretation:")
    print(f"  Even with a positive test, there's only a {p_disease_given_positive*100:.2f}% chance")
    print(f"  of actually having the disease due to its low prevalence.")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prior vs Posterior
    categories = ['Prior\nP(Disease)', 'Posterior\nP(Disease|Positive)']
    probabilities = [p_disease, p_disease_given_positive]
    colors = ['#E63946', '#06FFA5']
    
    bars = axes[0].bar(categories, probabilities, color=colors, 
                       edgecolor='black', alpha=0.7, width=0.6)
    axes[0].set_ylabel('Probability')
    axes[0].set_title("Bayes' Theorem: Updating Beliefs with Evidence")
    axes[0].set_ylim([0, 0.12])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    f'{prob:.4f}\n({prob*100:.2f}%)',
                    ha='center', va='bottom', fontweight='bold')
    
    # Flow diagram visualization
    axes[1].axis('off')
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    
    # Add text explanation
    explanation = [
        "Bayes' Theorem Formula:",
        "",
        "P(Disease|Positive) = ",
        "    P(Positive|Disease) × P(Disease)",
        "    ────────────────────────────────",
        "              P(Positive)",
        "",
        f"= {p_positive_given_disease} × {p_disease}",
        f"  ─────────────",
        f"     {p_positive:.4f}",
        "",
        f"= {p_disease_given_positive:.4f}"
    ]
    
    y_pos = 9
    for line in explanation:
        axes[1].text(1, y_pos, line, fontsize=11, family='monospace',
                    verticalalignment='top')
        y_pos -= 0.6
    
    plt.tight_layout()
    plt.savefig('examples/bayes_theorem.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: examples/bayes_theorem.png")


def bayesian_updating_example():
    """Demonstrate sequential Bayesian updating"""
    print("\n" + "=" * 60)
    print("BAYESIAN UPDATING: COIN FLIP EXAMPLE")
    print("=" * 60)
    
    # Scenario: Estimating probability of heads for an unknown coin
    # Start with uniform prior, update as we observe flips
    
    observations = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]  # 1=heads, 0=tails
    
    # Beta distribution parameters (conjugate prior for Binomial)
    alpha = 1  # Prior "successes"
    beta = 1   # Prior "failures"
    
    priors = [(alpha, beta)]
    
    print("\nObserved coin flips:", ['H' if x == 1 else 'T' for x in observations])
    print("\nBayesian Updates:")
    print(f"Initial prior: Beta(α={alpha}, β={beta}) - Uniform distribution")
    
    for i, obs in enumerate(observations, 1):
        if obs == 1:
            alpha += 1
        else:
            beta += 1
        priors.append((alpha, beta))
        
        mean = alpha / (alpha + beta)
        print(f"After flip {i}: Beta(α={alpha}, β={beta}), E[p] = {mean:.4f}")
    
    # Visualization
    x = np.linspace(0, 1, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    steps_to_plot = [0, 2, 5, 10]
    
    for idx, step in enumerate(steps_to_plot):
        a, b = priors[step]
        pdf = stats.beta.pdf(x, a, b)
        
        axes[idx].plot(x, pdf, 'b-', linewidth=2)
        axes[idx].fill_between(x, pdf, alpha=0.3)
        axes[idx].axvline(a/(a+b), color='red', linestyle='--', 
                         linewidth=2, label=f'Mean = {a/(a+b):.3f}')
        axes[idx].set_title(f'After {step} observations: Beta(α={a}, β={b})')
        axes[idx].set_xlabel('Probability of Heads (p)')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig('examples/bayesian_updating.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: examples/bayesian_updating.png")


if __name__ == "__main__":
    print("Conditional Probability and Bayes' Theorem Demonstration")
    print("=" * 60)
    
    joint_and_conditional_probabilities()
    contingency_table_example()
    bayes_theorem_example()
    bayesian_updating_example()
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)
