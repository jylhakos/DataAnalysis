"""
McNemar's Test for Model Comparison on a Single Test Set

McNemar's test is particularly valuable for comparing models on a single test set
without retraining. It's a non-parametric test that focuses on the
discordant cases where models make different predictions.

This is especially useful for:
- Comparing large, computationally expensive models (e.g., deep learning)
- When you have limited computational resources
- When cross-validation is not feasible
"""

import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    Perform McNemar's test to compare two classifiers.
    
    McNemar's test focuses on the 2x2 contingency table of disagreements:
    
                    Model B Correct | Model B Wrong
    Model A Correct       a         |      b
    Model A Wrong         c         |      d
    
    The test statistic focuses only on b and c (discordant cases).
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_a : array-like
        Predictions from Model A
    y_pred_b : array-like
        Predictions from Model B
    
    Returns:
    --------
    dict : Test results including contingency table and p-value
    """
    # Build contingency table
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    
    # Count outcomes
    a = np.sum(a_correct & b_correct)      # Both correct
    b = np.sum(a_correct & ~b_correct)     # A correct, B wrong
    c = np.sum(~a_correct & b_correct)     # A wrong, B correct
    d = np.sum(~a_correct & ~b_correct)    # Both wrong
    
    # McNemar's test statistic (with continuity correction)
    # Only considers discordant cases (b and c)
    if b + c == 0:
        # No disagreements - models perform identically
        chi2_stat = 0
        p_value = 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
    
    results = {
        'contingency_table': {
            'both_correct': a,
            'a_correct_b_wrong': b,
            'a_wrong_b_correct': c,
            'both_wrong': d
        },
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'total_disagreements': b + c
    }
    
    return results


def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy."""
    return np.mean(y_true == y_pred)


def main():
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "McNEMAR'S TEST FOR MODEL COMPARISON" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Load and split data
    print("Step 1: Loading and splitting dataset...")
    data = load_iris()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print()
    
    # Train models
    print("Step 2: Training models...")
    
    model_a = RandomForestClassifier(n_estimators=20, random_state=42)
    model_b = LogisticRegression(max_iter=200, random_state=42)
    
    print(f"   Model A: {model_a.__class__.__name__}")
    model_a.fit(X_train, y_train)
    print("      ✓ Trained")
    
    print(f"   Model B: {model_b.__class__.__name__}")
    model_b.fit(X_train, y_train)
    print("      ✓ Trained")
    print()
    
    # Make predictions on the SAME test set
    print("Step 3: Making predictions on test set...")
    y_pred_a = model_a.predict(X_test)
    y_pred_b = model_b.predict(X_test)
    
    acc_a = calculate_accuracy(y_test, y_pred_a)
    acc_b = calculate_accuracy(y_test, y_pred_b)
    
    print(f"   Model A Accuracy: {acc_a:.4f}")
    print(f"   Model B Accuracy: {acc_b:.4f}")
    print()
    
    # Perform McNemar's test
    print("Step 4: Performing McNemar's Test...")
    print()
    print("   Hypotheses:")
    print("   H0: The two classifiers have similar error rates")
    print("   H1: The two classifiers have different error rates")
    print()
    
    results = mcnemar_test(y_test, y_pred_a, y_pred_b)
    
    # Display contingency table
    print("=" * 80)
    print("CONTINGENCY TABLE")
    print("=" * 80)
    print()
    
    ct = results['contingency_table']
    
    print("                           Model B Correct    Model B Wrong")
    print(f"   Model A Correct             {ct['both_correct']:5d}            {ct['a_correct_b_wrong']:5d}")
    print(f"   Model A Wrong               {ct['a_wrong_b_correct']:5d}            {ct['both_wrong']:5d}")
    print()
    
    print("Key Values:")
    print(f"   Both correct (a): {ct['both_correct']}")
    print(f"   A correct, B wrong (b): {ct['a_correct_b_wrong']}")
    print(f"   A wrong, B correct (c): {ct['a_wrong_b_correct']}")
    print(f"   Both wrong (d): {ct['both_wrong']}")
    print()
    print(f"   Total disagreements (b + c): {results['total_disagreements']}")
    print()
    
    # Explain the test
    print("=" * 80)
    print("McNEMAR'S TEST EXPLANATION")
    print("=" * 80)
    print()
    print("McNemar's test focuses ONLY on the discordant cases (b and c):")
    print()
    print("• b = Cases where Model A is correct but Model B is wrong")
    print("• c = Cases where Model B is correct but Model A is wrong")
    print()
    print("If the models have similar error rates, we expect b ≈ c.")
    print("If b ≠ c significantly, one model is systematically better.")
    print()
    print(f"Test Statistic (χ² with continuity correction):")
    print(f"   χ² = (|b - c| - 1)² / (b + c)")
    print(f"   χ² = (|{ct['a_correct_b_wrong']} - {ct['a_wrong_b_correct']}| - 1)² / ({ct['a_correct_b_wrong']} + {ct['a_wrong_b_correct']})")
    print(f"   χ² = {results['chi2_statistic']:.4f}")
    print()
    
    # Test results
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print()
    print(f"Chi-square statistic: {results['chi2_statistic']:.4f}")
    print(f"Degrees of freedom: 1")
    print(f"P-value: {results['p_value']:.4f}")
    print(f"Significance level (α): 0.05")
    print()
    
    # Interpretation
    print("=" * 80)
    print("INFERENCE AND CONCLUSION")
    print("=" * 80)
    print()
    
    alpha = 0.05
    
    if results['p_value'] < alpha:
        print(f"✓ P-value ({results['p_value']:.4f}) < α ({alpha})")
        print("✓ Decision: REJECT the null hypothesis")
        print()
        print("Conclusion:")
        print("   The two classifiers have STATISTICALLY SIGNIFICANT differences")
        print("   in their error patterns.")
        print()
        
        if ct['a_correct_b_wrong'] > ct['a_wrong_b_correct']:
            print(f"   Model A ({model_a.__class__.__name__}) is superior:")
            print(f"   - A is correct while B is wrong: {ct['a_correct_b_wrong']} times")
            print(f"   - B is correct while A is wrong: {ct['a_wrong_b_correct']} times")
        elif ct['a_wrong_b_correct'] > ct['a_correct_b_wrong']:
            print(f"   Model B ({model_b.__class__.__name__}) is superior:")
            print(f"   - B is correct while A is wrong: {ct['a_wrong_b_correct']} times")
            print(f"   - A is correct while B is wrong: {ct['a_correct_b_wrong']} times")
    else:
        print(f"✗ P-value ({results['p_value']:.4f}) ≥ α ({alpha})")
        print("✗ Decision: FAIL TO REJECT the null hypothesis")
        print()
        print("Conclusion:")
        print("   There is NO statistically significant difference in the error")
        print("   patterns of the two classifiers.")
        print()
        print("   The models perform similarly, and the observed difference in")
        print("   accuracy could be due to random variation.")
    
    print()
    print("=" * 80)
    print()
    
    # Additional insights
    print("KEY INSIGHTS ABOUT McNEMAR'S TEST:")
    print("-" * 80)
    print()
    print("✓ Advantages:")
    print("  • Only requires a SINGLE test set (no cross-validation needed)")
    print("  • Computationally efficient for large models")
    print("  • Non-parametric (no normality assumption)")
    print("  • Focuses on where models disagree (discordant cases)")
    print()
    print("✗ Limitations:")
    print("  • Requires both models trained on SAME training data")
    print("  • Requires both models evaluated on SAME test instances")
    print("  • Only compares TWO models at a time")
    print("  • Less powerful with very few disagreements")
    print()
    print("When to use McNemar's test:")
    print("  • Comparing expensive models (e.g., deep learning)")
    print("  • Limited computational resources")
    print("  • Single train/test split scenario")
    print("  • Paired predictions on same instances")
    print()
    print("When NOT to use McNemar's test:")
    print("  • Models trained on different data")
    print("  • Comparing > 2 models (use Cochran's Q instead)")
    print("  • When cross-validation is feasible (use paired t-test instead)")
    print()
    print("=" * 80)
    print()
    
    # Comparison with other methods
    print("COMPARISON WITH OTHER METHODS:")
    print("-" * 80)
    print()
    print("Method                  | Data Requirements       | Use Case")
    print("-" * 80)
    print("Paired T-Test          | CV scores from both     | General model comparison")
    print("                        | models                  | with cross-validation")
    print()
    print("McNemar's Test         | Single test set         | Comparing models on")
    print("                        | predictions from both   | one fixed test set")
    print()
    print("5x2 CV Paired T-Test   | 5 repetitions of 2-fold | More robust than")
    print("                        | CV                      | standard paired t-test")
    print()
    print("Cochran's Q Test       | Single test set         | Comparing > 2 models")
    print("                        | predictions from all    | simultaneously")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
