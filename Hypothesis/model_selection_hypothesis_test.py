"""
Model Selection Using Hypothesis Testing and Statistical Inference

This script demonstrates how to use statistical hypothesis testing to compare
two machine learning models and determine whether their performance difference
is statistically significant.

Statistical Framework:
- Null Hypothesis (H0): There is no significant difference in performance 
  between Model A and Model B
- Alternative Hypothesis (H1): There is a statistically significant difference 
  in performance
- Significance Level (α): 0.05

Methodology:
We use a Paired T-Test because both models are evaluated on the same data splits
(via cross-validation), making the performance samples paired/dependent.
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

def main():
    print("=" * 70)
    print("Model Selection Using Statistical Hypothesis Testing")
    print("=" * 70)
    print()
    
    # 1. Load data
    print("Step 1: Loading Iris dataset...")
    data = load_iris()
    X, y = data.data, data.target
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    print()
    
    # 2. Define models
    print("Step 2: Defining models...")
    model_a = RandomForestClassifier(n_estimators=10, random_state=42)
    model_b = LogisticRegression(max_iter=200, random_state=42)
    print(f"   Model A: {model_a.__class__.__name__}")
    print(f"   Model B: {model_b.__class__.__name__}")
    print()
    
    # 3. Obtain performance samples (e.g., Accuracy across 10-fold CV)
    # Using cross_val_score provides the distribution needed for inference
    print("Step 3: Performing 10-fold cross-validation...")
    print("   This creates a distribution of performance scores for each model")
    scores_a = cross_val_score(model_a, X, y, cv=10, scoring='accuracy')
    scores_b = cross_val_score(model_b, X, y, cv=10, scoring='accuracy')
    print(f"   Model A scores: {scores_a}")
    print(f"   Model B scores: {scores_b}")
    print()
    
    # 4. Statistical Inference: Paired T-Test
    # Checks if the mean difference in accuracy is significantly different from zero
    print("Step 4: Performing statistical hypothesis test...")
    print()
    print("   Hypotheses:")
    print("   H0 (Null): μA - μB = 0 (no difference in performance)")
    print("   H1 (Alternative): μA - μB ≠ 0 (significant difference exists)")
    print()
    
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Calculate additional statistics
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    std_a = np.std(scores_a, ddof=1)
    std_b = np.std(scores_b, ddof=1)
    mean_diff = mean_a - mean_b
    
    # 5. Report results
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    print()
    print(f"Model A ({model_a.__class__.__name__}):")
    print(f"   Mean Accuracy: {mean_a:.4f}")
    print(f"   Std Deviation: {std_a:.4f}")
    print(f"   95% CI: [{mean_a - 1.96*std_a/np.sqrt(10):.4f}, "
          f"{mean_a + 1.96*std_a/np.sqrt(10):.4f}]")
    print()
    
    print(f"Model B ({model_b.__class__.__name__}):")
    print(f"   Mean Accuracy: {mean_b:.4f}")
    print(f"   Std Deviation: {std_b:.4f}")
    print(f"   95% CI: [{mean_b - 1.96*std_b/np.sqrt(10):.4f}, "
          f"{mean_b + 1.96*std_b/np.sqrt(10):.4f}]")
    print()
    
    print("Statistical Test Results:")
    print(f"   Test Statistic (t): {t_stat:.4f}")
    print(f"   P-Value: {p_value:.4f}")
    print(f"   Mean Difference: {mean_diff:.4f}")
    print(f"   Significance Level (α): 0.05")
    print()
    
    print("=" * 70)
    print("INFERENCE AND CONCLUSION")
    print("=" * 70)
    
    alpha = 0.05
    if p_value < alpha:
        print(f"✓ P-value ({p_value:.4f}) < α ({alpha})")
        print("✓ Decision: REJECT the null hypothesis (H0)")
        print()
        print("Conclusion:")
        print(f"   The performance difference between {model_a.__class__.__name__}")
        print(f"   and {model_b.__class__.__name__} is STATISTICALLY SIGNIFICANT.")
        print()
        
        if mean_a > mean_b:
            print(f"   {model_a.__class__.__name__} outperforms "
                  f"{model_b.__class__.__name__}")
            print(f"   by an average of {abs(mean_diff):.4f} accuracy points.")
        else:
            print(f"   {model_b.__class__.__name__} outperforms "
                  f"{model_a.__class__.__name__}")
            print(f"   by an average of {abs(mean_diff):.4f} accuracy points.")
    else:
        print(f"✗ P-value ({p_value:.4f}) ≥ α ({alpha})")
        print("✗ Decision: FAIL TO REJECT the null hypothesis (H0)")
        print()
        print("Conclusion:")
        print(f"   There is NO statistically significant difference in performance")
        print(f"   between {model_a.__class__.__name__} and "
              f"{model_b.__class__.__name__}.")
        print("   The observed difference could be due to random variation.")
    
    print()
    print("=" * 70)
    print()
    
    # Additional interpretation
    print("INTERPRETATION NOTES:")
    print("-" * 70)
    print("• The paired t-test compares the MEAN difference in accuracy")
    print("  across the 10 cross-validation folds.")
    print()
    print("• P-value interpretation:")
    print("  - P-value is the probability of observing a difference at least")
    print("    as extreme as ours, assuming H0 is true.")
    print(f"  - Small p-value (< {alpha}) = strong evidence against H0")
    print(f"  - Large p-value (≥ {alpha}) = insufficient evidence against H0")
    print()
    print("• Statistical significance ≠ Practical significance")
    print("  - A statistically significant difference might be too small")
    print("    to matter in practice.")
    print("  - Always consider domain knowledge and business requirements!")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
