"""
Hyperparameter Tuning with Statistical Validation

This script demonstrates:
1. Using GridSearchCV for systematic hyperparameter tuning
2. Statistically validating whether hyperparameter changes improve performance
3. Comparing baseline vs optimized model using hypothesis testing
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def evaluate_with_cv(model, X, y, cv=10):
    """
    Evaluate a model using cross-validation.
    
    Parameters:
    -----------
    model : estimator
        Scikit-learn model
    X : array-like
        Features
    y : array-like
        Target
    cv : int
        Number of CV folds
    
    Returns:
    --------
    array : Cross-validation scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores


def compare_models_statistically(baseline_scores, optimized_scores, alpha=0.05):
    """
    Perform statistical comparison between baseline and optimized model.
    
    Parameters:
    -----------
    baseline_scores : array-like
        CV scores for baseline model
    optimized_scores : array-like
        CV scores for optimized model
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Statistical test results
    """
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(optimized_scores, baseline_scores)
    
    # Compute effect size (Cohen's d for paired samples)
    mean_diff = np.mean(optimized_scores - baseline_scores)
    std_diff = np.std(optimized_scores - baseline_scores, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    results = {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'cohens_d': cohens_d,
        'significant': p_value < alpha
    }
    
    return results


def interpret_effect_size(cohens_d):
    """
    Interpret Cohen's d effect size.
    
    Parameters:
    -----------
    cohens_d : float
        Cohen's d value
    
    Returns:
    --------
    str : Interpretation
    """
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def main():
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "HYPERPARAMETER TUNING WITH STATISTICAL VALIDATION" + " " * 10 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Load data
    print("Step 1: Loading dataset...")
    data = load_iris()
    X, y = data.data, data.target
    print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print()
    
    # Define baseline model
    print("Step 2: Defining baseline model...")
    baseline_model = RandomForestClassifier(
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    print(f"   Baseline parameters: {baseline_model.get_params()}")
    print()
    
    # Evaluate baseline model
    print("Step 3: Evaluating baseline model with 10-fold CV...")
    baseline_scores = evaluate_with_cv(baseline_model, X, y, cv=10)
    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores, ddof=1)
    print(f"   Baseline Mean Accuracy: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"   Baseline Scores: {baseline_scores}")
    print()
    
    # Define hyperparameter grid
    print("Step 4: Defining hyperparameter grid for tuning...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    total_combinations = (len(param_grid['n_estimators']) * 
                         len(param_grid['max_depth']) * 
                         len(param_grid['min_samples_split']) * 
                         len(param_grid['min_samples_leaf']))
    
    print(f"   Total hyperparameter combinations: {total_combinations}")
    print(f"   Parameter grid: {param_grid}")
    print()
    
    # Perform grid search
    print("Step 5: Performing Grid Search with 5-fold CV...")
    print("   (This may take a moment...)")
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X, y)
    
    print(f"   ✓ Grid search complete!")
    print()
    
    # Display best parameters
    print("Step 6: Best hyperparameters found:")
    print("=" * 80)
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print()
    print(f"   Best CV Score (from Grid Search): {grid_search.best_score_:.4f}")
    print()
    
    # Evaluate optimized model with same CV as baseline
    print("Step 7: Evaluating optimized model with 10-fold CV...")
    optimized_model = grid_search.best_estimator_
    optimized_scores = evaluate_with_cv(optimized_model, X, y, cv=10)
    optimized_mean = np.mean(optimized_scores)
    optimized_std = np.std(optimized_scores, ddof=1)
    print(f"   Optimized Mean Accuracy: {optimized_mean:.4f} ± {optimized_std:.4f}")
    print(f"   Optimized Scores: {optimized_scores}")
    print()
    
    # Statistical comparison
    print("Step 8: Statistical Hypothesis Testing")
    print("=" * 80)
    print()
    print("Hypotheses:")
    print("   H0: Hyperparameter tuning does NOT improve performance")
    print("       (μ_optimized - μ_baseline ≤ 0)")
    print("   H1: Hyperparameter tuning DOES improve performance")
    print("       (μ_optimized - μ_baseline > 0)")
    print()
    
    test_results = compare_models_statistically(baseline_scores, optimized_scores, alpha=0.05)
    
    print("Statistical Test Results:")
    print("-" * 80)
    print(f"   T-statistic: {test_results['t_statistic']:.4f}")
    print(f"   P-value: {test_results['p_value']:.4f}")
    print(f"   Mean Improvement: {test_results['mean_difference']:.4f}")
    print(f"   Cohen's d (effect size): {test_results['cohens_d']:.4f}")
    print(f"   Effect size interpretation: {interpret_effect_size(test_results['cohens_d'])}")
    print()
    
    # Interpretation
    print("=" * 80)
    print("INFERENCE AND CONCLUSION")
    print("=" * 80)
    print()
    
    if test_results['significant']:
        print(f"✓ P-value ({test_results['p_value']:.4f}) < α (0.05)")
        print("✓ Decision: REJECT the null hypothesis")
        print()
        print("Conclusion:")
        print(f"   Hyperparameter tuning resulted in a STATISTICALLY SIGNIFICANT")
        print(f"   improvement in model performance.")
        print(f"   Average improvement: {test_results['mean_difference']:.4f} accuracy points")
        print(f"   Effect size: {interpret_effect_size(test_results['cohens_d']).upper()}")
        print()
        print("Recommendation:")
        print(f"   ✓ Use the optimized model with tuned hyperparameters")
    else:
        print(f"✗ P-value ({test_results['p_value']:.4f}) ≥ α (0.05)")
        print("✗ Decision: FAIL TO REJECT the null hypothesis")
        print()
        print("Conclusion:")
        print(f"   Hyperparameter tuning did NOT result in a statistically significant")
        print(f"   improvement in model performance.")
        print()
        print("Recommendation:")
        print(f"   • Consider using the baseline model (simpler, faster)")
        print(f"   • The observed improvement ({test_results['mean_difference']:.4f}) could be")
        print(f"     due to random variation rather than true performance gain")
    
    print()
    print("=" * 80)
    print()
    
    # Detailed comparison table
    print("DETAILED COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<30}{'Baseline':<20}{'Optimized':<20}{'Difference'}")
    print("-" * 80)
    print(f"{'Mean Accuracy':<30}{baseline_mean:<20.4f}{optimized_mean:<20.4f}"
          f"{optimized_mean - baseline_mean:+.4f}")
    print(f"{'Std Deviation':<30}{baseline_std:<20.4f}{optimized_std:<20.4f}"
          f"{optimized_std - baseline_std:+.4f}")
    print(f"{'Min Score':<30}{np.min(baseline_scores):<20.4f}"
          f"{np.min(optimized_scores):<20.4f}"
          f"{np.min(optimized_scores) - np.min(baseline_scores):+.4f}")
    print(f"{'Max Score':<30}{np.max(baseline_scores):<20.4f}"
          f"{np.max(optimized_scores):<20.4f}"
          f"{np.max(optimized_scores) - np.max(baseline_scores):+.4f}")
    print()
    print("=" * 80)
    print()
    
    print("KEY TAKEAWAYS:")
    print("-" * 80)
    print("• Hyperparameter tuning should be validated statistically.")
    print()
    print("• A small improvement in mean accuracy doesn't guarantee")
    print("  statistical significance.")
    print()
    print("• Effect size (Cohen's d) indicates practical significance:")
    print("  - Small: 0.2 - 0.5")
    print("  - Medium: 0.5 - 0.8")
    print("  - Large: > 0.8")
    print()
    print("• Combine statistical significance AND effect size when making")
    print("  decisions about model selection.")
    print()
    print("• Grid search explores parameter space systematically but can be")
    print("  computationally expensive. Consider RandomizedSearchCV or")
    print("  Bayesian optimization for large parameter spaces.")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
