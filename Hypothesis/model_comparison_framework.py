"""
Model Comparison Framework

This script demonstrates comparing multiple machine learning models using
cross-validation and displaying their performance with confidence intervals.

It showcases:
1. Evaluating multiple models on the same dataset
2. Computing mean accuracy and standard deviation for each model
3. Ranking models based on performance
4. Visualizing results with error bars (confidence intervals)
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

def compute_confidence_interval(scores, confidence=0.95):
    """
    Compute confidence interval for the mean of scores.
    
    Parameters:
    -----------
    scores : array-like
        Performance scores from cross-validation
    confidence : float
        Confidence level (default: 0.95 for 95% CI)
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    n = len(scores)
    mean = np.mean(scores)
    std_err = stats.sem(scores)  # Standard error of the mean
    
    # Use t-distribution for small samples
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return (mean - margin, mean + margin)


def evaluate_models(X, y, cv_folds=10):
    """
    Evaluate multiple models using cross-validation.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv_folds : int
        Number of cross-validation folds
    
    Returns:
    --------
    dict : Model names mapped to their CV scores
    """
    # Define candidate models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB()
    }
    
    results = {}
    
    print(f"Evaluating {len(models)} models using {cv_folds}-fold cross-validation...")
    print("=" * 80)
    print()
    
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        results[name] = scores
        print(f"✓ Complete")
    
    print()
    return results


def display_results(results, confidence=0.95):
    """
    Display results in a formatted table with rankings.
    
    Parameters:
    -----------
    results : dict
        Model names mapped to their CV scores
    confidence : float
        Confidence level for intervals
    """
    print("=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    # Compute statistics for each model
    stats_data = []
    for name, scores in results.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)
        ci_lower, ci_upper = compute_confidence_interval(scores, confidence)
        
        stats_data.append({
            'name': name,
            'mean': mean_score,
            'std': std_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'scores': scores
        })
    
    # Sort by mean score (descending)
    stats_data.sort(key=lambda x: x['mean'], reverse=True)
    
    # Print table header
    print(f"{'Rank':<6}{'Model':<25}{'Mean Acc':<12}{'Std Dev':<12}"
          f"{confidence*100:.0f}% CI")
    print("-" * 80)
    
    # Print each model
    for rank, data in enumerate(stats_data, 1):
        ci_str = f"[{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]"
        print(f"{rank:<6}{data['name']:<25}{data['mean']:<12.4f}"
              f"{data['std']:<12.4f}{ci_str}")
    
    print()
    print("=" * 80)
    print()
    
    return stats_data


def perform_pairwise_comparison(stats_data, alpha=0.05):
    """
    Perform pairwise statistical comparisons between top models.
    
    Parameters:
    -----------
    stats_data : list
        List of dictionaries containing model statistics
    alpha : float
        Significance level
    """
    print("PAIRWISE STATISTICAL COMPARISONS (Top 3 Models)")
    print("=" * 80)
    print()
    
    # Compare top 3 models
    top_models = stats_data[:min(3, len(stats_data))]
    
    for i in range(len(top_models)):
        for j in range(i + 1, len(top_models)):
            model_i = top_models[i]
            model_j = top_models[j]
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(model_i['scores'], model_j['scores'])
            
            print(f"{model_i['name']} vs {model_j['name']}")
            print(f"   Mean difference: {model_i['mean'] - model_j['mean']:.4f}")
            print(f"   T-statistic: {t_stat:.4f}")
            print(f"   P-value: {p_value:.4f}")
            
            if p_value < alpha:
                print(f"   ✓ Statistically significant difference (p < {alpha})")
            else:
                print(f"   ✗ No statistically significant difference (p ≥ {alpha})")
            print()
    
    print("=" * 80)
    print()


def main():
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "COMPREHENSIVE MODEL COMPARISON FRAMEWORK" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Load dataset
    print("Loading Iris dataset...")
    data = load_iris()
    X, y = data.data, data.target
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print()
    
    # Evaluate models
    results = evaluate_models(X, y, cv_folds=10)
    
    # Display results
    stats_data = display_results(results, confidence=0.95)
    
    # Perform pairwise comparisons
    perform_pairwise_comparison(stats_data, alpha=0.05)
    
    # Summary and recommendations
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print(f"Best Model: {stats_data[0]['name']}")
    print(f"   Mean Accuracy: {stats_data[0]['mean']:.4f} ± {stats_data[0]['std']:.4f}")
    print()
    
    # Check if best model is significantly better than second best
    if len(stats_data) > 1:
        t_stat, p_value = stats.ttest_rel(
            stats_data[0]['scores'], 
            stats_data[1]['scores']
        )
        
        if p_value < 0.05:
            print(f"✓ {stats_data[0]['name']} is statistically significantly better than")
            print(f"  {stats_data[1]['name']} (p = {p_value:.4f})")
            print()
            print(f"Recommendation: Use {stats_data[0]['name']} for this task.")
        else:
            print(f"✗ {stats_data[0]['name']} is NOT statistically significantly better")
            print(f"  than {stats_data[1]['name']} (p = {p_value:.4f})")
            print()
            print("Recommendation: Consider other factors (interpretability, speed,")
            print(f"                complexity) when choosing between top models.")
    
    print()
    print("=" * 80)
    print()
    
    print("KEY INSIGHTS:")
    print("-" * 80)
    print("• Cross-validation provides a distribution of performance scores,")
    print("  enabling statistical inference about model performance.")
    print()
    print("• Confidence intervals quantify the uncertainty in performance estimates.")
    print()
    print("• Pairwise t-tests reveal whether performance differences are")
    print("  statistically significant or could be due to random variation.")
    print()
    print("• The best-performing model should be selected based on both")
    print("  statistical evidence AND practical considerations.")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
