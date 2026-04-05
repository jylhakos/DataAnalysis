# Quick Start

## Project Setup

This guide will help you get started with the **Hypothesis and Inference in Data Analysis**.

---

## 📁 Project Structure

```
Hypothesis-Inference/
│
├── README.md                           # Documentation with mathematical formulas
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore file (excludes venv, binaries)
│
├── venv/                               # Virtual environment (already created and activated)
│
├── model_selection_hypothesis_test.py  # Main example: Paired t-test for model comparison
├── model_comparison_framework.py       # Compare multiple models using CV
├── hyperparameter_tuning.py            # Statistical validation of hyperparameter tuning
└── mcnemar_test_example.py            # McNemar's test for single test set comparison
```

---

## Virtual Environment Setup

### ✓ Already Completed

The virtual environment has been created and is currently **active**. You can see `(venv)` in your terminal prompt.

### To Activate Later (Linux/Mac)

```bash
source venv/bin/activate
```

### To Deactivate

```bash
deactivate
```

---

## 📦 Installed Packages

All required packages are already installed:

- **NumPy** 2.4.4 - Numerical computing
- **SciPy** 1.17.1 - Scientific computing and statistical tests
- **Scikit-learn** 1.8.0 - Machine learning algorithms
- **Matplotlib** 3.10.8 - Data visualization
- **Pandas** 3.0.2 - Data manipulation

To verify installation:

```bash
pip list
```

---

## Running the Examples

All scripts are ready to run in the virtual environment!

### 1. Model Selection Using Hypothesis Testing (Paired T-Test)

**What it does:** Compares Random Forest and Logistic Regression using a paired t-test on cross-validation scores.

```bash
python model_selection_hypothesis_test.py
```

**Key Concepts:**
- Null hypothesis: No difference in performance
- Alternative hypothesis: Significant difference exists
- Uses 10-fold cross-validation
- Applies paired t-test
- Interprets p-value at α = 0.05

---

### 2. Model Comparison Framework

**What it does:** Evaluates 7 different models and performs pairwise statistical comparisons.

```bash
python model_comparison_framework.py
```

**Key Concepts:**
- Compares multiple models simultaneously
- Computes confidence intervals
- Ranks models by performance
- Performs pairwise t-tests on top models

---

### 3. Hyperparameter Tuning with Statistical Validation

**What it does:** Uses Grid Search to find optimal hyperparameters, then validates the improvement statistically.

```bash
python hyperparameter_tuning.py
```

**Key Concepts:**
- Grid search for hyperparameter optimization
- Statistical comparison of baseline vs optimized model
- Effect size calculation (Cohen's d)
- Interpretation of practical vs statistical significance

---

### 4. McNemar's Test for Model Comparison

**What it does:** Compares two models on a single test set using McNemar's test.

```bash
python mcnemar_test_example.py
```

**Key Concepts:**
- Non-parametric test for paired nominal data
- Uses 2×2 contingency table
- Focuses on discordant cases (where models disagree)
- Efficient for expensive models (no cross-validation needed)

---

## 📊 Understanding the Output

Each script provides:

1. **Step-by-step execution** - Shows what's happening at each stage
2. **Statistical test results** - t-statistic, p-value, effect size
3. **Hypothesis decision** - Reject or fail to reject null hypothesis
4. **Interpretation** - What the results mean in practical terms
5. **Visual formatting** - Easy-to-read tables and sections

---

## Statistical Concepts

### P-Value Interpretation

- **p < 0.05**: Strong evidence against H₀ → Reject H₀
- **p ≥ 0.05**: Insufficient evidence against H₀ → Fail to reject H₀

### Confidence Intervals

A 95% confidence interval means that if we repeated the experiment many times, 95% of the constructed intervals would contain the true parameter.

### Type I and Type II Errors

| Reality | Decision | Result |
|---------|----------|--------|
| H₀ true | Reject H₀ | **Type I Error (α)** |
| H₀ false | Fail to reject H₀ | **Type II Error (β)** |

---

## Learning

1. **Start with README.md** - Read the comprehensive documentation
2. **Run `model_selection_hypothesis_test.py`** - Learn basic hypothesis testing
3. **Run `model_comparison_framework.py`** - See how to compare multiple models
4. **Run `hyperparameter_tuning.py`** - Understand tuning validation
5. **Run `mcnemar_test_example.py`** - Learn McNemar's test for single test sets

---

## 🛠️ Modifying the Examples

### Change the Dataset

All examples use the Iris dataset. To use a different dataset:

```python
from sklearn.datasets import load_wine  # or load_breast_cancer, etc.

data = load_wine()
X, y = data.data, data.target
```

### Change Models

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

model_a = GradientBoostingClassifier(random_state=42)
model_b = SVC(kernel='rbf', random_state=42)
```

### Adjust Significance Level

```python
alpha = 0.01  # More strict (99% confidence)
# or
alpha = 0.10  # More lenient (90% confidence)
```

---

## Troubleshooting

### Virtual Environment Not Activated

**Symptom:** Packages not found, or using system Python

**Solution:**
```bash
source venv/bin/activate
```

### Module Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Script Permission Errors

**Solution:**
```bash
chmod +x *.py
```

---

---

## Notes

- **Statistical significance ≠ Practical significance**: Always consider domain knowledge
- **Cross-validation** provides more robust estimates than single train/test splits
- **Effect size** matters as much as p-values for practical decisions
- **Multiple comparisons**: Be cautious when testing many hypotheses (consider Bonferroni correction)

---
