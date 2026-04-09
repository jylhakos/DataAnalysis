# Logistic Regression

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
  - [Sigmoid Function](#sigmoid-function)
  - [Model Equation](#model-equation)
  - [Decision Boundary](#decision-boundary)
  - [Loss Function](#loss-function)
  - [Parameter Estimation](#parameter-estimation)
- [Machine Learning Implementation](#machine-learning-implementation)
  - [Scikit-Learn Approach](#scikit-learn-approach)
  - [Hyperparameter Configuration](#hyperparameter-configuration)
  - [Feature Scaling](#feature-scaling)
- [Deep Learning Implementation](#deep-learning-implementation)
  - [TensorFlow/Keras Approach](#tensorflowkeras-approach)
  - [Neural Network Interpretation](#neural-network-interpretation)
- [Use Cases](#use-cases)
  - [Traditional Applications](#traditional-applications)
  - [Modern AI Applications](#modern-ai-applications)
  - [LLM-Related Applications](#llm-related-applications)
- [Implementation Files](#implementation-files)
- [Running the Examples](#running-the-examples)
- [Performance Metrics](#performance-metrics)
- [Multiclass Classification](#multiclass-classification)
- [References](#references)

## Overview

Logistic regression is a fundamental classification algorithm used to predict binary outcomes (0 or 1, True or False, Yes or No). Despite its name containing "regression," it is a **classification method**, not a regression technique. The model outputs probabilities between 0 and 1 using the sigmoid (logistic) function, which are then thresholded to produce discrete class predictions.

Logistic regression serves as the foundation for understanding binary classification in machine learning and provides the basis for more complex models. In deep learning, it corresponds to a single-layer neural network with a sigmoid activation function, making it the simplest possible neural classifier.

The model is widely used in medical diagnosis, spam detection, credit scoring, and as a final classification layer in large language models (LLMs) for tasks like sentiment analysis and content moderation.

## Mathematical Foundation

### Sigmoid Function

The **sigmoid function** (also called the **logistic function**) is the core of logistic regression:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**

1. **Output Range**: $\sigma(z) \in (0, 1)$ for all $z \in \mathbb{R}$
2. **Monotonicity**: Strictly increasing function
3. **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$
4. **Limits**:
   - $\lim_{z \to +\infty} \sigma(z) = 1$
   - $\lim_{z \to -\infty} \sigma(z) = 0$
   - $\sigma(0) = 0.5$

**Derivative:**

$$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$

This convenient derivative form is crucial for gradient-based optimization.

**Visualization:**

The sigmoid function smoothly maps any real number to a probability:

```
σ(z)
 1 |           ___________
   |         /
0.5|       /
   |     /
 0 | __/
   -----|-----|-----
      -6    0    6    z
```

### Model Equation

For binary classification with $n$ features, logistic regression models the probability that an instance belongs to the positive class (class 1):

$$P(y=1|\mathbf{x}) = \sigma(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n)$$

In compact form:

$$P(y=1|\mathbf{x}) = \sigma(\boldsymbol{\beta}^T\mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\beta}^T\mathbf{x}}}$$

Where:
- $\mathbf{x} = [1, x_1, x_2, \ldots, x_n]^T$ is the feature vector (including intercept term)
- $\boldsymbol{\beta} = [\beta_0, \beta_1, \beta_2, \ldots, \beta_n]^T$ are the model parameters
- $z = \boldsymbol{\beta}^T\mathbf{x} = \beta_0 + \sum_{i=1}^{n} \beta_i x_i$ is the **logit** (log-odds)

**Probability Interpretation:**

$$P(y=1|\mathbf{x}) = \hat{y} = \sigma(z)$$
$$P(y=0|\mathbf{x}) = 1 - \hat{y} = 1 - \sigma(z) = \sigma(-z)$$

**Logit (Log-Odds):**

The inverse of the sigmoid function gives the log-odds:

$$\text{logit}(p) = \log\left(\frac{p}{1-p}\right) = z = \boldsymbol{\beta}^T\mathbf{x}$$

Where $\frac{p}{1-p}$ is the **odds ratio**.

### Decision Boundary

**Classification Rule:**

Given a threshold $t$ (typically $t = 0.5$):

$$\hat{y}_{\text{class}} = \begin{cases}
1 & \text{if } P(y=1|\mathbf{x}) \geq t \\
0 & \text{if } P(y=1|\mathbf{x}) < t
\end{cases}$$

For $t = 0.5$:

$$\hat{y}_{\text{class}} = \begin{cases}
1 & \text{if } \boldsymbol{\beta}^T\mathbf{x} \geq 0 \\
0 & \text{if } \boldsymbol{\beta}^T\mathbf{x} < 0
\end{cases}$$

**Decision Boundary:**

The decision boundary is the hyperplane where $\boldsymbol{\beta}^T\mathbf{x} = 0$:

$$\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n = 0$$

- For $n=1$ (one feature): A point on the $x$-axis
- For $n=2$ (two features): A line in 2D space
- For $n>2$ (multiple features): A hyperplane in $n$-dimensional space

### Loss Function

Logistic regression uses the **Binary Cross-Entropy (BCE)** loss (also called **log-loss** or **negative log-likelihood**):

$$\mathcal{L}(\boldsymbol{\beta}) = -\frac{1}{m}\sum_{i=1}^{m} \left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

Where:
- $m$ is the number of training examples
- $y_i \in \{0, 1\}$ is the true label
- $\hat{y}_i = \sigma(\boldsymbol{\beta}^T\mathbf{x}_i)$ is the predicted probability

**Interpretation:**

- If $y_i = 1$: Loss contributes $-\log(\hat{y}_i)$ → Penalizes low predicted probabilities
- If $y_i = 0$: Loss contributes $-\log(1 - \hat{y}_i)$ → Penalizes high predicted probabilities

**Expanded Form:**

$$\mathcal{L}(\boldsymbol{\beta}) = -\frac{1}{m}\sum_{i=1}^{m} \left[y_i \log\left(\sigma(\boldsymbol{\beta}^T\mathbf{x}_i)\right) + (1-y_i)\log\left(1 - \sigma(\boldsymbol{\beta}^T\mathbf{x}_i)\right)\right]$$

### Parameter Estimation

**Gradient of Loss Function:**

The gradient of the BCE loss with respect to $\boldsymbol{\beta}$ is:

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}} = \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)\mathbf{x}_i = \frac{1}{m}\mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$$

Where:
- $\mathbf{X} \in \mathbb{R}^{m \times (n+1)}$ is the design matrix
- $\hat{\mathbf{y}} \in \mathbb{R}^m$ is the vector of predicted probabilities
- $\mathbf{y} \in \mathbb{R}^m$ is the vector of true labels

**Gradient Descent Update:**

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}}\bigg|_{\boldsymbol{\beta}^{(t)}}$$

Where $\alpha$ is the learning rate.

**Note:** Unlike linear regression, logistic regression has no closed-form solution due to the non-linearity of the sigmoid function. Parameters must be estimated using iterative optimization methods:
- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Limited-memory BFGS (L-BFGS)
- Liblinear (coordinate descent)

## Machine Learning Implementation

### Scikit-Learn Approach

**Basic Usage:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data: Tumor classification
X = np.array([[3.78], [2.44], [2.09], [0.14], [4.92], [5.88]])
y = np.array([0, 0, 0, 0, 1, 1])  # 0: Benign, 1: Malignant

# Feature scaling (important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Access parameters
print(f"Intercept (β₀): {model.intercept_[0]:.4f}")
print(f"Coefficient (β₁): {model.coef_[0][0]:.4f}")

# Predict class and probability
X_new = np.array([[3.46]])
X_new_scaled = scaler.transform(X_new)

predicted_class = model.predict(X_new_scaled)[0]
predicted_proba = model.predict_proba(X_new_scaled)[0]

print(f"Predicted Class: {predicted_class}")
print(f"P(class=0): {predicted_proba[0]:.4f}")
print(f"P(class=1): {predicted_proba[1]:.4f}")
```

### Hyperparameter Configuration

**Key Parameters for Production:**

```python
model = LogisticRegression(
    penalty='l2',           # Regularization: 'l1', 'l2', 'elasticnet', 'none'
    C=1.0,                  # Inverse regularization strength (smaller = stronger)
    solver='lbfgs',         # Optimization algorithm
    max_iter=1000,          # Maximum iterations for convergence
    random_state=42,        # Reproducibility
    class_weight=None       # Handle class imbalance: 'balanced' or dict
)
```

**Regularization (Penalty):**

1. **L2 Ridge** (`penalty='l2'`):
   - Adds $\lambda \sum_{j=1}^{n}\beta_j^2$ to the loss
   - Shrinks coefficients towards zero
   - Default and most common

2. **L1 Lasso** (`penalty='l1'`):
   - Adds $\lambda \sum_{j=1}^{n}|\beta_j|$ to the loss
   - Performs feature selection (drives some coefficients to exactly zero)
   - Requires `solver='liblinear'` or `'saga'`

3. **Elastic Net** (`penalty='elasticnet'`):
   - Combines L1 and L2: $\lambda_1 \sum|\beta_j| + \lambda_2 \sum\beta_j^2$
   - Requires `solver='saga'` and `l1_ratio` parameter

**Solver Selection:**

| Solver | Dataset Size | Penalty Support | Speed |
|--------|--------------|-----------------|-------|
| `'liblinear'` | Small (<10k samples) | L1, L2 | Fast for small data |
| `'lbfgs'` | Medium to large | L2 only | Default, robust |
| `'saga'` | Very large | L1, L2, Elastic Net | Fastest for huge data |
| `'sag'` | Large | L2 only | Fast, but less stable |
| `'newton-cg'` | Medium | L2 only | Accurate, slower |

**Regularization Strength (C):**

- $C = \frac{1}{\lambda}$ (inverse of regularization strength)
- **Smaller $C$** → Stronger regularization → Simpler model (prevent overfitting)
- **Larger $C$** → Weaker regularization → More complex model (may overfit)
- Default: $C = 1.0$

**Class Imbalance:**

```python
# Automatically adjust weights inversely proportional to class frequencies
model = LogisticRegression(class_weight='balanced')

# Manual class weights
model = LogisticRegression(class_weight={0: 1, 1: 10})  # 10x weight for class 1
```

### Feature Scaling

**Critical Importance:**

Logistic regression is **sensitive to feature scales** because it uses gradient-based optimization. Always standardize features before training.

**StandardScaler (Z-score Normalization):**

$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

Where $\mu$ is the mean and $\sigma$ is the standard deviation.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaling parameters
```

**MinMaxScaler (Range Normalization):**

$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Scales features to $[0, 1]$ range.

## Deep Learning Implementation

### TensorFlow/Keras Approach

**Single-Layer Network:**

```python
from tensorflow import keras

# Build model: Single Dense layer with sigmoid activation
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(1,), activation='sigmoid')
])

# Compile with Binary Cross-Entropy loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',  # BCE loss
    metrics=['accuracy', 'Precision', 'Recall']
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=4,
    validation_split=0.2
)

# Extract learned parameters
weights, bias = model.layers[0].get_weights()
print(f"Weight (W): {weights[0][0]:.4f}")
print(f"Bias (b): {bias[0]:.4f}")

# Predict probabilities
y_pred_proba = model.predict(X_test_scaled).flatten()

# Predict classes (threshold at 0.5)
y_pred_class = (y_pred_proba > 0.5).astype(int)
```

### Neural Network Interpretation

**Mathematical Equivalence:**

A single dense layer with sigmoid activation implements:

$$\hat{y} = \sigma(Wx + b) = \frac{1}{1 + e^{-(Wx + b)}}$$

Where:
- $W$ (weight) corresponds to $\beta_1, \beta_2, \ldots, \beta_n$
- $b$ (bias) corresponds to $\beta_0$

**Multi-Input Version:**

$$\hat{y} = \sigma(\mathbf{W}^T\mathbf{x} + b) = \sigma\left(\sum_{i=1}^{n} W_i x_i + b\right)$$

**Loss Function:**

Binary Cross-Entropy:

$$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

**Optimizer:**

Adam (Adaptive Moment Estimation) combines:
- Momentum (moving average of gradients)
- RMSprop (adaptive learning rates)

**Extension to Deep Networks:**

Adding hidden layers with non-linear activations creates a **Multi-Layer Perceptron (MLP)**:

```python
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

This is no longer logistic regression, but a deep neural network classifier.

## Use Cases

### Traditional Applications

1. **Medical Diagnosis**
   - **Input**: Tumor size, patient age, biomarker levels
   - **Output**: Benign (0) vs. Malignant (1)
   - **Example**: Cancer screening, disease prediction

2. **Email Spam Detection**
   - **Input**: Word frequencies, sender reputation, link count
   - **Output**: Spam (1) vs. Not Spam (0)
   - **Example**: Email filtering systems

3. **Credit Scoring**
   - **Input**: Income, credit history, debt-to-income ratio
   - **Output**: Default (1) vs. No Default (0)
   - **Example**: Loan approval systems

4. **Customer Churn Prediction**
   - **Input**: Usage patterns, customer tenure, support tickets
   - **Output**: Churn (1) vs. Retain (0)
   - **Example**: Subscription service optimization

### Modern AI Applications

1. **Classification Heads for Pre-trained Models**
   - **Workflow**: Pre-trained Embeddings → Logistic Regression → Class
   - **Advantage**: Lightweight, interpretable, fast to train
   - **Example**: Fine-tuning BERT for sentiment analysis with a linear classifier

2. **Interpretable AI**
   - **Use**: Providing explainable baselines for complex models
   - **Benefit**: Coefficient inspection reveals feature importance
   - **Example**: Understanding which features drive predictions in a medical model

3. **A/B Testing and Experimentation**
   - **Use**: Predicting conversion probabilities
   - **Example**: Click-through rate (CTR) prediction in online advertising

4. **Anomaly Detection**
   - **Use**: Binary classification of normal vs. anomalous behavior
   - **Example**: Fraud detection in financial transactions

### LLM-Related Applications

1. **Sentiment Analysis**
   - **Architecture**: LLM Embeddings → Logistic Regression → Positive/Negative
   - **Example**: Social media sentiment monitoring

2. **Content Moderation**
   - **Architecture**: LLM Text Encoding → Logistic Classifier → Safe/Unsafe
   - **Example**: Detecting harmful content in user-generated text

3. **Intent Classification**
   - **Architecture**: LLM Sentence Embedding → Logistic Regression → Intent Label
   - **Example**: Chatbot intent recognition

4. **Binary NLI (Natural Language Inference)**
   - **Architecture**: LLM Sentence Pair Encoding → Logistic Classifier → Entailment/Contradiction
   - **Example**: Fact-checking and textual entailment

**Research Insight:**

Studies show that logistic regression trained on embeddings from **smaller LLMs** can match or exceed the performance of much larger LLMs on specific binary classification tasks, while providing:
- **Better interpretability** (coefficient inspection)
- **Lower computational cost** (no need to run large models at inference)
- **Faster training** (only the classifier is trained, embeddings are frozen)

## Implementation Files

This directory contains two implementation files:

### 🐍 [logistic_regression_ml.py](logistic_regression_ml.py)

**Machine Learning Implementation using Scikit-Learn**

- Uses `sklearn.linear_model.LogisticRegression`
- Demonstrates tumor classification (benign vs. malignant)
- Includes feature scaling with `StandardScaler`
- Visualizes sigmoid curve, ROC curve, and confusion matrix

**Key Features:**
- Binary classification with probability outputs
- Evaluation metrics (accuracy, precision, recall, F1)
- ROC curve and AUC score
- Confusion matrix visualization

**Run Command:**
```bash
python logistic_regression_ml.py
```

### 🐍 [logistic_regression_dl.py](logistic_regression_dl.py)

**Deep Learning Implementation using TensorFlow/Keras**

- Uses single `Dense` layer with `sigmoid` activation
- Same tumor classification problem
- Includes training history and loss curves
- Extracts learned weights and bias

**Key Features:**
- Single-layer neural network with sigmoid
- Binary cross-entropy loss
- Training and validation accuracy/loss plots
- Probability prediction and class assignment

**Run Command:**
```bash
python logistic_regression_dl.py
```

## Running the Examples

### Prerequisites

```bash
# Navigate to Regression folder
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Regression

# Activate virtual environment
source venv/bin/activate

# Ensure dependencies are installed
pip install -r requirements.txt
```

### Execute Scripts

```bash
# Navigate to Logistic folder
cd Logistic

# Run machine learning example
python logistic_regression_ml.py

# Run deep learning example
python logistic_regression_dl.py
```

### Expected Output

**Sample Output:**

```
======================================================================
Logistic Regression Model Results
======================================================================
Intercept (β₀): -0.1234
Coefficient (β₁): 1.5678

Model Performance Metrics:
  Accuracy:  0.8571
  Precision: 0.8333
  Recall:    1.0000
  F1 Score:  0.9091

Confusion Matrix:
  True Negatives:  5
  False Positives: 1
  False Negatives: 0
  True Positives:  1

======================================================================
Prediction for New Tumor (size: 3.46 cm):
======================================================================
  Predicted Class: Benign
  Probability [Benign, Malignant]: [0.6234, 0.3766]
======================================================================
```

## Performance Metrics

### Confusion Matrix

|                | Predicted Negative (0) | Predicted Positive (1) |
|----------------|------------------------|------------------------|
| **Actual Negative (0)** | True Negative (TN)     | False Positive (FP)    |
| **Actual Positive (1)** | False Negative (FN)    | True Positive (TP)     |

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Proportion of correct predictions overall.

### Precision (Positive Predictive Value)

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of all positive predictions, how many were correct?

### Recall (Sensitivity, True Positive Rate)

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of all actual positives, how many were correctly identified?

### F1 Score

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times TP}{2 \times TP + FP + FN}$$

Harmonic mean of precision and recall.

### ROC Curve and AUC

**Receiver Operating Characteristic (ROC) Curve:**

- Plots **True Positive Rate (TPR)** vs. **False Positive Rate (FPR)** at various threshold settings
- TPR = Recall = $\frac{TP}{TP + FN}$
- FPR = $\frac{FP}{FP + TN}$

**Area Under the Curve (AUC):**

- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC < 0.5: Worse than random (inverted predictions)

## Multiclass Classification

### One-vs-Rest (OvR)

Trains $K$ binary classifiers, one for each class:

```python
model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)  # y can have > 2 classes
```

### Multinomial (Softmax Regression)

Extends logistic regression to multiple classes using the **softmax function**:

$$P(y=k|\mathbf{x}) = \frac{e^{\boldsymbol{\beta}_k^T\mathbf{x}}}{\sum_{j=1}^{K} e^{\boldsymbol{\beta}_j^T\mathbf{x}}}$$

```python
model = LogisticRegression(multi_class='multinomial')
model.fit(X_train, y_train)
```

**Deep Learning Equivalent:**

```python
model = keras.Sequential([
    keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(loss='categorical_crossentropy', ...)
```

## References

1. **Scikit-learn LogisticRegression Documentation**
   [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

2. **University of Virginia - Logistic Regression Four Ways with Python**
   [https://library.virginia.edu/data/articles/logistic-regression-four-ways-with-python](https://library.virginia.edu/data/articles/logistic-regression-four-ways-with-python)

3. **AWS - Difference Between Linear and Logistic Regression**
   [https://aws.amazon.com/compare/the-difference-between-linear-regression-and-logistic-regression/](https://aws.amazon.com/compare/the-difference-between-linear-regression-and-logistic-regression/)

4. **Scikit-learn User Guide - Linear Models**
   [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)

5. **Google Machine Learning Crash Course**
   [https://developers.google.com/machine-learning/crash-course/linear-regression](https://developers.google.com/machine-learning/crash-course/linear-regression)

---

**See also:**
- [Main README](../README.md) - Project overview and setup
- [Linear Regression](../Linear/README.md) - Continuous value prediction
- [Multiple Regression](../Multiple/README.md) - Multi-feature continuous prediction