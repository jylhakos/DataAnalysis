# Multiple Linear Regression

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
  - [Model Equation](#model-equation)
  - [Matrix Formulation](#matrix-formulation)
  - [Parameter Estimation](#parameter-estimation)
  - [Assumptions](#assumptions)
- [Machine Learning Implementation](#machine-learning-implementation)
  - [Scikit-Learn Approach](#scikit-learn-approach)
  - [Feature Scaling](#feature-scaling)
  - [Feature Importance](#feature-importance)
- [Deep Learning Implementation](#deep-learning-implementation)
  - [TensorFlow/Keras Approach](#tensorflowkeras-approach)
  - [Neural Network Interpretation](#neural-network-interpretation)
- [Use Cases](#use-cases)
  - [Traditional Applications](#traditional-applications)
  - [Modern AI Applications](#modern-ai-applications)
- [Implementation Files](#implementation-files)
- [Running the Examples](#running-the-examples)
- [Performance Metrics](#performance-metrics)
- [Regularization Techniques](#regularization-techniques)
- [References](#references)

## Overview

Multiple linear regression extends simple linear regression to model the relationship between multiple independent variables (predictors) and a single dependent variable (response). It assumes that the dependent variable can be expressed as a linear combination of multiple features, making it suitable for real-world problems where outcomes depend on several factors.

This technique is fundamental in both traditional statistics and modern machine learning, serving as a baseline model for continuous value prediction and as a key component in more complex architectures. In neural networks, multiple linear regression corresponds to a single fully connected layer without activation, demonstrating the deep connection between classical statistics and deep learning.

## Mathematical Foundation

### Model Equation

The multiple linear regression model with $n$ independent variables is:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \varepsilon$$

Where:
- $y \in \mathbb{R}$ is the **dependent variable** (target)
- $x_1, x_2, \ldots, x_n \in \mathbb{R}$ are the **independent variables** (features, predictors)
- $\beta_0 \in \mathbb{R}$ is the **intercept** (bias term)
- $\beta_1, \beta_2, \ldots, \beta_n \in \mathbb{R}$ are the **partial regression coefficients**
- $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ is the **error term** (residual)

**Prediction Equation:**

$$\hat{y} = \beta_0 + \sum_{i=1}^{n} \beta_i x_i$$

Each coefficient $\beta_i$ represents the **partial effect** of feature $x_i$ on the target $y$, holding all other features constant.

### Matrix Formulation

For $m$ observations and $n$ features, the model can be expressed in matrix form:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

Where:

**Response Vector:**
$$\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{bmatrix} \in \mathbb{R}^{m \times 1}$$

**Design Matrix:**
$$\mathbf{X} = \begin{bmatrix} 
1 & x_{1,1} & x_{1,2} & \cdots & x_{1,n} \\
1 & x_{2,1} & x_{2,2} & \cdots & x_{2,n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{m,1} & x_{m,2} & \cdots & x_{m,n}
\end{bmatrix} \in \mathbb{R}^{m \times (n+1)}$$

The first column of ones accounts for the intercept term.

**Coefficient Vector:**
$$\boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_n \end{bmatrix} \in \mathbb{R}^{(n+1) \times 1}$$

**Error Vector:**
$$\boldsymbol{\varepsilon} = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_m \end{bmatrix} \in \mathbb{R}^{m \times 1}$$

### Parameter Estimation

**Objective Function:**

The optimal coefficients minimize the **Sum of Squared Errors (SSE)**:

$$\text{SSE}(\boldsymbol{\beta}) = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$$

**Normal Equation (Closed-Form Solution):**

Taking the gradient with respect to $\boldsymbol{\beta}$ and setting it to zero yields:

$$\frac{\partial \text{SSE}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0$$

Solving for $\boldsymbol{\beta}$:

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

$$\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

This is the **Ordinary Least Squares (OLS)** estimator.

**Note:** The matrix $\mathbf{X}^T\mathbf{X}$ must be invertible (non-singular), which requires:
- Number of observations $m > n+1$ (more data points than features)
- Features are linearly independent (no multicollinearity)

**Alternative: Gradient Descent**

For large datasets or when $\mathbf{X}^T\mathbf{X}$ is not invertible, iterative optimization is used:

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \frac{\partial \text{SSE}}{\partial \boldsymbol{\beta}}\bigg|_{\boldsymbol{\beta}^{(t)}}$$

Where $\alpha$ is the learning rate and:

$$\frac{\partial \text{SSE}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

### Assumptions

Multiple linear regression relies on five key assumptions (Gauss-Markov conditions):

1. **Linearity**: The relationship between features and target is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors across all levels of features
4. **Normality**: Errors are normally distributed: $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$
5. **No Multicollinearity**: Features are not highly correlated with each other

**Multicollinearity Detection:**

- **Variance Inflation Factor (VIF)**: $\text{VIF}_i = \frac{1}{1 - R_i^2}$
  - $R_i^2$ is the $R^2$ when regressing feature $x_i$ on all other features
  - $\text{VIF} > 10$ indicates problematic multicollinearity

## Machine Learning Implementation

### Scikit-Learn Approach

**Basic Usage:**

```python
from sklearn.linear_model import LinearRegression
import pandas as pd

# Sample data: RAM performance prediction
data = {
    'Size_GB': [8, 16, 32, 64],
    'Frequency_MHz': [2400, 2933, 3200, 3600],
    'Bandwidth_GBs': [19.2, 23.5, 25.6, 28.8],
    'Voltage_V': [1.2, 1.2, 1.35, 1.35],
    'Latency_CL': [16, 15, 16, 18],
    'Performance_Score': [65, 85, 92, 98]
}

df = pd.DataFrame(data)

# Separate features and target
X = df.drop('Performance_Score', axis=1)
y = df['Performance_Score']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Access parameters
print(f"Intercept (β₀): {model.intercept_:.4f}")
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")

# Make predictions
X_new = pd.DataFrame({
    'Size_GB': [32],
    'Frequency_MHz': [3200],
    'Bandwidth_GBs': [25.6],
    'Voltage_V': [1.2],
    'Latency_CL': [15]
})
y_pred = model.predict(X_new)
print(f"Predicted Performance: {y_pred[0]:.2f}")
```

### Feature Scaling

Feature scaling is important when features have different units or ranges:

```python
from sklearn.preprocessing import StandardScaler

# Standardization (z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# After scaling: mean = 0, std = 1 for each feature
# Formula: z = (x - μ) / σ
```

**When to Scale:**
- **Required**: For regularized models (Ridge, Lasso), gradient descent-based optimizers
- **Optional**: For OLS linear regression (results are invariant to scaling)
- **Recommended**: For interpretability and numerical stability

### Feature Importance

**Coefficient Magnitude:**

The absolute value of standardized coefficients indicates feature importance:

```python
# Fit model on scaled data
model.fit(X_scaled, y)

# Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(importance)
```

**Note:** Only compare coefficient magnitudes after standardization.

## Deep Learning Implementation

### TensorFlow/Keras Approach

**Model Architecture:**

```python
from tensorflow import keras

# Build model: Single Dense layer with 5 inputs, 1 output
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(5,), activation=None)
])

# Compile with MSE loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
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
print(f"Bias (β₀): {bias[0]:.4f}")
print("Weights (β₁, ..., βₙ):")
for feature, weight in zip(feature_names, weights.flatten()):
    print(f"  {feature}: {weight:.4f}")
```

### Neural Network Interpretation

**Mathematical Equivalence:**

A single dense layer implements:

$$y = \mathbf{W}^T\mathbf{x} + b = \sum_{i=1}^{n} W_i x_i + b$$

Where:
- $\mathbf{W} = [W_1, W_2, \ldots, W_n]^T$ are the **weights** (equivalent to $[\beta_1, \beta_2, \ldots, \beta_n]^T$)
- $b$ is the **bias** (equivalent to $\beta_0$)
- $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$ is the input feature vector

**Gradient Descent Optimization:**

The model updates parameters using backpropagation:

$$W_i \leftarrow W_i - \alpha \frac{\partial \mathcal{L}}{\partial W_i}$$
$$b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}$$

Where $\mathcal{L} = \frac{1}{m}\sum_{j=1}^{m}(y_j - \hat{y}_j)^2$ is the MSE loss.

**Advantages:**
- Seamlessly extends to non-linear models (add hidden layers with activations)
- Handles streaming data with mini-batch training
- Leverages GPU acceleration
- Integrates with complex neural architectures

## Use Cases

### Traditional Applications

1. **RAM Performance Prediction**
   - **Features**: Size (GB), Frequency (MHz), Bandwidth (GB/s), Voltage (V), Latency (CL)
   - **Target**: Performance Score
   - **Application**: Hardware selection and optimization

2. **Employee Salary Prediction**
   - **Features**: Years of experience, education level, location, department
   - **Target**: Annual salary
   - **Application**: Compensation analysis and budgeting

3. **Energy Consumption Forecasting**
   - **Features**: Temperature, humidity, time of day, occupancy
   - **Target**: Building energy usage (kWh)
   - **Application**: Smart building management

4. **Real Estate Valuation**
   - **Features**: Square footage, number of bedrooms, age, location score
   - **Target**: House price
   - **Application**: Property assessment

### Modern AI Applications

1. **Embedding Space Analysis**
   - Studying relationships in word embeddings or image features
   - Understanding which dimensions contribute most to specific properties

2. **Interpretable AI**
   - Using linear models on neural network embeddings for explainability
   - Providing human-understandable relationships in high-dimensional data

3. **Transfer Learning**
   - Training linear layers on frozen pre-trained representations
   - Establishing performance baselines before fine-tuning

4. **Multi-Modal Fusion**
   - Combining features from different modalities (text, image, audio)
   - Linear projection of concatenated embeddings

## Implementation Files

This directory contains two implementation files:

### 🐍 [multiple_regression_ml.py](multiple_regression_ml.py)

**Machine Learning Implementation using Scikit-Learn**

- Uses `sklearn.linear_model.LinearRegression`
- Demonstrates RAM performance prediction based on multiple characteristics
- Includes feature scaling with `StandardScaler`
- Visualizes actual vs. predicted, residuals, and feature importance

**Key Features:**
- Multi-feature regression
- Feature importance analysis
- Visualizations (4 plots)
- Performance metrics (MSE, MAE, R²)

**Run Command:**
```bash
python multiple_regression_ml.py
```

### 🐍 [multiple_regression_dl.py](multiple_regression_dl.py)

**Deep Learning Implementation using TensorFlow/Keras**

- Uses single `Dense` layer with 5 inputs and 1 output
- Same RAM performance prediction problem
- Includes training history visualization
- Extracts and displays learned weights

**Key Features:**
- Single-layer neural network (no activation)
- Adam optimizer with MSE loss
- Training and validation loss curves
- Feature weight analysis

**Run Command:**
```bash
python multiple_regression_dl.py
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
# Navigate to Multiple folder
cd Multiple

# Run machine learning example
python multiple_regression_ml.py

# Run deep learning example
python multiple_regression_dl.py
```

### Expected Output

**Sample Output:**

```
======================================================================
Multiple Linear Regression Model Results
======================================================================
Intercept (β₀): 12.3456

Coefficients (β₁, β₂, ..., βₙ):
  Size_GB             :   0.5234
  Frequency_MHz       :   0.0123
  Bandwidth_GBs       :   1.2345
  Voltage_V           :  -2.3456
  Latency_CL          :  -0.4567

Model Performance Metrics:
  Mean Squared Error:  15.2345
  Mean Absolute Error: 3.1234
  R² Score:            0.9456
======================================================================

Feature Importance (sorted by absolute coefficient):
           Feature  Coefficient  Abs_Coefficient
       Voltage_V      -2.3456          2.3456
    Bandwidth_GBs      1.2345          1.2345
       Size_GB         0.5234          0.5234
      Latency_CL      -0.4567          0.4567
   Frequency_MHz       0.0123          0.0123
```

## Performance Metrics

### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2$$

### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}}$$

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{m}\sum_{i=1}^{m}|y_i - \hat{y}_i|$$

### Adjusted R²

For multiple regression, adjusted $R^2$ accounts for the number of features:

$$R_{\text{adj}}^2 = 1 - \frac{(1 - R^2)(m - 1)}{m - n - 1}$$

Where:
- $m$ is the number of observations
- $n$ is the number of features

Adjusted $R^2$ penalizes adding irrelevant features, unlike regular $R^2$ which always increases with more features.

## Regularization Techniques

### Ridge Regression (L2 Regularization)

Adds a penalty term to prevent overfitting:

$$\mathcal{L}_{\text{Ridge}} = \sum_{i=1}^{m}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{n}\beta_j^2$$

**Scikit-Learn Implementation:**

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # alpha is λ
model.fit(X_train, y_train)
```

### Lasso Regression (L1 Regularization)

Performs feature selection by driving some coefficients to zero:

$$\mathcal{L}_{\text{Lasso}} = \sum_{i=1}^{m}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{n}|\beta_j|$$

**Scikit-Learn Implementation:**

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)  # alpha is λ
model.fit(X_train, y_train)
```

### Elastic Net (L1 + L2)

Combines both penalties:

$$\mathcal{L}_{\text{ElasticNet}} = \sum_{i=1}^{m}(y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{n}|\beta_j| + \lambda_2 \sum_{j=1}^{n}\beta_j^2$$

## References

1. **Scikit-learn Linear Models Documentation**
   [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)

2. **Scikit-learn API - Linear Regression**
   [https://scikit-learn.org/stable/api/sklearn.linear_model.html](https://scikit-learn.org/stable/api/sklearn.linear_model.html)

3. **Google Machine Learning Crash Course**
   [https://developers.google.com/machine-learning/crash-course/linear-regression](https://developers.google.com/machine-learning/crash-course/linear-regression)

4. **AWS - Difference Between Linear and Logistic Regression**
   [https://aws.amazon.com/compare/the-difference-between-linear-regression-and-logistic-regression/](https://aws.amazon.com/compare/the-difference-between-linear-regression-and-logistic-regression/)

---

**See also:**
- [Main README](../README.md) - Project overview and setup
- [Linear Regression](../Linear/README.md) - Simple single-feature regression
- [Logistic Regression](../Logistic/README.md) - Binary classification