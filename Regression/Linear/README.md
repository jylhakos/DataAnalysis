# Linear Regression

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
  - [Model Equation](#model-equation)
  - [Parameter Estimation](#parameter-estimation)
  - [Assumptions](#assumptions)
- [Machine Learning Implementation](#machine-learning-implementation)
  - [Scikit-Learn Approach](#scikit-learn-approach)
  - [Key Functions and Methods](#key-functions-and-methods)
- [Deep Learning Implementation](#deep-learning-implementation)
  - [TensorFlow/Keras Approach](#tensorflowkeras-approach)
  - [Neural Network Interpretation](#neural-network-interpretation)
- [Use Cases](#use-cases)
  - [Traditional Applications](#traditional-applications)
  - [Modern AI Applications](#modern-ai-applications)
- [Implementation Files](#implementation-files)
- [Running the Examples](#running-the-examples)
- [Performance Metrics](#performance-metrics)
- [References](#references)

## Overview

Linear regression is the fundamental regression technique in statistics and machine learning. It models the relationship between a single independent variable (predictor) $x$ and a dependent variable (response) $y$ by fitting a linear equation to observed data. The method assumes that the relationship between the input and output can be approximated by a straight line.

Linear regression serves as the foundation for understanding more complex regression techniques and provides insights into how basic statistical methods translate into neural network architectures. In the context of modern machine learning, linear regression represents the simplest possible supervised learning algorithm for continuous value prediction.

## Mathematical Foundation

### Model Equation

The linear regression model expresses the relationship between the independent variable $x$ and the dependent variable $y$ as:

$$y = \beta_0 + \beta_1 x + \varepsilon$$

Where:
- $y \in \mathbb{R}$ is the **dependent variable** (target, response variable)
- $x \in \mathbb{R}$ is the **independent variable** (predictor, feature)
- $\beta_0 \in \mathbb{R}$ is the **intercept** (bias term, y-intercept when $x = 0$)
- $\beta_1 \in \mathbb{R}$ is the **slope coefficient** (rate of change in $y$ per unit change in $x$)
- $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ is the **error term** (random noise, residual)

**Prediction Equation:**

For a given value of $x$, the predicted value $\hat{y}$ (y-hat) is:

$$\hat{y} = \beta_0 + \beta_1 x$$

The difference between the actual value $y$ and the predicted value $\hat{y}$ is the **residual**:

$$\text{residual} = y - \hat{y} = \varepsilon$$

### Parameter Estimation

The optimal parameters $\beta_0$ and $\beta_1$ are estimated by minimizing the **Sum of Squared Errors (SSE)** or equivalently the **Mean Squared Error (MSE)**.

**Sum of Squared Errors:**

$$\text{SSE}(\beta_0, \beta_1) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2$$

**Mean Squared Error:**

$$\text{MSE}(\beta_0, \beta_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where $n$ is the number of observations (samples).

**Ordinary Least Squares (OLS) Solution:**

By taking partial derivatives of SSE with respect to $\beta_0$ and $\beta_1$ and setting them to zero, we obtain the closed-form solutions:

$$\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$$

$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

Where:
- $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ is the mean of the independent variable
- $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$ is the mean of the dependent variable
- $\text{Cov}(x, y)$ is the covariance between $x$ and $y$
- $\text{Var}(x)$ is the variance of $x$

### Assumptions

Linear regression relies on several key assumptions:

1. **Linearity**: The relationship between $x$ and $y$ is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Constant variance of errors (residuals have uniform spread)
4. **Normality**: Errors are normally distributed: $\varepsilon \sim \mathcal{N}(0, \sigma^2)$
5. **No Multicollinearity**: (Not applicable for simple linear regression, relevant for multiple regression)

## Machine Learning Implementation

### Scikit-Learn Approach

Scikit-learn provides the `LinearRegression` class for implementing simple and multiple linear regression models.

**Basic Usage:**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
X = np.array([[1000], [1500], [2000], [2500]])  # Square footage
y = np.array([200000, 250000, 300000, 350000])  # House prices

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Access learned parameters
print(f"Intercept (β₀): {model.intercept_}")
print(f"Coefficient (β₁): {model.coef_[0]}")

# Make predictions
X_new = np.array([[1800]])
y_pred = model.predict(X_new)
print(f"Predicted price: ${y_pred[0]:,.2f}")
```

### Key Functions and Methods

**Training:**
- `model.fit(X, y)`: Fit the model to training data

**Prediction:**
- `model.predict(X)`: Predict target values for new data
- `model.score(X, y)`: Return the coefficient of determination $R^2$

**Parameters:**
- `model.intercept_`: The intercept term $\beta_0$
- `model.coef_`: The coefficient(s) $\beta_1$ (array for multiple features)

**Evaluation Metrics:**

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Mean Squared Error
mse = mean_squared_error(y_true, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)

# R² Score (Coefficient of Determination)
r2 = r2_score(y_true, y_pred)
```

**R² Score Interpretation:**

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

- $R^2 = 1$: Perfect fit
- $R^2 = 0$: Model performs no better than predicting the mean
- $R^2 < 0$: Model performs worse than predicting the mean

## Deep Learning Implementation

### TensorFlow/Keras Approach

In deep learning, linear regression is implemented as a **single-layer neural network** with **no activation function** (or equivalently, a linear activation).

**Basic Architecture:**

```python
from tensorflow import keras

# Build model: Single Dense layer, no activation
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(1,), activation=None)
])

# Compile with MSE loss
model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # Mean Absolute Error
)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
y_pred = model.predict(X_new)
```

### Neural Network Interpretation

**Mathematical Equivalence:**

A single dense layer with no activation implements the equation:

$$y = Wx + b$$

Where:
- $W$ (weight) is equivalent to $\beta_1$ in traditional linear regression
- $b$ (bias) is equivalent to $\beta_0$ in traditional linear regression

**Loss Function:**

The model minimizes the Mean Squared Error:

$$\mathcal{L}(W, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - (Wx_i + b))^2$$

**Optimization:**

Instead of the closed-form OLS solution, deep learning uses **gradient descent** or its variants (Adam, RMSprop):

1. Initialize $W$ and $b$ randomly
2. Compute predictions: $\hat{y}_i = Wx_i + b$
3. Compute loss: $\mathcal{L} = \frac{1}{n}\sum_{i}(y_i - \hat{y}_i)^2$
4. Compute gradients: $\frac{\partial \mathcal{L}}{\partial W}$, $\frac{\partial \mathcal{L}}{\partial b}$
5. Update parameters: $W \leftarrow W - \alpha \frac{\partial \mathcal{L}}{\partial W}$, $b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}$
6. Repeat steps 2-5 until convergence

Where $\alpha$ is the learning rate.

**Advantages of Deep Learning Approach:**

- Easily extensible to non-linear models (add layers, activations)
- Handles large datasets efficiently with mini-batch gradient descent
- Integrates with other neural network components (embeddings, attention)
- GPU acceleration for faster training

## Use Cases

### Traditional Applications

1. **Real Estate Price Prediction**
   - **Feature**: Square footage
   - **Target**: House price
   - **Example**: $\text{Price} = 100 \times \text{SqFt} + 50,000$

2. **Sales Forecasting**
   - **Feature**: Advertising spend
   - **Target**: Sales revenue
   - **Example**: $\text{Sales} = 5 \times \text{Ad Spend} + 10,000$

3. **Temperature Prediction**
   - **Feature**: Time of day (hours)
   - **Target**: Temperature (°C)
   - **Example**: $\text{Temp} = 2 \times \text{Hour} + 10$

4. **Agricultural Yield**
   - **Feature**: Fertilizer amount (kg)
   - **Target**: Crop yield (tons)
   - **Example**: $\text{Yield} = 0.5 \times \text{Fertilizer} + 5$

### Modern AI Applications

1. **Feature Engineering for Deep Learning**
   - Analyzing linear relationships in preprocessed data
   - Determining feature importance before neural network training

2. **Transfer Learning Baselines**
   - Using linear regression on frozen embeddings from pre-trained models
   - Establishing performance baselines for complex models

3. **Interpretable AI**
   - Providing explainable alternatives to black-box models
   - Identifying key relationships in high-dimensional data

4. **Hybrid Models**
   - Combining neural networks with linear regression layers
   - Using linear projections in attention mechanisms (as in Transformers)

## Implementation Files

This directory contains two implementation files:

### 🐍 [linear_regression_ml.py](linear_regression_ml.py)

**Machine Learning Implementation using Scikit-Learn**

- Uses `sklearn.linear_model.LinearRegression`
- Demonstrates house price prediction based on square footage
- Includes data splitting, model training, evaluation, and visualization
- Outputs regression line plot and performance metrics

**Key Features:**
- OLS parameter estimation
- MSE and R² score calculation
- Matplotlib visualization of regression line

**Run Command:**
```bash
python linear_regression_ml.py
```

### 🐍 [linear_regression_dl.py](linear_regression_dl.py)

**Deep Learning Implementation using TensorFlow/Keras**

- Uses `tensorflow.keras.layers.Dense` with no activation
- Demonstrates the same house price prediction problem
- Includes feature scaling with `StandardScaler`
- Visualizes training loss and predictions

**Key Features:**
- Single-layer neural network
- Adam optimizer with MSE loss
- Training history visualization
- Learned weight and bias extraction

**Run Command:**
```bash
python linear_regression_dl.py
```

## Running the Examples

### Prerequisites

Ensure the virtual environment is activated and dependencies are installed:

```bash
# Navigate to Regression folder
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Regression

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### Execute Scripts

```bash
# Navigate to Linear folder
cd Linear

# Run machine learning example
python linear_regression_ml.py

# Run deep learning example
python linear_regression_dl.py
```

### Expected Output

Both scripts will:
1. Train a linear regression model on sample data
2. Print learned parameters (intercept and coefficient)
3. Display performance metrics (MSE, R²)
4. Make a prediction for a new input
5. Generate and save visualization plots

**Sample Output:**

```
==================================================
Linear Regression Model Results
==================================================
Intercept (β₀): $100,000.00
Coefficient (β₁): $100.00 per sqft
Mean Squared Error: 1,250,000.00
R² Score: 0.9850
==================================================

Prediction for 1800 sqft house:
Predicted Price: $280,000.00
```

## Performance Metrics

### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- Measures average squared difference between actual and predicted values
- **Lower is better**
- Sensitive to outliers (penalizes large errors heavily)

### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}}$$

- Has the same units as the target variable
- Easier to interpret than MSE

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- Measures average absolute difference
- Less sensitive to outliers than MSE

### Coefficient of Determination (R²)

$$R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

- Indicates proportion of variance in $y$ explained by $x$
- Range: $(-\infty, 1]$, typically interpreted in $[0, 1]$
- $R^2 = 0.95$ means 95% of variance is explained by the model

## References

1. **Scikit-learn LinearRegression Documentation**
   [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

2. **Google Machine Learning Crash Course - Linear Regression**
   [https://developers.google.com/machine-learning/crash-course/linear-regression](https://developers.google.com/machine-learning/crash-course/linear-regression)

3. **LabXchange - Linear Regression**
   [https://www.labxchange.org/library/items/lb:LabXchange:98e0c993:html:1](https://www.labxchange.org/library/items/lb:LabXchange:98e0c993:html:1)

4. **Towards Data Science - Linear Regression to GPT in Seven Steps**
   [https://towardsdatascience.com/linear-regression-to-gpt-in-seven-steps-cb3ab3173a14/](https://towardsdatascience.com/linear-regression-to-gpt-in-seven-steps-cb3ab3173a14/)

---

**See also:**
- [Main README](../README.md) - Project overview and setup
- [Multiple Regression](../Multiple/README.md) - Extension to multiple features
- [Logistic Regression](../Logistic/README.md) - Binary classification