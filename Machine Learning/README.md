# Machine Learning

A tutorial to learn machine learning concepts, algorithms, and practical implementations.

## Table of Contents

- [What is Machine Learning?](#what-is-machine-learning)
- [What is a Model?](#what-is-a-model)
- [Types of Machine Learning](#types-of-machine-learning)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Reinforcement Learning](#reinforcement-learning)
- [Predictive Models](#predictive-models)
- [Classification](#classification)
  - [What is a Classifier?](#what-is-a-classifier)
  - [k-Nearest Neighbors (k-NN)](#k-nearest-neighbors-k-nn)
- [Dimensionality](#dimensionality)
- [Model Evaluation](#model-evaluation)
  - [Accuracy and Correctness](#accuracy-and-correctness)
  - [Overfitting and Underfitting](#overfitting-and-underfitting)
  - [Bias-Variance Tradeoff](#bias-variance-tradeoff)
- [Feature Engineering](#feature-engineering)
  - [Feature Extraction and Selection](#feature-extraction-and-selection)
- [Machine Learning and Generative AI](#machine-learning-and-generative-ai)
- [Time-Series Forecasting with Deep Learning](#time-series-forecasting-with-deep-learning)
  - [Introduction to Time-Series Analysis](#introduction-to-time-series-analysis)
  - [How Time-Series Differs from Standard Predictive Models](#how-time-series-differs-from-standard-predictive-models)
  - [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
  - [Python Implementation Examples](#python-implementation-examples)
  - [Time-Series Model Validation](#time-series-model-validation)
  - [Debugging RNNs and Time-Series Models in VS Code](#debugging-rnns-and-time-series-models-in-vs-code)
  - [Weather Forecasting Case Study](#weather-forecasting-case-study)
  - [Step-by-Step DevOps Instructions](#step-by-step-devops-instructions)
  - [Testing Best Practices for RNNs](#testing-best-practices-for-rnns)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Resources](#resources)

---

## What is Machine Learning?

**Machine Learning (ML)** is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Instead of following rigid, pre-programmed rules, ML systems use algorithms to identify patterns, learn from experience, and improve their performance over time.

Machine learning creates data models by:
1. **Training**: Feeding data into algorithms to identify patterns
2. **Model Generalization**: Applying learned knowledge to new, unseen data
3. **Pattern Recognition**: Identifying trends within large datasets that humans might overlook

### How ML Differs from Traditional Programming

Traditional programming requires developers to write explicit instructions for every scenario. In contrast, ML algorithms discover patterns and rules autonomously by analyzing data.

**Example**: To predict rainfall:
- **Traditional Approach**: Create physics-based representations with massive fluid dynamics equations
- **ML Approach**: Provide historical weather data to a model, which learns mathematical relationships between weather patterns and rainfall amounts

---

## What is a Model?

A **model** in machine learning is a mathematical representation derived from data that an ML system uses to make predictions or generate content. Think of it as:
- A learned function that maps inputs to outputs
- A compressed representation of patterns found in training data
- The "knowledge" gained from the training process

**Formula Representation**:

$$
y = f(X; \theta)
$$

Where:
- $y$ = predicted output
- $X$ = input features
- $\theta$ = model parameters (learned during training)
- $f$ = the model function

---

## Types of Machine Learning

Machine learning models are categorized by how they learn, primarily falling into three main categories:

### Supervised Learning

The model trains on **labeled data** (input-output pairs) to predict outcomes or classify new data. The system learns by comparing its predictions to actual outcomes and adjusting accordingly.

**Key Characteristics**:
- Requires labeled training data
- Learns the mapping between inputs and known outputs
- Used for prediction and classification tasks

**Common Applications**:
- Fraud detection
- Image recognition
- Spam filtering
- Medical diagnosis
- House price prediction

**Mathematical Representation**:

Given training data $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$, find function $f$ such that:

$$
f(x_i) \approx y_i \text{ for all } i
$$

### Unsupervised Learning

The algorithm explores **unlabeled data** to discover hidden structures, patterns, or groupings without human guidance. The model identifies natural clusters and relationships within the data.

**Key Characteristics**:
- No labeled data required
- Discovers hidden patterns autonomously
- Groups similar data points together

**Common Applications**:
- Customer segmentation
- Anomaly detection
- Market basket analysis
- Data compression

**Clustering Example**:

The algorithm minimizes within-cluster variance:

$$
\min \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$$

Where:
- $k$ = number of clusters
- $C_i$ = cluster $i$
- $\mu_i$ = centroid of cluster $i$

### Reinforcement Learning

A model learns by **trial and error**, receiving rewards or penalties for its actions. The system discovers optimal strategies through interaction with an environment.

**Key Characteristics**:
- Learns from feedback (rewards/punishments)
- Makes sequential decisions
- Optimizes for long-term cumulative reward

**Common Applications**:
- Robotics and autonomous navigation
- Game playing (chess, Go, video games)
- Resource optimization
- Autonomous vehicles

**Core Components**:

| Component | Description |
|-----------|-------------|
| Agent | The decision-maker (the code/model) |
| Environment | The world the agent interacts with |
| State | Current situation of the environment |
| Action | What the agent does |
| Reward | Numerical feedback for training |
| Policy | Strategy to determine next action |

**Q-Learning Update Formula**:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

Where:
- $Q(s,a)$ = expected reward for action $a$ in state $s$
- $\alpha$ = learning rate
- $r$ = immediate reward
- $\gamma$ = discount factor
- $s'$ = next state
- $a'$ = next action

---

## Predictive Models

Predictive models use historical data to forecast future outcomes. These models are fundamental to decision-making in business, healthcare, finance, and many other domains.

### Regression Models

Regression models predict **continuous numeric values**.

**Linear Regression Formula**:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon
$$

Where:
- $y$ = predicted value
- $\beta_0$ = intercept
- $\beta_i$ = coefficients
- $x_i$ = input features
- $\epsilon$ = error term

**Examples**:
- Predicting house prices based on square footage, location, bedrooms
- Estimating ride time based on traffic conditions and distance
- Forecasting energy consumption

### Classification Models

Classification models predict **categorical outcomes** (discrete classes).

**Examples**:
- Binary Classification: Spam vs. Not Spam, Fraud vs. Legitimate
- Multiclass Classification: Classifying images into categories (dog, cat, bird)

---

## Classification

Classification is the task of predicting discrete class labels for new instances based on learned patterns from labeled training data.

### What is a Classifier?

A **classifier** is a supervised learning model that assigns input data to predefined categories.

**Decision Boundary**: Classifiers create decision boundaries that separate different classes in the feature space.

**Probability Estimation**: Many classifiers output probability scores:

$$
P(y = c | X) = \text{Probability that input } X \text{ belongs to class } c
$$

### k-Nearest Neighbors (k-NN)

k-NN is a simple, non-parametric classification algorithm that classifies new instances based on the majority class among the $k$ closest training examples.

**Algorithm Steps**:
1. Calculate distance between new point and all training points
2. Select $k$ nearest neighbors
3. Assign the most common class among those neighbors

**Distance Metric (Euclidean Distance)**:

$$
d(x, x') = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2}
$$

**Advantages**:
- Simple and intuitive
- No training phase
- Works well with small datasets

**Disadvantages**:
- Computationally expensive for large datasets
- Sensitive to irrelevant features
- Requires careful choice of $k$

---

## Dimensionality

**Dimensionality** refers to the number of input features (variables) in a dataset. Each feature represents one dimension in the feature space.

### The Curse of Dimensionality

As the number of dimensions increases:
- Data becomes sparse in the high-dimensional space
- Distance metrics become less meaningful
- More training data is required
- Computational complexity increases

**Volume of Hypercube**:

For a unit hypercube in $d$ dimensions:

$$
V_d = 1^d = 1
$$

But points become exponentially more distant as $d$ increases.

### Dimensionality Reduction

Techniques to reduce the number of features while preserving important information:

**Principal Component Analysis (PCA)**:

Finds orthogonal directions of maximum variance:

$$
\text{Maximize: } \text{Var}(X\mathbf{w}) \text{ subject to } ||\mathbf{w}|| = 1
$$

**Benefits**:
- Reduces computational cost
- Removes noise and redundant features
- Enables visualization (reducing to 2D or 3D)
- Helps prevent overfitting

---

## Model Evaluation

### Accuracy and Correctness

**Accuracy** measures the proportion of correct predictions:

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

**Important Metrics Beyond Accuracy**:

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Precision** | $\frac{TP}{TP + FP}$ | When false positives are costly |
| **Recall** | $\frac{TP}{TP + FN}$ | When false negatives are costly |
| **F1-Score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Balance between precision and recall |

Where:
- $TP$ = True Positives
- $FP$ = False Positives
- $FN$ = False Negatives
- $TN$ = True Negatives

**Correctness** goes beyond simple accuracy to ensure:
- The model works reliably on diverse data
- Predictions are made for the right reasons
- The model doesn't exploit spurious correlations

### Overfitting and Underfitting

**Overfitting** occurs when a model learns the training data too well, including noise and outliers, resulting in poor performance on new data.

**Underfitting** occurs when a model is too simple to capture underlying patterns in the data.

**Visual Representation**:
```
Error
  ^
  |     Underfitting
  |         \
  |          \    Optimal
  |           \   /
  |            \ /
  |             X
  |            / \
  |           /   \  Overfitting
  |          /     \
  |_________________> Model Complexity
```

**Training vs. Validation Error**:

$$
\text{Overfitting: } E_{\text{train}} \ll E_{\text{validation}}
$$

$$
\text{Underfitting: } E_{\text{train}} \approx E_{\text{validation}} \text{ (both high)}
$$

**Solutions to Overfitting**:
- Collect more training data
- Reduce model complexity
- Use regularization techniques
- Apply cross-validation
- Use dropout (for neural networks)

**Solutions to Underfitting**:
- Increase model complexity
- Add more features
- Reduce regularization
- Train longer

### Bias-Variance Tradeoff

The **Bias-Variance Tradeoff** is a fundamental concept describing the tradeoff between two sources of error in ML models.

**Total Error Decomposition**:

$$
\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

**Bias**: Error from incorrect assumptions in the model
- High bias → Underfitting
- Model is too simple
- Fails to capture patterns

**Variance**: Error from sensitivity to fluctuations in training data
- High variance → Overfitting
- Model is too complex
- Captures noise as patterns

**The Tradeoff**:
- Increasing model complexity: Decreases bias, increases variance
- Decreasing model complexity: Increases bias, decreases variance

**Optimal Model**: Balances bias and variance to minimize total error.

```
Error
  ^
  |
  |              Total Error
  |             /‾‾‾‾‾\
  |            /       \
  |    Bias²  /         \  Variance
  |          /           \
  |_________/_____________\________> Model Complexity
```

---

## Feature Engineering

### Feature Extraction and Selection

**Feature Engineering** is the process of creating, selecting, and transforming features to improve model performance.

**Feature Extraction**: Creating new features from raw data
- Combining existing features
- Transforming features (log, polynomial, etc.)
- Extracting domain-specific information

**Examples**:
- From date: Extract day, month, year, day_of_week
- From text: Word counts, TF-IDF scores
- From images: Edges, corners, textures

**Feature Selection**: Choosing the most relevant features

**Methods**:

1. **Filter Methods**: Use statistical measures
   - Correlation coefficient
   - Chi-squared test
   - Mutual information

2. **Wrapper Methods**: Use model performance
   - Forward selection
   - Backward elimination
   - Recursive feature elimination

3. **Embedded Methods**: Built into the model
   - LASSO regression (L1 regularization)
   - Decision tree feature importance

**L1 Regularization (LASSO)**:

$$
\min_{\beta} \sum_{i=1}^{n} (y_i - \beta^T x_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|
$$

Forces some coefficients to exactly zero, effectively selecting features.

**Benefits**:
- Reduces overfitting
- Improves model interpretability
- Decreases training time
- Reduces computational cost

---

## Machine Learning and Generative AI

### Understanding the Relationship

**Machine Learning (ML)** and **Generative AI (GenAI)** are related yet distinct subsets of artificial intelligence.

**Machine Learning**:
- Focuses on analyzing data to make predictions or decisions
- Learns patterns from historical data
- Classifies, predicts, and optimizes

**Generative AI**:
- A subset of ML that creates new, original content
- Uses ML techniques to generate text, images, code, audio, or video
- Learns underlying data structure to produce similar but unique content

**Key Distinction**:
> Traditional ML classifies or analyzes existing data, while Generative AI creates new content based on learned patterns.

### Core Technologies

**Machine Learning Technologies**:
- Neural networks for pattern recognition
- Recommendation systems
- Fraud detection algorithms
- Diagnostic tools
- Predictive analytics

**Generative AI Technologies**:
- **Large Language Models (LLMs)**: Generate human-like text (e.g., ChatGPT, Bard)
- **Generative Adversarial Networks (GANs)**: Generate realistic images
- **Variational Autoencoders (VAEs)**: Create new samples from learned distributions
- **Diffusion Models**: Generate high-quality images (e.g., DALL-E, Midjourney)

### How They Work Together

**ML serves as the engine that powers GenAI**. Generative models use deep learning (a subset of ML) to:
1. Learn patterns from massive datasets
2. Understand statistical properties of data
3. Generate new content with similar characteristics

### Use Cases Comparison

| **Machine Learning** | **Generative AI** |
|----------------------|-------------------|
| Fraud detection | Content creation (ChatGPT) |
| Spam filtering | Image generation (DALL-E, Midjourney) |
| Recommendation engines | Code generation (GitHub Copilot) |
| Predictive maintenance | Video synthesis |
| Customer churn prediction | Synthetic data generation |
| Medical diagnosis | Automated copywriting |
| Credit scoring | Drug discovery molecule design |

### When to Use Together

**1. Augmenting ML Models**:
- Generate synthetic training data when real data is scarce
- Create realistic test scenarios
- Balance imbalanced datasets

**2. Enhancing Predictions**:
- Use ML to analyze customer behavior
- Use GenAI to create personalized content recommendations
- Combine predictive analytics with content generation

**3. Accelerating Development**:
- GenAI generates code scaffolding
- ML optimizes and tests the code
- Continuous improvement loop

### The AI Relationship Hierarchy

```
Artificial Intelligence (AI)
    ├── Machine Learning (ML)
    │   ├── Supervised Learning
    │   ├── Unsupervised Learning
    │   ├── Reinforcement Learning
    │   └── Deep Learning
    │       └── Generative AI
    │           ├── Large Language Models (LLMs)
    │           ├── GANs
    │           ├── VAEs
    │           └── Diffusion Models
    ├── Robotics
    ├── Expert Systems
    └── Natural Language Processing
```

**Key Insight**: Generative AI is not separate from ML—it is a specialized application of deep learning, which itself is a subset of machine learning.

---

## Time-Series Forecasting with Deep Learning

### Introduction to Time-Series Analysis

**Time-series data** consists of observations recorded sequentially over time, where the temporal ordering of observations is critical. Unlike traditional supervised learning where data points are assumed to be independent and identically distributed (i.i.d.), time-series data exhibits temporal dependencies that must be preserved during modeling.

Time-series forecasting is the task of predicting future values based on historical observations, with applications spanning weather prediction, financial market analysis, energy consumption forecasting, and sensor monitoring systems. The fundamental challenge lies in capturing temporal patterns, trends, seasonality, and long-term dependencies inherent in sequential data.

### How Time-Series Differs from Standard Predictive Models

The temporal nature of time-series data introduces unique characteristics that distinguish it from conventional machine learning tasks:

#### 1. Temporal Dependence

Standard ML models assume independence between observations, allowing random shuffling during train-test splits. Time-series forecasting must preserve temporal order, as future predictions depend on past observations.

**Mathematical Representation**:

In standard supervised learning:
$$P(y_1, y_2, \ldots, y_n | X) = \prod_{i=1}^{n} P(y_i | x_i)$$

In time-series forecasting:
$$P(y_t | y_{t-1}, y_{t-2}, \ldots, y_1, X)$$

where predictions at time $t$ depend on previous timesteps.

#### 2. Autocorrelation

Time-series exhibit **autocorrelation**, where observations are correlated with their own lagged values. This property is quantified by the autocorrelation function (ACF):

$$\rho(\tau) = \frac{\text{Cov}(y_t, y_{t-\tau})}{\text{Var}(y_t)}$$

where $\tau$ is the lag, measuring correlation between observations separated by $\tau$ timesteps.

#### 3. Sequential Input Structure

Time-series models consume sequences of observations rather than single data points. For a sequence of length $n$, the input is:

$$X = [x_{t-n}, x_{t-n+1}, \ldots, x_{t-1}]$$

predicting the next value(s):

$$\hat{y}_t = f(X)$$

This windowing approach, called **sliding window** or **lookback window**, transforms time-series into a supervised learning problem.

#### 4. Stationarity Requirements

Many time-series models assume **stationarity**—constant statistical properties (mean, variance) over time. Non-stationary series require preprocessing:

**Differencing**:
$$\Delta y_t = y_t - y_{t-1}$$

**Detrending**:
$$y_t^{\text{detrended}} = y_t - \text{Trend}(t)$$

**Log Transformation**:
$$y_t^{\text{stab}} = \log(y_t)$$

#### 5. Specialized Validation Techniques

Standard k-fold cross-validation is inappropriate for time-series due to temporal leakage (training on future data to predict the past). Instead, time-series requires:

- **Walk-Forward Validation**
- **Time Series Split**
- **Blocked Cross-Validation**

These techniques are detailed in the validation section below.

---

### Recurrent Neural Networks (RNN)

**Recurrent Neural Networks (RNNs)** are a class of neural networks designed to process sequential data by maintaining an internal **hidden state** that captures information from previous timesteps. Unlike feedforward networks that process inputs independently, RNNs have recurrent connections allowing information to persist across timesteps.

#### RNN Architecture

An RNN processes a sequence $x_1, x_2, \ldots, x_T$ by updating a hidden state $h_t$ at each timestep:

**Hidden State Update**:

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

**Output**:

$$y_t = W_{hy} \cdot h_t + b_y$$

**Parameters**:
- $W_{xh} \in \mathbb{R}^{H \times D}$: Input-to-hidden weight matrix
- $W_{hh} \in \mathbb{R}^{H \times H}$: Hidden-to-hidden (recurrent) weight matrix  
- $W_{hy} \in \mathbb{R}^{O \times H}$: Hidden-to-output weight matrix
- $b_h \in \mathbb{R}^{H}$: Hidden bias vector
- $b_y \in \mathbb{R}^{O}$: Output bias vector
- $D$: Input dimension
- $H$: Hidden state dimension
- $O$: Output dimension
- $x_t \in \mathbb{R}^{D}$: Input at time $t$
- $h_{t-1} \in \mathbb{R}^{H}$: Previous hidden state
- $h_t \in \mathbb{R}^{H}$: Current hidden state
- $\tanh$: Hyperbolic tangent activation function

#### RNN Information Flow

The recurrent connection $W_{hh} \cdot h_{t-1}$ allows the network to maintain a memory of previous inputs. At each timestep:

1. Current input $x_t$ and previous hidden state $h_{t-1}$ are combined
2. Linear transformation applies weights
3. Non-linear activation ($\tanh$) introduces expressiveness
4. New hidden state $h_t$ encodes information from current and past inputs

**Computational Graph** (unrolled across time):

```
x₁ → [RNN] → h₁ → y₁
      ↓
x₂ → [RNN] → h₂ → y₂
      ↓
x₃ → [RNN] → h₃ → y₃
```

#### Vanishing Gradient Problem

Training RNNs via backpropagation through time (BPTT) suffers from the **vanishing gradient problem**. Gradients diminish exponentially as they propagate backward through many timesteps, preventing the model from learning long-term dependencies.

**Gradient Propagation**:

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=1}^{k} \frac{\partial h_{t-i+1}}{\partial h_{t-i}} = \prod_{i=1}^{k} W_{hh} \cdot \text{diag}(\tanh'(h_{t-i}))$$

When $|W_{hh}| < 1$ and $k$ is large, $\left|\frac{\partial h_t}{\partial h_{t-k}}\right| \to 0$, causing vanishing gradients.

Conversely, $|W_{hh}| > 1$ leads to **exploding gradients**.

**Solutions**:
- Gradient clipping (for exploding gradients)
- **Long Short-Term Memory (LSTM)** networks (for vanishing gradients)
- Gated Recurrent Units (GRU)

---

### Long Short-Term Memory (LSTM)

**LSTM networks** are a specialized RNN architecture designed to address the vanishing gradient problem, enabling the learning of long-term dependencies in sequential data. Introduced by Hochreiter and Schmidhuber (1997), LSTMs use a gating mechanism to control information flow through the network.

#### LSTM Architecture

An LSTM unit consists of:
- **Cell State** $C_t$: Long-term memory, running through the entire chain
- **Hidden State** $h_t$: Short-term memory, output at each timestep
- **Three Gates**: Forget, Input, and Output gates

#### Mathematical Formulation

At each timestep $t$, given input $x_t$ and previous hidden state $h_{t-1}$:

**1. Forget Gate** (decides what information to discard from cell state):

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**2. Input Gate** (decides what new information to store):

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Cell State** (new information to potentially add):

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**3. Cell State Update** (update long-term memory):

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**4. Output Gate** (decides what to output from cell state):

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**5. Hidden State Update** (final output):

$$h_t = o_t \odot \tanh(C_t)$$

**Parameters**:
- $\sigma$: Sigmoid activation function $\sigma(x) = \frac{1}{1 + e^{-x}}$ (outputs in range $[0, 1]$)
- $\odot$: Element-wise multiplication (Hadamard product)
- $[\cdot, \cdot]$: Vector concatenation
- $W_f, W_i, W_C, W_o \in \mathbb{R}^{H \times (H+D)}$: Weight matrices for gates
- $b_f, b_i, b_C, b_o \in \mathbb{R}^{H}$: Bias vectors for gates
- $C_t \in \mathbb{R}^{H}$: Cell state (long-term memory)
- $h_t \in \mathbb{R}^{H}$: Hidden state (short-term memory)

#### Gate Functionality Explained

**Forget Gate** ($f_t$):
- Sigmoid output in $[0, 1]$ acts as a multiplier
- Value of $0$ → completely forget information
- Value of $1$ → completely retain information
- Controls what to forget from previous cell state $C_{t-1}$

**Input Gate** ($i_t$) & **Candidate Cell State** ($\tilde{C}_t$):
- Input gate decides which values to update
- Candidate state creates new information
- Together, determine what new information to store

**Cell State Update**:
- $f_t \odot C_{t-1}$: Forget irrelevant information
- $i_t \odot \tilde{C}_t$: Add new relevant information
- Cell state acts as a "conveyor belt" carrying information across timesteps

**Output Gate** ($o_t$):
- Filters cell state to produce hidden state
- Determines which parts of cell state to expose

#### LSTM vs. Simple RNN Comparison

| Feature | Simple RNN | LSTM |
|---------|-----------|------|
| **Memory** | Short-term only | Long-term & Short-term |
| **Architecture Complexity** | Low (1 tanh layer) | High (3-4 gated layers) |
| **Parameters** | $W_{xh}, W_{hh}, b_h$ | $W_f, W_i, W_C, W_o, b_f, b_i, b_C, b_o$ |
| **Gradient Flow** | Prone to vanishing/exploding | Stable via gating mechanism |
| **Training Difficulty** | Easy but limited capacity | More complex but powerful |
| **Best For** | Short sequences (<10 steps) | Long sequences, complex patterns |
| **Applications** | Simple sequence tasks | Text, music, time-series forecasting |
| **Computational Cost** | Low | High (4x parameters of RNN) |

#### Why LSTMs Solve Vanishing Gradients

The cell state $C_t$ provides an uninterrupted gradient highway. During backpropagation, gradients flow through the cell state with minimal transformation (only element-wise multiplication):

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

Since forget gate $f_t$ is learned and can be close to $1$, gradients can flow backward through many timesteps without vanishing, enabling long-term dependency learning.

---

### Python Implementation Examples

#### Basic RNN Step (From Scratch)

```python
import numpy as np

def rnn_step(xt, h_prev, Wxh, Whh, bh):
    """
    Single RNN forward step.
    
    Args:
        xt: Input at time t, shape (input_size,)
        h_prev: Previous hidden state, shape (hidden_size,)
        Wxh: Input-to-hidden weights, shape (hidden_size, input_size)
        Whh: Hidden-to-hidden weights, shape (hidden_size, hidden_size)
        bh: Hidden bias, shape (hidden_size,)
    
    Returns:
        h_next: New hidden state, shape (hidden_size,)
    """
    # Calculate new hidden state using tanh activation
    h_next = np.tanh(np.dot(Wxh, xt) + np.dot(Whh, h_prev) + bh)
    return h_next

# Example usage
input_size, hidden_size = 3, 5
Wxh = np.random.randn(hidden_size, input_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
bh = np.zeros(hidden_size)

# Process a sequence
sequence = np.random.randn(10, input_size)  # 10 timesteps
h = np.zeros(hidden_size)

for t in range(10):
    h = rnn_step(sequence[t], h, Wxh, Whh, bh)
    print(f"Timestep {t+1}, Hidden state norm: {np.linalg.norm(h):.4f}")
```

#### LSTM for Weather Forecasting (TensorFlow/Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Load weather data
df = pd.read_csv('data/raw/mpi_roof.csv')
data = df[['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']].values

# Normalize data to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences (sliding window)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])  # Past seq_length timesteps
        y.append(data[i + seq_length, 0])      # Predict temperature
    return np.array(X), np.array(y)

sequence_length = 24  # Use 24 observations to predict next
X, y = create_sequences(data_scaled, sequence_length)

# Split data (80% train, 20% test - maintaining temporal order)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build LSTM model
model = Sequential([
    LSTM(units=50, activation='tanh', return_sequences=True,
         input_shape=(sequence_length, X.shape[2])),
    Dropout(0.2),
    LSTM(units=50, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(units=1)  # Single output (temperature prediction)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")
```

---

### Time-Series Model Validation

Time-series forecasting requires specialized validation techniques to prevent **temporal leakage**—information from the future inadvertently leaking into training data.

#### Walk-Forward Validation (Time Series Split)

**Walk-Forward Validation** is the gold standard for time-series evaluation. The model is trained on a window of past data and tested on the immediate subsequent period. This process repeats, progressively expanding or sliding the training window.

**Algorithm**:

1. Split data into initial train and test sets
2. Train model on training data
3. Predict on test set (next $h$ timesteps)
4. Expand training window to include test set
5. Repeat steps 2-4 until data exhausted

**Expanding Window**:
```
Train: [1, 2, 3, 4, 5] → Test: [6]
Train: [1, 2, 3, 4, 5, 6] → Test: [7]
Train: [1, 2, 3, 4, 5, 6, 7] → Test: [8]
```

**Sliding Window (Rolling)**:
```
Train: [1, 2, 3, 4, 5] → Test: [6]
Train: [2, 3, 4, 5, 6] → Test: [7]
Train: [3, 4, 5, 6, 7] → Test: [8]
```

**Expanding vs. Sliding Windows**:

| Approach | Training Data | Advantages | Disadvantages |
|----------|---------------|------------|---------------|
| **Expanding Window** | Includes all historical data | Captures long-term trends, more training data | Computationally expensive, old data may be irrelevant |
| **Sliding Window** | Fixed-size window moves forward | Focuses on recent trends, faster training | Discards historical information, requires sufficient window size |

**Python Implementation**:

```python
from sklearn.model_selection import TimeSeriesSplit

# Create time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on test fold
    predictions = model.predict(X_test)
    score = evaluate_metrics(y_test, predictions)
    print(f"Fold score: {score}")
```

#### Blocked Cross-Validation

**Blocked Cross-Validation** divides the time-series into blocks with **gaps (margins)** between training and validation folds to prevent the model from observing lagged values that appear in both sets.

**Structure**:
```
[Train Block 1] [Gap] [Test Block 1] ... [Train Block k] [Gap] [Test Block k]
```

**Purpose**: Ensures that lagged features in the test set do not overlap with training set observations, further reducing leakage.

**Example**:
- Lag features: Use past 7 days to predict next day
- Gap size: 7 days (matching lag length)
- This prevents training data from contaminating test predictions

#### Prohibition of Shuffled Cross-Validation

**Standard k-fold cross-validation is invalid for time-series** because:
1. Shuffling destroys temporal ordering
2. Training on future data to predict the past violates causality
3. Artificially inflates performance metrics

**Incorrect**:
```python
from sklearn.model_selection import cross_val_score, KFold

# WRONG: Shuffles data, breaks temporal order
kfold = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(model, X, y, cv=kfold)
```

**Correct**:
```python
from sklearn.model_selection import TimeSeriesSplit

# CORRECT: Preserves temporal order
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

---

### Debugging RNNs and Time-Series Models in VS Code

Developing and debugging Recurrent Neural Networks presents unique challenges due to their sequential nature, vanishing/exploding gradients, and high dimensionality. Visual Studio Code, combined with specialized extensions and debugging techniques, provides a robust environment for troubleshooting ML models on Linux.

#### VS Code Python Debugger

**Conditional Breakpoints**:

Set breakpoints that trigger when anomalies occur:

1. Open Python file in VS Code
2. Click left margin to set breakpoint (red dot appears)
3. Right-click breakpoint → **Edit Breakpoint** → **Expression**
4. Enter condition: `loss != loss` (detects NaN) or `loss > 1000` (detects exploding loss)

**Variables Explorer**:

During debugging, inspect tensor shapes and values:
- **Variables pane**: Shows local/global variables
- **Watch expressions**: Add custom expressions like `X.shape`, `np.isnan(y_pred).sum()`
- **Debug Console**: Interactively query variables: `print(model.summary())`

**Debugging Workflow**:

1. Press `F5` or **Run → Start Debugging**
2. Select **Python File** debugger
3. Execution pauses at breakpoints
4. Use **Step Over** (`F10`), **Step Into** (`F11`), **Continue** (`F5`)
5. Inspect variables in real-time

**Example Debug Configuration** (`.vscode/launch.json`):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: LSTM Training",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/supervised/lstm_timeseries.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [],
            "env": {"PYTHONPATH": "${workspaceFolder}"}
        }
    ]
}
```

#### Jupyter Extension for Tensor Visualization

**Data Viewer**:

VS Code's Jupyter extension provides graphical data inspection:

1. Install: `ms-toolsai.jupyter`
2. Open notebook or Python file
3. Click on variable in Variables pane
4. Select **View Value** → Data Viewer opens
5. Inspect DataFrames, NumPy arrays, tensors in spreadsheet format

**Real-Time Training Monitoring**:

```python
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Training loop with live plotting
losses = []
for epoch in range(epochs):
    loss = train_one_epoch(model, X_train, y_train)
    losses.append(loss)
    
    # Update plot every 10 epochs
    if epoch % 10 == 0:
        clear_output(wait=True)
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.show()
```

#### TensorFlow/Keras Specific Debugging

**TensorBoard Integration**:

Visualize training metrics, computational graphs, and model architecture:

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Create TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train with callback
model.fit(X_train, y_train, 
          epochs=50, 
          callbacks=[tensorboard_callback])
```

**Launch TensorBoard in VS Code terminal**:

```bash
tensorboard --logdir=logs/fit
```

Navigate to `http://localhost:6006` to view dashboards.

**Model Summary and Shape Validation**:

```python
# Print model architecture
model.summary()

# Check output shape compatibility
print(f"Input shape: {X_train.shape}")
print(f"Expected output shape: {y_train.shape}")
print(f"Model output shape: {model.output_shape}")

# Validate with single batch
test_batch = X_train[:1]
test_output = model.predict(test_batch)
print(f"Test output shape: {test_output.shape}")
```

#### Logging and Trace Analysis

**Structured Logging**:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log training progress
for epoch in range(epochs):
    loss = train_one_epoch(model, X_train, y_train)
    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
    
    # Log gradient statistics
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            grads = layer.kernel.numpy()
            logger.debug(f"Layer {layer.name} - "
                        f"Grad mean: {grads.mean():.6f}, "
                        f"Grad std: {grads.std():.6f}")
```

**Gradient Monitoring**:

Detect vanishing/exploding gradients:

```python
from tensorflow.keras.callbacks import Callback

class GradientLogger(Callback):
    def on_batch_end(self, batch, logs=None):
        # Compute gradients
        with tf.GradientTape() as tape:
            predictions = self.model(X_train[:32], training=True)
            loss = self.model.loss(y_train[:32], predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Check for vanishing/exploding
        for grad, var in zip(gradients, self.model.trainable_variables):
            if grad is not None:
                grad_norm = tf.norm(grad).numpy()
                if grad_norm < 1e-7:
                    print(f"WARNING: Vanishing gradient in {var.name}")
                elif grad_norm > 10:
                    print(f"WARNING: Exploding gradient in {var.name}")

# Use callback
model.fit(X_train, y_train, callbacks=[GradientLogger()])
```

#### Advanced Debugging Tools

**Cockpit (PyTorch Visual Debugger)**:

Tracks statistical quantities during training:
- Gradient norms
- Layer activations
- Parameter updates

**Installation**:
```bash
pip install cockpit-for-pytorch
```

**Deep Seer (RNN State Abstraction)**:

Abstracts RNN hidden states into finite state machines for interpretability:
- Trace which input features influenced decisions
- Identify problematic sequence patterns

**AI/ML Debugger Extension**:

VS Code Marketplace extension providing:
- Model auto-detection
- Real-time state exploration
- Unified dashboard for training metrics

**Installation**: Search "AI/ML Debugger" in Extensions (`Ctrl+Shift+X`)

#### Common RNN Debugging Checklist

1. **Check tensor shapes at each layer** → Mismatched dimensions
2. **Monitor loss for NaN/Inf** → Learning rate too high, gradient explosion
3. **Validate input normalization** → Unnormalized inputs cause instability  
4. **Inspect gradient magnitudes** → Detect vanishing/exploding gradients
5. **Verify sequence alignment** → Off-by-one errors in windowing
6. **Test on tiny dataset** → Model should overfit perfectly (sanity check)
7. **Check activation functions** → ReLU can die, prefer tanh/sigmoid for RNNs
8. **Validate train/test split** → No temporal leakage
9. **Profile training time** → Identify bottlenecks (use `cProfile`)
10. **Examine prediction distribution** → All zeros/constants indicate model collapse

---

### Weather Forecasting Case Study

#### Dataset: MPI Roof Weather Station

The Max Planck Institute for Biogeochemistry operates a weather station on a rooftop in Jena, Germany, recording meteorological observations at 10-minute intervals. The dataset contains:

**Features**:
- **Temperature (T)**: Air temperature in °C
- **Pressure (p)**: Atmospheric pressure in mbar
- **Humidity (rh)**: Relative humidity in %
- **Wind Speed (wv)**: Wind velocity in m/s
- **Rainfall (rain)**: Precipitation in mm
- **Solar Radiation (SWDR)**: Shortwave downward radiation in W/m²
- **CO₂ (CO2)**: Carbon dioxide concentration in ppm

**Temporal Resolution**: 10-minute intervals
**Duration**: Multiple years of continuous recordings
**Application**: Temperature forecasting for next-day prediction

#### Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv('data/raw/mpi_roof.csv')

# Parse datetime
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df = df.set_index('Date Time')

# Select features (temperature, pressure, humidity, wind speed)
features = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']
df_clean = df[features].copy()

# Handle missing values
df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')

# Normalize features to [0, 1]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_clean.values)

print(f"Dataset shape: {data_scaled.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
```

#### Creating Sequences with Sliding Window

Transform time-series into supervised learning format:

```python
def create_sequences(data, seq_length, pred_horizon):
    """
    Create input-output pairs for time-series forecasting.
    
    Args:
        data: Scaled feature array, shape (n_samples, n_features)
        seq_length: Lookback window (number of past timesteps)
        pred_horizon: Number of future timesteps to predict
        
    Returns:
        X: Input sequences, shape (n_sequences, seq_length, n_features)
        y: Target values, shape (n_sequences, pred_horizon)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - pred_horizon + 1):
        # Input: past seq_length timesteps (all features)
        X.append(data[i:(i + seq_length), :])
        
        # Output: next pred_horizon timesteps (temperature only)
        y.append(data[(i + seq_length):(i + seq_length + pred_horizon), 0])
    
    return np.array(X), np.array(y)

# Use 24 observations (4 hours) to predict next timestep (10 minutes)
SEQ_LENGTH = 24
PRED_HORIZON = 1

X, y = create_sequences(data_scaled, SEQ_LENGTH, PRED_HORIZON)
print(f"X shape: {X.shape}")  # (n_sequences, 24, 4)
print(f"y shape: {y.shape}")  # (n_sequences, 1)
```

#### Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Build LSTM model
model = Sequential([
    # First LSTM layer - returns sequences for stacking
    LSTM(units=64, activation='tanh', return_sequences=True,
         input_shape=(SEQ_LENGTH, X.shape[2])),
    Dropout(0.2),
    
    # Second LSTM layer - returns final hidden state
    LSTM(units=32, activation='tanh', return_sequences=False),
    Dropout(0.2),
    
    # Output layer
    Dense(units=PRED_HORIZON)
])

# Compile
model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['mae'])

model.summary()
```

#### Training and Evaluation

```python
# Split data (80% train, 20% test - maintaining temporal order)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                           restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Predict
y_pred = model.predict(X_test)

# Inverse transform to original scale
# Create dummy array for inverse transform
dummy_test = np.zeros((len(y_test), df_clean.shape[1]))
dummy_pred = np.zeros((len(y_pred), df_clean.shape[1]))

dummy_test[:, 0] = y_test.flatten()
dummy_pred[:, 0] = y_pred.flatten()

y_test_orig = scaler.inverse_transform(dummy_test)[:, 0]
y_pred_orig = scaler.inverse_transform(dummy_pred)[:, 0]

# Evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)

print(f"Test RMSE: {rmse:.4f} °C")
print(f"Test MAE: {mae:.4f} °C")
print(f"Test R²: {r2:.4f}")
```

#### Visualization

```python
import matplotlib.pyplot as plt

# Plot predictions vs actual
plt.figure(figsize=(14, 6))
plt.plot(y_test_orig[:200], label='Actual', color='blue', alpha=0.7)
plt.plot(y_pred_orig[:200], label='Predicted', color='red', alpha=0.7)
plt.xlabel('Timestep')
plt.ylabel('Temperature (°C)')
plt.title('LSTM Weather Forecast: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('weather_forecast.png', dpi=300)
plt.show()
```

---

### Dataset Acquisition

Before running any machine learning examples, you must obtain the required datasets. This section provides detailed instructions for downloading and preparing both the weather forecasting dataset and the Iris classification dataset.

#### Weather Dataset (mpi_roof.csv)

The weather forecasting examples use historical weather data from the Max Planck Institute for Biogeochemistry weather station in Jena, Germany. This dataset contains meteorological measurements recorded at 10-minute intervals.

**Download Steps**:

1. **Navigate to the MPI Data Portal**:
   
   Open the official data download page in your browser:
   ```
   https://www.bgc-jena.mpg.de/wetter/weather_data.html
   ```

2. **Locate the Dataset**:
   
   Look for the section titled "Download weather data" or similar. The dataset is typically available as a CSV file containing columns such as:
   - Date Time
   - Temperature (T in °C)
   - Pressure (p in mbar)
   - Humidity (rh in %)
   - Wind speed and direction
   - Precipitation

3. **Download the CSV File**:
   
   Download the complete weather dataset. The file may be named `jena_climate_2009_2016.csv` or similar. For this project, rename it to `mpi_roof.csv`.

4. **Place in Project Directory**:
   
   ```bash
   # Create data directories if they don't exist
   mkdir -p data/raw data/processed
   
   # Move the downloaded file to the correct location
   # Replace ~/Downloads/jena_climate_2009_2016.csv with your actual download path
   mv ~/Downloads/jena_climate_2009_2016.csv data/raw/mpi_roof.csv
   
   # Verify file exists
   ls -lh data/raw/mpi_roof.csv
   ```
   
   Expected output:
   ```
   -rw-rw-r-- 1 user user 15M Apr  9 13:15 data/raw/mpi_roof.csv
   ```

5. **Verify Data Format**:
   
   ```bash
   # Check first few lines
   head -5 data/raw/mpi_roof.csv
   ```
   
   Expected columns (exact names may vary):
   ```
   Date Time,p (mbar),T (degC),Tpot (K),Tdew (degC),rh (%),VPmax (mbar),VPact (mbar),VPdef (mbar),sh (g/kg),H2OC (mmol/mol),rho (g/m**3),wv (m/s),max. wv (m/s),wd (deg)
   01.01.2009 00:10:00,996.52,-8.02,265.40,-8.90,93.30,3.33,3.11,0.22,1.94,3.12,1307.75,1.03,1.75,152.30
   ...
   ```

#### Iris Dataset (iris.csv)

The Iris dataset is a classic classification dataset containing measurements of iris flowers across three species. It is widely used for demonstrating supervised learning algorithms.

**Option 1: Download from UCI Machine Learning Repository**

1. **Access UCI Repository**:
   
   Navigate to the Iris dataset page:
   ```
   https://archive.ics.uci.edu/ml/datasets/iris
   ```

2. **Download Data File**:
   
   Click on "Data Folder" and download `iris.data`. This file contains comma-separated values without a header row.

3. **Add Header and Save**:
   
   ```bash
   # Download to data/raw/
   cd data/raw/
   
   # Add header row and save as iris.csv
   echo "sepal_length,sepal_width,petal_length,petal_width,species" > iris.csv
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -O - >> iris.csv
   
   # Return to project root
   cd ../..
   ```

**Option 2: Generate from Scikit-Learn (Recommended)**

This option is simpler and ensures compatibility with the project code:

```bash
# Activate virtual environment first
source venv/bin/activate

# Install scikit-learn if not already installed
pip install scikit-learn

# Run Python script to generate iris.csv
python3 << 'EOF'
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset from sklearn
iris = load_iris()
df = pd.DataFrame(
    data=iris.data,
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
)
df['species'] = iris.target_names[iris.target]

# Save to CSV
df.to_csv('data/raw/iris.csv', index=False)
print(f"Iris dataset saved to data/raw/iris.csv")
print(f"Shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")
EOF
```

Expected output:
```
Iris dataset saved to data/raw/iris.csv
Shape: (150, 5)
First 5 rows:
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
```

#### Dataset Verification Checklist

After downloading both datasets, verify their integrity:

```bash
# Check both files exist
ls -lh data/raw/

# Expected output:
# -rw-rw-r-- 1 user user 3.9K Apr  9 13:15 iris.csv
# -rw-rw-r-- 1 user user  15M Apr  9 13:15 mpi_roof.csv

# Count rows in each dataset
echo "Iris dataset rows:"
wc -l data/raw/iris.csv  # Expected: 151 (including header)

echo "Weather dataset rows:"
wc -l data/raw/mpi_roof.csv  # Should be > 100,000 rows

# Preview both datasets
echo "Iris dataset preview:"
head -3 data/raw/iris.csv

echo "Weather dataset preview:"
head -3 data/raw/mpi_roof.csv
```

**Important Notes**:

- The `.gitignore` file is configured to exclude `*.csv` files from version control to prevent committing large datasets.
- If you encounter download issues with the MPI weather data portal, alternative sources include Kaggle's "Jena Climate" dataset or TensorFlow's weather dataset tutorial.
- The weather dataset should contain at least 50,000 rows for meaningful time-series analysis. Smaller datasets may not demonstrate the LSTM model's temporal learning capabilities effectively.

---

### Step-by-Step DevOps Instructions

This section provides instructions for DevOps engineers to set up, run, test, and debug time-series forecasting models in a Linux + VS Code environment.

#### Environment Setup

**1. Verify Prerequisites**:

```bash
# Check Python version (requires 3.8+)
python3 --version

# Check pip
pip --version

# Check VS Code installation
code --version
```

**2. Clone or Navigate to Repository**:

```bash
cd "/home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Machine Learning"
```

**3. Create Virtual Environment**:

```bash
# Create virtual environment
python3 -m venv venv

# Verify creation
ls -la venv
```

**4. Activate Virtual Environment**:

```bash
# Activate (Linux/macOS)
source venv/bin/activate

# Verify activation (prompt shows (venv))
which python3
# Expected: /path/to/project/venv/bin/python3
```

**5. Upgrade pip and Install Dependencies**:

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify TensorFlow installation
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
# Expected: TensorFlow version: 2.13.x or higher
```

**6. Verify Dataset**:

```bash
# Check mpi_roof.csv exists
ls -lh data/raw/mpi_roof.csv

# Preview first 5 lines
head -5 data/raw/mpi_roof.csv
```

#### Running Time-Series Forecasting

**Execute LSTM Weather Forecasting**:

```bash
# Run from project root with virtual environment activated
python src/supervised/lstm_timeseries.py
```

**Expected Output**:

```
================================================================================
LSTM TIME-SERIES WEATHER FORECASTING
================================================================================

==================================================
BASIC RNN DEMONSTRATION
==================================================

Input sequence shape: (3, 3)
...
Final hidden state:
...

==================================================
Loading weather data from data/raw/mpi_roof.csv
Loaded 1000 records with 4 features
...
Created sequences: X shape=(952, 24, 4), y shape=(952, 1)

Dataset splits:
  Training: 609 sequences
  Validation: 152 sequences
  Testing: 191 sequences

Training model...
Epoch 1/50
...
Test RMSE: 1.2345 °C
Test MAE: 0.9876 °C
Test R²: 0.9234
```

**Output Artifacts**:
- Training logs in console
- Model architecture summary
- Evaluation metrics (RMSE, MAE, R²)
- Visualization plot: `notebooks/weather_forecast_results.png`

#### Running Unit Tests

**Execute Pytest Test Suite**:

```bash
# Run all tests with verbose output
pytest tests/test_lstm_timeseries.py -v

# Run specific test class
pytest tests/test_lstm_timeseries.py::TestRNNBasic -v

# Run with coverage report
pytest tests/test_lstm_timeseries.py --cov=src/supervised --cov-report=html

# View coverage report
xdg-open htmlcov/index.html  # Linux
```

**Expected Test Output**:

```
======================================== test session starts ========================================
collected 25 items

tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_initialization PASSED                  [  4%]
tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_different_sizes[1-5] PASSED           [  8%]
tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_different_sizes[3-10] PASSED          [ 12%]
...
tests/test_lstm_timeseries.py::TestIntegration::test_full_pipeline PASSED                   [100%]

======================================== 25 passed in 45.23s =========================================
```

#### Debugging in VS Code

**1. Open Project in VS Code**:

```bash
code .
```

**2. Configure Python Interpreter**:

- Press `Ctrl+Shift+P`
- Type "Python: Select Interpreter"
- Choose `./venv/bin/python`

**3. Create Debug Configuration**:

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug LSTM Training",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/supervised/lstm_timeseries.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Debug Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "tests/test_lstm_timeseries.py",
                "-v",
                "-s"
            ],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

**4. Set Breakpoints**:

- Open `src/supervised/lstm_timeseries.py`
- Click left margin at line (e.g., line 350 in `train()` method)
- Red dot appears indicating breakpoint

**5. Start Debugging**:

- Press `F5` or select **Run → Start Debugging**
- Choose "Debug LSTM Training"
- Execution pauses at breakpoints
- Inspect variables in **Variables** pane
- Use **Debug Console** to query: `X_train.shape`, `model.summary()`

**6. Conditional Breakpoints**:

Right-click breakpoint → **Edit Breakpoint** → **Expression**:

```python
# Trigger when loss becomes NaN
loss != loss

# Trigger when loss exceeds threshold
loss > 10.0

# Trigger on specific epoch
epoch == 25
```

#### Monitoring Training with TensorBoard

**1. Launch TensorBoard** (in separate terminal):

```bash
# Activate virtual environment
source venv/bin/activate

# Launch TensorBoard
tensorboard --logdir=logs/fit

# Output: TensorBoard 2.x.x at http://localhost:6006/
```

**2. View Dashboards**:

Open browser: `http://localhost:6006`

- **Scalars**: Loss, MAE over epochs
- **Graphs**: Model architecture visualization
- **Histograms**: Weight distributions
- **Distributions**: Activation distributions

**3. Stop TensorBoard**:

Press `Ctrl+C` in terminal.

#### Logging and Error Tracing

**View Training Logs**:

```bash
# Real-time log monitoring
tail -f training.log

# Search for errors
grep -i "error\|warning\|nan" training.log

# Count warnings
grep -c "WARNING" training.log
```

**Python Traceback Analysis**:

When errors occur, analyze traceback:

```
Traceback (most recent call last):
  File "src/supervised/lstm_timeseries.py", line 650, in <module>
    main()
  File "src/supervised/lstm_timeseries.py", line 580, in main
    history = forecaster.train(X_train, y_train, X_val, y_val)
  File "src/supervised/lstm_timeseries.py", line 280, in train
    history = self.model.fit(...)
ValueError: Input shape mismatch
```

**Common Issues**:

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Input shape mismatch` | Wrong sequence length | Verify `X.shape[1] == sequence_length` |
| `NaN loss during training` | Learning rate too high | Reduce learning rate, check input normalization |
| `ResourceExhaustedError` | GPU out of memory | Reduce batch size, simplify model |
| `FileNotFoundError` | Dataset missing | Ensure `mpi_roof.csv` in `data/raw/` |

#### Performance Profiling

**CPU Profiling with cProfile**:

```bash
# Profile execution
python -m cProfile -o profile_output.prof src/supervised/lstm_timeseries.py

# Analyze with snakeviz
pip install snakeviz
snakeviz profile_output.prof
```

**Memory Profiling**:

```bash
# Install memory profiler
pip install memory_profiler

# Run with memory profiling
python -m memory_profiler src/supervised/lstm_timeseries.py
```

#### Deactivating Environment

**After work completion**:

```bash
# Deactivate virtual environment
deactivate

# Prompt returns to normal (no (venv) prefix)
```

---

### Testing Best Practices for RNNs

#### Fixture-Based Testing

Use pytest fixtures to avoid repeatedly loading models:

```python
import pytest
import torch

@pytest.fixture(scope="session")
def loaded_lstm_model():
    """Load LSTM model once per test session."""
    from tensorflow.keras.models import load_model
    model = load_model("models/lstm_weather.h5")
    return model

def test_prediction_shape(loaded_lstm_model):
    """Test model output shape consistency."""
    X_test = np.random.randn(10, 24, 4)
    predictions = loaded_lstm_model.predict(X_test)
    assert predictions.shape == (10, 1)
```

#### Parameterized Testing

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("batch_size,seq_len", [
    (1, 10),
    (32, 24),
    (64, 50)
])
def test_lstm_various_inputs(loaded_lstm_model, batch_size, seq_len):
    """Test LSTM with different batch sizes and sequence lengths."""
    forecaster = WeatherTimeSeriesForecaster(sequence_length=seq_len)
    model = forecaster.build_model(n_features=4)
    
    X_test = np.random.randn(batch_size, seq_len, 4)
    output = model.predict(X_test)
    
    assert output.shape == (batch_size, 1)
    assert not np.any(np.isnan(output))
```

#### Numerical Stability Tests

Ensure predictions remain stable:

```python
def test_prediction_consistency(forecaster):
    """Verify predictions are deterministic."""
    X = np.random.randn(10, 24, 4)
    
    pred1 = forecaster.predict(X)
    pred2 = forecaster.predict(X)
    pred3 = forecaster.predict(X)
    
    np.testing.assert_array_equal(pred1, pred2)
    np.testing.assert_array_equal(pred2, pred3)

def test_no_nan_predictions(forecaster, temp_csv_file):
    """Ensure model never produces NaN."""
    df = forecaster.load_weather_data(temp_csv_file)
    data_scaled = forecaster.scaler.fit_transform(df.values)
    X, y = forecaster.create_sequences(data_scaled)
    
    forecaster.model = forecaster.build_model(n_features=4)
    predictions = forecaster.predict(X[:100])
    
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))
```

#### Overfitting Sanity Check

Verify model can learn:

```python
def test_overfitting_tiny_dataset(forecaster):
    """Model should perfectly fit tiny dataset (training logic check)."""
    # Create tiny synthetic dataset
    X_tiny = np.random.randn(2, 10, 4)
    y_tiny = np.random.randn(2, 1)
    
    forecaster.model = forecaster.build_model(n_features=4)
    
    # Train for many epochs
    history = forecaster.model.fit(X_tiny, y_tiny, epochs=100, verbose=0)
    
    # Should achieve very low loss
    final_loss = history.history['loss'][-1]
    assert final_loss < 0.1, f"Model failed to overfit (loss={final_loss})"
```

#### Bias and Fairness Tests

```python
def test_prediction_bias():
    """Ensure mean predictions are unbiased."""
    y_true = np.random.randn(1000)
    y_pred = y_true + np.random.normal(0, 0.1, 1000)  # Small noise
    
    mean_error = np.mean(y_pred - y_true)
    assert abs(mean_error) < 0.01, f"Prediction bias detected: {mean_error}"
```

#### Inference Latency Tests

```python
import time

def test_inference_latency(loaded_lstm_model):
    """Ensure prediction meets SLA (< 100ms per batch)."""
    X_test = np.random.randn(32, 24, 4)
    
    start_time = time.time()
    predictions = loaded_lstm_model.predict(X_test, verbose=0)
    elapsed = time.time() - start_time
    
    latency_per_sample = elapsed / len(X_test) * 1000  # ms
    assert latency_per_sample < 100, f"Latency {latency_per_sample:.2f}ms exceeds SLA"
```

---

### References and External Resources

**Foundational Papers**:
- Hochreiter & Schmidhuber (1997). [Long Short-Term Memory](http://www.bioinf.jku.at/publications/older/2604.pdf). Neural Computation 9(8).
- Cho et al. (2014). [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078). arXiv:1406.1078.

**Tutorials**:
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Visual explanation of LSTM architecture
- [Stanford CS230: Recurrent Neural Networks Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/) - Mathematical foundations
- [Dive into Deep Learning: LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html) - Implementation details
- [Dive into Deep Learning: RNN from Scratch](https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html) - RNN implementation
- [TensorFlow Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series) - Official TensorFlow guide

**Weather Forecasting Resources**:
- [MPI Biogeochemistry Weather Data](https://www.bgc-jena.mpg.de/wetter/) - Weather station dataset
- [Data Download Portal](https://www.bgc-jena.mpg.de/wetter/weather_data.html) - Historical weather data
- [RNN for Weather Prediction](https://rpubs.com/DarrenKeeley/weather_rnn) - Case study with R

**Debugging and Development**:
- [VS Code Data Science Tutorial](https://code.visualstudio.com/docs/datascience/data-science-tutorial) - VS Code Python debugging
- [AWS SageMaker GluonTS Debugging](https://aws.amazon.com/blogs/machine-learning/training-debugging-and-running-time-series-forecasting-models-with-the-gluonts-toolkit-on-amazon-sagemaker/) - Time-series debugging strategies

**Testing Frameworks**:
- [pytest Documentation](https://docs.pytest.org/en/latest/) - Python testing framework
- [Real Python: pytest Tutorial](https://realpython.com/pytest-python-testing/) - A pytest guide

---

## Project Structure

```
Machine Learning/
│
├── 📄 README.md                          # This documentation
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                         # Git ignore file
│
├── 📁 data/                              # Dataset storage
│   ├── 📁 raw/                           # Original datasets
│   │   ├── 📄 iris.csv                   # Iris dataset
│   │   └── 📄 mpi_roof.csv               # Weather station time-series data
│   └── 📁 processed/                     # Cleaned/transformed data
│
├── 📁 notebooks/                         # Jupyter notebooks for exploration
│   ├── 📓 exploratory_analysis.ipynb     # Data exploration notebook
│   └── 🖼️ weather_forecast_results.png  # LSTM prediction visualization
│
├── 📁 src/                               # Source code
│   ├── 📁 supervised/                    # Supervised learning examples
│   │   ├── 📄 __init__.py
│   │   ├── 📄 linear_regression.py       # Linear regression implementation
│   │   ├── 📄 logistic_regression.py     # Logistic regression classifier
│   │   ├── 📄 knn_classifier.py          # k-NN classifier
│   │   └── 📄 lstm_timeseries.py         # ⭐ LSTM time-series forecasting
│   │
│   ├── 📁 unsupervised/                  # Unsupervised learning examples
│   │   ├── 📄 __init__.py
│   │   ├── 📄 kmeans_clustering.py       # K-Means clustering
│   │   └── 📄 pca_dimensionality.py      # PCA dimensionality reduction
│   │
│   ├── 📁 reinforcement/                 # Reinforcement learning examples
│   │   ├── 📄 __init__.py
│   │   ├── 📄 q_learning.py              # Q-learning algorithm
│   │   └── 📄 taxi_environment.py        # Taxi environment
│   │
│   └── 📁 utils/                         # Utility functions
│       ├── 📄 __init__.py
│       ├── 📄 data_loader.py             # Data loading utilities
│       ├── 📄 preprocessing.py           # Data preprocessing
│       └── 📄 visualization.py           # Visualization utilities
│
├── 📁 tests/                             # Unit tests
│   ├── 📄 __init__.py
│   ├── 📄 test_data_loader.py            # Data loader tests
│   ├── 📄 test_preprocessing.py          # Preprocessing tests
│   └── 📄 test_lstm_timeseries.py        # ⭐ LSTM time-series tests
│
├── 📁 models/                            # Saved trained models
│   └── 📄 .gitkeep
│
├── 📁 logs/                              # Training logs and TensorBoard
│   └── 📁 fit/                           # TensorBoard logs
│
└── 📁 venv/                              # Virtual environment (excluded from Git)
```

**Key Additions for Time-Series Forecasting**:
- ⭐ `src/supervised/lstm_timeseries.py`: Complete LSTM implementation for weather forecasting
- ⭐ `tests/test_lstm_timeseries.py`: Test suite with 25+ tests
- 📄 `data/raw/mpi_roof.csv`: MPI weather station dataset
- 🖼️ `notebooks/weather_forecast_results.png`: Prediction visualization output
- 📁 `logs/fit/`: TensorBoard training visualization logs

---

## Getting Started

This section provides step-by-step instructions for setting up the project, installing dependencies, and running the machine learning examples using VS Code and Terminal.

### Prerequisites

Before starting, ensure you have the following installed:

- Python 3.8 or higher
- pip package manager
- VS Code (recommended IDE)

**Verify Python installation**:

Open a terminal and check your Python version:

```bash
python3 --version
```

Expected output: `Python 3.8.x` or higher

**Verify pip installation**:

```bash
pip --version
```

or

```bash
python3 -m pip --version
```

Expected output: `pip 20.x.x` or higher

### Step 1: Open Project in VS Code

1. Launch VS Code
2. Select `File > Open Folder`
3. Navigate to and select the "Machine Learning" directory
4. Click `Open`

Alternatively, from terminal:

```bash
cd "/home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Machine Learning"
code .
```

### Step 2: Open Integrated Terminal in VS Code

In VS Code:
- Press `Ctrl + ` (backtick) or
- Select `Terminal > New Terminal` from the menu
- Or press `Ctrl + Shift + ` `

The terminal will open at the project root directory.

### Step 3: Create Virtual Environment

A virtual environment isolates project dependencies from system-wide Python packages.

**Create the virtual environment**:

```bash
python3 -m venv venv
```

This creates a `venv` folder containing an isolated Python environment.

**Verify creation**:

```bash
ls -la venv
```

You should see directories: `bin/`, `lib/`, `include/`, etc.

### Step 4: Activate Virtual Environment

**On Linux/macOS**:

```bash
source venv/bin/activate
```

**On Windows (Command Prompt)**:

```bash
venv\Scripts\activate
```

**On Windows (PowerShell)**:

```bash
venv\Scripts\Activate.ps1
```

**Verification**:

After activation, your terminal prompt should show `(venv)` prefix:

```
(venv) laptop@machine:~/Machine Learning$
```

Verify the Python path:

```bash
which python3
```

Expected output: `/path/to/project/venv/bin/python3`

### Step 5: Install Required Libraries

With the virtual environment activated, install all dependencies:

```bash
pip install -r requirements.txt
```

**Expected installation output**:
```
Collecting numpy==1.24.0
  Downloading numpy-1.24.0-...
Collecting pandas==2.0.0
  Downloading pandas-2.0.0-...
Collecting scikit-learn==1.3.0
  ...
Successfully installed numpy-1.24.0 pandas-2.0.0 scikit-learn-1.3.0 ...
```

**Verify installation**:

List installed packages:

```bash
pip list
```

Expected output includes:
```
Package         Version
--------------- -------
numpy           1.24.0
pandas          2.0.0
scikit-learn    1.3.0
matplotlib      3.7.0
seaborn         0.12.0
gymnasium       0.29.0
jupyter         1.0.0
```

**Verify specific libraries**:

Test NumPy:

```bash
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
```

Expected output: `NumPy version: 1.24.0`

Test scikit-learn:

```bash
python3 -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
```

Expected output: `scikit-learn version: 1.3.0`

Test all critical imports:

```bash
python3 -c "import numpy, pandas, sklearn, matplotlib, seaborn, gymnasium; print('All libraries imported successfully')"
```

Expected output: `All libraries imported successfully`

### Step 6: Verify Dataset

Check that the Iris dataset is present:

```bash
ls -lh data/raw/iris.csv
```

Expected output:
```
-rw-rw-r-- 1 user user 3.9K Apr 9 13:15 data/raw/iris.csv
```

View first few lines:

```bash
head -5 data/raw/iris.csv
```

Expected output:
```
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
...
```

### Step 7: Run Supervised Learning Examples

#### Example 1: k-Nearest Neighbors Classifier

**Run the script**:

```bash
python src/supervised/knn_classifier.py
```

**Expected output**:
```
==================================================
 k-NEAREST NEIGHBORS CLASSIFICATION
==================================================

Loading Iris dataset...
Dataset loaded successfully
  Total samples: 150
  Features: 4
  Classes: 3

Training k-NN classifier...
  Training samples: 105
  Test samples: 45
  k = 5
Training complete

==================================================
MODEL EVALUATION
==================================================

Accuracy: 0.9778 (97.78%)
...
```

**Verify plots appear**: Multiple matplotlib windows should open showing confusion matrix and decision boundaries.

#### Example 2: Linear Regression

**Run the script**:

```bash
python src/supervised/linear_regression.py
```

**Expected output**:
```
==================================================
 LINEAR REGRESSION
==================================================

Loading dataset...
Preparing regression task: Predicting petal_length

Training linear regression model...
  Training samples: 105
  Features: 3
  Target: petal_length
Training complete

Mean Squared Error (MSE): 0.0234
Root Mean Squared Error (RMSE): 0.1530
Mean Absolute Error (MAE): 0.1156
R-squared (R²): 0.9873
...
```

**Verify plots**: Actual vs Predicted and Residuals plots should display.

#### Example 3: Logistic Regression

**Run the script**:

```bash
python src/supervised/logistic_regression.py
```

**Expected output**:
```
==================================================
 LOGISTIC REGRESSION CLASSIFICATION 
==================================================

Standardizing features...

Training Logistic Regression...
  Training samples: 105
  Features: 4
  Classes: 3
Training complete

Accuracy: 0.9778 (97.78%)
...
```

### Step 8: Run Unsupervised Learning Examples

#### Example 4: K-Means Clustering

**Run the script**:

```bash
python src/unsupervised/kmeans_clustering.py
```

**Expected output**:
```
==================================================
 K-MEANS CLUSTERING
==================================================

Loading dataset...

Finding optimal number of clusters...
Testing k from 2 to 10...
k=2: Inertia=152.35, Silhouette=0.6810
k=3: Inertia=78.85, Silhouette=0.5529
k=4: Inertia=57.23, Silhouette=0.4977
...
Optimal k based on Silhouette Score: 2
Recommended k based on Elbow Method: 3
...
```

**Verify plots**: Elbow curve, silhouette analysis, and cluster visualizations should appear.

#### Example 5: PCA Dimensionality Reduction

**Run the script**:

```bash
python src/unsupervised/pca_dimensionality.py
```

**Expected output**:
```
==================================================
 PCA DIMENSIONALITY REDUCTION
==================================================

Standardizing features...

Applying PCA...
  Original dimensions: 4
  Target dimensions: 4
  Transformed shape: (150, 4)

Explained Variance by Component:
  PC1: 0.7296 (72.96%) | Cumulative: 0.7296 (72.96%)
  PC2: 0.2285 (22.85%) | Cumulative: 0.9581 (95.81%)
...
```

**Verify plots**: Explained variance, cumulative variance, 2D and 3D PCA visualizations.

### Step 9: Run Reinforcement Learning Examples

#### Example 6: Q-Learning (Custom GridWorld)

**Run the script**:

```bash
python src/reinforcement/q_learning.py
```

**Expected output**:
```
==================================================
 Q-LEARNING: GRIDWORLD ENVIRONMENT
==================================================

Environment Setup:
  Grid size: 4x4
  States: 16
  Actions: 4 (up, down, left, right)
  Goal state: (3, 3)

Training Q-Learning Agent...
Episode 100/1000 | Avg Reward: -15.32
Episode 200/1000 | Avg Reward: -12.45
...
Training complete

Learned Policy:
→ → → ↓
→ ● → ↓
→ → → ↓
→ → → G
...
```

**Verify plots**: Training progress and Q-values heatmap should display.

#### Example 7: Q-Learning with Gymnasium Taxi

**Run the script**:

```bash
python src/reinforcement/taxi_environment.py
```

**Expected output**:
```
==================================================
 Q-LEARNING: TAXI ENVIRONMENT
==================================================

Environment: Taxi-v3 (OpenAI Gymnasium)

Training agent for 10000 episodes...
Episode 1000/10000 | Epsilon: 0.91 | Avg Reward: -200.5
Episode 2000/10000 | Epsilon: 0.82 | Avg Reward: -150.3
Episode 5000/10000 | Epsilon: 0.55 | Avg Reward: -50.2
Episode 10000/10000 | Epsilon: 0.01 | Avg Reward: 8.5
Training complete

Evaluating trained agent...
Success Rate: 95.0%
Average Steps per Episode: 12.8
...
```

**Note**: This script takes longer to run (2-5 minutes).

### Step 10: Run Time-Series Forecasting (LSTM)

#### Example 8: LSTM Weather Forecasting

**Run the script**:

```bash
python src/supervised/lstm_timeseries.py
```

**Expected output**:
```
================================================================================
LSTM TIME-SERIES WEATHER FORECASTING
================================================================================

==================================================
BASIC RNN DEMONSTRATION
==================================================

Input sequence shape: (3, 3)
Input sequence:
[[0.5 0.2 0.8]
 [0.3 0.6 0.1]
 [0.9 0.4 0.7]]

Hidden states shape: (3, 5, 1)
...
==================================================

2026-04-10 14:30:25 - INFO - Initialized forecaster: seq_len=24, horizon=1, units=64
2026-04-10 14:30:26 - INFO - Loading weather data from data/raw/mpi_roof.csv
2026-04-10 14:30:27 - INFO - Loaded 52560 records with 4 features
2026-04-10 14:30:27 - INFO - Features: ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']
2026-04-10 14:30:28 - INFO - Created sequences: X shape=(52512, 24, 4), y shape=(52512, 1)

Dataset splits:
  Training: 33607 sequences
  Validation: 8402 sequences
  Testing: 10503 sequences

2026-04-10 14:30:29 - INFO - Model architecture built:
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
lstm_layer_1 (LSTM)         (None, 24, 64)            17664     
dropout_1 (Dropout)         (None, 24, 64)            0         
lstm_layer_2 (LSTM)         (None, 32)                12416     
dropout_2 (Dropout)         (None, 32)                0         
output_layer (Dense)        (None, 1)                 33        
=================================================================
Total params: 30,113
Trainable params: 30,113
Non-trainable params: 0
_________________________________________________________________

2026-04-10 14:30:30 - INFO - Starting model training...
Epoch 1/50
1051/1051 - 15s - loss: 0.0125 - mae: 0.0842 - val_loss: 0.0089 - val_mae: 0.0713
Epoch 2/50
1051/1051 - 14s - loss: 0.0078 - mae: 0.0665 - val_loss: 0.0067 - val_mae: 0.0612
...
Epoch 35/50
1051/1051 - 14s - loss: 0.0032 - mae: 0.0421 - val_loss: 0.0041 - val_mae: 0.0487
Restoring model weights from the end of the best epoch: 25.
Epoch 00035: early stopping

2026-04-10 14:38:15 - INFO - Training completed

==================================================
TRAINING SET METRICS
==================================================
2026-04-10 14:38:20 - INFO - Evaluation Metrics:
2026-04-10 14:38:20 - INFO -   MSE: 0.8234
2026-04-10 14:38:20 - INFO -   RMSE: 0.9074
2026-04-10 14:38:20 - INFO -   MAE: 0.6821
2026-04-10 14:38:20 - INFO -   R2: 0.9856

==================================================
TEST SET METRICS
==================================================
2026-04-10 14:38:22 - INFO - Evaluation Metrics:
2026-04-10 14:38:22 - INFO -   MSE: 1.2456
2026-04-10 14:38:22 - INFO -   RMSE: 1.1161
2026-04-10 14:38:22 - INFO -   MAE: 0.8234
2026-04-10 14:38:22 - INFO -   R2: 0.9782

2026-04-10 14:38:25 - INFO - Plot saved to notebooks/weather_forecast_results.png
```

**Verify output artifacts**:
- Console logs with training progress
- Model architecture summary
- Evaluation metrics (RMSE, MAE, R²)
- Visualization plot saved: `notebooks/weather_forecast_results.png`

**View the prediction plot**:

```bash
# Linux
xdg-open notebooks/weather_forecast_results.png

# Or use VS Code
code notebooks/weather_forecast_results.png
```

**Interpretation of Results**:
- **RMSE ≈ 1.1°C**: Average prediction error
- **MAE ≈ 0.8°C**: Average absolute error
- **R² ≈ 0.98**: Model explains 98% of temperature variance

**Note**:
- First run takes longer (downloads TensorFlow components)
- Training time: 7-10 minutes on CPU, 2-3 minutes on GPU
- Memory usage: ~2-3 GB during training

### Step 11: Run Unit Tests

Verify that all components work correctly by running tests.

#### Test Data Loader

**Run the test**:

```bash
python tests/test_data_loader.py
```

**Expected output**:
```
test_load_iris_dataset (__main__.TestDataLoader) ... ok
test_split_data (__main__.TestDataLoader) ... ok
test_split_data_reproducibility (__main__.TestDataLoader) ... ok
test_split_data_stratification (__main__.TestDataLoader) ... ok
test_feature_correlations (__main__.TestDataIntegrity) ... ok
test_class_separability (__main__.TestDataIntegrity) ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.123s

OK

==================================================
TEST SUMMARY
==================================================
Tests run: 6
Successes: 6
Failures: 0
Errors: 0
==================================================
```

#### Test Preprocessing

**Run the test**:

```bash
python tests/test_preprocessing.py
```

**Expected output**:
```
test_standardize_output_shape (__main__.TestStandardization) ... ok
test_standardize_mean_zero (__main__.TestStandardization) ... ok
test_standardize_std_one (__main__.TestStandardization) ... ok
...
----------------------------------------------------------------------
Ran 15 tests in 0.234s

OK

==================================================
TEST SUMMARY
==================================================
Tests run: 15
Successes: 15
Failures: 0
Errors: 0
==================================================
```

#### Test LSTM Time-Series Forecasting

**Run the LSTM test suite**:

```bash
pytest tests/test_lstm_timeseries.py -v
```

**Expected output**:
```
======================== test session starts =========================
collected 25 items

tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_initialization PASSED                          [  4%]
tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_different_sizes[1-5] PASSED                    [  8%]
tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_different_sizes[3-10] PASSED                   [ 12%]
tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_different_sizes[10-20] PASSED                  [ 16%]
tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_different_sizes[50-100] PASSED                 [ 20%]
tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_step_output_range PASSED                       [ 24%]
tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_forward_sequence PASSED                        [ 28%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_forecaster_initialization PASSED [ 32%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_load_weather_data PASSED        [ 36%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_create_sequences_shapes[5-1] PASSED [ 40%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_create_sequences_shapes[10-1] PASSED [ 44%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_create_sequences_shapes[20-3] PASSED [ 48%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_create_sequences_shapes[30-5] PASSED [ 52%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_create_sequences_temporal_order PASSED [ 56%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_build_model_architecture PASSED [ 60%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_model_output_shapes[1-10-4] PASSED [ 64%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_model_output_shapes[16-10-4] PASSED [ 68%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_model_output_shapes[32-20-1] PASSED [ 72%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_model_output_shapes[64-5-10] PASSED [ 76%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_overfitting_on_tiny_dataset PASSED [ 80%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_prediction_consistency PASSED   [ 84%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_evaluation_metrics PASSED       [ 88%]
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_perfect_prediction_metrics PASSED [ 92%]
tests/test_lstm_timeseries.py::TestNumericalStability::test_no_nan_in_predictions PASSED             [ 96%]
tests/test_lstm_timeseries.py::TestIntegration::test_full_pipeline PASSED                            [100%]

======================== 25 passed in 48.52s =========================
```

**Test Coverage**:
- ✓ Basic RNN functionality
- ✓ LSTM model architecture validation
- ✓ Data preprocessing and sequence creation
- ✓ Training stability (overfitting sanity check)
- ✓ Prediction consistency and numerical stability
- ✓ Evaluation metrics correctness
- ✓ End-to-end integration pipeline

**Run with coverage report**:

```bash
pytest tests/test_lstm_timeseries.py --cov=src/supervised/lstm_timeseries --cov-report=term-missing
```

**Expected coverage report**:
```
---------- coverage: platform linux, python 3.10.x -----------
Name                                 Stmts   Miss  Cover   Missing
------------------------------------------------------------------
src/supervised/lstm_timeseries.py      425     12    97%   125-127, 340-342
------------------------------------------------------------------
TOTAL                                  425     12    97%
```

**Note**: Tests require TensorFlow. First run downloads model components (~500MB).

#### Run All Tests with pytest (Optional)

If pytest is installed:

```bash
python -m pytest tests/ -v
```

**Expected output**:
```
tests/test_data_loader.py::TestDataLoader::test_load_iris_dataset PASSED
tests/test_data_loader.py::TestDataLoader::test_split_data PASSED
tests/test_lstm_timeseries.py::TestRNNBasic::test_rnn_initialization PASSED
tests/test_lstm_timeseries.py::TestWeatherTimeSeriesForecaster::test_load_weather_data PASSED
...
======================== 40+ passed in 65.23s ========================
```

### Step 12: Launch Jupyter Notebook

For interactive exploration and visualization:

**Start Jupyter**:

```bash
jupyter notebook notebooks/
```

**Expected output**:
```
[I 13:45:23.456 NotebookApp] Serving notebooks from local directory: .../notebooks
[I 13:45:23.456 NotebookApp] Jupyter Notebook 6.x.x is running at:
[I 13:45:23.456 NotebookApp] http://localhost:8888/?token=abc123...
[I 13:45:23.456 NotebookApp] Use Control-C to stop this server
```

Your browser should automatically open to `http://localhost:8888/`

**Open the notebook**:
1. Click on `exploratory_analysis.ipynb`
2. Run cells individually with `Shift + Enter`
3. Or run all cells: `Cell > Run All`

**Stop Jupyter**:

```bash
# Press Ctrl+C in the terminal running Jupyter
# Confirm with 'y' when prompted
```

### Step 13: Verify Output Files

Some scripts generate visualizations. Check if plots are displayed:

**If plots don't appear**, the scripts save them. Check for generated files:

```bash
ls -la *.png 2>/dev/null || echo "No PNG files in current directory"
```

### Step 13: Deactivate Virtual Environment

When finished working:

```bash
deactivate
```

The `(venv)` prefix should disappear from your terminal prompt.

**Verify deactivation**:

```bash
which python3
```

Should now point to system Python (not venv): `/usr/bin/python3`

### Troubleshooting

#### Issue: "python3: command not found"

**Solution**: Use `python` instead of `python3`:

```bash
python --version
python -m venv venv
```

#### Issue: "Permission denied" when activating venv

**Solution**: Make activation script executable:

```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

#### Issue: "ModuleNotFoundError: No module named 'xxx'"

**Solution**: Ensure virtual environment is activated and reinstall:

```bash
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
```

#### Issue: "AttributeError: module 'numpy' has no attribute..."

**Solution**: Version conflict. Upgrade pip and reinstall:

```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

#### Issue: Plots don't display

**Solution**: Backend issue. Set matplotlib backend:

```bash
export MPLBACKEND=TkAgg  # Linux/macOS
python src/supervised/knn_classifier.py
```

Or modify the script to save plots:

Add at the beginning of any script:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

#### Issue: Jupyter kernel not found

**Solution**: Install IPython kernel in venv:

```bash
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name="Python (venv)"
```

Then select the "Python (venv)" kernel in Jupyter.

#### Issue: "FileNotFoundError: data/raw/iris.csv"

**Solution**: Ensure you're in the project root:

```bash
pwd  # Should show: .../Machine Learning
ls data/raw/iris.csv  # Should exist
```

If running from different directory:

```bash
cd "/home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Machine Learning"
```

### Quick Reference Commands

**Setup**:
```bash
cd "Machine Learning"
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
pip install -r requirements.txt
```

**Run Examples**:
```bash
# Supervised Learning
python src/supervised/knn_classifier.py
python src/supervised/linear_regression.py
python src/supervised/logistic_regression.py
python src/supervised/lstm_timeseries.py          # Time-Series Forecasting

# Unsupervised Learning
python src/unsupervised/kmeans_clustering.py
python src/unsupervised/pca_dimensionality.py

# Reinforcement Learning
python src/reinforcement/q_learning.py
python src/reinforcement/taxi_environment.py
```

**Run Tests**:
```bash
# Unit tests
python tests/test_data_loader.py
python tests/test_preprocessing.py

# LSTM Time-Series tests (with pytest)
pytest tests/test_lstm_timeseries.py -v

# All tests
pytest tests/ -v
```

**TensorBoard (for LSTM training visualization)**:
```bash
tensorboard --logdir=logs/fit
# Open browser: http://localhost:6006
```

**Jupyter**:
```bash
jupyter notebook notebooks/
```

**Cleanup**:
```bash
deactivate
```

---

## Resources

### External Learning Materials

**General Machine Learning**:
- [Google ML Introduction](https://developers.google.com/machine-learning/intro-to-ml/what-is-ml) - An introduction to ML concepts
- [MIT: Machine Learning Explained](https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-explained) - Business-focused ML overview
- [Google Cloud: AI vs. ML](https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning) - Understanding the relationship
- [Azure: What is Machine Learning?](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-machine-learning-platform) - ML platform guide

**Reinforcement Learning**:
- [AWS: What is Reinforcement Learning?](https://aws.amazon.com/what-is/reinforcement-learning/) - Deep dive into RL
- [Azure: Reinforcement Learning](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-reinforcement-learning) - RL concepts and applications

**Time-Series and Deep Learning**:
- [Understanding LSTM Networks (Chris Olah)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Visual LSTM explanation
- [Stanford CS230: RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/) - RNN mathematical foundations
- [Dive into Deep Learning: LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html) - LSTM implementation details
- [Dive into Deep Learning: RNN from Scratch](https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html) - RNN implementation
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series) - Official TensorFlow time-series guide
- [MPI Weather Data](https://www.bgc-jena.mpg.de/wetter/) - Weather station dataset
- [AWS SageMaker: GluonTS Debugging](https://aws.amazon.com/blogs/machine-learning/training-debugging-and-running-time-series-forecasting-models-with-the-gluonts-toolkit-on-amazon-sagemaker/) - Time-series debugging
- [RNN Weather Forecasting Case Study](https://rpubs.com/DarrenKeeley/weather_rnn) - Practical weather prediction example

**Development Tools**:
- [VS Code Data Science Tutorial](https://code.visualstudio.com/docs/datascience/data-science-tutorial) - VS Code Python debugging
- [pytest Documentation](https://docs.pytest.org/en/latest/) - Python testing framework
- [Real Python: pytest Tutorial](https://realpython.com/pytest-python-testing/) - A pytest guide

**Generative AI**:
- [Salesforce: GenAI vs. ML](https://www.salesforce.com/artificial-intelligence/what-is-generative-ai/generative-ai-vs-machine-learning/) - Comparing generative AI and ML

### Books and Papers

**Foundational Texts**:
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Reinforcement Learning: An Introduction" by Sutton and Barto

**Deep Learning and RNNs**:
- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory". Neural Computation 9(8)
- Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder". arXiv:1406.1078

### Tools and Libraries

- **scikit-learn**: General-purpose ML library
- **TensorFlow**: Deep learning framework
- **PyTorch**: Deep learning framework
- **Gymnasium**: RL environments (successor to OpenAI Gym)
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization

---

**Last Updated**: April 2026