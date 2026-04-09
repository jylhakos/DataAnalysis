# Regression Analysis: Linear, Multiple, and Logistic

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [DevOps Setup Instructions](#devops-setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Virtual Environment Setup](#virtual-environment-setup)
  - [Installing Dependencies](#installing-dependencies)
  - [Running the Scripts](#running-the-scripts)
- [Regression Types](#regression-types)
  - [Linear Regression](#linear-regression)
  - [Multiple Linear Regression](#multiple-linear-regression)
  - [Logistic Regression](#logistic-regression)
- [Regression in Generative AI and Large Language Models](#regression-in-generative-ai-and-large-language-models)
  - [Linear and Multiple Regression in AI Context](#linear-and-multiple-regression-in-ai-context)
  - [Logistic Regression in LLM Context](#logistic-regression-in-llm-context)
  - [In-Context Learning and Regression](#in-context-learning-and-regression)
- [Transformer Architecture and Regression](#transformer-architecture-and-regression)
  - [Analysis of "Attention Is All You Need"](#analysis-of-attention-is-all-you-need)
  - [Linear Projections in Multi-Head Attention](#linear-projections-in-multi-head-attention)
  - [Position-wise Feed-Forward Networks](#position-wise-feed-forward-networks)
  - [Final Linear Projection and Softmax](#final-linear-projection-and-softmax)
  - [Comparison to Traditional Regression](#comparison-to-traditional-regression)
- [References](#references)

## Overview

This repository contains implementations and academic explanations of three fundamental regression techniques: Linear Regression, Multiple Linear Regression, and Logistic Regression. Each technique is demonstrated using both traditional machine learning approaches (scikit-learn) and deep learning frameworks (TensorFlow/Keras), illustrating the connection between classical statistical models and modern neural network architectures.

The project explores how regression methods relate to contemporary advances in artificial intelligence, particularly in the context of generative AI and large language models (LLMs). Special attention is given to understanding how linear transformations and regression-like components manifest in transformer architectures, as introduced in the seminal paper "Attention Is All You Need" by Vaswani et al. (2017).

## Project Structure

```
📁 Regression/
│
├── 📄 README.md                    # This file - Documentation
├── 📄 .gitignore                   # Git ignore rules for virtual environments
├── 📄 requirements.txt             # Python dependencies
│
├── 📁 Linear/
│   ├── 📄 README.md                # Linear regression detailed documentation
│   ├── 🐍 linear_regression_ml.py  # Machine learning implementation
│   └── 🐍 linear_regression_dl.py  # Deep learning implementation
│
├── 📁 Multiple/
│   ├── 📄 README.md                # Multiple regression detailed documentation
│   ├── 🐍 multiple_regression_ml.py # Machine learning implementation
│   └── 🐍 multiple_regression_dl.py # Deep learning implementation
│
└── 📁 Logistic/
    ├── 📄 README.md                # Logistic regression detailed documentation
    ├── 🐍 logistic_regression_ml.py # Machine learning implementation
    └── 🐍 logistic_regression_dl.py # Deep learning implementation
```

## DevOps Setup Instructions

### Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: Version 3.8 or higher
- **VS Code**: Latest version with Python extension
- **Git**: For version control

### Virtual Environment Setup

Virtual environments isolate project dependencies, preventing conflicts with system-wide Python packages. Follow these steps to create and activate a virtual environment:

#### Step 1: Navigate to the Project Directory

```bash
cd /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Regression
```

#### Step 2: Create Virtual Environment

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv
```

This creates a `venv/` directory containing the Python interpreter and package management tools.

#### Step 3: Activate Virtual Environment

```bash
# Activate the virtual environment
source venv/bin/activate
```

After activation, your terminal prompt should be prefixed with `(venv)`, indicating the virtual environment is active.

#### Step 4: Verify Activation

```bash
# Verify Python is using the virtual environment
which python
# Output should be: /home/laptop/EXERCISES/DATA-ANALYSIS/GIT/DataAnalysis/Regression/venv/bin/python

# Check Python version
python --version
```

#### Step 5: Configure VS Code Terminal

To automatically activate the virtual environment in VS Code terminals:

1. Open VS Code in the project directory: `code .`
2. Open the Command Palette: `Ctrl+Shift+P`
3. Type "Python: Select Interpreter"
4. Select the interpreter from `./venv/bin/python`

VS Code will now automatically activate the virtual environment for new terminals.

### Installing Dependencies

With the virtual environment activated, install all required packages:

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install project dependencies from requirements.txt
pip install -r requirements.txt
```

This installs:
- **numpy**: Numerical computing library
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **tensorflow**: Deep learning framework
- **matplotlib**: Data visualization

#### Verify Installation

```bash
# List installed packages
pip list

# Check specific package versions
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
```

### Running the Scripts

Each regression type has two implementations: machine learning (ML) and deep learning (DL).

#### Linear Regression

```bash
# Machine learning approach
cd Linear
python linear_regression_ml.py

# Deep learning approach
python linear_regression_dl.py
```

#### Multiple Linear Regression

```bash
# Machine learning approach
cd ../Multiple
python multiple_regression_ml.py

# Deep learning approach
python multiple_regression_dl.py
```

#### Logistic Regression

```bash
# Machine learning approach
cd ../Logistic
python logistic_regression_ml.py

# Deep learning approach
python logistic_regression_dl.py
```

#### Deactivating Virtual Environment

When finished working:

```bash
# Deactivate the virtual environment
deactivate
```

## Regression Types

### Linear Regression

Linear regression is a fundamental statistical method for modeling the relationship between a single independent variable $x$ and a dependent variable $y$. It assumes a linear relationship that can be represented by a straight line.

#### Mathematical Formula

The linear regression model is expressed as:

$$y = \beta_0 + \beta_1 x + \varepsilon$$

Where:
- $y$ is the predicted (dependent) variable
- $x$ is the independent (predictor) variable
- $\beta_0$ is the intercept (y-intercept when $x = 0$)
- $\beta_1$ is the slope (rate of change in $y$ with respect to $x$)
- $\varepsilon$ is the error term (residual)

The coefficients $\beta_0$ and $\beta_1$ are estimated by minimizing the **Mean Squared Error (MSE)**:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2$$

Where $\hat{y}_i$ represents the predicted value for the $i$-th observation.

#### Use Cases

**Machine Learning Applications:**
- House price prediction based on square footage
- Sales forecasting based on advertising spend
- Temperature prediction based on time of day

**Deep Learning Perspective:**
Linear regression can be viewed as a single-layer neural network with one neuron, no activation function, and MSE loss:

$$y = Wx + b$$

Where $W$ (weight) corresponds to $\beta_1$ and $b$ (bias) corresponds to $\beta_0$.

See [Linear/README.md](Linear/README.md) for detailed implementation and examples.

### Multiple Linear Regression

Multiple linear regression extends simple linear regression to model the relationship between multiple independent variables and a single dependent variable.

#### Mathematical Formula

The multiple linear regression model is:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \varepsilon$$

In matrix notation:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

Where:
- $\mathbf{y}$ is the $m \times 1$ vector of observations
- $\mathbf{X}$ is the $m \times (n+1)$ design matrix (including a column of ones for the intercept)
- $\boldsymbol{\beta}$ is the $(n+1) \times 1$ vector of coefficients $[\beta_0, \beta_1, \ldots, \beta_n]^T$
- $\boldsymbol{\varepsilon}$ is the $m \times 1$ vector of errors

The optimal coefficients are found using the **Normal Equation**:

$$\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

Or through iterative optimization methods like gradient descent.

#### Use Cases

**Machine Learning Applications:**
- RAM performance prediction based on size, frequency, bandwidth, voltage, and latency
- Employee salary prediction based on experience, education, and location
- Energy consumption forecasting based on multiple environmental factors

**Deep Learning Perspective:**
Multiple linear regression corresponds to a single dense layer with multiple inputs and one output (no activation):

$$y = \mathbf{W}^T\mathbf{x} + b = \sum_{i=1}^{n} W_i x_i + b$$

See [Multiple/README.md](Multiple/README.md) for detailed implementation and examples.

### Logistic Regression

Logistic regression is a classification algorithm used to predict binary outcomes (0 or 1, True or False). Despite its name, it is a classification method, not a regression technique.

#### Mathematical Formula

Logistic regression uses the **sigmoid function** (also called the logistic function) to map any real-valued input to a probability between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The model predicts the probability that the output belongs to the positive class:

$$P(y=1|x) = \sigma(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n)$$

Or in compact form:

$$P(y=1|\mathbf{x}) = \sigma(\boldsymbol{\beta}^T\mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\beta}^T\mathbf{x}}}$$

Where $z = \boldsymbol{\beta}^T\mathbf{x} = \beta_0 + \sum_{i=1}^{n} \beta_i x_i$.

The model is trained by maximizing the **log-likelihood** or equivalently minimizing the **Binary Cross-Entropy (BCE) loss**:

$$\text{BCE} = -\frac{1}{m}\sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

Where $\hat{y}_i = P(y=1|x_i)$ is the predicted probability.

#### Use Cases

**Machine Learning Applications:**
- Medical diagnosis (tumor classification: benign vs. malignant)
- Email spam detection
- Customer churn prediction
- Credit default prediction

**Deep Learning Perspective:**
Logistic regression is equivalent to a single-layer neural network with a sigmoid activation function:

$$y = \sigma(Wx + b)$$

For multiclass classification, the sigmoid is replaced with the **softmax function**:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

See [Logistic/README.md](Logistic/README.md) for detailed implementation and examples.

## Regression in Generative AI and Large Language Models

### Linear and Multiple Regression in AI Context

**Definition and Fundamental Difference:**

Linear and multiple regression models predict continuous or discrete outputs based on fixed, structured data with explicitly defined features. In contrast, Large Language Models (LLMs) are generative models that create new content (text, code, images) by learning complex, high-dimensional probability distributions over sequences.

**Key Distinctions:**

1. **Task Type**:
   - Regression: Predictive AI - mapping known inputs to known outputs
   - LLMs: Generative AI - creating novel outputs from learned patterns

2. **Mathematical Relationship**:
   - Linear Regression: $y = \beta_0 + \beta_1 x$ (direct functional mapping)
   - LLMs: $P(\text{next token} | \text{context})$ (probabilistic sequence modeling)

3. **Model Complexity**:
   - Regression: Few parameters (coefficients for each feature)
   - LLMs: Billions of parameters (e.g., GPT-3 has 175 billion parameters)

**Role in AI Systems:**

While linear models are not directly used for text generation, they serve important auxiliary functions:

- **Feature Analysis**: Understanding correlations in training data that inform AI architectures
- **Interpretability**: Providing explainable baselines for complex model decisions
- **Embeddings**: Analyzing relationships in vector spaces generated by LLMs
- **Fine-tuning**: Simple linear classifiers on top of frozen LLM embeddings

### Logistic Regression in LLM Context

Logistic regression plays a more direct role in modern LLM systems as a classification and decision-making component.

**Classification Heads:**

Many LLM applications use logistic regression as a final classification layer:

```
LLM Embeddings → Logistic Regression → Binary/Multiclass Decision
```

**Example Applications:**

1. **Sentiment Analysis**:
   - Input: Text → LLM → Embedding vector
   - Logistic Regression: Embedding → P(positive) vs. P(negative)

2. **Content Moderation**:
   - Input: User message → LLM → Semantic representation
   - Logistic Regression: Representation → P(safe) vs. P(unsafe)

3. **Intent Classification**:
   - Input: User query → LLM → Contextual embedding
   - Logistic Regression: Embedding → Intent category

**Advantages:**

- **Interpretability**: Logistic regression provides clear decision boundaries
- **Efficiency**: Lightweight compared to fine-tuning entire LLMs
- **Transfer Learning**: Frozen LLM backbone + trainable logistic head

**Research Finding:**

According to research on explainable AI, logistic regression applied to embeddings from smaller LLMs can match or exceed the performance of large LLMs on specific classification tasks while providing better interpretability and reduced computational cost.

### In-Context Learning and Regression

Recent research demonstrates that LLMs can perform regression tasks directly through **in-context learning** without explicit training:

**Mechanism:**

When provided with examples in the prompt (few-shot learning), LLMs can:
1. Recognize the pattern as a regression problem
2. Infer the underlying relationship
3. Generate predictions for new inputs

**Example:**

```
Prompt:
x: 10 → y: 25
x: 20 → y: 45
x: 30 → y: 65
x: 15 → y: ?

LLM Output: 35 (implicitly learning y = 2x + 5)
```

This suggests that LLMs internally develop mechanisms analogous to regression algorithms, enabling them to perform numerical reasoning tasks that traditional regression models handle explicitly.

**Reward Models for LLM Training:**

Large language models often improve through **Reinforcement Learning from Human Feedback (RLHF)**, where a reward model is trained to predict human preferences:

$$R(\text{prompt}, \text{response}) = \text{reward score}$$

This reward model often uses regression techniques (linear or more complex) to map prompt-response pairs to scalar reward values, guiding the LLM's training process.

Reference: [Simulating Large Systems with Regression Language Models](https://research.google/blog/simulating-large-systems-with-regression-language-models/)

## Transformer Architecture and Regression

### Analysis of "Attention Is All You Need"

The seminal paper "Attention Is All You Need" by Vaswani et al. (2017) introduced the Transformer architecture, which revolutionized natural language processing and became the foundation for modern LLMs (BERT, GPT, T5, etc.). Importantly, the Transformer **does not use traditional statistical regression models** like simple linear regression, multiple regression, or logistic regression in their textbook forms.

Instead, the architecture extensively employs **linear transformations** (matrix multiplications) and **feed-forward networks** (fully connected layers) as fundamental building blocks. These components can be understood as generalized, learnable extensions of regression concepts, adapted for high-dimensional sequence modeling.

### Linear Projections in Multi-Head Attention

**Core Mechanism:**

The multi-head attention mechanism is the defining feature of Transformers. For each token in the input sequence, three distinct representations are created: **Query (Q)**, **Key (K)**, and **Value (V)**. These are generated through linear transformations of the input embeddings.

**Mathematical Formulation:**

Given an input matrix $\mathbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}$ where $n$ is the sequence length and $d_{\text{model}}$ is the embedding dimension:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$$
$$\mathbf{K} = \mathbf{X}\mathbf{W}^K$$
$$\mathbf{V} = \mathbf{X}\mathbf{W}^V$$

Where:
- $\mathbf{W}^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$ is the learned query projection matrix
- $\mathbf{W}^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ is the learned key projection matrix
- $\mathbf{W}^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ is the learned value projection matrix
- $d_k$ and $d_v$ are the dimensions of the key and value vectors (typically $d_k = d_v = d_{\text{model}} / h$ where $h$ is the number of heads)

**Multi-Head Attention:**

The original paper uses $h = 8$ parallel attention heads, meaning each head has its own set of unique weight matrices $(W_i^Q, W_i^K, W_i^V)$ for $i = 1, \ldots, 8$.

The attention score for a single head is computed as:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

All heads are then concatenated and projected:

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

Where $\mathbf{W}^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ is the output projection matrix.

**Connection to Multiple Linear Regression:**

Each linear projection $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$ is mathematically equivalent to applying multiple linear regression to project the input into a different subspace. However, unlike traditional regression that predicts a fixed target, these projections are:
- **Learned end-to-end** through backpropagation
- **Task-adaptive** (optimized for the specific NLP task)
- **Multi-dimensional** (projecting to high-dimensional spaces rather than scalar outputs)

### Position-wise Feed-Forward Networks

**Architecture:**

Each encoder and decoder layer contains a position-wise feed-forward network (FFN) applied identically to each token position independently.

**Mathematical Formulation:**

$$\text{FFN}(x) = \max(0, x\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

Or equivalently:

$$\text{FFN}(x) = \text{ReLU}(x\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

Where:
- $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ is the first layer weight matrix
- $\mathbf{b}_1 \in \mathbb{R}^{d_{ff}}$ is the first layer bias
- $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$ is the second layer weight matrix
- $\mathbf{b}_2 \in \mathbb{R}^{d_{\text{model}}}$ is the second layer bias
- $d_{ff} = 2048$ in the original paper (intermediate dimension)
- $d_{\text{model}} = 512$ (model dimension)
- $\text{ReLU}(z) = \max(0, z)$ is the Rectified Linear Unit activation

**Connection to Regression:**

- The first linear transformation $x\mathbf{W}_1 + \mathbf{b}_1$ is a multiple linear regression expanding the representation
- The ReLU activation introduces non-linearity, enabling the network to learn complex patterns beyond linear relationships
- The second linear transformation $\mathbf{W}_2 + \mathbf{b}_2$ projects back to the original dimension

This two-layer FFN with non-linear activation can be viewed as a **non-linear regression model**, capable of approximating complex functions through learned weight matrices.

### Final Linear Projection and Softmax

**Decoder Output Layer:**

Before producing the final word probabilities, the decoder's output passes through a linear projection layer followed by a softmax activation.

**Mathematical Formulation:**

$$\text{logits} = \mathbf{h}_{\text{decoder}}\mathbf{W}_{\text{output}} + \mathbf{b}_{\text{output}}$$

$$P(\text{word}_i) = \text{softmax}(\text{logits})_i = \frac{e^{\text{logits}_i}}{\sum_{j=1}^{|\mathcal{V}|} e^{\text{logits}_j}}$$

Where:
- $\mathbf{h}_{\text{decoder}} \in \mathbb{R}^{d_{\text{model}}}$ is the final decoder hidden state
- $\mathbf{W}_{\text{output}} \in \mathbb{R}^{d_{\text{model}} \times |\mathcal{V}|}$ is the output projection matrix
- $|\mathcal{V}|$ is the vocabulary size (e.g., 30,000-50,000 tokens)
- $\text{logits} \in \mathbb{R}^{|\mathcal{V}|}$ are the unnormalized scores for each vocabulary word

**Connection to Multinomial Logistic Regression:**

This final layer is mathematically equivalent to **multinomial logistic regression** (also called softmax regression):
- The linear projection computes scores for each class (vocabulary word)
- The softmax function converts scores to a probability distribution
- The model is trained to maximize the log-likelihood of the correct next word

However, instead of predicting a simple categorical outcome, the Transformer predicts over a vast vocabulary while conditioning on the entire preceding sequence through the attention mechanism.

### Comparison to Traditional Regression

The following table summarizes how Transformer components relate to classical regression models:

| Transformer Component | Traditional Regression Analogue | Key Difference |
|----------------------|--------------------------------|----------------|
| **Linear Projection (Q, K, V)** | Multiple Linear Regression | Projects inputs into high-dimensional semantic subspaces for similarity computation rather than predicting a single target variable |
| **Feed-Forward Network (FFN)** | Non-linear Regression / Multi-Layer Perceptron | Uses stacked linear layers with ReLU activation to learn complex, non-linear transformations rather than fitting a simple polynomial |
| **Final Linear + Softmax** | Multinomial Logistic Regression | Predicts probability distribution over large vocabulary (30K+ words) conditioned on full sequence context, rather than simple multi-class classification |
| **Overall Architecture** | Composite Function Approximation | Combines hundreds of linear transformations and non-linearities across multiple layers and attention heads to model sequential dependencies |

**Summary:**

The Transformer architecture replaces traditional statistical regression with:
1. **Learnable linear projections**: Adaptively mapping inputs to task-specific representations
2. **Multi-head attention**: Enabling each token to attend to relevant context through learned query-key-value mechanisms
3. **Position-wise feed-forward networks**: Applying non-linear transformations to enhance representational capacity
4. **Softmax output layer**: Predicting next-token probabilities over large vocabularies

These components, while rooted in linear algebra and regression principles, are generalized and composed to handle the complexity of natural language understanding and generation.

## References

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** "Attention Is All You Need." *Advances in Neural Information Processing Systems* (NeurIPS), 30.
   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

2. **Google Research Blog.** "Simulating Large Systems with Regression Language Models."
   [https://research.google/blog/simulating-large-systems-with-regression-language-models/](https://research.google/blog/simulating-large-systems-with-regression-language-models/)

3. **Scikit-learn Documentation.** "Linear Models."
   [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)

4. **Scikit-learn API.** "sklearn.linear_model.LinearRegression."
   [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

5. **Scikit-learn API.** "sklearn.linear_model.LogisticRegression."
   [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

6. **Google Developers.** "Linear Regression - Machine Learning Crash Course."
   [https://developers.google.com/machine-learning/crash-course/linear-regression](https://developers.google.com/machine-learning/crash-course/linear-regression)

7. **LabXchange.** "Linear Regression."
   [https://www.labxchange.org/library/items/lb:LabXchange:98e0c993:html:1](https://www.labxchange.org/library/items/lb:LabXchange:98e0c993:html:1)

8. **University of Virginia Library.** "Logistic Regression Four Ways with Python."
   [https://library.virginia.edu/data/articles/logistic-regression-four-ways-with-python](https://library.virginia.edu/data/articles/logistic-regression-four-ways-with-python)

9. **AWS.** "What's the Difference Between Linear Regression and Logistic Regression?"
   [https://aws.amazon.com/compare/the-difference-between-linear-regression-and-logistic-regression/](https://aws.amazon.com/compare/the-difference-between-linear-regression-and-logistic-regression/)

10. **Towards Data Science.** "Linear Regression to GPT in Seven Steps."
    [https://towardsdatascience.com/linear-regression-to-gpt-in-seven-steps-cb3ab3173a14/](https://towardsdatascience.com/linear-regression-to-gpt-in-seven-steps-cb3ab3173a14/)

---

**Note:** Each sub-folder (Linear, Multiple, Logistic) contains detailed README files with specific implementations, mathematical derivations, and use case examples.