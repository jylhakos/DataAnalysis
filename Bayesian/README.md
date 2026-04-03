# Data Analysis with Bayesian

This repository contains a practical implementation of Naive Bayes classification for email spam filtering, demonstrating the application of Bayesian statistical methods in data analysis.

## Table of Contents

- [What is Bayes' Theorem?](#what-is-bayes-theorem)
- [Bayesian vs. Naive Bayes](#bayesian-vs-naive-bayes)
- [What is Naive Bayes Classification?](#what-is-naive-bayes-classification)
- [Why is Naive Bayes Called "Naive"?](#why-is-naive-bayes-called-naive)
- [Real-World Use Case: Email Spam Filtering](#real-world-use-case-email-spam-filtering)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Running the Code](#running-the-code)
- [References](#references)

## What is Bayes' Theorem?

**Bayes' theorem** is a mathematical formula used to calculate **conditional probability** — the likelihood of an event occurring based on prior knowledge of conditions related to that event. It updates the probability of a hypothesis as more evidence becomes available.

### Bayes' Theorem Formula

$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$

### Formula Components

- **$P(H|E)$ (Posterior)**: The probability of hypothesis $H$ given evidence $E$
- **$P(E|H)$ (Likelihood)**: The probability of evidence $E$ given that hypothesis $H$ is true
- **$P(H)$ (Prior)**: The initial probability of hypothesis $H$ before seeing evidence $E$
- **$P(E)$ (Marginal/Evidence)**: The total probability of observing evidence $E$

### Intuitive Explanation

Bayes' theorem allows us to **update our beliefs** based on new information. It answers the question: "Given that I've observed some evidence, what's the probability that my hypothesis is true?"

## Bayesian vs. Naive Bayes

### Bayesian Methods

**Bayesian methods** refer to a broad probabilistic approach that uses Bayes' Theorem to update beliefs and make inferences. They encompass various techniques including:

- Bayesian inference
- Bayesian networks
- Hierarchical Bayesian models
- Bayesian optimization

### Naive Bayes

**Naive Bayes** is a **specific, simplified classification algorithm** that applies Bayes' theorem with a strong assumption: all features are conditionally independent given the class label.

**Key Difference**: Bayesian methods are general probabilistic tools, while Naive Bayes is a particular classification algorithm that makes simplifying assumptions for computational efficiency.

## What is Naive Bayes Classification?

**Naive Bayes classification** is a probabilistic machine learning algorithm based on Bayes' theorem. It is used for classification tasks where we want to assign a label (class) to an input based on its features.

### How It Works

The Naive Bayes Classifier:

1. **Calculates prior probabilities** of each class: $P(\text{Class})$
2. **Calculates likelihoods** of each feature given a class: $P(\text{Feature}|\text{Class})$
3. **Computes posterior probabilities** using Bayes' theorem
4. **Classifies** the input based on the highest posterior probability

### Mathematical Formulation

For a set of features $X = (x_1, x_2, ..., x_n)$ and classes $C_k$:

$$P(C_k|X) = \frac{P(C_k) \cdot P(X|C_k)}{P(X)}$$

With the naive assumption of independence:

$$P(X|C_k) = \prod_{i=1}^{n} P(x_i|C_k)$$

The classification decision is:

$$\hat{y} = \arg\max_{C_k} P(C_k) \prod_{i=1}^{n} P(x_i|C_k)$$

## Why is Naive Bayes Called "Naive"?

The algorithm is called **"naive"** because of a **strong independence assumption** it makes:

### The Naive Assumption

**All features are conditionally independent of each other given the class label.**

This means:
- The presence of a particular feature in a class is **unrelated** to the presence of any other feature
- Each feature contributes **independently** to the probability of a certain outcome

### Example: Email Spam Classification

In spam filtering:
- The word "Winner" and "Prize" often appear together in spam emails
- However, Naive Bayes treats them as **independent**
- It assumes knowing that "Winner" appears tells us nothing about whether "Prize" appears

### Why is This Assumption "Naive"?

In **real-world scenarios**, features are often **correlated or dependent** on each other:

- In text: words often appear together in patterns
- In medical diagnosis: symptoms are often related
- In finance: market indicators are interconnected

Despite this "naive" assumption, the algorithm often performs **surprisingly well** in practice, especially for:
- Text classification
- Spam filtering
- Sentiment analysis
- Document categorization

## Real-World Use Case: Email Spam Filtering

A classic application of Naive Bayes is **email spam filtering**, where emails are classified as either **"Spam"** (junk) or **"Ham"** (legitimate) based on word occurrences.

### Why Naive Bayes Works Well for Spam Filtering

1. **Computationally fast**: Can process emails in real-time
2. **High-dimensional data**: Handles thousands of unique words efficiently
3. **Robust performance**: Works well even with the independence assumption
4. **Simple to implement**: Requires minimal training data

### How It Works

1. **Training Phase**:
   - Count word occurrences in spam and ham emails
   - Calculate prior probabilities: $P(\text{Spam})$ and $P(\text{Ham})$
   - Calculate likelihoods: $P(\text{word}|\text{Spam})$ and $P(\text{word}|\text{Ham})$

2. **Prediction Phase**:
   - For a new email, extract words
   - Calculate: $P(\text{Spam}|\text{words})$ and $P(\text{Ham}|\text{words})$
   - Classify as spam if $P(\text{Spam}|\text{words}) > P(\text{Ham}|\text{words})$

### Implementation Steps

The example in `naive_bayes.py` demonstrates:

1. **Data Preparation**: Convert text into tokens (words)
2. **Model Training**: Count word frequencies in spam vs. ham
3. **Smoothing**: Use Laplace smoothing (k-smoothing) to handle unseen words
4. **Prediction**: Calculate log probabilities to avoid numerical underflow

### Naive Bayes Variants

The `scikit-learn` library provides several Naive Bayes implementations:

- **MultinomialNB**: Used when features represent word counts (most common for spam)
- **BernoulliNB**: Used if features are binary (word present vs. not present)
- **GaussianNB**: Used for continuous/numeric features with Gaussian distribution

## Virtual Environment Setup

A virtual environment isolates your Python project dependencies, preventing conflicts with system-wide packages. Follow these steps to set up and use a virtual environment on Linux.

### Step 1: Create Virtual Environment

Open a terminal in the project directory and run:

```bash
python3 -m venv venv
```

This creates a folder named `venv` containing the isolated Python environment.

### Step 2: Activate Virtual Environment

**Always activate the virtual environment before installing libraries or running scripts:**

```bash
source venv/bin/activate
```

You'll see `(venv)` prefix in your terminal prompt, indicating the environment is active.

### Step 3: Upgrade pip (Optional but Recommended)

```bash
pip install --upgrade pip
```

### Step 4: Install Required Dependencies

The `naive_bayes.py` script uses only Python standard library, but for enhanced functionality, you can install:

```bash
pip install scikit-learn numpy pandas
```

For the basic version, no external dependencies are needed.

### Step 5: Deactivate Virtual Environment

When you're done working:

```bash
deactivate
```

### Using Virtual Environment with VS Code

#### Method 1: VS Code Integrated Terminal

1. Open VS Code in the project folder:
   ```bash
   code .
   ```

2. Open the integrated terminal: **View** → **Terminal** or press `` Ctrl+` ``

3. The virtual environment should auto-activate. If not, activate it manually:
   ```bash
   source venv/bin/activate
   ```

#### Method 2: Select Python Interpreter

1. Press `Ctrl+Shift+P` to open the command palette
2. Type: **Python: Select Interpreter**
3. Choose the interpreter from `./venv/bin/python`

VS Code will now use this virtual environment for all Python operations.

## Running the Code

### Prerequisites

- Python 3.6 or higher
- Virtual environment activated (see above)

### Step-by-Step Execution

#### 1. Activate Virtual Environment

```bash
source venv/bin/activate
```

#### 2. Run the Naive Bayes Script

```bash
python naive_bayes.py
```

#### 3. Expected Output

The script will:
- Train a Naive Bayes classifier on sample messages
- Run assertions to verify correctness
- Demonstrate spam probability prediction
- Display example classifications

Example output:
```
Training completed successfully!
Vocabulary size: 4
Spam messages: 1
Ham messages: 2

Testing message: 'hello spam'
Spam probability: 0.83

All tests passed!
```

### Testing the Classifier

The script includes several test cases:

1. **Tokenization Test**: Verifies text is properly split into words
2. **Training Test**: Confirms message counts and word frequencies
3. **Prediction Test**: Checks probability calculations

### Extending the Example

To test with your own data, modify the `messages` list:

```python
messages = [
    Message("Get rich quick! Winner winner!", is_spam=True),
    Message("Meeting scheduled for tomorrow", is_spam=False),
    Message("Click here for free prize", is_spam=True),
    Message("Project deadline reminder", is_spam=False),
]
```

### Using Scikit-Learn (Alternative)

For production use, consider using `scikit-learn`:

```bash
pip install scikit-learn
```

Example with scikit-learn:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Prepare data
texts = ["spam rules", "ham rules", "hello ham"]
labels = [1, 0, 0]  # 1 = spam, 0 = ham

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Predict
test_text = vectorizer.transform(["hello spam"])
prediction = model.predict(test_text)
```

## References

### Academic Resources

1. **Cornell University - CS 4780 Lecture Notes**
   [Naive Bayes Classification](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html)
   The lecture notes on Bayesian methods and Naive Bayes classification

2. **Scikit-Learn Documentation**
   [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
   Official documentation for Naive Bayes implementations in scikit-learn

3. **Stanford CS229 - Machine Learning**
   [Generative Learning Algorithms](http://cs229.stanford.edu/notes/cs229-notes2.pdf)
   Detailed mathematical derivations of Naive Bayes

### Books

4. **"Data Science from Scratch" by Joel Grus**
   Chapter on Naive Bayes with Python implementation

5. **"Pattern Recognition and Machine Learning" by Christopher Bishop**
   Coverage of Bayesian methods (Chapter 4)

6. **"Machine Learning: A Probabilistic Perspective" by Kevin Murphy**
   In-depth treatment of probabilistic models including Naive Bayes

### Online Tutorials

7. **Towards Data Science**
   [Naive Bayes Classifier Explained](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c)

### Python Documentation

8. **Python Virtual Environments**
   [venv — Creation of virtual environments](https://docs.python.org/3/library/venv.html)

9. **Python Math Module**
    [Mathematical functions](https://docs.python.org/3/library/math.html)

---
