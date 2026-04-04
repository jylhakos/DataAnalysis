# Probability in Data Analysis

## Overview

Probability formulas are applied in data analysis to quantify uncertainty, model randomness, and make predictions based on data. Descriptive statistics is about describing and summarizing data. The quantitative approach describes and summarizes data numerically. Probability formulas in data analysis quantify uncertainty and likelihood, transforming raw data into actionable predictions and decisions.

We view the sample of data that we collect and analyze as a selection from a larger population, and our goal in data analysis is to make statements about the population, not only about the sample. In a probability-based approach to statistics, the sample is viewed as a random "draw" from the population.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
  - [Troubleshooting](#troubleshooting)
- [Probability Terminology](#-probability-terminology)
  - [Basic Concepts](#basic-concepts)
  - [Probability Distribution](#probability-distribution)
  - [Joint and Conditional Distributions](#joint-and-conditional-distributions)
- [Key Probability Formulas](#-key-probability-formulas)
  - [Conditional Probability](#conditional-probability)
  - [Bayes' Theorem](#bayes-theorem)
  - [Probability Laws](#probability-laws)
  - [Expected Value](#expected-value)
  - [Variance and Standard Deviation](#variance-and-standard-deviation)
- [Probability Distributions](#-probability-distributions)
  - [Random Variables](#random-variables)
  - [Distribution Functions](#distribution-functions)
  - [Discrete Distributions](#discrete-distributions)
  - [Continuous Distributions](#continuous-distributions)
- [Applications in Data Analysis](#applications-in-data-analysis)
- [Bias-Variance Tradeoff](#bias-variance-tradeoff)
- [Python Libraries for Probability and Statistics](#python-libraries-for-probability-and-statistics)
- [Python Code Examples](#python-code-examples)
  - [Generating Random Variables](#generating-random-variables)
  - [Working with Probability Distributions](#working-with-probability-distributions)
  - [Calculating Joint and Conditional Probabilities](#calculating-joint-and-conditional-probabilities)
  - [Visualizing Distributions](#visualizing-distributions)
  - [Parameter Estimation](#parameter-estimation-maximum-likelihood-estimation)
- [Statistical Inference](#statistical-inference)
  - [Overview: Statistical Analysis vs Probability Analysis](#overview-statistical-analysis-vs-probability-analysis)
  - [Population vs Sample](#population-vs-sample)
  - [Parameters vs Statistics](#parameters-vs-statistics)
  - [Point Estimation](#point-estimation)
  - [Interval Estimation](#interval-estimation)
  - [Confidence Intervals](#confidence-intervals)
  - [Hypothesis Testing](#hypothesis-testing)
  - [P-value](#p-value)
  - [Statistical Models and Inference](#statistical-models-and-inference)
  - [Statistical Significance vs Practical Significance](#statistical-significance-vs-practical-significance)
  - [Bayesian Inference](#bayesian-inference)
- [Project Structure](#-project-structure)
- [Example Scripts](#example-scripts)
- [Using the Data Folder](#using-the-data-folder)
  - [Generating Sample Datasets](#generating-sample-datasets)
  - [Available Datasets](#available-datasets)
  - [Loading Datasets in Python](#loading-datasets-in-python)
- [Further Reading](#further-reading)
- [Reference](#reference)
  - [Common Commands](#common-commands)
  - [Dataset Example](#dataset-example)
  - [Project Workflow](#project-workflow)

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setting Up the Virtual Environment

Follow these steps to set up and run the probability examples:

#### 1. Create Virtual Environment

```bash
# Navigate to the project directory
cd /path/to/Probability

# Create a virtual environment named 'probability_env'
python3 -m venv probability_env
```

#### 2. Activate Virtual Environment

**On Linux/MacOS:**
```bash
source probability_env/bin/activate
```

**On Windows:**
```bash
probability_env\Scripts\activate
```

After activation, you should see `(probability_env)` in your terminal prompt.

#### 3. Install Required Packages

```bash
# Update pip to the latest version
pip install --upgrade pip

# Install all required libraries from requirements.txt
pip install -r requirements.txt
```

This will install:
- numpy (≥1.21.0) - Numerical computing
- scipy (≥1.7.0) - Scientific computing and probability distributions
- pandas (≥1.3.0) - Data manipulation and analysis
- matplotlib (≥3.4.0) - Data visualization
- scikit-learn (≥1.0.0) - Machine learning and statistical modeling

**Verify Installation:**
```bash
python check_setup.py
```

This script will check that all required packages are properly installed.

#### 4. Generate Sample Datasets (Optional)

Generate realistic datasets for practicing probability analysis:

```bash
python generate_data.py
```

This creates five CSV datasets in `data/sample_datasets/`:
- **customer_purchases.csv** - Customer purchase data for conditional probability (1,000 records)
- **student_test_scores.csv** - Test scores for normal distribution analysis (500 students)
- **website_traffic.csv** - Hourly traffic for Poisson distribution (365 days)
- **machine_failures.csv** - Failure times for exponential distribution (200 machines)
- **ab_test_results.csv** - A/B test data for hypothesis testing (10,000 visitors)

**View dataset summary:**
```bash
python generate_data.py --summary
```

**Analyze the generated datasets:**
```bash
python examples/analyze_datasets.py
```

#### 5. Run the Examples

**Option A: Run all examples at once**
```bash
python run_examples.py
```

**Option B: Run individual examples**
```bash
# Probability distributions
python examples/distributions.py

# Conditional probability and Bayes' Theorem
python examples/conditional_probability.py

# Bias-variance tradeoff
python examples/bias_variance.py

# Analyze generated datasets
python examples/analyze_datasets.py
```

#### 6. Visualizations

All generated plots are saved in the `examples/` directory as PNG files:
- `normal_distribution.png`
- `binomial_distribution.png`
- `poisson_distribution.png`
- `exponential_distribution.png`
- `uniform_distribution.png`
- `fitted_distribution.png`
- `conditional_probability.png`
- `bayes_theorem.png`
- `bayesian_updating.png`
- `bias_variance_models.png`
- `bias_variance_tradeoff.png`

**From dataset analysis (if generated):**
- `customer_analysis.png`
- `test_scores_analysis.png`
- `website_traffic_analysis.png`

#### 7. Deactivate Virtual Environment

When you're done working:
```bash
deactivate
```

### Troubleshooting

**Issue: "python3: command not found"**
- Try using `python` instead of `python3`
- Ensure Python is installed and added to your PATH

**Issue: "pip: command not found"**
- Install pip: `python -m ensurepip --upgrade`

**Issue: Permission denied**
- Use `python3 -m venv probability_env` instead of `sudo`
- Ensure you have write permissions in the directory

**Issue: Module import errors**
- Verify the virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## 📁 Probability Terminology

### Basic Concepts

- **Outcome**: The result of a single trial in a probability experiment.
- **Sample Space**: The set of all possible outcomes in a probability experiment.
- **Event**: Some subset of the sample space.
- **Independence**: Events that don't affect each other.

### Probability Distribution

A probability distribution is formally defined by two entities:
1. **Sample Space**: The collection of values that can be drawn (can be either finite or infinite)
2. **Probability Mass Function (pmf)**: A function on the sample space that tells us the probability of observing each point when we take a draw

An event is a subset of the sample space. We say that the event "occurs" if the value we draw is in the event. The probability of an event is the sum of all probabilities in the event.

Another term for a probability distribution is a **random variable**, usually denoted with a capital letter, such as $X$.

### Joint and Conditional Distributions

- **Joint Distribution**: A probability distribution when we observe multiple random variables together (e.g., $X$ and $Y$ observed jointly)
- **Independence**: Informally, $X$ and $Y$ are independent if knowing $X$ does not tell us anything about the value of $Y$, and vice versa
- **Conditional Distribution**: A distribution obtained by starting with a joint distribution, and renormalizing it so that each row or each column sums to 1
- **Conditional Independence**: Combining the notions of conditioning and independence

## 📊 Key Probability Formulas

### Conditional Probability

Evaluates the likelihood of an event occurring based on prior knowledge.

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

where $P(B) > 0$

### Bayes' Theorem

Allows analysts to revise probability estimates when new data becomes available.

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Or in expanded form:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|\neg A) \cdot P(\neg A)}$$

Bayes' Theorem is a statistical technique that allows for the revision of probability estimates based on new information or evidence, enabling more accurate and efficient decision-making in uncertain situations.

### Probability Laws

**Additive Law** (for mutually exclusive events):
$$P(A \cup B) = P(A) + P(B)$$

**General Addition Rule**:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Multiplicative Law** (for independent events):
$$P(A \cap B) = P(A) \cdot P(B)$$

**Complement Rule**:
$$P(A^c) = 1 - P(A)$$

### Relative Frequency Probability

A method of determining the likelihood of an event occurring based on the observed frequency of its occurrence in a given sample or population:

$$P(A) = \frac{\text{Number of times Event A occurs}}{\text{Total number of trials}}$$

### Expected Value

Determines the average outcome, such as forecasting expected revenue.

For a discrete random variable:
$$E[X] = \sum_{i} x_i \cdot P(X = x_i)$$

For a continuous random variable:
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

### Variance and Standard Deviation

Measures data spread to evaluate risk.

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

$$\sigma = \sqrt{\text{Var}(X)}$$

## 📈 Probability Distributions

### Random Variables

- **Random Variable**: A variable whose value is a numerical outcome of a random phenomenon. It can be discrete or continuous.

### Distribution Functions

- **Probability Mass Function (PMF)**: Gives the probability of a random variable taking on a specific value for discrete random variables.
- **Probability Density Function (PDF)**: Describes the relative likelihood of a random variable taking on a specific value for continuous random variables.
- **Cumulative Distribution Function (CDF)**: Gives the probability that a random variable is less than or equal to a specific value.

$$F(x) = P(X \leq x)$$

### Discrete Distributions

#### Bernoulli Distribution
Models a binary random variable with two possible outcomes: success (1) or failure (0).

$$P(X = k) = p^k(1-p)^{1-k}, \quad k \in \{0,1\}$$

#### Binomial Distribution
Models the number of successes in fixed trials, such as calculating the probability of a specific number of clicks on an ad.

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

where $n$ is the number of trials, $k$ is the number of successes, and $p$ is the probability of success.

#### Poisson Distribution
Models the number of times an event occurs within a specific interval, such as website traffic per hour.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

where $\lambda$ is the average rate of occurrence.

### Continuous Distributions

#### Uniform Distribution
Models a random variable equally likely to take on any value within a specified range.

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

#### Normal Distribution
Models continuous data (e.g., heights, test scores), allowing analysts to estimate the probability of data falling within certain standard deviations.

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

where $\mu$ is the mean and $\sigma$ is the standard deviation.

**Why is the normal distribution so important?**

The normal distribution (the bell curve) is fundamental in statistics due to the **Central Limit Theorem**. This theorem states that the distribution of sample means tends to be normal, regardless of the underlying population distribution, as the sample size increases.

#### Exponential Distribution
Models the time between events in a Poisson process.

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

## Applications in Data Analysis

### 1. Predicting Outcomes (Conditional Probability)
Evaluates the likelihood of an event occurring based on prior knowledge, such as customer reorder rates.

### 2. Modeling Uncertainty (Distributions)
- **Normal Distribution**: Used for continuous data (e.g., heights, test scores)
- **Binomial Distribution**: Models the number of successes in fixed trials
- **Poisson Distribution**: Models the number of events within specific intervals

### 3. Bayesian Inference
Allows analysts to revise probability estimates when new data becomes available.

### 4. Risk Assessment (Expectation & Variance)
- **Expected Value**: Determines the average outcome
- **Variance/Standard Deviation**: Measures data spread to evaluate risk

### 5. Hypothesis Testing (Central Limit Theorem)
Determines if results are statistically significant, relying on the probability that a sample mean approximates the population mean.

## Bias-Variance Tradeoff

The expectation of the squared error can be decomposed into bias, variance, and noise.

Consider a random variable $Z$ with a probability distribution given by $P(Z)$. We denote the expectation value of $Z$ as $E[Z]$. The expectation value of $(Z - E[Z])^2$ gives us the variance.

The expected prediction error when estimating a model $\hat{f}(x)$ can be decomposed as:

$$E[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2$$

where:
- **Variance** tells us how sensitive the model is to small fluctuations in the training set
- **Bias** is related to the difference between the expected value of our estimator and its true value
- **Noise** ($\sigma^2$) is the irreducible error

**Key Insights:**
- High variance results in overfitting
- High bias results in underfitting
- High variance gives us more complex models
- High bias yields simpler ones
- Finding a good model is a matter of balancing the bias and the variance

## Python Libraries for Probability and Statistics

### Built-in Library

- **statistics**: Python's built-in library for descriptive statistics
  - Documentation: https://docs.python.org/3/library/statistics.html

### Third-Party Libraries

- **NumPy**: Third-party library for numerical computing, optimized for working with single- and multi-dimensional arrays
  - Documentation: https://docs.scipy.org/doc/numpy/user/index.html

- **SciPy**: Third-party library for scientific computing based on NumPy
  - Getting Started: https://www.scipy.org/getting-started.html
  - Probability Distributions: https://docs.scipy.org/doc/scipy/tutorial/stats/probability_distributions.html

- **Pandas**: Third-party library for numerical computing based on NumPy. Excels in handling labeled one-dimensional (1D) data with Series objects and two-dimensional (2D) data with DataFrame objects
  - Documentation: https://pandas.pydata.org/pandas-docs/stable/

- **Matplotlib**: Third-party library for data visualization
  - Documentation: https://matplotlib.org/

- **PyMC3** and **Stan**: Powerful tools for implementing Bayesian models using Markov Chain Monte Carlo (MCMC)

### Additional Resources

- Datasets: https://github.com/realpython/python-data-cleaning

## Python Code Examples

> **Note**: Full working examples are available in the `examples/` directory. The code snippets below demonstrate key concepts. Run the complete scripts for demonstrations with visualizations.

### Setting Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv probability_env

# Activate virtual environment (Linux/MacOS)
source probability_env/bin/activate

# Activate virtual environment (Windows)
probability_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Generating Random Variables

```python
import numpy as np

# Generate 100 random numbers from a standard normal distribution
random_numbers = np.random.randn(100)

# Generate 100 random numbers from a uniform distribution between 0 and 1
uniform_numbers = np.random.rand(100)

# Generate 100 random numbers from an exponential distribution with rate parameter 2
exponential_numbers = np.random.exponential(scale=0.5, size=100)
```

### Working with Probability Distributions

```python
from scipy.stats import norm

# Probability of a z-score less than 1.96
probability = norm.cdf(1.96)

# PDF of a standard normal distribution at x = 1
pdf_value = norm.pdf(1)
```

```python
from scipy.stats import expon

# CDF of an exponential distribution with rate parameter 2 at x = 3
cdf_value = expon.cdf(3, scale=1/2)
```

```python
from scipy.stats import chi2

# 95th percentile of a chi-squared distribution with 10 degrees of freedom
percentile = chi2.ppf(0.95, 10)
```

### Calculating Joint and Conditional Probabilities

**Joint Probability** is the likelihood of two events occurring simultaneously:

```python
import numpy as np

# Example: Boolean arrays for two events
event_A = np.array([1, 0, 1, 1, 0])
event_B = np.array([1, 1, 1, 0, 0])

# P(A and B)
joint_prob = np.mean((event_A == 1) & (event_B == 1))
```

**Conditional Probability** is the likelihood of event A occurring given that event B has already occurred:

```python
# P(B) is the marginal probability of event B
p_B = np.mean(event_B == 1)

# P(A|B) = P(A and B) / P(B)
conditional_prob = joint_prob / p_B
```

Alternative method using filtering:

```python
# Filter data where B is true, then find the average of A
conditional_prob = event_A[event_B == 1].mean()
```

### Visualizing Distributions

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Generate x-axis values
x = np.linspace(-3, 3, 100)

# Plot the PDF of a standard normal distribution
plt.plot(x, norm.pdf(x))
plt.xlabel('x')
plt.ylabel('PDF')
plt.title('Normal Distribution')
plt.show()
```

### Parameter Estimation (Maximum Likelihood Estimation)

**Maximum Likelihood Estimation (MLE)**: Finds the parameter values that maximise the likelihood of observing the given data.

Example: Fitting a normal distribution to a dataset:

```python
from scipy.stats import norm
import numpy as np

# Sample data
data = np.random.randn(100)

# Fit a normal distribution
params = norm.fit(data)
mean, std = params

print("Estimated mean:", mean)
print("Estimated standard deviation:", std)
```

## Statistical Inference

### Overview: Statistical Analysis vs Probability Analysis

While **statistical analysis** is concerned with actual observations of phenomena, **probability analysis** is concerned with hypothetical observations of phenomena (said to be events). Both statistical estimation studies and statistical association studies rely on logic based on probability analysis.

**Statistical Inference** allows us to ask questions about what we can learn from the data we record. Given a data sample, we use statistical inference to understand some characteristics of the unobserved (probability) distribution that generated the data.

### Population vs Sample

- **Population**: The entire set of items or individuals of interest. For example, all customers of a company, or all possible measurements of a phenomenon.
- **Sample**: A subset of the population used for analysis. Samples are collected because it's often impractical or impossible to measure the entire population.

We view the sample data we collect and analyze as a selection from a larger population. Our goal in data analysis is to make statements about the population, not only about the sample.

### Parameters vs Statistics

- **Parameter**: A numerical characteristic of a population. Parameters are typically unknown and denoted with Greek letters.
  - Example: Population mean $\mu$, population standard deviation $\sigma$, population proportion $p$

- **Statistic**: A numerical characteristic of a sample, calculated from sample data. Statistics are used to estimate parameters.
  - Example: Sample mean $\bar{x}$, sample standard deviation $s$, sample proportion $\hat{p}$

### Point Estimation

**Point Estimation** involves using a sample statistic to estimate a population parameter with a single numerical value.

**Examples of Point Estimates:**
- **Sample Mean** as an estimate for the population mean:
  $$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

- **Sample Proportion** as an estimate for the population proportion:
  $$\hat{p} = \frac{x}{n}$$
  where $x$ is the number of successes and $n$ is the sample size.

- **Sample Variance** as an estimate for the population variance:
  $$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

Point estimates provide a single "best guess" but do not convey uncertainty about the estimate.

### Interval Estimation

**Interval Estimates** give a range of values as the estimate for a parameter, rather than a single number. This approach acknowledges uncertainty and provides more information than point estimates alone.

Interval estimates are built around point estimates, which is why understanding point estimates is essential to understanding interval estimates.

### Confidence Intervals

A **Confidence Interval (CI)** is an interval of values computed from sample data that is likely to contain the true population parameter of interest.

**General Form:**
$$\text{Confidence Interval} = \text{Point Estimate} \pm \text{Margin of Error}$$

**Confidence Interval for Population Mean** (when population standard deviation $\sigma$ is known):
$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

where:
- $\bar{x}$ is the sample mean
- $z_{\alpha/2}$ is the critical value from the standard normal distribution
- $\sigma$ is the population standard deviation
- $n$ is the sample size

**Confidence Interval for Population Mean** (when $\sigma$ is unknown, using t-distribution):
$$\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

where $s$ is the sample standard deviation and $t_{\alpha/2, n-1}$ is the critical value from the t-distribution with $n-1$ degrees of freedom.

**Confidence Interval for Population Proportion:**
$$\hat{p} \pm z_{\alpha/2} \cdot \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

**Interpretation:**
A 95% confidence interval means that if we repeated the sampling process many times, approximately 95% of the resulting intervals would contain the true population parameter. It does NOT mean there is a 95% probability that the true parameter lies within any specific interval.

**Common Confidence Levels:**
- 90% CI: $z_{\alpha/2} = 1.645$
- 95% CI: $z_{\alpha/2} = 1.96$
- 99% CI: $z_{\alpha/2} = 2.576$

### Hypothesis Testing

Hypothesis testing is a structured statistical method to decide if there is enough evidence to reject a research claim about a population parameter.

**Components:**
- **Null Hypothesis** ($H_0$): The default assumption or claim to be tested (e.g., "there is no effect" or "no difference").
- **Alternative Hypothesis** ($H_a$ or $H_1$): The claim we want to find evidence for (e.g., "there is an effect" or "there is a difference").

**Test Statistic:**
A value calculated from sample data used to make a decision about the hypotheses. Common test statistics include:

- **Z-statistic** (for large samples or known population standard deviation):
  $$z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

- **T-statistic** (for small samples with unknown population standard deviation):
  $$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

**Decision Rules:**
- If the p-value $< \alpha$ (significance level), reject $H_0$
- If the p-value $\geq \alpha$, fail to reject $H_0$

**Common Significance Levels:**
- $\alpha = 0.05$ (5% chance of Type I error)
- $\alpha = 0.01$ (1% chance of Type I error)
- $\alpha = 0.10$ (10% chance of Type I error)

Probability distributions are utilized in statistical inference, which involves drawing conclusions about a population based on sample data.

### P-value

The **p-value** represents the probability of obtaining results at least as extreme as the observed results, assuming the null hypothesis ($H_0$) is true.

**Interpretation:**
- A small p-value (typically $p < 0.05$) suggests that the observed data is unlikely under the null hypothesis, providing evidence to reject $H_0$.
- A large p-value suggests that the observed data is consistent with the null hypothesis, so we fail to reject $H_0$.

**Common Thresholds:**
- $p < 0.05$: Statistically significant at the 5% level (common threshold)
- $p < 0.01$: Statistically significant at the 1% level (strong evidence)
- $p < 0.001$: Statistically significant at the 0.1% level (very strong evidence)

**Important Notes:**
- The p-value is NOT the probability that $H_0$ is true.
- The p-value is NOT the probability that the results occurred by chance.
- A statistically significant result (small p-value) does not necessarily mean the effect is large or practically important.

### Statistical Models and Inference

Given a data sample, a **statistical model** is a probability distribution that describes how this data is generated. Using the concept of statistical models, we can say that statistical inference is how we use observed data to infer characteristics of the unobserved probability distribution.

**Two Common Forms of Inference:**
1. **Confidence Intervals**: Provide a range of plausible values for a parameter
2. **Tests of Significance**: Evaluate evidence against a null hypothesis

### Statistical Significance vs Practical Significance

**Statistical Significance** and **Practical Significance** are two distinct concepts that should both be considered when interpreting research results.

#### Statistical Significance

**Definition**: Indicates that research results (such as a difference between groups) are unlikely due to random chance, typically determined by a p-value.

**Key Question**: "Is there evidence of an effect?"

**Characteristics:**
- Probability-based measure using p-values
- Determines if a result is probably not due to random coincidence
- Does not measure the size or importance of the effect
- Can be achieved with very large sample sizes even for tiny, unimportant effects

**Use statistical significance to**: Determine if your data has reliable, non-random patterns.

#### Practical Significance (Clinical Significance)

**Definition**: Refers to the importance, size, or real-world impact of an effect.

**Key Question**: "Is the effect large or important enough to matter in practice?"

**Characteristics:**
- Focuses on effect size and real-world relevance
- Context-dependent (what's meaningful varies by field and application)
- Not determined by p-values alone
- Considers costs, risks, benefits, and feasibility

**Use practical significance to**: Determine if those patterns are valuable for decision-making.

#### Key Distinctions

| Aspect | Statistical Significance | Practical Significance |
|--------|-------------------------|------------------------|
| **Measures** | Probability (p-value) | Effect size and importance |
| **Question** | Is it real? | Does it matter? |
| **Influenced by** | Sample size, variability | Context, domain knowledge |
| **Example** | p = 0.001 | Drug reduces blood pressure by 2 mmHg |

**Important Consideration**: A result can be statistically significant but have no practical value. For example, with a very large sample size, you might detect a statistically significant difference of 0.1% in conversion rates, but such a small difference may not be worth implementing in practice.

**Best Practice**: Always consider both statistical and practical significance when making decisions based on data analysis.

#### Additional Resources

For more detailed information on confidence intervals and statistical inference, see:
- Penn State STAT 500: https://online.stat.psu.edu/stat500/Lesson05

### Bayesian Inference

Bayesian inference is a statistical method that uses Bayes' theorem to update beliefs about a parameter or hypothesis as new evidence is observed. Probability distributions are fundamental to Bayesian inference, representing prior and posterior beliefs.

Python libraries like **PyMC3** and **Stan** allow you to:
- Define probabilistic models
- Specify prior distributions
- Perform Bayesian inference using techniques like Markov Chain Monte Carlo (MCMC)

## 📄 Project Structure

```
Probability/
├── README.md                          # This documentation file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── check_setup.py                     # Installation verification script
├── generate_data.py                   # Dataset generation script
├── run_examples.py                    # Main script to run all examples
├── probability_env/                   # Virtual environment (after setup)
├── examples/                          # Python example scripts
│   ├── distributions.py              # Probability distribution examples
│   ├── conditional_probability.py    # Conditional probability & Bayes
│   ├── bias_variance.py              # Bias-variance tradeoff demo
│   ├── analyze_datasets.py           # Analyze generated datasets
│   └── *.png                         # Generated visualization files
└── data/
    └── sample_datasets/              # Directory for generated CSV datasets
        ├── customer_purchases.csv    # Customer data (conditional probability)
        ├── student_test_scores.csv   # Test scores (normal distribution)
        ├── website_traffic.csv       # Traffic data (Poisson distribution)
        ├── machine_failures.csv      # Failure times (exponential distribution)
        └── ab_test_results.csv       # A/B test data (hypothesis testing)
```

## Example Scripts

### 1. distributions.py
Demonstrates various probability distributions:
- Normal (Gaussian) Distribution
- Binomial Distribution
- Poisson Distribution
- Exponential Distribution
- Uniform Distribution
- Fitting distributions to data using Maximum Likelihood Estimation

### 2. conditional_probability.py
Covers conditional probability and Bayesian inference:
- Joint and conditional probabilities
- Contingency table analysis
- Bayes' Theorem with medical test example
- Sequential Bayesian updating

### 3. bias_variance.py
Illustrates the bias-variance tradeoff:
- Polynomial regression with varying degrees
- Visualization of underfitting vs overfitting
- Error decomposition into bias, variance, and noise
- Learning curves

### 4. analyze_datasets.py
Analyzes the generated datasets:
- Customer purchase patterns using conditional probability
- Test score distributions with normality tests
- Website traffic patterns using Poisson models
- Real-world applications of probability theory

## Using the Data Folder

### Generating Sample Datasets

The `data/sample_datasets/` folder is designed to store realistic datasets for practicing probability analysis. Use the dataset generator to create sample data:

```bash
# Generate all datasets
python generate_data.py

# View summary of generated datasets
python generate_data.py --summary
```

### Available Datasets

#### 1. customer_purchases.csv
**Use Case:** Conditional probability, Bayes' Theorem, contingency tables

**Features:**
- `customer_id` - Unique identifier
- `age` - Customer age (18-80)
- `purchased_before` - Previous purchase history (Yes/No)
- `received_email` - Marketing email received (Yes/No)
- `made_purchase` - Purchase this month (Yes/No)
- `purchase_amount` - Amount spent ($)

**Example Analysis:**
- Calculate P(Purchase | Email) vs P(Purchase | No Email)
- Measure email marketing effectiveness
- Build contingency tables

#### 2. student_test_scores.csv
**Use Case:** Normal distribution, parameter estimation, hypothesis testing

**Features:**
- `student_id` - Unique identifier
- `study_hours` - Hours spent studying (0-50)
- `previous_score` - Previous test score (0-100)
- `test_score` - Current test score (0-100)
- `passed` - Pass/fail status (≥60 = pass)

**Example Analysis:**
- Fit normal distribution to test scores
- Test for normality using Q-Q plots
- Analyze correlation between study time and performance

#### 3. website_traffic.csv
**Use Case:** Poisson distribution, time series patterns

**Features:**
- `date` - Date of observation
- `day_of_week` - Day (0=Monday, 6=Sunday)
- `hour` - Hour of day (0-23)
- `visitors` - Number of visitors
- `conversions` - Number of conversions

**Example Analysis:**
- Model hourly traffic using Poisson distribution
- Identify peak traffic hours
- Calculate conversion rates

#### 4. machine_failures.csv
**Use Case:** Exponential distribution, reliability analysis

**Features:**
- `machine_id` - Unique identifier
- `machine_type` - Type (A, B, C)
- `days_until_failure` - Time to first failure
- `failure_count` - Number of failures observed
- `last_maintenance` - Days since maintenance

**Example Analysis:**
- Model time-to-failure with exponential distribution
- Compare reliability across machine types
- Predict maintenance schedules

#### 5. ab_test_results.csv
**Use Case:** Hypothesis testing, binomial distribution, statistical significance

**Features:**
- `visitor_id` - Unique identifier
- `variant` - Test variant (A or B)
- `clicked` - Click status (Yes/No)
- `time_on_page` - Time spent (seconds)
- `converted` - Conversion status (Yes/No)

**Example Analysis:**
- Test if variant B performs better than A
- Calculate statistical significance
- Estimate conversion rate differences

### Loading Datasets in Python

```python
import pandas as pd

# Load a dataset
df = pd.read_csv('data/sample_datasets/customer_purchases.csv')

# View first few rows
print(df.head())

# Basic statistics
print(df.describe())

# Calculate probabilities
p_purchased = (df['made_purchase'] == 'Yes').mean()
print(f"Purchase probability: {p_purchased:.3f}")
```

### Custom Datasets

You can also add your own CSV files to `data/sample_datasets/` for analysis. The folder structure supports any tabular data for probability and statistical analysis.

## Further Reading

- Central Limit Theorem
- Monte Carlo methods for probability estimation
- Bayesian modeling
- Time series analysis with probability distributions

## Reference

### Common Commands

```bash
# Setup (one-time)
python3 -m venv probability_env
source probability_env/bin/activate  # Linux/MacOS
pip install -r requirements.txt

# Verify installation
python check_setup.py

# Generate datasets (optional but recommended)
python generate_data.py

# Run all examples
python run_examples.py

# Run individual examples
python examples/distributions.py
python examples/conditional_probability.py
python examples/bias_variance.py
python examples/analyze_datasets.py

# View dataset info
python generate_data.py --summary

# Deactivate environment when done
deactivate
```

### Dataset Example

```python
import pandas as pd
import numpy as np

# Load customer purchase data
df = pd.read_csv('data/sample_datasets/customer_purchases.csv')

# Calculate conditional probability
email_customers = df[df['received_email'] == 'Yes']
p_purchase_given_email = (email_customers['made_purchase'] == 'Yes').mean()

print(f"P(Purchase | Email) = {p_purchase_given_email:.3f}")
```

### Project Workflow

1. **Setup** → Create virtual environment and install dependencies
2. **Generate** → Create sample datasets with `python generate_data.py`

---

*This repository provides practical examples to illustrate probability calculations with Python and mathematical libraries for data analysis, plotting figures, and processing statistical data.*
