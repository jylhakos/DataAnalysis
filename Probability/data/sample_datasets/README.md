# Sample Datasets

This folder contains generated datasets for probability and statistical analysis examples.

## Generating Datasets

From the project root directory, run:

```bash
python generate_data.py
```

This will create 5 CSV files in this directory.

## Available Datasets

### 1. customer_purchases.csv (1,000 records)
Customer purchase behavior for conditional probability analysis.

**Columns:**
- customer_id, age, purchased_before, received_email, made_purchase, purchase_amount

**Use for:** Conditional probability, Bayes' Theorem, contingency tables

### 2. student_test_scores.csv (500 records)
Student test scores for normal distribution analysis.

**Columns:**
- student_id, study_hours, previous_score, test_score, passed

**Use for:** Normal distribution, parameter estimation, correlation analysis

### 3. website_traffic.csv (8,760 records - 365 days × 24 hours)
Hourly website traffic data for Poisson distribution analysis.

**Columns:**
- date, day_of_week, hour, visitors, conversions

**Use for:** Poisson distribution, time patterns, conversion analysis

### 4. machine_failures.csv (200 records)
Machine failure times for exponential distribution analysis.

**Columns:**
- machine_id, machine_type, days_until_failure, failure_count, last_maintenance

**Use for:** Exponential distribution, reliability engineering, survival analysis

### 5. ab_test_results.csv (10,000 records)
A/B testing results for hypothesis testing.

**Columns:**
- visitor_id, variant, clicked, time_on_page, converted

**Use for:** Hypothesis testing, binomial distribution, statistical significance

## Loading Data

```python
import pandas as pd

# Load a dataset
df = pd.read_csv('data/sample_datasets/customer_purchases.csv')

# View first few rows
print(df.head())

# Get basic statistics
print(df.describe())
```

## Analyzing Data

Run the dataset analysis examples:

```bash
python examples/analyze_datasets.py
```

This will generate probability analyses and visualizations for all datasets.
