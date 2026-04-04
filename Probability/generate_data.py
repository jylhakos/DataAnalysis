"""
Dataset Generator for Probability Examples
This script generates sample datasets for probability and statistical analysis
"""

import numpy as np
import pandas as pd
import os


def ensure_data_directory():
    """Create data directories if they don't exist"""
    os.makedirs('data/sample_datasets', exist_ok=True)


def generate_customer_purchase_data(n_customers=1000):
    """
    Generate customer purchase dataset for conditional probability analysis
    
    Features:
    - customer_id: Unique customer identifier
    - age: Customer age (18-80)
    - purchased_before: Whether customer has purchased before (Yes/No)
    - received_email: Whether customer received marketing email (Yes/No)
    - made_purchase: Whether customer made a purchase this month (Yes/No)
    - purchase_amount: Amount spent (0 if no purchase)
    """
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'age': np.random.randint(18, 81, n_customers),
        'purchased_before': np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6]),
    }
    
    # Customers who purchased before are more likely to receive emails
    email_prob = np.where(
        np.array(data['purchased_before']) == 'Yes',
        0.7,  # 70% of previous customers get emails
        0.3   # 30% of new customers get emails
    )
    data['received_email'] = np.array([
        np.random.choice(['Yes', 'No'], p=[p, 1-p])
        for p in email_prob
    ])
    
    # Purchase likelihood depends on both factors
    purchase_prob = []
    for prev, email in zip(data['purchased_before'], data['received_email']):
        if prev == 'Yes' and email == 'Yes':
            prob = 0.6  # High likelihood
        elif prev == 'Yes' and email == 'No':
            prob = 0.15
        elif prev == 'No' and email == 'Yes':
            prob = 0.25
        else:
            prob = 0.05  # Low likelihood
        purchase_prob.append(prob)
    
    data['made_purchase'] = [
        np.random.choice(['Yes', 'No'], p=[p, 1-p])
        for p in purchase_prob
    ]
    
    # Generate purchase amounts
    data['purchase_amount'] = [
        np.random.lognormal(4.5, 0.5) if purchase == 'Yes' else 0
        for purchase in data['made_purchase']
    ]
    
    df = pd.DataFrame(data)
    df['purchase_amount'] = df['purchase_amount'].round(2)
    
    return df


def generate_test_scores_data(n_students=500):
    """
    Generate student test scores for normal distribution analysis
    
    Features:
    - student_id: Unique student identifier
    - study_hours: Hours spent studying (0-50)
    - previous_score: Score on previous test (0-100)
    - test_score: Current test score (0-100)
    - passed: Whether student passed (score >= 60)
    """
    np.random.seed(123)
    
    # Study hours affects test score
    study_hours = np.random.gamma(3, 3, n_students)
    study_hours = np.clip(study_hours, 0, 50).round(1)
    
    # Previous scores
    previous_score = np.random.normal(70, 15, n_students)
    previous_score = np.clip(previous_score, 0, 100).round(1)
    
    # Current test score depends on study hours and previous performance
    base_score = 50
    study_effect = study_hours * 0.8
    previous_effect = previous_score * 0.2
    noise = np.random.normal(0, 8, n_students)
    
    test_score = base_score + study_effect + previous_effect + noise
    test_score = np.clip(test_score, 0, 100).round(1)
    
    df = pd.DataFrame({
        'student_id': range(1, n_students + 1),
        'study_hours': study_hours,
        'previous_score': previous_score,
        'test_score': test_score,
        'passed': test_score >= 60
    })
    
    return df


def generate_website_traffic_data(n_days=365):
    """
    Generate website traffic data for Poisson distribution analysis
    
    Features:
    - date: Date of observation
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - hour: Hour of day (0-23)
    - visitors: Number of visitors in that hour
    - conversions: Number of conversions (follows Binomial)
    """
    np.random.seed(456)
    
    start_date = pd.Timestamp('2025-01-01')
    dates = []
    hours = []
    visitors = []
    conversions = []
    
    for day in range(n_days):
        current_date = start_date + pd.Timedelta(days=day)
        day_of_week = current_date.dayofweek
        
        # Weekend traffic is different
        is_weekend = day_of_week >= 5
        
        for hour in range(24):
            # Traffic patterns by hour
            if 9 <= hour <= 17:  # Business hours
                base_rate = 50
            elif 18 <= hour <= 22:  # Evening
                base_rate = 80
            else:  # Night/early morning
                base_rate = 20
            
            # Weekend adjustment
            if is_weekend:
                base_rate *= 0.7
            
            # Generate visitors (Poisson)
            n_visitors = np.random.poisson(base_rate)
            
            # Generate conversions (Binomial with ~3% conversion rate)
            n_conversions = np.random.binomial(n_visitors, 0.03)
            
            dates.append(current_date)
            hours.append(hour)
            visitors.append(n_visitors)
            conversions.append(n_conversions)
    
    df = pd.DataFrame({
        'date': dates,
        'day_of_week': [d.dayofweek for d in dates],
        'hour': hours,
        'visitors': visitors,
        'conversions': conversions
    })
    
    return df


def generate_machine_failure_data(n_machines=200, n_days=180):
    """
    Generate machine failure time data for exponential distribution analysis
    
    Features:
    - machine_id: Unique machine identifier
    - machine_type: Type of machine (A, B, C)
    - days_until_failure: Days until first failure
    - failure_count: Number of failures in observation period
    - last_maintenance: Days since last maintenance
    """
    np.random.seed(789)
    
    machine_types = np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_machines, p=[0.4, 0.35, 0.25])
    
    # Different failure rates by type
    failure_rates = {
        'Type_A': 1/60,  # Mean time to failure: 60 days
        'Type_B': 1/90,  # 90 days
        'Type_C': 1/45   # 45 days
    }
    
    days_until_failure = []
    failure_counts = []
    
    for machine_type in machine_types:
        rate = failure_rates[machine_type]
        
        # Time until first failure (Exponential)
        time_to_fail = np.random.exponential(1/rate)
        days_until_failure.append(min(time_to_fail, n_days))
        
        # Count failures during observation period (Poisson)
        expected_failures = rate * n_days
        failures = np.random.poisson(expected_failures)
        failure_counts.append(failures)
    
    df = pd.DataFrame({
        'machine_id': range(1, n_machines + 1),
        'machine_type': machine_types,
        'days_until_failure': np.array(days_until_failure).round(1),
        'failure_count': failure_counts,
        'last_maintenance': np.random.randint(0, 90, n_machines)
    })
    
    return df


def generate_ab_test_data(n_visitors=10000):
    """
    Generate A/B test data for hypothesis testing
    
    Features:
    - visitor_id: Unique visitor identifier
    - variant: Test variant (A or B)
    - clicked: Whether visitor clicked (Yes/No)
    - time_on_page: Time spent on page (seconds)
    - converted: Whether visitor converted (Yes/No)
    """
    np.random.seed(321)
    
    # Randomly assign variants (50/50 split)
    variants = np.random.choice(['A', 'B'], n_visitors)
    
    # Variant B has slightly higher click rate
    click_prob = np.where(variants == 'A', 0.10, 0.12)
    clicked = [np.random.choice(['Yes', 'No'], p=[p, 1-p]) for p in click_prob]
    
    # Time on page (log-normal distribution)
    time_on_page = np.random.lognormal(3.5, 1.2, n_visitors)
    time_on_page = np.clip(time_on_page, 5, 600).round(0)
    
    # Conversion depends on clicking and time on page
    conversion_prob = []
    for click, time_val in zip(clicked, time_on_page):
        if click == 'Yes' and time_val > 60:
            prob = 0.25
        elif click == 'Yes':
            prob = 0.10
        else:
            prob = 0.01
        conversion_prob.append(prob)
    
    converted = [np.random.choice(['Yes', 'No'], p=[p, 1-p]) for p in conversion_prob]
    
    df = pd.DataFrame({
        'visitor_id': range(1, n_visitors + 1),
        'variant': variants,
        'clicked': clicked,
        'time_on_page': time_on_page,
        'converted': converted
    })
    
    return df


def generate_all_datasets():
    """Generate all sample datasets and save to CSV files"""
    ensure_data_directory()
    
    print("=" * 70)
    print("GENERATING SAMPLE DATASETS FOR PROBABILITY ANALYSIS")
    print("=" * 70)
    
    datasets = [
        ('customer_purchases.csv', generate_customer_purchase_data, 1000, 
         'Customer purchase data for conditional probability'),
        
        ('student_test_scores.csv', generate_test_scores_data, 500,
         'Student test scores for normal distribution analysis'),
        
        ('website_traffic.csv', generate_website_traffic_data, 365,
         'Website traffic data for Poisson distribution'),
        
        ('machine_failures.csv', generate_machine_failure_data, 200,
         'Machine failure data for exponential distribution'),
        
        ('ab_test_results.csv', generate_ab_test_data, 10000,
         'A/B test results for hypothesis testing'),
    ]
    
    print("\nGenerating datasets in 'data/sample_datasets/':\n")
    
    for filename, generator, size, description in datasets:
        filepath = f'data/sample_datasets/{filename}'
        
        if len(generator.__code__.co_varnames) > 0:
            # Function takes argument
            df = generator(size)
        else:
            df = generator()
        
        df.to_csv(filepath, index=False)
        
        print(f"✓ {filename}")
        print(f"  Description: {description}")
        print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")
        print(f"  Features: {', '.join(df.columns.tolist())}")
        print()
    
    print("=" * 70)
    print("ALL DATASETS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("\nYou can now use these datasets in your probability examples.")
    print("To load a dataset in Python:")
    print("  import pandas as pd")
    print("  df = pd.read_csv('data/sample_datasets/customer_purchases.csv')")
    print("=" * 70)


def show_dataset_summary():
    """Display summary of available datasets"""
    print("\n" + "=" * 70)
    print("AVAILABLE DATASETS SUMMARY")
    print("=" * 70)
    
    datasets = [
        'customer_purchases.csv',
        'student_test_scores.csv',
        'website_traffic.csv',
        'machine_failures.csv',
        'ab_test_results.csv'
    ]
    
    for filename in datasets:
        filepath = f'data/sample_datasets/{filename}'
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"\n📄 {filename}")
            print(f"   Dimensions: {len(df)} rows × {len(df.columns)} columns")
            print(f"   Columns: {', '.join(df.columns.tolist())}")
        else:
            print(f"\n📄 {filename} - NOT FOUND")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--summary':
        show_dataset_summary()
    else:
        generate_all_datasets()
