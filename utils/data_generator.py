import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_customer_data(n_samples=500):
    """Generates synthetic customer data for segmentation."""
    np.random.seed(42)
    data = {
        'CustomerID': [f'C{i:03d}' for i in range(1, n_samples + 1)],
        'Age': np.random.randint(18, 70, n_samples),
        'Income': np.random.randint(3000000, 20000000, n_samples),
        'SpendingScore': np.random.randint(1, 100, n_samples),
        'Recency_Days': np.random.randint(1, 365, n_samples),
        'Frequency': np.random.randint(1, 50, n_samples),
        'Monetary': np.random.randint(50000, 5000000, n_samples)
    }
    return pd.DataFrame(data)

def generate_sales_data(days=365):
    """Generates synthetic daily sales data for forecasting."""
    np.random.seed(42)
    dates = [datetime.today() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    base_sales = 10000000
    trend = np.linspace(0, 5000000, days)
    seasonality = 2000000 * np.sin(np.linspace(0, 3.14 * 8, days))
    noise = np.random.normal(0, 1000000, days)
    
    sales = base_sales + trend + seasonality + noise
    
    return pd.DataFrame({'Date': dates, 'Sales': sales.astype(int)})

def generate_funnel_data():
    """Generates synthetic funnel data."""
    stages = ['Awareness', 'Interest', 'Consideration', 'Intent', 'Purchase']
    users = [10000, 6500, 3000, 1200, 450]
    return pd.DataFrame({'Stage': stages, 'Users': users})

def generate_sentiment_data():
    """Generates synthetic social media comments."""
    comments = [
        "Love this product! Best purchase ever.",
        "Terrible service, very disappointed.",
        "It's okay, but a bit expensive.",
        "Great quality, highly recommend!",
        "Not what I expected, returned it.",
        "Amazing customer support, very helpful.",
        "Delivery was late, but product is good.",
        "Worst experience, never buying again.",
        "Five stars! Will buy more.",
        "Average product, nothing special."
    ]
    sentiments = ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Positive', 'Neutral', 'Negative', 'Positive', 'Neutral']
    return pd.DataFrame({'Comment': comments * 10, 'Sentiment': sentiments * 10})

def generate_transaction_log(n_customers=500, n_transactions=2000):
    """Generates granular transaction logs for Cohort Analysis."""
    np.random.seed(42)
    
    # Generate Customer Pool
    customer_ids = [f'C{i:03d}' for i in range(1, n_customers + 1)]
    
    # Generate Transactions
    data = []
    start_date = datetime.today() - timedelta(days=365)
    
    for _ in range(n_transactions):
        cust_id = np.random.choice(customer_ids)
        # Random date within the last year
        days_offset = np.random.randint(0, 365)
        txn_date = start_date + timedelta(days=days_offset)
        
        amount = np.random.randint(50000, 2000000) # IDR 50k - 2M
        
        data.append({
            'CustomerID': cust_id,
            'TransactionDate': txn_date,
            'Amount': amount
        })
        
    df = pd.DataFrame(data)
    return df.sort_values('TransactionDate')
