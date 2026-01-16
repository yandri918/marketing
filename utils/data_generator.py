import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_customer_data(n_samples=500):
    """Generates synthetic customer data for segmentation."""
    np.random.seed(42)
    data = {
        'CustomerID': [f'C{i:03d}' for i in range(1, n_samples + 1)],
        'Age': np.random.randint(18, 70, n_samples),
        'Income': np.random.randint(20000, 150000, n_samples),
        'SpendingScore': np.random.randint(1, 100, n_samples),
        'Recency_Days': np.random.randint(1, 365, n_samples),
        'Frequency': np.random.randint(1, 50, n_samples),
        'Monetary': np.random.randint(100, 10000, n_samples)
    }
    return pd.DataFrame(data)

def generate_sales_data(days=365):
    """Generates synthetic daily sales data for forecasting."""
    np.random.seed(42)
    dates = [datetime.today() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    base_sales = 1000
    trend = np.linspace(0, 500, days)
    seasonality = 200 * np.sin(np.linspace(0, 3.14 * 8, days))
    noise = np.random.normal(0, 100, days)
    
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
