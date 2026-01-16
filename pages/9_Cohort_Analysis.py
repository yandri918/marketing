import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_transaction_log

st.set_page_config(page_title="Cohort Analysis | Retention", page_icon="📅", layout="wide")

st.title("📅 Cohort Analysis: Retention Heatmap")
st.markdown("""
Analyze user retention by grouping customers into **Cohorts** based on their first purchase month.
*   **Vertical Axis**: Cohort Month (When they joined).
*   **Horizontal Axis**: Months since first purchase.
*   **Cell Value**: Percentage of users who came back to buy again.
""")

# Data Source Controls
st.sidebar.header("Data Configuration")
data_source = st.sidebar.radio("Select Data Source", ["Generate Synthetic Data", "Upload CSV File"])

if data_source == "Generate Synthetic Data":
    n_cust = st.sidebar.slider("Number of Customers", 100, 2000, 800)
    n_txn = st.sidebar.slider("Transaction Volume", 500, 10000, 3000)
    
    if st.sidebar.button("Generate New Data"):
        st.cache_data.clear()
        
    @st.cache_data
    def get_synthetic_data(n_c, n_t):
        return generate_transaction_log(n_customers=n_c, n_transactions=n_t)
        
    df = get_synthetic_data(n_cust, n_txn)
    st.info(f"Using Synthetic Data: {len(df)} transactions from {df['CustomerID'].nunique()} customers.")

else:
    uploaded_file = st.sidebar.file_uploader("Upload Transaction CSV", type=["csv"])
    st.sidebar.caption("Required columns: `CustomerID`, `TransactionDate` (YYYY-MM-DD), `Amount`")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Ensure datetime conversion
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
            st.success(f"Loaded {len(df)} transactions from uploaded file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.warning("Please upload a CSV file to proceed. Showing placeholder data (Empty).")
        st.stop()

# Data Preprocessing for Cohort Analysis
def get_month(x): 
    return datetime(x.year, x.month, 1)

from datetime import datetime

df['TransactionMonth'] = df['TransactionDate'].apply(lambda x: datetime(x.year, x.month, 1))
df['CohortMonth'] = df.groupby('CustomerID')['TransactionMonth'].transform('min')

def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    return year, month

txn_year, txn_month = get_date_int(df, 'TransactionMonth')
cohort_year, cohort_month = get_date_int(df, 'CohortMonth')

years_diff = txn_year - cohort_year
months_diff = txn_month - cohort_month

df['CohortIndex'] = years_diff * 12 + months_diff + 1

# 1. Retention Matrix
grouping = df.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique).reset_index()
cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
cohort_sizes = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_sizes, axis=0)

# 2. Average Quantity Matrix (Proxied by Count of Txns)
cohort_data_qty = grouping['TransactionDate'].count().reset_index() # Count transactions
avg_qty = cohort_data_qty.pivot(index='CohortMonth', columns='CohortIndex', values='TransactionDate')

# Visualization
st.divider()

option = st.selectbox("Select Metric to Visualize", ["Retention Rate (%)", "Active Users (Count)"])

if option == "Retention Rate (%)":
    st.subheader("🔥 User Retention Heatmap")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Format Cohort Month index for readability
    yticklabels = [x.strftime('%Y-%m') for x in retention.index]
    
    sns.heatmap(retention, annot=True, fmt='.0%', cmap='YlGnBu', vmin=0.0, vmax=0.5, ax=ax, yticklabels=yticklabels)
    ax.set_title('Cohort Analysis - Retention Rate')
    ax.set_ylabel('Cohort Month')
    ax.set_xlabel('Months Since First Purchase')
    
    st.pyplot(fig)
    
    st.info("**Insight:** Darker blue cells indicate higher retention. Look for vertical consistency (product health) or horizontal improvements (better onboarding).")

elif option == "Active Users (Count)":
    st.subheader("👥 Active Users Heatmap")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    yticklabels = [x.strftime('%Y-%m') for x in cohort_counts.index]
    
    sns.heatmap(cohort_counts, annot=True, fmt='.0f', cmap='Blues', ax=ax, yticklabels=yticklabels)
    ax.set_title('Cohort Analysis - Active Users Count')
    st.pyplot(fig)

# Cohort Details
st.divider()
st.subheader("📋 Cohort Performance Table")

# Calculate metrics per cohort
cohort_metrics = df.groupby('CohortMonth').agg({
    'CustomerID': 'nunique',
    'Amount': ['sum', 'mean']
}).reset_index()
cohort_metrics.columns = ['CohortMonth', 'New Users', 'Total Revenue', 'Avg LTV (Initial)']

cohort_metrics['CohortMonth'] = cohort_metrics['CohortMonth'].dt.strftime('%Y-%m')

st.dataframe(cohort_metrics.style.format({
    "Total Revenue": "Rp {:,.0f}",
    "Avg LTV (Initial)": "Rp {:,.0f}"
}))

# Specific Insight
best_cohort = cohort_metrics.loc[cohort_metrics['Total Revenue'].idxmax()]
st.success(f"🏆 **Best Performing Cohort:** {best_cohort['CohortMonth']} with Total Revenue of **Rp {best_cohort['Total Revenue']:,.0f}**")
