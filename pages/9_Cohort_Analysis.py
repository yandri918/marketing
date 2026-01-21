import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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
    
    # Convert to long format for Altair
    retention_long = retention.reset_index().melt(
        id_vars='CohortMonth',
        var_name='CohortIndex',
        value_name='Retention'
    )
    retention_long['CohortMonth'] = retention_long['CohortMonth'].dt.strftime('%Y-%m')
    retention_long['Retention'] = retention_long['Retention'].fillna(0)
    
    # Create Altair heatmap
    heatmap = alt.Chart(retention_long).mark_rect().encode(
        x=alt.X('CohortIndex:O', title='Months Since First Purchase'),
        y=alt.Y('CohortMonth:N', title='Cohort Month'),
        color=alt.Color('Retention:Q',
                      scale=alt.Scale(scheme='yellowgreenblue', domain=[0, 0.5]),
                      title='Retention Rate'),
        tooltip=[
            alt.Tooltip('CohortMonth:N', title='Cohort'),
            alt.Tooltip('CohortIndex:O', title='Month'),
            alt.Tooltip('Retention:Q', title='Retention', format='.1%')
        ]
    ).properties(
        width=700,
        height=400,
        title='Cohort Analysis - Retention Rate'
    )
    
    st.altair_chart(heatmap, use_container_width=True)
    
    st.info("**Insight:** Darker blue cells indicate higher retention. Look for vertical consistency (product health) or horizontal improvements (better onboarding).")

elif option == "Active Users (Count)":
    st.subheader("👥 Active Users Heatmap")
    
    # Convert to long format for Altair
    counts_long = cohort_counts.reset_index().melt(
        id_vars='CohortMonth',
        var_name='CohortIndex',
        value_name='Users'
    )
    counts_long['CohortMonth'] = counts_long['CohortMonth'].dt.strftime('%Y-%m')
    counts_long['Users'] = counts_long['Users'].fillna(0)
    
    heatmap = alt.Chart(counts_long).mark_rect().encode(
        x=alt.X('CohortIndex:O', title='Months Since First Purchase'),
        y=alt.Y('CohortMonth:N', title='Cohort Month'),
        color=alt.Color('Users:Q',
                      scale=alt.Scale(scheme='blues'),
                      title='Active Users'),
        tooltip=[
            alt.Tooltip('CohortMonth:N', title='Cohort'),
            alt.Tooltip('CohortIndex:O', title='Month'),
            alt.Tooltip('Users:Q', title='Users', format='.0f')
        ]
    ).properties(
        width=700,
        height=400,
        title='Cohort Analysis - Active Users Count'
    )
    
    st.altair_chart(heatmap, use_container_width=True)

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
