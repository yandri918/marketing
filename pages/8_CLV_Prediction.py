import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Try to import lifetimes, gracefully handle if not installed
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False
    st.error("⚠️ **CLV Prediction Unavailable**: The `lifetimes` library is not installed. Please contact the administrator to add it to requirements.txt")
    st.info("💡 The platform is rebuilding with updated dependencies. Please refresh in a few minutes.")
    st.stop()

st.set_page_config(page_title="CLV Prediction | AI Analytics", page_icon="🔮", layout="wide")

st.title("🔮 Customer Lifetime Value (CLV) Prediction")
st.markdown("""
Predict the future value of your customers using **Probabilistic Models** (BG/NBD & Gamma-Gamma).
This advanced method estimates:
1.  **Probability of Alive**: Is the customer still active or have they churned?
2.  **Expected Transactions**: How many times will they buy in the future?
3.  **CLV**: Total expected revenue over a specific period.
""")

# Generate Synthetic RFM Data tailored for Lifetimes
@st.cache_data
def generate_rfm_summary(n_customers=2000):
    np.random.seed(42)
    data = {
        'frequency': np.random.negative_binomial(n=2, p=0.2, size=n_customers),  # transactions (repeat)
        'recency': np.random.randint(1, 365, size=n_customers), # days since first purchase
        'T': np.random.randint(200, 700, size=n_customers), # customer age (days)
        'monetary_value': np.random.gamma(shape=10, scale=100000, size=n_customers) # Avg transaction value
    }
    df = pd.DataFrame(data)
    # Filter logical constraints
    # Filter logical constraints & Ensure valid BG/NBD input
    df = df[df['frequency'] > 0]
    df['recency'] = df.apply(lambda row: min(row['recency'], row['T']), axis=1) # Recency cannot be > Age
    df = df[df['recency'] > 0] # Recency must be > 0 for valid fit
    df = df[df['T'] > df['recency']] # Ideally T >= Recency, dropping equal cases to avoid edge errors
    
    # Add CustomerID
    df['CustomerID'] = [f'C{i:03d}' for i in range(1, len(df) + 1)]
    return df


# Initialize Session State
if 'clv_data' not in st.session_state:
    st.session_state.clv_data = generate_rfm_summary()

# Manual Input Form
with st.sidebar.expander("➕ Add Single Customer Data"):
    with st.form("clv_input_form"):
        new_freq = st.number_input("Frequency (transactions)", min_value=1, value=5)
        new_recency = st.number_input("Recency (days)", min_value=1, value=30)
        new_T = st.number_input("Customer Age T (days)", min_value=1, value=100)
        new_monetary = st.number_input("Avg Monetary Value (Rp)", min_value=0, value=1000000, step=50000)
        
        submitted = st.form_submit_button("Add to Model")
        
        if submitted:
            if new_recency > new_T:
                st.error("Error: Recency cannot be greater than Customer Age (T).")
            else:
                new_row = pd.DataFrame({
                    'CustomerID': [f"Manual_{len(st.session_state.clv_data) + 1}"],
                    'frequency': [new_freq],
                    'recency': [new_recency], 
                    'T': [new_T],
                    'monetary_value': [new_monetary]
                })
                # Add to session state
                st.session_state.clv_data = pd.concat([st.session_state.clv_data, new_row], ignore_index=True)
                st.success(f"Customer 'Manual_{len(st.session_state.clv_data) + 1}' added! Check the Top Customers table.")
                st.rerun()

df = st.session_state.clv_data

with st.expander("Show Sample Data (RFM Summary)"):
    st.dataframe(df.head())
    st.caption("Detailed transaction logs are aggregated into this RFM summary format for modeling.")

# 1. BG/NBD Model (Frequency & Recency)
st.divider()
st.subheader("1. BG/NBD Model: Churn & Future Transactions")
st.write("We fit the **Beta-Geometric/Negative Binomial Distribution (BG/NBD)** model to the data.")

bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(df['frequency'], df['recency'], df['T'])

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Frequency/Recency Matrix")
    st.caption("Expected number of future purchases. Yellow = Best customers (High Freq, High Recency).")
    fig = plt.figure(figsize=(10, 8))
    plot_frequency_recency_matrix(bgf)
    st.pyplot(fig)

with col2:
    st.markdown("#### Probability of Alive")
    st.caption("Likelihood that the customer is still active. Dark = Likely Churned.")
    fig = plt.figure(figsize=(10, 8))
    plot_probability_alive_matrix(bgf)
    st.pyplot(fig)

# 2. Gamma-Gamma Model (Monetary Value)
st.divider()
st.subheader("2. Gamma-Gamma Model: Monetary Value & CLV")
st.write("We fit the **Gamma-Gamma** model to estimate the average transaction value.")

# Check assumption: no correlation between frequency and monetary value
corr = df[['frequency', 'monetary_value']].corr().iloc[0,1]
if abs(corr) < 0.2:
    st.success(f"✅ Independence Assumption Met: Correlation between freq & value is low ({corr:.2f}).")
else:
    st.warning(f"⚠️ Correlation is high ({corr:.2f}). Gamma-Gamma model might be less accurate.")

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(df['frequency'], df['monetary_value'])

# Predict CLV
prediction_months = st.slider("Predict Value for Next X Months", 1, 24, 12)

df['predicted_clv'] = ggf.customer_lifetime_value(
    bgf,
    df['frequency'],
    df['recency'],
    df['T'],
    df['monetary_value'],
    time=prediction_months, # months
    freq='D', # data is in days
    discount_rate=0.01
)

df['prob_alive'] = bgf.conditional_probability_alive(df['frequency'], df['recency'], df['T'])

# Top Customers
st.subheader(f"🏆 Top Customers by Predicted CLV ({prediction_months} Months)")
top_customers = df.sort_values('predicted_clv', ascending=False).head(10)

st.dataframe(top_customers.style.format({
    'monetary_value': 'Rp {:,.0f}',
    'predicted_clv': 'Rp {:,.0f}',
    'prob_alive': '{:.1%}'
}))

# Aggregate Insights
total_future_revenue = df['predicted_clv'].sum()
avg_clv = df['predicted_clv'].mean()

m1, m2, m3 = st.columns(3)
m1.metric("Total Potential Revenue", f"Rp {total_future_revenue:,.0f}", help=f"Total CLV of all customers for next {prediction_months} months")
m2.metric("Avg Predicted CLV", f"Rp {avg_clv:,.0f}")
m3.metric("Active Customer Base", f"{len(df[df['prob_alive'] > 0.5])} / {len(df)}", help="Customers with >50% chance of being alive")

# Visualization
st.subheader("Distribution of Customer Value")
fig_dist = px.histogram(df, x='predicted_clv', nbins=50, title="Predicted CLV Distribution", 
                        labels={'predicted_clv': 'Predicted CLV (Rp)'}, template="plotly_white")
st.plotly_chart(fig_dist, use_container_width=True)

st.info("""
**Business Action Plan:**
1.  **High CLV, High Prob Alive**: VIPs. Invest in retention (Personalized Service).
2.  **High CLV, Low Prob Alive**: At Risk. Immediate Win-Back Campaign needed!
3.  **Low CLV**: Standard customers. Automate engagement.
""")
