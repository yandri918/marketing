import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# Add parent directory to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_funnel_data

st.set_page_config(page_title="Funnel Analysis | Growth", page_icon="🔻", layout="wide")

st.title("🔻 Marketing Funnel Performance")
st.markdown("Visualize conversion rates and identify drop-off points in the customer journey.")

# Sidebar Configuration
st.sidebar.header("Filter Segment")
segment = st.sidebar.selectbox("Select Segment", ["All Users", "Mobile", "Desktop", "Organic Traffic", "Paid Ads"])

# Data Loading
df = generate_funnel_data()

# Calculate Conversion Rates
df['Conversion Rate'] = df['Users'].pct_change().apply(lambda x: f"{ (x+1)*100:.1f}%" if pd.notnull(x) else "100%")
df['Drop-off'] = df['Users'].diff().apply(lambda x: f"{abs(x):.0f} users lost" if pd.notnull(x) and x < 0 else "-")

# Funnel Visualization
col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure(go.Funnel(
        y=df['Stage'],
        x=df['Users'],
        textinfo="value+percent initial",
        marker={"color": ["#3498db", "#e67e22", "#e74c3c", "#9b59b6", "#2ecc71"]}
    ))
    fig.update_layout(title="Conversion Funnel", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Performance Metrics")
    st.dataframe(df[['Stage', 'Users', 'Conversion Rate', 'Drop-off']], hide_index=True)
    
    # Insights based on data
    st.info("💡 **Insight:** The biggest drop-off (53%) occurs between **Interest** and **Consideration**. Consider optimizing the landing page content or adding retargeting ads.")

st.divider()

# Growth Recommendations
st.header("🚀 Optimization Strategy")
c1, c2, c3 = st.columns(3)
with c1:
    st.success("**Top of Funnel (ToFu)**")
    st.write("Current Awareness is strong. Focus on improving ad click-through rates (CTR).")
with c2:
    st.warning("**Middle of Funnel (MoFu)**")
    st.write("High drop-off at Interest phase. Implement email drip campaigns to nurture leads.")
with c3:
    st.error("**Bottom of Funnel (BoFu)**")
    st.write("Conversion to Purchase is steady. Test urgency triggers (e.g., limited time offers).")

st.divider()

# Advanced Financial Metrics
st.header("💰 Paid Media Performance & ROI")
st.markdown("Calculate the profitability of your advertising campaigns.")

col_input1, col_input2, col_input3, col_input4 = st.columns(4)

with col_input1:
    ad_spend = st.number_input("Total Ad Spend ($)", value=5000, step=100)
with col_input2:
    impressions = st.number_input("Impressions", value=150000, step=1000)
with col_input3:
    clicks = st.number_input("Clicks", value=3500, step=50)
with col_input4:
    conversions = st.number_input("Total Conversions", value=120, step=5)

# Calculate Metrics
cpm = (ad_spend / impressions) * 1000 if impressions > 0 else 0
ctr = (clicks / impressions) * 100 if impressions > 0 else 0
cpc = ad_spend / clicks if clicks > 0 else 0
cpa = ad_spend / conversions if conversions > 0 else 0
conversion_rate = (conversions / clicks) * 100 if clicks > 0 else 0

# Revenue Calculation
avg_order_value = st.number_input("Average Order Value (AOV) $", value=150, step=10)
total_revenue = conversions * avg_order_value
roas = total_revenue / ad_spend if ad_spend > 0 else 0
profit = total_revenue - ad_spend

# Display Metrics
st.subheader("📊 Campaign KPIs")
m1, m2, m3, m4 = st.columns(4)
m1.metric("CPM (Cost/1k Views)", f"${cpm:.2f}")
m2.metric("CTR (Click-Through)", f"{ctr:.2f}%")
m3.metric("CPC (Cost/Click)", f"${cpc:.2f}")
m4.metric("CPA (Cost/Acquisition)", f"${cpa:.2f}")

st.divider()

k1, k2, k3 = st.columns(3)
k1.metric("Total Revenue", f"${total_revenue:,.2f}")
k2.metric("ROAS (Return on Ad Spend)", f"{roas:.2f}x", delta="Positive" if roas > 4 else "Low")
k3.metric("Net Profit (Ads Only)", f"${profit:,.2f}", delta_color="normal")

# Dynamic Scenario
with st.expander("⚖️ Budget Scalability Calculator"):
    st.write("Estimate impact of increasing budget (assuming constant efficiency).")
    budget_increase = st.slider("Increase Budget by (%)", 0, 200, 50)
    
    projected_spend = ad_spend * (1 + budget_increase/100)
    projected_conversions = conversions * (1 + budget_increase/100)
    projected_revenue = projected_conversions * avg_order_value
    projected_profit = projected_revenue - projected_spend
    
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Projected Spend", f"${projected_spend:,.0f}")
    sc2.metric("Projected Revenue", f"${projected_revenue:,.0f}")
    sc3.metric("Projected Profit", f"${projected_profit:,.0f}", delta=f"{projected_profit - profit:,.0f}")
