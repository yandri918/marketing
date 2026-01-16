import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Competitive Analysis | Economics", page_icon="⚖️", layout="wide")

st.title("⚖️ Competitive Market Analysis")
st.markdown("Analyze pricing strategies and market dynamics using **Economic Modeling**.")

# Pricing Simulation
st.sidebar.header("Price Elasticity Model")
base_price = st.sidebar.number_input("Base Product Price ($)", 50, 500, 100)
elasticity = st.sidebar.slider("Price Elasticity of Demand (PED)", -3.0, -0.1, -1.5, step=0.1)

st.sidebar.info("**Note:** Elasticity < -1 means demand is sensitive to price (elastic). Elasticity > -1 means demand is less sensitive (inelastic).")

# Generate Demand Curve
prices = np.linspace(base_price * 0.5, base_price * 1.5, 100)
base_demand = 1000  # Initial demand at base price
# Q = Q0 * (P / P0) ^ Elasticity
demands = base_demand * (prices / base_price) ** elasticity
revenues = prices * demands

df_econ = pd.DataFrame({'Price': prices, 'Demand': demands, 'Revenue': revenues})

col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Demand Curve")
    fig_demand = px.line(df_econ, x='Price', y='Demand', title='Price vs. Quantity Demanded', template="plotly_white")
    
    # Mark current point
    current_demand = base_demand * (base_price / base_price) ** elasticity
    fig_demand.add_scatter(x=[base_price], y=[current_demand], mode='markers', marker=dict(size=10, color='red'), name='Current Price')
    
    st.plotly_chart(fig_demand, use_container_width=True)

with col2:
    st.subheader("💰 Revenue Optimization")
    fig_rev = px.line(df_econ, x='Price', y='Revenue', title='Price vs. Total Revenue', template="plotly_white")
    
    # Identify Max Revenue
    max_rev_row = df_econ.loc[df_econ['Revenue'].idxmax()]
    fig_rev.add_scatter(x=[max_rev_row['Price']], y=[max_rev_row['Revenue']], mode='markers+text', 
                        marker=dict(size=12, color='green'), name='Optimal Price',
                        text=[f"Optimal: ${max_rev_row['Price']:.2f}"], textposition="top center")
    
    st.plotly_chart(fig_rev, use_container_width=True)

st.divider()

# Market Share Estimation
st.subheader("🍰 Market Share Simulation")
competitors = ['Your Brand', 'Competitor A', 'Competitor B', 'Competitor C']
initial_shares = [25, 30, 20, 25]

# Dynamic inputs
with st.expander("Adjust Market Conditions"):
    your_price_change = st.slider("Your Price Change (%)", -20, 20, 0)
    ad_spend_change = st.slider("Ad Spend Increase (%)", 0, 100, 0)

# logic: price increase lowers share, ad spend increases share
share_change_factor = 1 + (ad_spend_change * 0.2 - your_price_change * 0.5) / 100
new_your_share = initial_shares[0] * share_change_factor

# Re-normalize
total_other_shares = sum(initial_shares[1:])
remaining_share = 100 - new_your_share
normalization_factor = remaining_share / total_other_shares
new_competitor_shares = [s * normalization_factor for s in initial_shares[1:]]

final_shares = [new_your_share] + new_competitor_shares

fig_share = go.Figure(data=[
    go.Bar(name='Initial Market Share', x=competitors, y=initial_shares),
    go.Bar(name='Projected Market Share', x=competitors, y=final_shares)
])
fig_share.update_layout(barmode='group', title="Market Share Shift Prediction", template="plotly_white")
st.plotly_chart(fig_share, use_container_width=True)

st.caption(f"Projected Share for Your Brand: {new_your_share:.1f}% (Change: {new_your_share - initial_shares[0]:.1f}%)")
