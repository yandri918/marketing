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
base_price = st.sidebar.number_input("Base Product Price (Rp)", 50000, 500000, 150000, step=5000)
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
                        text=[f"Optimal: Rp {max_rev_row['Price']:,.0f}"], textposition="top center")
    
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

st.divider()

# Advanced Unit Economics
st.header("🧮 Unit Economics & Break-Even Analysis")
st.markdown("Determine the precise sales volume needed to cover costs and achieve profitability.")

ue_col1, ue_col2 = st.columns(2)

with ue_col1:
    st.subheader("💰 Cost Structure")
    price_per_unit = st.number_input("Selling Price per Unit (Rp)", value=float(base_price), step=5000.0)
    cogs = st.number_input("COGS (Material + Labor) Rp", value=60000.0, step=1000.0)
    shipping = st.number_input("Shipping/Fulfillment (Rp)", value=15000.0, step=1000.0)
    marketing_cost = st.number_input("CAC (Marketing per Unit) Rp", value=25000.0, step=1000.0)
    
    variable_cost = cogs + shipping + marketing_cost
    contribution_margin = price_per_unit - variable_cost
    margin_percent = (contribution_margin / price_per_unit) * 100

with ue_col2:
    st.subheader("🏢 Fixed Costs (Monthly)")
    rent = st.number_input("Rent & Utilities (Rp)", value=20000000, step=1000000)
    salaries = st.number_input("Salaries & Overhead (Rp)", value=80000000, step=5000000)
    software = st.number_input("Software & Tools (Rp)", value=5000000, step=500000)
    
    fixed_costs = rent + salaries + software

# Break-Even Calculation
if contribution_margin > 0:
    break_even_units = fixed_costs / contribution_margin
    break_even_revenue = break_even_units * price_per_unit
    
    st.divider()
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Contribution Margin", f"Rp {contribution_margin:,.0f}")
    kpi2.metric("Margin %", f"{margin_percent:.1f}%")
    kpi3.metric("Break-Even Units", f"{break_even_units:,.0f} units", help="Units needed to sell to cover all costs")
    kpi4.metric("Break-Even Revenue", f"Rp {break_even_revenue:,.0f}")
    
    # Visualization: Cost-Volume-Profit (CVP) Analysis
    units_range = np.linspace(0, break_even_units * 2, 50)
    total_revenue_line = units_range * price_per_unit
    total_variable_line = units_range * variable_cost
    total_cost_line = fixed_costs + total_variable_line
    
    fig_cvp = go.Figure()
    
    # Fixed Cost Line
    fig_cvp.add_trace(go.Scatter(x=units_range, y=[fixed_costs]*50, mode='lines', name='Fixed Costs', line=dict(dash='dash', color='red')))
    
    # Total Cost Line
    fig_cvp.add_trace(go.Scatter(x=units_range, y=total_cost_line, mode='lines', name='Total Costs', line=dict(color='orange')))
    
    # Revenue Line
    fig_cvp.add_trace(go.Scatter(x=units_range, y=total_revenue_line, mode='lines', name='Total Revenue', line=dict(color='green', width=3)))
    
    # BEP Marker
    fig_cvp.add_trace(go.Scatter(x=[break_even_units], y=[break_even_revenue], mode='markers+text', 
                                 marker=dict(size=12, color='black'),
                                 text=["Break-Even Point"], textposition="top left", name='BEP'))

    fig_cvp.update_layout(title="Cost-Volume-Profit (CVP) Analysis", xaxis_title="Units Sold", yaxis_title="Rupiah (Rp)", template="plotly_white")
    st.plotly_chart(fig_cvp, use_container_width=True)

else:
    st.error("⚠️ Selling price is lower than variable costs! You are losing money on every sale.")
