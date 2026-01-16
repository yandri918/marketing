import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Channel Effectiveness | Digital vs Field", page_icon="⚔️", layout="wide")

st.title("⚔️ Channel Effectiveness Analysis")
st.markdown("Compare the ROI and Efficiency of **Digital Marketing** vs **Field Officer (Offline Acquisition)**.")

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.header("📱 Digital Marketing")
    st.info("Scalable, but often lower engagement/activation.")
    
    dig_ad_spend = st.number_input("Ad Spend ($)", value=5000, step=100)
    dig_creative = st.number_input("Creative & Content Costs ($)", value=1000, step=100)
    dig_tools = st.number_input("Software/Tools Cost ($)", value=500, step=50)
    
    dig_leads = st.number_input("Digital Leads Generated", value=2000, step=100)
    dig_conversion_rate = st.slider("Digital Conversion Rate (%)", 0.0, 20.0, 5.0, step=0.1)
    
    dig_stores = int(dig_leads * (dig_conversion_rate/100))
    dig_total_cost = dig_ad_spend + dig_creative + dig_tools
    dig_cac = dig_total_cost / dig_stores if dig_stores > 0 else 0

with col2:
    st.header("👔 Field Officer (Offline)")
    st.info("High trust, relationship-based, higher activation.")
    
    field_officers = st.number_input("Number of Field Officers", value=5, step=1)
    field_salary = st.number_input("Monthly Salary per Officer ($)", value=800, step=50)
    field_transport = st.number_input("Transport & Logistics per Officer ($)", value=200, step=20)
    field_commission = st.number_input("Commission per Store Acquired ($)", value=20, step=5)
    
    field_stores_per_officer = st.number_input("Avg Stores Acquired per Officer/Month", value=30, step=1)
    
    field_total_stores = field_officers * field_stores_per_officer
    field_base_cost = (field_salary + field_transport) * field_officers
    field_comm_cost = field_commission * field_total_stores
    field_total_cost = field_base_cost + field_comm_cost
    field_cac = field_total_cost / field_total_stores if field_total_stores > 0 else 0

st.divider()

# Comparative Analysis
st.header("📊 Head-to-Head Comparison")

# Metrics Comparison
m1, m2, m3, m4 = st.columns(4)

# CAC Winner
cac_delta = field_cac - dig_cac
winner_cac = "Digital" if dig_cac < field_cac else "Field"
m1.metric("Digital CAC", f"${dig_cac:.2f}")
m2.metric("Field CAC", f"${field_cac:.2f}", delta=f"${-cac_delta:.2f}" if winner_cac=="Digital" else f"${cac_delta:.2f}", delta_color="inverse")

# Volume Winner
vol_delta = field_total_stores - dig_stores
winner_vol = "Field" if field_total_stores > dig_stores else "Digital"
m3.metric("Digital Stores", f"{dig_stores}")
m4.metric("Field Stores", f"{field_total_stores}", delta=f"{vol_delta}", delta_color="normal")

# Data for Charts
comparison_data = pd.DataFrame({
    'Metric': ['Total Cost', 'Total Stores Acquired', 'CAC (Cost per Store)'],
    'Digital': [dig_total_cost, dig_stores, dig_cac],
    'Field Officer': [field_total_cost, field_total_stores, field_cac]
})

c1, c2 = st.columns(2)

with c1:
    st.subheader("Cost vs Volume")
    fig_bar = go.Figure(data=[
        go.Bar(name='Digital', x=['Total Cost', 'Total Stores'], y=[dig_total_cost, dig_stores], marker_color='#3498db'),
        go.Bar(name='Field Officer', x=['Total Cost', 'Total Stores'], y=[field_total_cost, field_total_stores], marker_color='#2ecc71')
    ])
    fig_bar.update_layout(barmode='group', title="Budget & Output Comparison", template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.subheader("Cost Efficiency (CAC)")
    fig_cac = px.bar(
        x=['Digital', 'Field Officer'], 
        y=[dig_cac, field_cac], 
        color=['Digital', 'Field Officer'],
        color_discrete_map={'Digital': '#3498db', 'Field Officer': '#2ecc71'},
        title="Cost Per Acquisition (Lower is Better)",
        labels={'y': 'CAC ($)', 'x': 'Channel'}
    )
    st.plotly_chart(fig_cac, use_container_width=True)

# Quality / Activation Adjustment
st.divider()
st.subheader("💎 Quality Adjusted Cost")
st.markdown("Offline stores often have higher **Activation Rates** (actual transactions) than Digital ones. Adjust below to see the 'Real' cost.")

q1, q2 = st.columns(2)
with q1:
    dig_activation = st.slider("Digital Store Activation Rate (%)", 0, 100, 40)
with q2:
    field_activation = st.slider("Field Store Activation Rate (%)", 0, 100, 80)

real_dig_stores = dig_stores * (dig_activation/100)
real_field_stores = field_total_stores * (field_activation/100)

real_dig_cac = dig_total_cost / real_dig_stores if real_dig_stores > 0 else 0
real_field_cac = field_total_cost / real_field_stores if real_field_stores > 0 else 0

k1, k2, k3 = st.columns(3)
k1.metric("Effective Digital CAC", f"${real_dig_cac:.2f}", delta=f"{real_dig_cac - dig_cac:.2f} impact")
k2.metric("Effective Field CAC", f"${real_field_cac:.2f}", delta=f"{real_field_cac - field_cac:.2f} impact")

cost_gap = real_field_cac - real_dig_cac
if cost_gap < 0:
    st.success(f"🏆 **Field Officer is actually {abs(cost_gap):.2f} cheaper** per *Active* Store!")
else:
    st.info(f"🏆 **Digital is still {cost_gap:.2f} cheaper** per *Active* Store.")

# Hybrid Recommendation
st.subheader("💡 Strategic Recommendation")
total_budget = dig_total_cost + field_total_cost
if real_field_cac < real_dig_cac:
    st.markdown(f"""
    **Focus on Offline Scale:**  
    Since Field Officers deliver higher quality at a better effective rate, allocate **70%** of budget to expanding the field team. 
    Use the remaining **30%** of Digital for brand awareness to support the officers.
    """)
else:
    st.markdown(f"""
    **Digital First Strategy:**  
    Digital is significantly deeper. Use Digital for volume acquisition (**60-70%** budget). 
    Use Field Officers as a "Special Ops" team for high-value strategic partners only.
    """)
