import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go

st.set_page_config(page_title="A/B Testing | Experimentation", page_icon="🧪", layout="wide")

st.title("🧪 A/B Testing Simulator")
st.markdown("Statistically validate marketing experiments to make data-driven decisions.")

# Experiment Setup
col1, col2 = st.columns(2)

with col1:
    st.subheader("Control Group (A)")
    visitors_a = st.number_input("Visitors (Control)", value=1000, step=100)
    conversions_a = st.number_input("Conversions (Control)", value=120, step=10)
    
with col2:
    st.subheader("Variant Group (B)")
    visitors_b = st.number_input("Visitors (Variant)", value=1000, step=100)
    conversions_b = st.number_input("Conversions (Variant)", value=150, step=10)

# Calculations
rate_a = conversions_a / visitors_a
rate_b = conversions_b / visitors_b
uplift = (rate_b - rate_a) / rate_a

# Statistical Significance (Z-Test)
# Standard error
se_a = np.sqrt(rate_a * (1 - rate_a) / visitors_a)
se_b = np.sqrt(rate_b * (1 - rate_b) / visitors_b)
se_diff = np.sqrt(se_a**2 + se_b**2)

# Z-score and P-value
z_score = (rate_b - rate_a) / se_diff
p_value = stats.norm.sf(abs(z_score)) * 2  # Two-tailed test

confidence = (1 - p_value) * 100

st.divider()

# Results Display
st.header("📊 Experiment Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Conversion Rate A", f"{rate_a:.2%}")
m2.metric("Conversion Rate B", f"{rate_b:.2%}")
m3.metric("Uplift", f"{uplift:.2%}", delta_color="normal")
m4.metric("Confidence Level", f"{confidence:.2f}%")

# Decision Logic
if confidence >= 95:
    st.success(f"✅ **Statistically Significant!** You can be {confidence:.1f}% confident that the Variant B performs better than Control A.")
else:
    st.warning("⚠️ **Not Significant Yet.** The difference could be due to random chance. Collect more data.")

# Visualizing Distributions
st.subheader("Distribution Analysis")
x = np.linspace(min(rate_a, rate_b) - 0.05, max(rate_a, rate_b) + 0.05, 1000)
y_a = stats.norm.pdf(x, rate_a, se_a)
y_b = stats.norm.pdf(x, rate_b, se_b)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_a, mode='lines', name='Control A', fill='tozeroy', line=dict(color='gray')))
fig.add_trace(go.Scatter(x=x, y=y_b, mode='lines', name='Variant B', fill='tozeroy', line=dict(color='blue')))

fig.update_layout(title="Probability Density Functions of Conversion Rates", xaxis_title="Conversion Rate", yaxis_title="Probability Density", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)
