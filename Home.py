import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Marketing Portfolio | Data-Driven Growth",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #1E1E1E;
        text-align: center;
    }
    .sub-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #4B4B4B;
        text-align: center;
    }
    .card {
        background-color: #f9f9fab0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://avatars.githubusercontent.com/u/yandri918", width=250, caption="Data-Driven Marketer")  # Placeholder or GitHub avatar if available
    
with col2:
    st.markdown("<h1 class='main-header'>Hi, I'm Yandri</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Marketing Analyst | Growth Engineer | Data Scientist</h3>", unsafe_allow_html=True)
    st.write("""
    I bridge the gap between **Marketing Strategy**, **Economics**, and **Data Science**. 
    I build tools that automate insights, predict market trends, and optimize conversion analytics using Python & Streamlit.
    """)
    
    # Social Links
    st.markdown("""
    [<img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white' width=100>](https://github.com/yandri918)
    &nbsp;&nbsp;
    [<img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' width=100>](https://linkedin.com/in/yandri-s)
    """, unsafe_allow_html=True)

st.divider()

# Expertise Section
st.markdown("## 🎯 My Expertise")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
    <h3 style="text-align:center;">📊 Marketing Analytics</h3>
    <ul style="list-style-type:none; padding:0; text-align:center;">
        <li>Customer Segmentation</li>
        <li>Lifetime Value (CLV)</li>
        <li>Campaign Performance</li>
        <li>Churn Prediction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
    <h3 style="text-align:center;">📈 Growth Marketing</h3>
    <ul style="list-style-type:none; padding:0; text-align:center;">
        <li>Funnel Analysis</li>
        <li>A/B Testing</li>
        <li>Conversion Optimization</li>
        <li>Predictive Modeling</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
    <h3 style="text-align:center;">💡 Product Marketing</h3>
    <ul style="list-style-type:none; padding:0; text-align:center;">
        <li>Market Research</li>
        <li>Competitive Analysis</li>
        <li>Pricing Strategy</li>
        <li>Data Storytelling</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Portfolio Highlights
st.markdown("## 🚀 Portfolio Projects")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Marketing Analytics")
    st.info("**1. Customer Segmentation Engine**\n\nUsing K-Means clustering to identify high-value customer segments based on RFM analysis.")
    st.info("**2. Market Demand Forecasting**\n\nPredicting sales trends using Prophet & ARIMA time-series modeling for better inventory planning.")

with col_b:
    st.subheader("🤖 AI & Economics")
    st.success("**3. Social Media Sentiment Analysis**\n\nReal-time brand sentiment tracking using NLP transformers on social comments.")
    st.success("**4. Competitive Market Analysis**\n\nEconomic modeling of price elasticity and market share estimation.")

st.warning("👈 **Select a project from the sidebar to explore the interactive demos.**")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© 2026 Built with Streamlit by Yandri</p>", unsafe_allow_html=True)
