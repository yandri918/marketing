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
