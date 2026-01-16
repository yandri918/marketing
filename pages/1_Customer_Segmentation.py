import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_customer_data

st.set_page_config(page_title="Customer Segmentation | Marketing Analytics", page_icon="👥", layout="wide")

st.title("👥 Customer Segmentation Dashboard")
st.markdown("Identify distinct customer groups using **K-Means Clustering** to tailor marketing strategies.")

# Sidebar
st.sidebar.header("Configuration")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 3)
st.sidebar.info("Adjust the number of clusters to see different segmentation scenarios.")

# Load Data
@st.cache_data
def load_data():
    return generate_customer_data()

df = load_data()

# Data Preview
with st.expander("Show Raw Data"):
    st.dataframe(df.head())

# Feature Selection
features = ['Income', 'SpendingScore', 'Age']
X = df[features]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster'] = df['Cluster'].astype(str)

# Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Income vs Spending Score")
    fig1 = px.scatter(
        df, x='Income', y='SpendingScore', color='Cluster',
        title='Customer Segments: Income vs Spending',
        hover_data=['Age', 'Recency_Days'],
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("3D View (Age, Income, Spending)")
    fig2 = px.scatter_3d(
        df, x='Age', y='Income', z='SpendingScore', color='Cluster',
        title='3D Customer Segments',
        opacity=0.8,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig2, use_container_width=True)

# Cluster Analysis
st.divider()
st.subheader("📊 Cluster Insights")

# Calculate metrics per cluster
cluster_metrics = df.groupby('Cluster')[['Age', 'Income', 'SpendingScore', 'Recency_Days', 'Frequency']].mean().reset_index()

# Radar Chart
categories = ['Age', 'Income', 'SpendingScore', 'Recency_Days', 'Frequency']
fig_radar = go.Figure()

for i, row in cluster_metrics.iterrows():
    # Normalize for radar chart visibility
    normalized_values = row[categories] / cluster_metrics[categories].max()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name=f"Cluster {row['Cluster']}"
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    showlegend=True,
    title="Cluster Profile Comparison (Normalized)"
)

c1, c2 = st.columns([1, 1])
with c1:
    st.plotly_chart(fig_radar, use_container_width=True)

with c2:
    st.write("**Average Metrics per Cluster:**")
    st.dataframe(cluster_metrics.style.format("{:.1f}"))
    
    st.markdown("---")
    st.write("**Marketing Recommendations:**")
    for i, row in cluster_metrics.iterrows():
        st.write(f"**Cluster {row['Cluster']}:**")
        if row['SpendingScore'] > 70:
            st.caption("🌟 VIP Customers: Target with exclusive loyalty programs and luxury product launches.")
        elif row['Income'] > 80000 and row['SpendingScore'] < 40:
            st.caption("💰 High Potential: High income but low spending. Target with premium offers to unlock value.")
        elif row['Recency_Days'] > 100:
            st.caption("💤 At Risk: Haven't visited recently. Send win-back campaigns and discounts.")
        else:
            st.caption("🛒 Standard: Regular customers. Maintain engagement with newsletters.")

