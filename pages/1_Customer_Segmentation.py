import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_customer_data

st.set_page_config(page_title="Advanced Customer Segmentation | Marketing Analytics", page_icon="👥", layout="wide")

st.title("👥 Advanced Customer Segmentation & Analytics")
st.markdown("Enterprise-grade customer analytics with **multiple algorithms**, **RFM analysis**, **CLV prediction**, and **churn modeling**.")

# ========== HELPER FUNCTIONS ==========

def calculate_rfm_scores(df):
    """Calculate RFM scores (1-5 scale)"""
    df = df.copy()
    
    # Recency Score (lower is better, so reverse)
    df['R_Score'] = pd.qcut(df['Recency_Days'], 5, labels=[5,4,3,2,1], duplicates='drop')
    
    # Frequency Score (higher is better)
    df['F_Score'] = pd.qcut(df['Frequency'], 5, labels=[1,2,3,4,5], duplicates='drop')
    
    # Monetary Score (higher is better)
    df['M_Score'] = pd.qcut(df['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
    
    # Convert to int
    df['R_Score'] = df['R_Score'].astype(int)
    df['F_Score'] = df['F_Score'].astype(int)
    df['M_Score'] = df['M_Score'].astype(int)
    
    # Combined RFM Score
    df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
    
    return df

def assign_rfm_segment(row):
    """Assign RFM segment based on scores"""
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    
    if r >= 4 and f >= 4 and m >= 4:
        return '🏆 Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return '💎 Loyal Customers'
    elif r >= 4 and f <= 2:
        return '🆕 New Customers'
    elif r >= 3 and f >= 2 and m <= 2:
        return '🌱 Potential Loyalists'
    elif r <= 2 and f >= 3:
        return '⚠️ At Risk'
    elif r <= 2 and f <= 2:
        return '💤 Hibernating'
    elif m >= 4:
        return '💰 Big Spenders'
    else:
        return '🛒 Standard'

def calculate_clv(df):
    """Calculate Customer Lifetime Value"""
    df = df.copy()
    
    # Historical CLV (simple)
    df['CLV_Historical'] = df['Monetary'] * df['Frequency']
    
    # Predictive CLV (using linear model on historical data)
    # Features: Recency, Frequency, Monetary, Age
    X = df[['Recency_Days', 'Frequency', 'Monetary', 'Age']].fillna(0)
    y = df['CLV_Historical']
    
    model = LinearRegression()
    model.fit(X, y)
    
    df['CLV_Predicted'] = model.predict(X)
    df['CLV_Predicted'] = df['CLV_Predicted'].clip(lower=0)  # No negative CLV
    
    return df

def predict_churn(df):
    """Predict churn probability"""
    df = df.copy()
    
    # Simple rule-based churn (for demo)
    # High recency + low frequency = high churn risk
    churn_score = (df['Recency_Days'] / df['Recency_Days'].max()) * 0.6 + \
                  (1 - df['Frequency'] / df['Frequency'].max()) * 0.4
    
    df['Churn_Probability'] = churn_score
    
    # Risk level
    df['Churn_Risk'] = pd.cut(df['Churn_Probability'], 
                               bins=[0, 0.3, 0.6, 1.0],
                               labels=['🟢 Low', '🟡 Medium', '🔴 High'])
    
    return df

def create_cohort_data(df):
    """Create cohort analysis data"""
    # For demo, create synthetic cohort data
    cohorts = ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06']
    months = list(range(6))
    
    # Synthetic retention rates (decreasing over time)
    cohort_matrix = []
    for i, cohort in enumerate(cohorts):
        retention = [100]  # Month 0 always 100%
        for month in range(1, 6):
            # Retention decreases over time
            retention.append(max(20, 100 - (month * 15) - np.random.randint(0, 10)))
        cohort_matrix.append(retention[:6-i])
    
    return cohorts, cohort_matrix

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Configuration")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"]
)

if algorithm == "K-Means":
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 3)
elif algorithm == "Hierarchical":
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 3)
elif algorithm == "Gaussian Mixture":
    n_clusters = st.sidebar.slider("Number of Components", 2, 6, 3)
else:  # DBSCAN
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
    min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)

st.sidebar.divider()

# ========== DATA LOADING ==========
if 'customer_df' not in st.session_state:
    st.session_state.customer_df = generate_customer_data()

# Add New Customer Form
with st.sidebar.expander("➕ Add New Customer"):
    with st.form("add_customer_form"):
        new_age = st.number_input("Age", 18, 100, 30)
        new_income = st.number_input("Income (Rp)", 3000000, 50000000, 10000000, step=1000000)
        new_score = st.number_input("Spending Score (1-100)", 1, 100, 50)
        new_recency = st.number_input("Recency (Days)", 0, 365, 10)
        new_freq = st.number_input("Frequency (Visits)", 1, 100, 5)
        
        submitted = st.form_submit_button("Add Customer")
        if submitted:
            new_data = pd.DataFrame([{
                'CustomerID': f"C{len(st.session_state.customer_df) + 1:03d}",
                'Age': new_age,
                'Income': new_income,
                'SpendingScore': new_score,
                'Recency_Days': new_recency,
                'Frequency': new_freq,
                'Monetary': new_income * 0.1
            }])
            st.session_state.customer_df = pd.concat([st.session_state.customer_df, new_data], ignore_index=True)
            st.success("Customer Added!")
            st.rerun()

df = st.session_state.customer_df.copy()

# ========== ANALYTICS CALCULATIONS ==========

# RFM Analysis
df = calculate_rfm_scores(df)
df['RFM_Segment'] = df.apply(assign_rfm_segment, axis=1)

# CLV Calculation
df = calculate_clv(df)

# Churn Prediction
df = predict_churn(df)

# Clustering
features = ['Income', 'SpendingScore', 'Age', 'Frequency', 'Monetary']
X = df[features].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply selected algorithm
if algorithm == "K-Means":
    model = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)
elif algorithm == "DBSCAN":
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = model.fit_predict(X_scaled)
    # DBSCAN uses -1 for noise, convert to positive
    df['Cluster'] = df['Cluster'] + 1
elif algorithm == "Hierarchical":
    model = AgglomerativeClustering(n_clusters=n_clusters)
    df['Cluster'] = model.fit_predict(X_scaled)
else:  # Gaussian Mixture
    model = GaussianMixture(n_components=n_clusters, random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)

df['Cluster'] = df['Cluster'].astype(str)

# Calculate cluster quality metrics
if len(df['Cluster'].unique()) > 1:
    silhouette_avg = silhouette_score(X_scaled, df['Cluster'].astype(int))
    davies_bouldin = davies_bouldin_score(X_scaled, df['Cluster'].astype(int))
else:
    silhouette_avg = 0
    davies_bouldin = 0

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "🎯 Clustering",
    "💎 RFM Analysis",
    "💰 CLV & Value",
    "⚠️ Churn & Retention",
    "🔮 Predictive",
    "🎯 Action Plan"
])

# ========== TAB 1: OVERVIEW ==========
with tab1:
    st.subheader("📊 Customer Analytics Overview")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Avg CLV", f"Rp {df['CLV_Predicted'].mean():,.0f}")
    col3.metric("High Churn Risk", f"{(df['Churn_Risk'] == '🔴 High').sum()}")
    col4.metric("Champions", f"{(df['RFM_Segment'] == '🏆 Champions').sum()}")
    col5.metric("Total Revenue", f"Rp {df['Monetary'].sum():,.0f}")
    
    st.divider()
    
    # Segment Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Customer Segments Distribution")
        segment_counts = df['RFM_Segment'].value_counts()
        
        fig_pie = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="RFM Segments",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### Churn Risk Distribution")
        churn_counts = df['Churn_Risk'].value_counts()
        
        fig_churn = px.bar(
            x=churn_counts.index,
            y=churn_counts.values,
            title="Customers by Churn Risk",
            labels={'x': 'Risk Level', 'y': 'Count'},
            color=churn_counts.index,
            color_discrete_map={'🟢 Low': '#2ECC71', '🟡 Medium': '#F39C12', '🔴 High': '#E74C3C'}
        )
        st.plotly_chart(fig_churn, use_container_width=True)
    
    # CLV Distribution
    st.markdown("### Customer Lifetime Value Distribution")
    
    fig_clv = px.histogram(
        df,
        x='CLV_Predicted',
        nbins=30,
        title="CLV Distribution",
        labels={'CLV_Predicted': 'Predicted CLV (Rp)'},
        color_discrete_sequence=['#3498DB']
    )
    st.plotly_chart(fig_clv, use_container_width=True)

# ========== TAB 2: CLUSTERING ==========
with tab2:
    st.subheader(f"🎯 {algorithm} Clustering Analysis")
    
    # Cluster Quality Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Algorithm", algorithm)
    m2.metric("Silhouette Score", f"{silhouette_avg:.3f}", help="Higher is better (range: -1 to 1)")
    m3.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}", help="Lower is better")
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Income vs Spending Score")
        fig1 = px.scatter(
            df, x='Income', y='SpendingScore', color='Cluster',
            title=f'{algorithm} Clusters: Income vs Spending',
            hover_data=['Age', 'RFM_Segment', 'CLV_Predicted'],
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("### 3D Cluster View")
        fig2 = px.scatter_3d(
            df, x='Age', y='Income', z='SpendingScore', color='Cluster',
            title='3D Customer Segments',
            opacity=0.8,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Elbow Method (for K-Means)
    if algorithm == "K-Means":
        st.markdown("### Elbow Method - Optimal K Selection")
        
        inertias = []
        silhouettes = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            kmeans_temp.fit(X_scaled)
            inertias.append(kmeans_temp.inertia_)
            silhouettes.append(silhouette_score(X_scaled, kmeans_temp.labels_))
        
        fig_elbow = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Elbow Curve (Inertia)", "Silhouette Score")
        )
        
        fig_elbow.add_trace(
            go.Scatter(x=list(K_range), y=inertias, mode='lines+markers', name='Inertia'),
            row=1, col=1
        )
        
        fig_elbow.add_trace(
            go.Scatter(x=list(K_range), y=silhouettes, mode='lines+markers', name='Silhouette', marker=dict(color='orange')),
            row=1, col=2
        )
        
        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
        fig_elbow.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
        fig_elbow.update_yaxes(title_text="Inertia", row=1, col=1)
        fig_elbow.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    # Cluster Profiles
    st.markdown("### Cluster Profiles")
    
    cluster_profiles = df.groupby('Cluster')[['Age', 'Income', 'SpendingScore', 'Frequency', 'Monetary', 'CLV_Predicted']].mean()
    
    st.dataframe(cluster_profiles.style.format({
        'Age': '{:.1f}',
        'Income': 'Rp {:,.0f}',
        'SpendingScore': '{:.1f}',
        'Frequency': '{:.1f}',
        'Monetary': 'Rp {:,.0f}',
        'CLV_Predicted': 'Rp {:,.0f}'
    }), use_container_width=True)

# ========== TAB 3: RFM ANALYSIS ==========
with tab3:
    st.subheader("💎 RFM Analysis (Recency, Frequency, Monetary)")
    
    # RFM Segment Counts
    rfm_counts = df['RFM_Segment'].value_counts().reset_index()
    rfm_counts.columns = ['Segment', 'Count']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### RFM Segments")
        st.dataframe(rfm_counts, use_container_width=True, hide_index=True)
        
        st.markdown("### Segment Definitions")
        st.markdown("""
        - 🏆 **Champions**: Best customers (R:5, F:5, M:5)
        - 💎 **Loyal**: Regular high-value (R:3-5, F:3-5)
        - 🆕 **New**: Recent first-timers (R:5, F:1)
        - 🌱 **Potential**: Can become loyal (R:3-4, F:2-3)
        - ⚠️ **At Risk**: Used to buy, now inactive (R:1-2, F:3-4)
        - 💤 **Hibernating**: Long inactive (R:1-2, F:1-2)
        - 💰 **Big Spenders**: High monetary value
        - 🛒 **Standard**: Regular customers
        """)
    
    with col2:
        st.markdown("### RFM Heatmap")
        
        # Create RFM matrix
        rfm_matrix = df.pivot_table(
            values='CustomerID',
            index='F_Score',
            columns='M_Score',
            aggfunc='count',
            fill_value=0
        )
        
        fig_heatmap = px.imshow(
            rfm_matrix,
            labels=dict(x="Monetary Score", y="Frequency Score", color="Count"),
            title="RFM Matrix (Frequency vs Monetary)",
            color_continuous_scale='YlOrRd',
            text_auto=True
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # RFM Segment Value
    st.markdown("### Segment Value Analysis")
    
    segment_value = df.groupby('RFM_Segment').agg({
        'CustomerID': 'count',
        'Monetary': 'sum',
        'CLV_Predicted': 'mean',
        'Churn_Probability': 'mean'
    }).reset_index()
    
    segment_value.columns = ['Segment', 'Count', 'Total Revenue', 'Avg CLV', 'Avg Churn Risk']
    segment_value = segment_value.sort_values('Total Revenue', ascending=False)
    
    fig_segment_value = px.bar(
        segment_value,
        x='Segment',
        y='Total Revenue',
        title="Revenue by RFM Segment",
        color='Avg CLV',
        color_continuous_scale='Viridis',
        text_auto='.2s'
    )
    st.plotly_chart(fig_segment_value, use_container_width=True)
    
    st.dataframe(segment_value.style.format({
        'Count': '{:,.0f}',
        'Total Revenue': 'Rp {:,.0f}',
        'Avg CLV': 'Rp {:,.0f}',
        'Avg Churn Risk': '{:.2%}'
    }), use_container_width=True, hide_index=True)

# ========== TAB 4: CLV & VALUE ==========
with tab4:
    st.subheader("💰 Customer Lifetime Value Analysis")
    
    # CLV Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg CLV", f"Rp {df['CLV_Predicted'].mean():,.0f}")
    col2.metric("Total CLV", f"Rp {df['CLV_Predicted'].sum():,.0f}")
    col3.metric("Max CLV", f"Rp {df['CLV_Predicted'].max():,.0f}")
    col4.metric("High-Value (>1M)", f"{(df['CLV_Predicted'] > 1000000).sum()}")
    
    st.divider()
    
    # CLV Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### CLV Distribution by Segment")
        
        fig_clv_segment = px.box(
            df,
            x='RFM_Segment',
            y='CLV_Predicted',
            title="CLV by RFM Segment",
            color='RFM_Segment',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_clv_segment.update_xaxes(tickangle=45)
        st.plotly_chart(fig_clv_segment, use_container_width=True)
    
    with col2:
        st.markdown("### Top 20 High-Value Customers")
        
        top_customers = df.nlargest(20, 'CLV_Predicted')[['CustomerID', 'RFM_Segment', 'CLV_Predicted', 'Churn_Risk']]
        
        st.dataframe(top_customers.style.format({
            'CLV_Predicted': 'Rp {:,.0f}'
        }), use_container_width=True, hide_index=True)
    
    # CLV vs Churn
    st.markdown("### CLV vs Churn Risk")
    
    fig_clv_churn = px.scatter(
        df,
        x='CLV_Predicted',
        y='Churn_Probability',
        color='RFM_Segment',
        size='Monetary',
        title="Customer Value vs Churn Risk",
        labels={'CLV_Predicted': 'Predicted CLV (Rp)', 'Churn_Probability': 'Churn Probability'},
        hover_data=['CustomerID', 'Age'],
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Add quadrant lines
    fig_clv_churn.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="High Churn Threshold")
    fig_clv_churn.add_vline(x=df['CLV_Predicted'].median(), line_dash="dash", line_color="gray", annotation_text="Median CLV")
    
    st.plotly_chart(fig_clv_churn, use_container_width=True)
    
    st.info("""
    **Quadrant Analysis:**
    - **Top Right**: High CLV + High Churn = 🚨 URGENT - Retain these valuable customers!
    - **Top Left**: Low CLV + High Churn = Let go or minimal retention effort
    - **Bottom Right**: High CLV + Low Churn = ✅ Champions - Maintain relationship
    - **Bottom Left**: Low CLV + Low Churn = Standard customers
    """)

# ========== TAB 5: CHURN & RETENTION ==========
with tab5:
    st.subheader("⚠️ Churn Prediction & Retention Analysis")
    
    # Churn Metrics
    churn_high = (df['Churn_Risk'] == '🔴 High').sum()
    churn_medium = (df['Churn_Risk'] == '🟡 Medium').sum()
    churn_low = (df['Churn_Risk'] == '🟢 Low').sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("High Risk", churn_high, delta=f"{churn_high/len(df)*100:.1f}%", delta_color="inverse")
    col2.metric("Medium Risk", churn_medium, delta=f"{churn_medium/len(df)*100:.1f}%")
    col3.metric("Low Risk", churn_low, delta=f"{churn_low/len(df)*100:.1f}%", delta_color="normal")
    col4.metric("At-Risk Revenue", f"Rp {df[df['Churn_Risk']=='🔴 High']['Monetary'].sum():,.0f}")
    
    st.divider()
    
    # Churn Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Churn Risk by Segment")
        
        churn_segment = df.groupby(['RFM_Segment', 'Churn_Risk']).size().reset_index(name='Count')
        
        fig_churn_segment = px.bar(
            churn_segment,
            x='RFM_Segment',
            y='Count',
            color='Churn_Risk',
            title="Churn Risk Distribution by Segment",
            color_discrete_map={'🟢 Low': '#2ECC71', '🟡 Medium': '#F39C12', '🔴 High': '#E74C3C'},
            barmode='stack'
        )
        fig_churn_segment.update_xaxes(tickangle=45)
        st.plotly_chart(fig_churn_segment, use_container_width=True)
    
    with col2:
        st.markdown("### High-Risk Customers (Top 20)")
        
        high_risk = df[df['Churn_Risk'] == '🔴 High'].nlargest(20, 'CLV_Predicted')[
            ['CustomerID', 'RFM_Segment', 'CLV_Predicted', 'Recency_Days', 'Frequency']
        ]
        
        st.dataframe(high_risk.style.format({
            'CLV_Predicted': 'Rp {:,.0f}',
            'Recency_Days': '{:.0f}',
            'Frequency': '{:.0f}'
        }), use_container_width=True, hide_index=True)
    
    # Cohort Analysis
    st.markdown("### Cohort Retention Analysis")
    
    cohorts, cohort_matrix = create_cohort_data(df)
    
    # Create cohort heatmap
    cohort_df = pd.DataFrame(cohort_matrix, index=cohorts)
    cohort_df.columns = [f"Month {i}" for i in range(cohort_df.shape[1])]
    
    fig_cohort = px.imshow(
        cohort_df,
        labels=dict(x="Months Since First Purchase", y="Cohort", color="Retention %"),
        title="Cohort Retention Heatmap",
        color_continuous_scale='RdYlGn',
        text_auto='.0f',
        aspect="auto"
    )
    st.plotly_chart(fig_cohort, use_container_width=True)
    
    st.info("""
    **Cohort Analysis Insights:**
    - Green cells = High retention (good!)
    - Red cells = Low retention (needs attention)
    - Diagonal pattern shows retention decay over time
    - Compare cohorts to identify trends
    """)

# ========== TAB 6: PREDICTIVE ==========
with tab6:
    st.subheader("🔮 Predictive Analytics")
    
    st.markdown("### Next Purchase Prediction")
    
    # Simple next purchase prediction based on frequency and recency
    df_pred = df.copy()
    df_pred['Days_Since_Last'] = df_pred['Recency_Days']
    df_pred['Avg_Days_Between'] = 365 / df_pred['Frequency'].clip(lower=1)
    df_pred['Next_Purchase_Days'] = df_pred['Avg_Days_Between'] - df_pred['Days_Since_Last']
    df_pred['Next_Purchase_Days'] = df_pred['Next_Purchase_Days'].clip(lower=0)
    
    # Categorize
    df_pred['Purchase_Likelihood'] = pd.cut(
        df_pred['Next_Purchase_Days'],
        bins=[-1, 7, 30, 90, 999],
        labels=['🔥 This Week', '📅 This Month', '📆 Next 3 Months', '❓ Uncertain']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        likelihood_counts = df_pred['Purchase_Likelihood'].value_counts()
        
        fig_likelihood = px.pie(
            values=likelihood_counts.values,
            names=likelihood_counts.index,
            title="Next Purchase Likelihood",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_likelihood, use_container_width=True)
    
    with col2:
        st.markdown("### Likely to Purchase Soon (Top 20)")
        
        likely_soon = df_pred[df_pred['Purchase_Likelihood'] == '🔥 This Week'].nlargest(20, 'CLV_Predicted')[
            ['CustomerID', 'RFM_Segment', 'CLV_Predicted', 'Next_Purchase_Days']
        ]
        
        st.dataframe(likely_soon.style.format({
            'CLV_Predicted': 'Rp {:,.0f}',
            'Next_Purchase_Days': '{:.0f} days'
        }), use_container_width=True, hide_index=True)
    
    # Product Recommendation (Simplified)
    st.markdown("### Segment-Based Product Recommendations")
    
    recommendations = {
        '🏆 Champions': ['Premium Products', 'Exclusive Memberships', 'VIP Services', 'Early Access'],
        '💎 Loyal Customers': ['Loyalty Rewards', 'Bundle Deals', 'Referral Programs', 'Premium Upgrades'],
        '🆕 New Customers': ['Welcome Discounts', 'Starter Packs', 'Tutorials', 'Onboarding Guides'],
        '🌱 Potential Loyalists': ['Upsell Offers', 'Cross-sell Products', 'Loyalty Programs', 'Engagement Campaigns'],
        '⚠️ At Risk': ['Win-back Discounts', 'Re-engagement Emails', 'Special Offers', 'Feedback Surveys'],
        '💤 Hibernating': ['Aggressive Discounts', 'New Product Launches', 'Reactivation Campaigns', 'Limited-time Offers'],
        '💰 Big Spenders': ['Luxury Products', 'High-end Services', 'Personalized Experiences', 'Concierge Service'],
        '🛒 Standard': ['Regular Promotions', 'Seasonal Sales', 'Newsletter Content', 'Standard Products']
    }
    
    rec_df = pd.DataFrame([
        {'Segment': seg, 'Recommendations': ', '.join(recs)}
        for seg, recs in recommendations.items()
    ])
    
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

# ========== TAB 7: ACTION PLAN ==========
with tab7:
    st.subheader("🎯 Automated Action Plan & Recommendations")
    
    st.markdown("### 🚨 Priority Actions")
    
    # High-value at-risk customers
    high_value_risk = df[(df['CLV_Predicted'] > df['CLV_Predicted'].quantile(0.75)) & 
                         (df['Churn_Risk'] == '🔴 High')]
    
    if len(high_value_risk) > 0:
        st.error(f"""
        **URGENT: {len(high_value_risk)} High-Value Customers at Risk!**
        
        - Total at-risk revenue: Rp {high_value_risk['Monetary'].sum():,.0f}
        - Average CLV: Rp {high_value_risk['CLV_Predicted'].mean():,.0f}
        
        **Recommended Actions:**
        1. 📞 Personal outreach within 48 hours
        2. 🎁 Exclusive retention offer (20-30% discount)
        3. 📧 VIP win-back email campaign
        4. 🤝 Account manager assignment
        
        **Expected ROI:** 3-5x (retention cost vs. CLV)
        """)
    
    # New customers to nurture
    new_customers = df[df['RFM_Segment'] == '🆕 New Customers']
    
    if len(new_customers) > 0:
        st.info(f"""
        **Opportunity: {len(new_customers)} New Customers to Nurture**
        
        - Potential CLV: Rp {new_customers['CLV_Predicted'].sum():,.0f}
        
        **Recommended Actions:**
        1. 📨 Welcome email series (Days 1, 3, 7, 14)
        2. 🎓 Onboarding tutorial/guide
        3. 💝 First purchase discount (10-15%)
        4. 📱 Mobile app download incentive
        
        **Goal:** Convert to Loyal Customers within 90 days
        """)
    
    # Champions to reward
    champions = df[df['RFM_Segment'] == '🏆 Champions']
    
    if len(champions) > 0:
        st.success(f"""
        **Maintain: {len(champions)} Champion Customers**
        
        - Total revenue contribution: Rp {champions['Monetary'].sum():,.0f}
        - Average CLV: Rp {champions['CLV_Predicted'].mean():,.0f}
        
        **Recommended Actions:**
        1. 🏆 VIP loyalty program enrollment
        2. 🎁 Exclusive early access to new products
        3. 💎 Personalized thank-you gifts
        4. 🌟 Referral incentives (reward for bringing friends)
        
        **Goal:** Maintain 95%+ retention rate
        """)
    
    st.divider()
    
    # Campaign Budget Allocation
    st.markdown("### 💰 Recommended Marketing Budget Allocation")
    
    segment_priority = df.groupby('RFM_Segment').agg({
        'CustomerID': 'count',
        'CLV_Predicted': 'sum',
        'Churn_Probability': 'mean'
    }).reset_index()
    
    segment_priority.columns = ['Segment', 'Count', 'Total CLV', 'Avg Churn']
    
    # Calculate budget allocation (simplified)
    segment_priority['Budget_Weight'] = segment_priority['Total CLV'] * (1 + segment_priority['Avg Churn'])
    segment_priority['Budget_%'] = (segment_priority['Budget_Weight'] / segment_priority['Budget_Weight'].sum() * 100)
    
    fig_budget = px.pie(
        segment_priority,
        values='Budget_%',
        names='Segment',
        title="Recommended Budget Allocation by Segment",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_budget, use_container_width=True)
    
    st.dataframe(segment_priority[['Segment', 'Count', 'Total CLV', 'Budget_%']].style.format({
        'Count': '{:,.0f}',
        'Total CLV': 'Rp {:,.0f}',
        'Budget_%': '{:.1f}%'
    }), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Export Options
    st.markdown("### 📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_all = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download All Customer Data",
            data=csv_all,
            file_name="customer_analytics_full.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_risk = df[df['Churn_Risk'] == '🔴 High'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⚠️ Download High-Risk Customers",
            data=csv_risk,
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )
    
    with col3:
        csv_champions = df[df['RFM_Segment'] == '🏆 Champions'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="🏆 Download Champions",
            data=csv_champions,
            file_name="champion_customers.csv",
            mime="text/csv"
        )

# ========== FOOTER ==========
st.divider()
st.caption("💡 **Pro Tip:** Use the sidebar to add new customers and see real-time updates across all analytics!")
