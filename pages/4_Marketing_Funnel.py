import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_funnel_data

st.set_page_config(page_title="Advanced Funnel Analysis | Growth", page_icon="🔻", layout="wide")

st.title("🔻 Advanced Marketing Funnel Performance")
st.markdown("Enterprise-grade conversion analytics with **multi-channel analysis**, **attribution modeling**, and **optimization recommendations**.")

# ========== HELPER FUNCTIONS ==========

def calculate_funnel_metrics(df):
    """Calculate comprehensive funnel metrics"""
    metrics = {}
    
    # Overall conversion rate
    if len(df) > 0:
        metrics['overall_conversion'] = (df.iloc[-1]['Users'] / df.iloc[0]['Users']) * 100
        
        # Stage-by-stage conversion
        stage_conversions = []
        for i in range(1, len(df)):
            conv_rate = (df.iloc[i]['Users'] / df.iloc[i-1]['Users']) * 100
            drop_off = df.iloc[i-1]['Users'] - df.iloc[i]['Users']
            stage_conversions.append({
                'from': df.iloc[i-1]['Stage'],
                'to': df.iloc[i]['Stage'],
                'conversion_rate': conv_rate,
                'drop_off': drop_off
            })
        
        metrics['stage_conversions'] = stage_conversions
        
        # Find biggest drop-off
        if stage_conversions:
            biggest_drop = max(stage_conversions, key=lambda x: x['drop_off'])
            metrics['biggest_drop'] = biggest_drop
    
    return metrics

def calculate_attribution(touchpoints, model='linear'):
    """Calculate attribution based on model"""
    n = len(touchpoints)
    
    if n == 0:
        return []
    
    if model == 'first_touch':
        credits = [1.0] + [0.0] * (n - 1)
    elif model == 'last_touch':
        credits = [0.0] * (n - 1) + [1.0]
    elif model == 'linear':
        credits = [1.0 / n] * n
    elif model == 'time_decay':
        # More credit to recent touchpoints
        weights = [2 ** i for i in range(n)]
        total = sum(weights)
        credits = [w / total for w in weights]
    elif model == 'position_based':
        # 40% first, 40% last, 20% middle
        if n == 1:
            credits = [1.0]
        elif n == 2:
            credits = [0.5, 0.5]
        else:
            middle_credit = 0.2 / (n - 2)
            credits = [0.4] + [middle_credit] * (n - 2) + [0.4]
    else:
        credits = [1.0 / n] * n
    
    return credits

def generate_cohort_data(n_cohorts=6):
    """Generate synthetic cohort funnel data"""
    cohorts = []
    base_date = pd.Timestamp('2024-01-01')
    
    for i in range(n_cohorts):
        cohort_date = base_date + pd.DateOffset(months=i)
        
        # Simulate improving conversion over time
        improvement_factor = 1 + (i * 0.05)
        
        cohort_data = {
            'Cohort': cohort_date.strftime('%Y-%m'),
            'Awareness': 10000,
            'Interest': int(5000 * improvement_factor),
            'Consideration': int(2500 * improvement_factor),
            'Intent': int(1000 * improvement_factor),
            'Purchase': int(350 * improvement_factor)
        }
        cohorts.append(cohort_data)
    
    return pd.DataFrame(cohorts)

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Configuration")

# Channel selection
selected_channel = st.sidebar.selectbox(
    "Primary Channel",
    ["All Channels", "Organic", "Paid Search", "Social Media", "Email", "Direct"]
)

# Date range
date_range = st.sidebar.selectbox(
    "Time Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year"]
)

st.sidebar.divider()

# ========== DATA LOADING ==========
df_main = generate_funnel_data()

# Generate multi-channel data
channels = ['Organic', 'Paid Search', 'Social Media', 'Email', 'Direct']
channel_data = {}

for channel in channels:
    # Simulate different performance per channel
    multiplier = np.random.uniform(0.7, 1.3)
    channel_df = df_main.copy()
    channel_df['Users'] = (channel_df['Users'] * multiplier).astype(int)
    channel_data[channel] = channel_df

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔀 Multi-Channel",
    "👥 Cohorts",
    "🎯 Attribution",
    "📈 Trends",
    "💡 Optimization"
])

# ========== TAB 1: OVERVIEW ==========
with tab1:
    st.subheader("📊 Funnel Overview")
    
    # Calculate metrics
    funnel_metrics = calculate_funnel_metrics(df_main)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Visitors", f"{df_main.iloc[0]['Users']:,}")
    col2.metric("Conversions", f"{df_main.iloc[-1]['Users']:,}")
    col3.metric("Overall Conv. Rate", f"{funnel_metrics.get('overall_conversion', 0):.2f}%")
    
    # Calculate additional metrics
    total_visitors = df_main.iloc[0]['Users']
    conversions = df_main.iloc[-1]['Users']
    avg_order_value = 1500000  # Rp
    total_revenue = conversions * avg_order_value
    revenue_per_visitor = total_revenue / total_visitors if total_visitors > 0 else 0
    
    col4.metric("Revenue Per Visitor", f"Rp {revenue_per_visitor:,.0f}")
    col5.metric("Total Revenue", f"Rp {total_revenue/1e9:.2f}B")
    
    st.divider()
    
    # Main Funnel Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Conversion Funnel")
        
        fig_funnel = go.Figure(go.Funnel(
            y=df_main['Stage'],
            x=df_main['Users'],
            textinfo="value+percent initial",
            marker={
                "color": ["#3498DB", "#E67E22", "#E74C3C", "#9B59B6", "#2ECC71"],
                "line": {"width": 2, "color": "white"}
            }
        ))
        
        fig_funnel.update_layout(
            title="Customer Journey Funnel",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col2:
        st.markdown("### Stage Performance")
        
        # Editable funnel data
        edited_df = st.data_editor(
            df_main[['Stage', 'Users']],
            key="funnel_editor",
            num_rows="dynamic",
            use_container_width=True
        )
        
        # Stage-by-stage conversion
        st.markdown("### Conversion Rates")
        
        for i in range(1, len(df_main)):
            conv_rate = (df_main.iloc[i]['Users'] / df_main.iloc[i-1]['Users']) * 100
            drop_off = df_main.iloc[i-1]['Users'] - df_main.iloc[i]['Users']
            
            st.metric(
                f"{df_main.iloc[i-1]['Stage']} → {df_main.iloc[i]['Stage']}",
                f"{conv_rate:.1f}%",
                delta=f"-{drop_off:,} users",
                delta_color="inverse"
            )
    
    # Drop-off Analysis
    st.markdown("### Drop-off Analysis")
    
    if 'biggest_drop' in funnel_metrics:
        biggest = funnel_metrics['biggest_drop']
        st.error(f"""
        🚨 **Biggest Drop-off Point:**
        
        **{biggest['from']} → {biggest['to']}**
        - Conversion Rate: {biggest['conversion_rate']:.1f}%
        - Users Lost: {biggest['drop_off']:,}
        - Potential Revenue Lost: Rp {biggest['drop_off'] * avg_order_value / 1e9:.2f}B
        
        **Priority:** HIGH - Focus optimization efforts here!
        """)
    
    # Funnel Efficiency Score
    st.markdown("### Funnel Efficiency Score")
    
    # Calculate CRO score (0-100)
    cro_score = funnel_metrics.get('overall_conversion', 0) * 2  # Scale to 0-100
    cro_score = min(100, cro_score)
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cro_score,
        title={'text': "CRO Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "#E74C3C"},
                {'range': [33, 66], 'color': "#F39C12"},
                {'range': [66, 100], 'color': "#2ECC71"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

# ========== TAB 2: MULTI-CHANNEL ==========
with tab2:
    st.subheader("🔀 Multi-Channel Funnel Comparison")
    
    # Channel Performance Overview
    st.markdown("### Channel Performance Metrics")
    
    channel_metrics = []
    for channel, df_channel in channel_data.items():
        metrics = calculate_funnel_metrics(df_channel)
        channel_metrics.append({
            'Channel': channel,
            'Visitors': df_channel.iloc[0]['Users'],
            'Conversions': df_channel.iloc[-1]['Users'],
            'Conv. Rate': metrics.get('overall_conversion', 0),
            'Revenue': df_channel.iloc[-1]['Users'] * avg_order_value
        })
    
    channel_metrics_df = pd.DataFrame(channel_metrics)
    channel_metrics_df = channel_metrics_df.sort_values('Conv. Rate', ascending=False)
    
    st.dataframe(channel_metrics_df.style.format({
        'Visitors': '{:,.0f}',
        'Conversions': '{:,.0f}',
        'Conv. Rate': '{:.2f}%',
        'Revenue': 'Rp {:,.0f}'
    }).background_gradient(subset=['Conv. Rate'], cmap='RdYlGn'), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Side-by-side funnel comparison
    st.markdown("### Funnel Comparison")
    
    # Select channels to compare
    compare_channels = st.multiselect(
        "Select channels to compare",
        channels,
        default=channels[:3]
    )
    
    if compare_channels:
        fig_compare = go.Figure()
        
        for channel in compare_channels:
            df_channel = channel_data[channel]
            
            fig_compare.add_trace(go.Funnel(
                name=channel,
                y=df_channel['Stage'],
                x=df_channel['Users'],
                textinfo="value"
            ))
        
        fig_compare.update_layout(
            title="Channel Funnel Comparison",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # Channel ROI Analysis
    st.markdown("### Channel ROI Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by channel
        fig_revenue = px.bar(
            channel_metrics_df,
            x='Channel',
            y='Revenue',
            title="Revenue by Channel",
            color='Revenue',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        # Conversion rate comparison
        fig_conv = px.bar(
            channel_metrics_df,
            x='Channel',
            y='Conv. Rate',
            title="Conversion Rate by Channel",
            color='Conv. Rate',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_conv, use_container_width=True)

# ========== TAB 3: COHORTS ==========
with tab3:
    st.subheader("👥 Cohort Conversion Analysis")
    
    # Generate cohort data
    cohort_df = generate_cohort_data(n_cohorts=6)
    
    st.markdown("### Monthly Cohort Performance")
    
    # Display cohort table
    st.dataframe(cohort_df.style.format({
        'Awareness': '{:,.0f}',
        'Interest': '{:,.0f}',
        'Consideration': '{:,.0f}',
        'Intent': '{:,.0f}',
        'Purchase': '{:,.0f}'
    }).background_gradient(cmap='Blues'), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Cohort conversion rates
    st.markdown("### Cohort Conversion Trends")
    
    cohort_df['Conv. Rate'] = (cohort_df['Purchase'] / cohort_df['Awareness']) * 100
    
    fig_cohort_trend = px.line(
        cohort_df,
        x='Cohort',
        y='Conv. Rate',
        title="Conversion Rate by Cohort",
        markers=True,
        labels={'Conv. Rate': 'Conversion Rate (%)'}
    )
    
    fig_cohort_trend.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_cohort_trend, use_container_width=True)
    
    # Cohort heatmap
    st.markdown("### Cohort Performance Heatmap")
    
    cohort_matrix = cohort_df.set_index('Cohort')[['Awareness', 'Interest', 'Consideration', 'Intent', 'Purchase']]
    
    fig_heatmap = px.imshow(
        cohort_matrix.T,
        labels=dict(x="Cohort", y="Stage", color="Users"),
        title="Cohort Performance Heatmap",
        color_continuous_scale='YlOrRd',
        text_auto=True,
        aspect="auto"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ========== TAB 4: ATTRIBUTION ==========
with tab4:
    st.subheader("🎯 Attribution Modeling")
    
    st.markdown("""
    Compare different attribution models to understand which touchpoints deserve credit for conversions.
    """)
    
    # Simulated customer journey
    touchpoints = ['Social Media Ad', 'Google Search', 'Email Campaign', 'Direct Visit', 'Purchase']
    
    # Calculate attribution for each model
    models = {
        'First-Touch': 'first_touch',
        'Last-Touch': 'last_touch',
        'Linear': 'linear',
        'Time-Decay': 'time_decay',
        'Position-Based': 'position_based'
    }
    
    attribution_results = []
    
    for model_name, model_code in models.items():
        credits = calculate_attribution(touchpoints, model=model_code)
        
        for i, (touchpoint, credit) in enumerate(zip(touchpoints, credits)):
            attribution_results.append({
                'Model': model_name,
                'Touchpoint': touchpoint,
                'Credit': credit * 100,
                'Order': i + 1
            })
    
    attribution_df = pd.DataFrame(attribution_results)
    
    # Attribution comparison
    st.markdown("### Attribution Model Comparison")
    
    fig_attribution = px.bar(
        attribution_df,
        x='Touchpoint',
        y='Credit',
        color='Model',
        barmode='group',
        title="Attribution Credit by Model",
        labels={'Credit': 'Attribution Credit (%)'}
    )
    
    fig_attribution.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_attribution, use_container_width=True)
    
    # Model selector for detailed view
    st.markdown("### Detailed Attribution View")
    
    selected_model = st.selectbox("Select Attribution Model", list(models.keys()))
    
    model_data = attribution_df[attribution_df['Model'] == selected_model]
    
    fig_model = px.pie(
        model_data,
        values='Credit',
        names='Touchpoint',
        title=f"{selected_model} Attribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig_model, use_container_width=True)
    
    # Attribution insights
    st.info(f"""
    **{selected_model} Model Insights:**
    
    {
        "First-Touch: All credit to the first interaction. Good for measuring awareness campaigns." if selected_model == 'First-Touch' else
        "Last-Touch: All credit to the final interaction. Good for measuring conversion campaigns." if selected_model == 'Last-Touch' else
        "Linear: Equal credit to all touchpoints. Fair but doesn't account for importance." if selected_model == 'Linear' else
        "Time-Decay: More credit to recent interactions. Reflects recency bias." if selected_model == 'Time-Decay' else
        "Position-Based: 40% to first, 40% to last, 20% to middle. Balances awareness and conversion."
    }
    """)

# ========== TAB 5: TRENDS ==========
with tab5:
    st.subheader("📈 Funnel Trends & Patterns")
    
    # Generate time series data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
    
    trend_data = []
    for date in dates:
        # Simulate daily funnel with some variation
        base_visitors = 10000
        daily_variation = np.random.normal(1, 0.1)
        
        trend_data.append({
            'Date': date,
            'Awareness': int(base_visitors * daily_variation),
            'Interest': int(base_visitors * 0.5 * daily_variation),
            'Consideration': int(base_visitors * 0.25 * daily_variation),
            'Intent': int(base_visitors * 0.1 * daily_variation),
            'Purchase': int(base_visitors * 0.035 * daily_variation)
        })
    
    trend_df = pd.DataFrame(trend_data)
    trend_df['Conv. Rate'] = (trend_df['Purchase'] / trend_df['Awareness']) * 100
    
    # Conversion rate over time
    st.markdown("### Conversion Rate Trend")
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=trend_df['Date'],
        y=trend_df['Conv. Rate'],
        mode='lines+markers',
        name='Daily Conv. Rate',
        line=dict(color='#3498DB', width=2)
    ))
    
    # Add moving average
    trend_df['MA_7'] = trend_df['Conv. Rate'].rolling(window=7).mean()
    
    fig_trend.add_trace(go.Scatter(
        x=trend_df['Date'],
        y=trend_df['MA_7'],
        mode='lines',
        name='7-Day MA',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    fig_trend.update_layout(
        title="Conversion Rate Over Time",
        xaxis_title="Date",
        yaxis_title="Conversion Rate (%)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Day of week analysis
    st.markdown("### Day of Week Patterns")
    
    trend_df['DayOfWeek'] = trend_df['Date'].dt.day_name()
    
    dow_avg = trend_df.groupby('DayOfWeek')['Conv. Rate'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    fig_dow = px.bar(
        x=dow_avg.index,
        y=dow_avg.values,
        title="Average Conversion Rate by Day of Week",
        labels={'x': 'Day', 'y': 'Avg Conv. Rate (%)'},
        color=dow_avg.values,
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig_dow, use_container_width=True)
    
    # Volume trends
    st.markdown("### Traffic Volume Trends")
    
    fig_volume = go.Figure()
    
    for stage in ['Awareness', 'Interest', 'Consideration', 'Intent', 'Purchase']:
        fig_volume.add_trace(go.Scatter(
            x=trend_df['Date'],
            y=trend_df[stage],
            mode='lines',
            name=stage,
            stackgroup='one'
        ))
    
    fig_volume.update_layout(
        title="Funnel Volume Over Time (Stacked)",
        xaxis_title="Date",
        yaxis_title="Users",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)

# ========== TAB 6: OPTIMIZATION ==========
with tab6:
    st.subheader("💡 Funnel Optimization Recommendations")
    
    # Priority actions
    st.markdown("### 🎯 Priority Actions")
    
    if 'biggest_drop' in funnel_metrics:
        biggest = funnel_metrics['biggest_drop']
        
        # Calculate potential impact
        current_conv = biggest['conversion_rate']
        improved_conv_5 = current_conv * 1.05
        improved_conv_10 = current_conv * 1.10
        
        users_at_stage = df_main[df_main['Stage'] == biggest['from']].iloc[0]['Users']
        
        additional_conversions_5 = users_at_stage * (improved_conv_5 - current_conv) / 100
        additional_conversions_10 = users_at_stage * (improved_conv_10 - current_conv) / 100
        
        additional_revenue_5 = additional_conversions_5 * avg_order_value
        additional_revenue_10 = additional_conversions_10 * avg_order_value
        
        st.error(f"""
        🚨 **URGENT: Fix {biggest['from']} → {biggest['to']} Drop-off**
        
        **Current Performance:**
        - Conversion Rate: {current_conv:.1f}%
        - Users Lost: {biggest['drop_off']:,}
        
        **Impact of Improvement:**
        - **+5% improvement** → +{additional_conversions_5:,.0f} conversions → **+Rp {additional_revenue_5/1e9:.2f}B revenue**
        - **+10% improvement** → +{additional_conversions_10:,.0f} conversions → **+Rp {additional_revenue_10/1e9:.2f}B revenue**
        
        **Recommended Tests:**
        1. 🎨 Simplify user interface
        2. ⚡ Improve page load speed
        3. 📱 Optimize mobile experience
        4. 💬 Add live chat support
        5. 🎁 Offer limited-time incentive
        """)
    
    st.divider()
    
    # Optimization calculator
    st.markdown("### 📊 Optimization Impact Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stage_to_optimize = st.selectbox(
            "Select stage to optimize",
            [f"{df_main.iloc[i-1]['Stage']} → {df_main.iloc[i]['Stage']}" for i in range(1, len(df_main))]
        )
        
        improvement_pct = st.slider("Expected Improvement (%)", 1, 50, 10)
    
    with col2:
        # Calculate impact
        stage_idx = int(stage_to_optimize.split('→')[0].strip()[-1]) if stage_to_optimize else 1
        
        if stage_idx < len(df_main):
            current_users = df_main.iloc[stage_idx-1]['Users']
            current_conv = (df_main.iloc[stage_idx]['Users'] / current_users) * 100
            improved_conv = current_conv * (1 + improvement_pct / 100)
            
            additional_users = current_users * (improved_conv - current_conv) / 100
            additional_revenue = additional_users * avg_order_value
            
            st.metric("Current Conversion", f"{current_conv:.1f}%")
            st.metric("Improved Conversion", f"{improved_conv:.1f}%", delta=f"+{improvement_pct}%")
            st.metric("Additional Revenue", f"Rp {additional_revenue/1e9:.2f}B", delta="Potential Gain")
    
    st.divider()
    
    # Best practices
    st.markdown("### 📚 Optimization Best Practices")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **Top of Funnel (ToFu)**
        
        ✅ Improve ad targeting
        ✅ Optimize landing pages
        ✅ A/B test headlines
        ✅ Reduce bounce rate
        """)
    
    with col2:
        st.warning("""
        **Middle of Funnel (MoFu)**
        
        ⚡ Nurture with email
        ⚡ Provide social proof
        ⚡ Offer free trials
        ⚡ Retarget visitors
        """)
    
    with col3:
        st.info("""
        **Bottom of Funnel (BoFu)**
        
        🎯 Simplify checkout
        🎯 Add urgency (scarcity)
        🎯 Offer guarantees
        🎯 Reduce friction
        """)
    
    # ROI Calculator
    st.markdown("### 💰 Optimization ROI Calculator")
    
    with st.expander("Calculate Expected ROI"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            optimization_cost = st.number_input("Optimization Cost (Rp)", value=10000000, step=1000000)
        
        with col2:
            expected_lift = st.slider("Expected Conversion Lift (%)", 1, 30, 10)
        
        with col3:
            # Calculate ROI
            total_visitors = df_main.iloc[0]['Users']
            current_conversions = df_main.iloc[-1]['Users']
            current_conv_rate = (current_conversions / total_visitors) * 100
            
            improved_conv_rate = current_conv_rate * (1 + expected_lift / 100)
            additional_conversions = total_visitors * (improved_conv_rate - current_conv_rate) / 100
            additional_revenue = additional_conversions * avg_order_value
            
            roi = ((additional_revenue - optimization_cost) / optimization_cost) * 100
            
            st.metric("Expected ROI", f"{roi:.0f}%")
            st.metric("Payback Period", f"{optimization_cost / (additional_revenue / 12):.1f} months")

# ========== FOOTER ==========
st.divider()
st.caption("💡 **Pro Tip:** Focus on the biggest drop-off points first for maximum impact. Even small improvements can generate significant revenue!")
