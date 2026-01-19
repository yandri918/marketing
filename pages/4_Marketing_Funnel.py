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
# Initialize session state for editable data
if 'funnel_data' not in st.session_state:
    st.session_state.funnel_data = generate_funnel_data()

df_main = st.session_state.funnel_data

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
    
    # Editable AOV
    st.markdown("### ⚙️ Configuration")
    col_aov1, col_aov2 = st.columns([1, 3])
    
    with col_aov1:
        # Initialize AOV in session state
        if 'avg_order_value' not in st.session_state:
            st.session_state.avg_order_value = 1500000
        
        avg_order_value = st.number_input(
            "Average Order Value (AOV) - Rp",
            min_value=100000,
            max_value=100000000,
            value=st.session_state.avg_order_value,
            step=100000,
            key="aov_input",
            help="Edit this to change revenue calculations"
        )
        
        # Update session state
        st.session_state.avg_order_value = avg_order_value
    
    with col_aov2:
        st.info(f"💡 **Current AOV:** Rp {avg_order_value:,} - All revenue calculations will update automatically when you change this value.")
    
    st.divider()
    
    # Calculate metrics
    funnel_metrics = calculate_funnel_metrics(df_main)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Visitors", f"{df_main.iloc[0]['Users']:,}")
    col2.metric("Conversions", f"{df_main.iloc[-1]['Users']:,}")
    col3.metric("Overall Conv. Rate", f"{funnel_metrics.get('overall_conversion', 0):.2f}%")
    
    # Calculate additional metrics with editable AOV
    total_visitors = df_main.iloc[0]['Users']
    conversions = df_main.iloc[-1]['Users']
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
        st.markdown("### Stage Performance (Editable)")
        
        # Editable funnel data
        edited_df = st.data_editor(
            df_main[['Stage', 'Users']],
            key="funnel_editor",
            num_rows="dynamic",
            use_container_width=True
        )
        
        # Update session state if data changed
        if not edited_df.equals(df_main[['Stage', 'Users']]):
            st.session_state.funnel_data = edited_df
            st.rerun()
        
        # Stage-by-stage conversion
        st.markdown("### Conversion Rates")
        
        for i in range(1, len(df_main)):
            if i < len(df_main) and df_main.iloc[i-1]['Users'] > 0:
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
    
    # Initialize channel data in session state if not exists
    if 'channel_data' not in st.session_state:
        channel_data_init = {}
        for channel in channels:
            multiplier = np.random.uniform(0.7, 1.3)
            channel_df = df_main.copy()
            channel_df['Users'] = (channel_df['Users'] * multiplier).astype(int)
            channel_data_init[channel] = channel_df
        st.session_state.channel_data = channel_data_init
    
    channel_data = st.session_state.channel_data
    
    # Channel Performance Overview
    st.markdown("### Channel Performance Metrics (Editable)")
    
    # Create editable table for channel metrics
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
    
    # Make it editable
    st.info("💡 **Edit channel data below** - Change Visitors and Conversions for each channel. Conv. Rate and Revenue will auto-calculate.")
    
    edited_channel_metrics = st.data_editor(
        channel_metrics_df[['Channel', 'Visitors', 'Conversions']],
        key="channel_metrics_editor",
        use_container_width=True,
        hide_index=True
    )
    
    # Update channel data if edited
    data_changed = False
    for idx, row in edited_channel_metrics.iterrows():
        channel = row['Channel']
        new_visitors = row['Visitors']
        new_conversions = row['Conversions']
        
        # Check if data changed
        if channel in channel_data:
            old_visitors = channel_data[channel].iloc[0]['Users']
            old_conversions = channel_data[channel].iloc[-1]['Users']
            
            if old_visitors != new_visitors or old_conversions != new_conversions:
                data_changed = True
                
                # Update channel data proportionally
                if old_visitors > 0:
                    ratio = new_visitors / old_visitors
                    updated_df = channel_data[channel].copy()
                    updated_df['Users'] = (updated_df['Users'] * ratio).astype(int)
                    updated_df.iloc[-1, updated_df.columns.get_loc('Users')] = new_conversions
                    channel_data[channel] = updated_df
    
    if data_changed:
        st.session_state.channel_data = channel_data
        st.rerun()
    
    # Recalculate metrics with updated data
    channel_metrics_display = []
    for channel, df_channel in channel_data.items():
        metrics = calculate_funnel_metrics(df_channel)
        channel_metrics_display.append({
            'Channel': channel,
            'Visitors': df_channel.iloc[0]['Users'],
            'Conversions': df_channel.iloc[-1]['Users'],
            'Conv. Rate': metrics.get('overall_conversion', 0),
            'Revenue': df_channel.iloc[-1]['Users'] * avg_order_value
        })
    
    channel_metrics_display_df = pd.DataFrame(channel_metrics_display)
    channel_metrics_display_df = channel_metrics_display_df.sort_values('Conv. Rate', ascending=False)
    
    # Display full metrics table
    st.markdown("### 📊 Complete Channel Metrics")
    st.dataframe(channel_metrics_display_df.style.format({
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
    
    # Initialize cohort data in session state if not exists
    if 'cohort_data' not in st.session_state:
        st.session_state.cohort_data = generate_cohort_data(n_cohorts=6)
    
    cohort_df = st.session_state.cohort_data
    
    st.markdown("### Monthly Cohort Performance (Editable)")
    
    st.info("💡 **Edit cohort data below** - Modify user counts for each cohort and stage. Conversion rates will auto-calculate.")
    
    # Make cohort data editable
    edited_cohort_df = st.data_editor(
        cohort_df,
        key="cohort_editor",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cohort": st.column_config.TextColumn("Cohort (YYYY-MM)", disabled=True),
            "Awareness": st.column_config.NumberColumn("Awareness", min_value=0, max_value=1000000, step=100),
            "Interest": st.column_config.NumberColumn("Interest", min_value=0, max_value=1000000, step=100),
            "Consideration": st.column_config.NumberColumn("Consideration", min_value=0, max_value=1000000, step=100),
            "Intent": st.column_config.NumberColumn("Intent", min_value=0, max_value=1000000, step=100),
            "Purchase": st.column_config.NumberColumn("Purchase", min_value=0, max_value=1000000, step=10)
        }
    )
    
    # Update session state if data changed
    if not edited_cohort_df.equals(cohort_df):
        st.session_state.cohort_data = edited_cohort_df
        cohort_df = edited_cohort_df
        st.rerun()
    
    # Add new cohort button
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("➕ Add New Cohort"):
            # Get next month
            last_cohort = cohort_df.iloc[-1]['Cohort']
            last_date = pd.to_datetime(last_cohort)
            next_date = last_date + pd.DateOffset(months=1)
            
            new_cohort = pd.DataFrame([{
                'Cohort': next_date.strftime('%Y-%m'),
                'Awareness': 10000,
                'Interest': 5000,
                'Consideration': 2500,
                'Intent': 1000,
                'Purchase': 350
            }])
            
            st.session_state.cohort_data = pd.concat([cohort_df, new_cohort], ignore_index=True)
            st.rerun()
    
    st.divider()
    
    # Calculate conversion rates
    cohort_df['Conv. Rate'] = (cohort_df['Purchase'] / cohort_df['Awareness']) * 100
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cohorts", len(cohort_df))
    col2.metric("Avg Conv. Rate", f"{cohort_df['Conv. Rate'].mean():.2f}%")
    col3.metric("Best Cohort", f"{cohort_df.loc[cohort_df['Conv. Rate'].idxmax(), 'Cohort']} ({cohort_df['Conv. Rate'].max():.2f}%)")
    
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
    
    # Initialize touchpoints in session state
    if 'touchpoints' not in st.session_state:
        st.session_state.touchpoints = ['Social Media Ad', 'Google Search', 'Email Campaign', 'Direct Visit', 'Purchase']
    
    # Editable touchpoints
    st.markdown("### 🛣️ Customer Journey Touchpoints (Editable)")
    
    st.info("💡 **Customize your customer journey** - Add, edit, or remove touchpoints to match your actual marketing funnel.")
    
    # Create editable list
    touchpoints_df = pd.DataFrame({
        'Order': range(1, len(st.session_state.touchpoints) + 1),
        'Touchpoint': st.session_state.touchpoints
    })
    
    edited_touchpoints_df = st.data_editor(
        touchpoints_df,
        key="touchpoints_editor",
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Order": st.column_config.NumberColumn("Order", disabled=True),
            "Touchpoint": st.column_config.TextColumn("Touchpoint Name", max_chars=50)
        }
    )
    
    # Update touchpoints if changed
    new_touchpoints = edited_touchpoints_df['Touchpoint'].tolist()
    if new_touchpoints != st.session_state.touchpoints:
        st.session_state.touchpoints = new_touchpoints
        st.rerun()
    
    touchpoints = st.session_state.touchpoints
    
    # Quick add buttons
    st.markdown("**Quick Add Common Touchpoints:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("➕ Facebook Ad"):
            st.session_state.touchpoints.append("Facebook Ad")
            st.rerun()
    with col2:
        if st.button("➕ Instagram"):
            st.session_state.touchpoints.append("Instagram Post")
            st.rerun()
    with col3:
        if st.button("➕ YouTube"):
            st.session_state.touchpoints.append("YouTube Video")
            st.rerun()
    with col4:
        if st.button("➕ Webinar"):
            st.session_state.touchpoints.append("Webinar Attendance")
            st.rerun()
    with col5:
        if st.button("🔄 Reset"):
            st.session_state.touchpoints = ['Social Media Ad', 'Google Search', 'Email Campaign', 'Direct Visit', 'Purchase']
            st.rerun()
    
    st.divider()
    
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
    
    # Use actual funnel data for trends
    st.info("💡 **Trends are based on your current funnel data.** Edit the funnel in Tab 1 to see changes here.")
    
    # Generate time series data based on current funnel metrics
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
    
    # Get current conversion rate
    current_conv_rate = funnel_metrics.get('overall_conversion', 0)
    current_visitors = df_main.iloc[0]['Users']
    current_conversions = df_main.iloc[-1]['Users']
    
    trend_data = []
    for i, date in enumerate(dates):
        # Add realistic daily variation around current conversion rate
        daily_variation = np.random.normal(1, 0.05)  # ±5% variation
        
        # Simulate visitors with some variation
        daily_visitors = int(current_visitors * daily_variation)
        
        # Calculate conversions based on current conversion rate with variation
        conv_rate_variation = current_conv_rate * np.random.normal(1, 0.1)  # ±10% variation
        daily_conversions = int(daily_visitors * conv_rate_variation / 100)
        
        trend_data.append({
            'Date': date,
            'Awareness': daily_visitors,
            'Interest': int(daily_visitors * 0.5 * daily_variation),
            'Consideration': int(daily_visitors * 0.25 * daily_variation),
            'Intent': int(daily_visitors * 0.1 * daily_variation),
            'Purchase': daily_conversions,
            'Conv. Rate': (daily_conversions / daily_visitors * 100) if daily_visitors > 0 else 0
        })
    
    trend_df = pd.DataFrame(trend_data)
    
    # Display current metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Conv. Rate", f"{current_conv_rate:.2f}%")
    col2.metric("Avg Conv. Rate (30d)", f"{trend_df['Conv. Rate'].mean():.2f}%")
    col3.metric("Trend Volatility", f"{trend_df['Conv. Rate'].std():.2f}%")
    
    st.divider()
    
    # Conversion rate over time
    st.markdown("### Conversion Rate Trend (Based on Current Funnel)")
    
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
    
    # Add current conversion rate as reference line
    fig_trend.add_hline(
        y=current_conv_rate,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Current: {current_conv_rate:.2f}%"
    )
    
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
        # Create stage options
        stage_options = [f"{df_main.iloc[i-1]['Stage']} → {df_main.iloc[i]['Stage']}" for i in range(1, len(df_main))]
        
        if stage_options:
            stage_to_optimize = st.selectbox(
                "Select stage to optimize",
                stage_options
            )
            
            improvement_pct = st.slider("Expected Improvement (%)", 1, 50, 10)
        else:
            st.warning("Please add at least 2 stages to the funnel")
            stage_to_optimize = None
    
    with col2:
        if stage_to_optimize and len(df_main) >= 2:
            # Find the stage index from the selection
            stage_parts = stage_to_optimize.split('→')
            from_stage = stage_parts[0].strip()
            to_stage = stage_parts[1].strip()
            
            # Find indices
            from_idx = df_main[df_main['Stage'] == from_stage].index[0] if from_stage in df_main['Stage'].values else 0
            to_idx = df_main[df_main['Stage'] == to_stage].index[0] if to_stage in df_main['Stage'].values else 1
            
            if from_idx < len(df_main) and to_idx < len(df_main) and df_main.iloc[from_idx]['Users'] > 0:
                current_users = df_main.iloc[from_idx]['Users']
                current_conv = (df_main.iloc[to_idx]['Users'] / current_users) * 100
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
