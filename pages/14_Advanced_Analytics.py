import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(page_title="Advanced Web Analytics | GA4 Style", page_icon="📊", layout="wide")

# Custom CSS for professional look
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    .trend-up {
        color: #10b981;
        font-weight: bold;
    }
    .trend-down {
        color: #ef4444;
        font-weight: bold;
    }
    .section-header {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 20px 0 10px 0;
        font-size: 1.2em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Advanced Web Analytics Dashboard")
st.markdown("**Google Analytics 4 Style** - Enterprise-grade analytics with AI-powered insights")

# ========== SIDEBAR CONFIGURATION ==========
st.sidebar.header("⚙️ Analytics Configuration")

# Data source selection
data_mode = st.sidebar.radio(
    "Data Source",
    ["🤖 AI-Generated (Smart)", "✍️ Manual Input", "📤 Upload CSV", "🔗 API Integration"],
    help="Choose your data source"
)

# Advanced filters
st.sidebar.divider()
st.sidebar.header("🎯 Advanced Filters")

# Date comparison
compare_periods = st.sidebar.checkbox("Compare with Previous Period", value=True)

# Segment selection
segment = st.sidebar.multiselect(
    "User Segments",
    ["All Users", "New Users", "Returning Users", "High-Value Users", "Mobile Users", "Desktop Users"],
    default=["All Users"]
)

# Traffic source filter
traffic_filter = st.sidebar.multiselect(
    "Traffic Sources",
    ["Organic Search", "Direct", "Social Media", "Referral", "Paid Ads", "Email"],
    default=["Organic Search", "Direct", "Social Media"]
)

# Default values
default_sessions = 28337
default_users = 90
default_pageviews = 114103
default_duration = 180
default_bounce_rate = 0.144
default_conversion_rate = 0.0521
default_conversion_value = 500000

if data_mode == "✍️ Manual Input":
    st.sidebar.subheader("📊 Core Metrics")
    
    manual_sessions = st.sidebar.number_input("Total Sessions", min_value=0, value=default_sessions, step=100)
    manual_users = st.sidebar.number_input("Total Users", min_value=0, value=default_users, step=10)
    manual_pageviews = st.sidebar.number_input("Total Pageviews", min_value=0, value=default_pageviews, step=100)
    manual_duration = st.sidebar.number_input("Avg Duration (seconds)", min_value=0, value=default_duration, step=10)
    manual_bounce_rate = st.sidebar.slider("Bounce Rate (%)", 0.0, 100.0, default_bounce_rate * 100, 0.1) / 100
    
    st.sidebar.divider()
    st.sidebar.subheader("💰 Revenue Metrics")
    
    manual_conversion_rate = st.sidebar.slider("Conversion Rate (%)", 0.0, 100.0, default_conversion_rate * 100, 0.1) / 100
    manual_conversion_value = st.sidebar.number_input("Value per Conversion (Rp)", min_value=0, value=default_conversion_value, step=10000, format="%d")
    manual_revenue = st.sidebar.number_input("Total Revenue (Rp)", min_value=0, value=int(default_sessions * default_conversion_rate * default_conversion_value), step=100000, format="%d")

st.sidebar.divider()
st.sidebar.header("📅 Date Range")
date_range = st.sidebar.selectbox("Select Period", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom Range"], index=1)

days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
selected_days = days_map.get(date_range, 30)

# ========== DATA GENERATION ==========
@st.cache_data
def generate_advanced_analytics_data(days=30, sessions_target=28337, conversion_rate=0.0521):
    """Generate professional analytics data with advanced metrics"""
    np.random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    sources = ['Organic Search', 'Direct', 'Social Media', 'Referral', 'Paid Ads', 'Email']
    devices = ['Desktop', 'Mobile', 'Tablet']
    countries = ['Indonesia', 'USA', 'Singapore', 'Malaysia', 'Thailand', 'Philippines', 'Japan', 'Australia']
    pages = ['/home', '/products', '/pricing', '/about', '/contact', '/blog', '/checkout', '/thank-you']
    
    base_multiplier = sessions_target / (len(dates) * 50)
    
    data = []
    user_id = 1
    
    for date in dates:
        hour = date.hour
        hour_multiplier = 1.8 if 9 <= hour <= 17 else 0.4
        day_multiplier = 0.6 if date.weekday() >= 5 else 1.0
        
        base_sessions = int(np.random.poisson(50) * hour_multiplier * day_multiplier * base_multiplier)
        
        for _ in range(base_sessions):
            source = np.random.choice(sources, p=[0.35, 0.25, 0.15, 0.12, 0.08, 0.05])
            device = np.random.choice(devices, p=[0.5, 0.4, 0.1])
            country = np.random.choice(countries, p=[0.35, 0.20, 0.15, 0.10, 0.08, 0.07, 0.03, 0.02])
            
            # User journey
            num_pages = max(1, int(np.random.exponential(3)))
            landing_page = np.random.choice(pages[:6])  # Exclude checkout/thank-you as landing
            
            # Session metrics
            duration = max(10, np.random.exponential(180))
            bounce = 1 if num_pages == 1 else 0
            conversion = 1 if np.random.random() < conversion_rate else 0
            new_user = 1 if np.random.random() < 0.35 else 0
            
            # Revenue (if converted)
            revenue = np.random.uniform(100000, 2000000) if conversion else 0
            
            # Engagement score (0-100)
            engagement_score = min(100, (num_pages * 10) + (duration / 10) + (50 if conversion else 0))
            
            # Event tracking
            events = []
            if num_pages > 2:
                events.append('scroll_depth_75')
            if duration > 120:
                events.append('engaged_session')
            if conversion:
                events.append('purchase')
            
            data.append({
                'timestamp': date,
                'date': date.date(),
                'hour': hour,
                'user_id': user_id,
                'session_id': f"sess_{user_id}_{date.strftime('%Y%m%d%H')}",
                'source': source,
                'device': device,
                'country': country,
                'landing_page': landing_page,
                'pageviews': num_pages,
                'duration': duration,
                'bounce': bounce,
                'conversion': conversion,
                'revenue': revenue,
                'new_user': new_user,
                'engagement_score': engagement_score,
                'events': ','.join(events) if events else 'none'
            })
            
            user_id += 1
    
    return pd.DataFrame(data)

# Generate data
if data_mode == "✍️ Manual Input":
    df = generate_advanced_analytics_data(days=selected_days, sessions_target=manual_sessions, conversion_rate=manual_conversion_rate)
    conversion_value = manual_conversion_value
    total_revenue = manual_revenue
else:
    df = generate_advanced_analytics_data(days=selected_days, sessions_target=default_sessions, conversion_rate=default_conversion_rate)
    conversion_value = default_conversion_value
    total_revenue = df['revenue'].sum()

# Filter data
cutoff_date = df['date'].max() - timedelta(days=selected_days)
df_filtered = df[df['date'] > cutoff_date].copy()

# Previous period for comparison
if compare_periods:
    prev_start = cutoff_date - timedelta(days=selected_days)
    df_previous = df[(df['date'] > prev_start) & (df['date'] <= cutoff_date)].copy()

# ========== EXECUTIVE SUMMARY ==========
st.markdown('<div class="section-header">📊 Executive Summary</div>', unsafe_allow_html=True)

# Key metrics with comparison
col1, col2, col3, col4, col5 = st.columns(5)

total_sessions = len(df_filtered)
total_users = df_filtered['user_id'].nunique()
total_pageviews = df_filtered['pageviews'].sum()
avg_duration = df_filtered['duration'].mean()
bounce_rate = df_filtered['bounce'].mean()

if compare_periods:
    prev_sessions = len(df_previous)
    prev_users = df_previous['user_id'].nunique()
    prev_pageviews = df_previous['pageviews'].sum()
    prev_duration = df_previous['duration'].mean()
    prev_bounce = df_previous['bounce'].mean()
    
    sessions_change = ((total_sessions - prev_sessions) / prev_sessions * 100) if prev_sessions > 0 else 0
    users_change = ((total_users - prev_users) / prev_users * 100) if prev_users > 0 else 0
    pageviews_change = ((total_pageviews - prev_pageviews) / prev_pageviews * 100) if prev_pageviews > 0 else 0
    duration_change = ((avg_duration - prev_duration) / prev_duration * 100) if prev_duration > 0 else 0
    bounce_change = ((bounce_rate - prev_bounce) / prev_bounce * 100) if prev_bounce > 0 else 0
    
    col1.metric("Sessions", f"{total_sessions:,}", f"{sessions_change:+.1f}%")
    col2.metric("Users", f"{total_users:,}", f"{users_change:+.1f}%")
    col3.metric("Pageviews", f"{total_pageviews:,}", f"{pageviews_change:+.1f}%")
    col4.metric("Avg Duration", f"{int(avg_duration)}s", f"{duration_change:+.1f}%")
    col5.metric("Bounce Rate", f"{bounce_rate:.1%}", f"{bounce_change:+.1f}%", delta_color="inverse")
else:
    col1.metric("Sessions", f"{total_sessions:,}")
    col2.metric("Users", f"{total_users:,}")
    col3.metric("Pageviews", f"{total_pageviews:,}")
    col4.metric("Avg Duration", f"{int(avg_duration)}s")
    col5.metric("Bounce Rate", f"{bounce_rate:.1%}")

st.divider()

# ========== ADVANCED TABS ==========
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Real-Time",
    "🎯 User Journey",
    "🔄 Funnel Analysis",
    "👥 Cohort Retention",
    "💰 Revenue Analytics",
    "🤖 AI Insights",
    "📊 Custom Reports"
])

# ========== TAB 1: REAL-TIME ==========
with tab1:
    st.markdown('<div class="section-header">⚡ Real-Time Analytics</div>', unsafe_allow_html=True)
    
    # Last hour data
    last_hour = df_filtered[df_filtered['timestamp'] >= (datetime.now() - timedelta(hours=1))]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 Active Users (Last Hour)", len(last_hour))
    col2.metric("📄 Pageviews/min", f"{len(last_hour) / 60:.1f}")
    col3.metric("🌍 Countries", last_hour['country'].nunique())
    col4.metric("💰 Revenue (Last Hour)", f"Rp {last_hour['revenue'].sum():,.0f}")
    
    st.subheader("📊 Traffic by Minute (Last Hour)")
    
    # Minute-by-minute traffic
    last_hour['minute'] = last_hour['timestamp'].dt.floor('T')
    minute_traffic = last_hour.groupby('minute').size().reset_index(name='sessions')
    
    line_chart = alt.Chart(minute_traffic).mark_line(point=True, strokeWidth=3, color='#10b981').encode(
        x=alt.X('minute:T', title='Time'),
        y=alt.Y('sessions:Q', title='Sessions'),
        tooltip=[
            alt.Tooltip('minute:T', title='Time', format='%H:%M'),
            alt.Tooltip('sessions:Q', title='Sessions')
        ]
    ).properties(height=300)
    
    st.altair_chart(line_chart, use_container_width=True)
    
    # Top pages right now
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Top Active Pages")
        top_pages = last_hour.groupby('landing_page').size().reset_index(name='views').nlargest(5, 'views')
        
        page_chart = alt.Chart(top_pages).mark_bar().encode(
            x=alt.X('views:Q', title='Active Views'),
            y=alt.Y('landing_page:N', title='Page', sort='-x'),
            color=alt.Color('views:Q', scale=alt.Scale(scheme='greens'), legend=None),
            tooltip=['landing_page', 'views']
        ).properties(height=250)
        
        st.altair_chart(page_chart, use_container_width=True)
    
    with col2:
        st.subheader("🌐 Live Traffic Sources")
        live_sources = last_hour.groupby('source').size().reset_index(name='sessions')
        
        source_pie = alt.Chart(live_sources).mark_arc(innerRadius=50).encode(
            theta='sessions:Q',
            color=alt.Color('source:N', scale=alt.Scale(scheme='category20')),
            tooltip=['source', 'sessions']
        ).properties(height=250)
        
        st.altair_chart(source_pie, use_container_width=True)

# ========== TAB 2: USER JOURNEY ==========
with tab2:
    st.markdown('<div class="section-header">🎯 User Journey & Behavior Flow</div>', unsafe_allow_html=True)
    
    st.subheader("📍 Landing Page Performance")
    
    landing_stats = df_filtered.groupby('landing_page').agg({
        'session_id': 'count',
        'bounce': 'mean',
        'conversion': 'mean',
        'duration': 'mean'
    }).reset_index()
    landing_stats.columns = ['page', 'sessions', 'bounce_rate', 'conv_rate', 'avg_duration']
    landing_stats = landing_stats.nlargest(8, 'sessions')
    
    # Multi-metric visualization
    col1, col2 = st.columns(2)
    
    with col1:
        sessions_chart = alt.Chart(landing_stats).mark_bar().encode(
            x=alt.X('sessions:Q', title='Sessions'),
            y=alt.Y('page:N', title='Landing Page', sort='-x'),
            color=alt.Color('sessions:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=['page', alt.Tooltip('sessions:Q', format=',')]
        ).properties(height=350, title='Sessions by Landing Page')
        
        st.altair_chart(sessions_chart, use_container_width=True)
    
    with col2:
        conv_chart = alt.Chart(landing_stats).mark_bar().encode(
            x=alt.X('conv_rate:Q', title='Conversion Rate', axis=alt.Axis(format='%')),
            y=alt.Y('page:N', title='Landing Page', sort='-x'),
            color=alt.Color('conv_rate:Q', scale=alt.Scale(scheme='greens'), legend=None),
            tooltip=['page', alt.Tooltip('conv_rate:Q', title='Conv Rate', format='.2%')]
        ).properties(height=350, title='Conversion Rate by Page')
        
        st.altair_chart(conv_chart, use_container_width=True)
    
    # Engagement distribution
    st.subheader("💎 User Engagement Distribution")
    
    df_filtered['engagement_tier'] = pd.cut(
        df_filtered['engagement_score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    engagement_dist = df_filtered.groupby('engagement_tier').size().reset_index(name='users')
    
    engagement_chart = alt.Chart(engagement_dist).mark_arc(innerRadius=60).encode(
        theta='users:Q',
        color=alt.Color('engagement_tier:N', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(title='Engagement')),
        tooltip=['engagement_tier', alt.Tooltip('users:Q', format=',')]
    ).properties(height=350)
    
    st.altair_chart(engagement_chart, use_container_width=True)

# ========== TAB 3: FUNNEL ANALYSIS ==========
with tab3:
    st.markdown('<div class="section-header">🔄 Conversion Funnel Analysis</div>', unsafe_allow_html=True)
    
    # Define funnel stages
    total_visitors = len(df_filtered)
    engaged_users = len(df_filtered[df_filtered['pageviews'] > 2])
    product_viewers = len(df_filtered[df_filtered['landing_page'].isin(['/products', '/pricing'])])
    converters = df_filtered['conversion'].sum()
    
    funnel_data = pd.DataFrame({
        'stage': ['All Visitors', 'Engaged (3+ pages)', 'Product Viewers', 'Conversions'],
        'users': [total_visitors, engaged_users, product_viewers, int(converters)],
        'percentage': [100, engaged_users/total_visitors*100, product_viewers/total_visitors*100, converters/total_visitors*100]
    })
    
    # Funnel visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        funnel_chart = alt.Chart(funnel_data).mark_bar().encode(
            x=alt.X('users:Q', title='Number of Users'),
            y=alt.Y('stage:N', title='Funnel Stage', sort=['All Visitors', 'Engaged (3+ pages)', 'Product Viewers', 'Conversions']),
            color=alt.Color('percentage:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=[
                'stage',
                alt.Tooltip('users:Q', format=','),
                alt.Tooltip('percentage:Q', title='% of Total', format='.1f')
            ]
        ).properties(height=300, title='Conversion Funnel')
        
        st.altair_chart(funnel_chart, use_container_width=True)
    
    with col2:
        st.subheader("📊 Drop-off Rates")
        
        for i in range(len(funnel_data) - 1):
            current = funnel_data.iloc[i]
            next_stage = funnel_data.iloc[i + 1]
            drop_off = ((current['users'] - next_stage['users']) / current['users'] * 100)
            
            st.metric(
                f"{current['stage']} → {next_stage['stage']}",
                f"{drop_off:.1f}% drop-off",
                delta=f"-{int(current['users'] - next_stage['users'])} users",
                delta_color="inverse"
            )
    
    # Time to conversion
    st.subheader("⏱️ Time to Conversion Analysis")
    
    converters_df = df_filtered[df_filtered['conversion'] == 1]
    converters_df['time_bucket'] = pd.cut(
        converters_df['duration'],
        bins=[0, 60, 180, 300, 600, float('inf')],
        labels=['<1min', '1-3min', '3-5min', '5-10min', '10min+']
    )
    
    time_dist = converters_df.groupby('time_bucket').size().reset_index(name='conversions')
    
    time_chart = alt.Chart(time_dist).mark_bar().encode(
        x=alt.X('time_bucket:N', title='Time to Convert', sort=['<1min', '1-3min', '3-5min', '5-10min', '10min+']),
        y=alt.Y('conversions:Q', title='Number of Conversions'),
        color=alt.Color('conversions:Q', scale=alt.Scale(scheme='oranges'), legend=None),
        tooltip=['time_bucket', 'conversions']
    ).properties(height=300)
    
    st.altair_chart(time_chart, use_container_width=True)

# ========== TAB 4: COHORT RETENTION ==========
with tab4:
    st.markdown('<div class="section-header">👥 Cohort Retention Analysis</div>', unsafe_allow_html=True)
    
    st.info("💡 **Cohort Analysis**: Track how user groups behave over time based on their first visit date")
    
    # Create cohorts based on first visit
    df_filtered['cohort'] = pd.to_datetime(df_filtered.groupby('user_id')['date'].transform('min'))
    df_filtered['date_dt'] = pd.to_datetime(df_filtered['date'])
    df_filtered['days_since_first'] = (df_filtered['date_dt'] - df_filtered['cohort']).dt.days
    
    # Cohort retention matrix
    cohort_data = df_filtered.groupby(['cohort', 'days_since_first'])['user_id'].nunique().reset_index()
    cohort_data.columns = ['cohort', 'period', 'users']
    
    # Pivot for heatmap
    cohort_pivot = cohort_data.pivot(index='cohort', columns='period', values='users')
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100
    
    # Show only first 14 days
    retention_display = retention.iloc[:, :min(15, len(retention.columns))]
    
    # Convert to long format for Altair
    retention_long = retention_display.reset_index().melt(
        id_vars='cohort',
        var_name='days',
        value_name='retention_rate'
    )
    retention_long['cohort'] = retention_long['cohort'].astype(str)
    retention_long['retention_rate'] = retention_long['retention_rate'].fillna(0)
    
    heatmap = alt.Chart(retention_long).mark_rect().encode(
        x=alt.X('days:O', title='Days Since First Visit'),
        y=alt.Y('cohort:N', title='Cohort (First Visit Date)'),
        color=alt.Color('retention_rate:Q', 
                       scale=alt.Scale(scheme='blues', domain=[0, 100]),
                       title='Retention %'),
        tooltip=[
            alt.Tooltip('cohort:N', title='Cohort'),
            alt.Tooltip('days:O', title='Day'),
            alt.Tooltip('retention_rate:Q', title='Retention', format='.1f')
        ]
    ).properties(
        height=400,
        title='Cohort Retention Heatmap (First 14 Days)'
    )
    
    st.altair_chart(heatmap, use_container_width=True)
    
    # Retention curve
    st.subheader("📉 Average Retention Curve")
    
    avg_retention = retention_display.mean().reset_index()
    avg_retention.columns = ['days', 'retention']
    
    retention_curve = alt.Chart(avg_retention).mark_line(point=True, strokeWidth=3, color='#6366f1').encode(
        x=alt.X('days:O', title='Days Since First Visit'),
        y=alt.Y('retention:Q', title='Retention Rate (%)', scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip('days:O', title='Day'),
            alt.Tooltip('retention:Q', title='Retention %', format='.1f')
        ]
    ).properties(height=300)
    
    st.altair_chart(retention_curve, use_container_width=True)

# ========== TAB 5: REVENUE ANALYTICS ==========
with tab5:
    st.markdown('<div class="section-header">💰 Revenue & E-commerce Analytics</div>', unsafe_allow_html=True)
    
    # Revenue metrics
    total_revenue = df_filtered['revenue'].sum()
    total_conversions = df_filtered['conversion'].sum()
    avg_order_value = total_revenue / total_conversions if total_conversions > 0 else 0
    revenue_per_session = total_revenue / len(df_filtered) if len(df_filtered) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Total Revenue", f"Rp {total_revenue:,.0f}")
    col2.metric("🛒 Total Orders", f"{int(total_conversions):,}")
    col3.metric("📊 Avg Order Value", f"Rp {avg_order_value:,.0f}")
    col4.metric("💎 Revenue/Session", f"Rp {revenue_per_session:,.0f}")
    
    st.divider()
    
    # Revenue trend
    st.subheader("📈 Revenue Trend")
    
    daily_revenue = df_filtered.groupby('date').agg({
        'revenue': 'sum',
        'conversion': 'sum'
    }).reset_index()
    daily_revenue.columns = ['date', 'revenue', 'orders']
    
    revenue_chart = alt.Chart(daily_revenue).mark_area(
        line={'color': '#10b981'},
        color=alt.Gradient(
            gradient='linear',
            stops=[
                alt.GradientStop(color='#10b981', offset=0),
                alt.GradientStop(color='#10b98120', offset=1)
            ],
            x1=0, x2=0, y1=0, y2=1
        )
    ).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('revenue:Q', title='Revenue (Rp)'),
        tooltip=[
            alt.Tooltip('date:T', format='%Y-%m-%d'),
            alt.Tooltip('revenue:Q', title='Revenue', format=',.0f'),
            alt.Tooltip('orders:Q', title='Orders', format=',')
        ]
    ).properties(height=350)
    
    st.altair_chart(revenue_chart, use_container_width=True)
    
    # Revenue by source
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Revenue by Traffic Source")
        
        source_revenue = df_filtered.groupby('source')['revenue'].sum().reset_index()
        source_revenue = source_revenue.nlargest(6, 'revenue')
        
        source_rev_chart = alt.Chart(source_revenue).mark_bar().encode(
            x=alt.X('revenue:Q', title='Total Revenue (Rp)'),
            y=alt.Y('source:N', title='Source', sort='-x'),
            color=alt.Color('revenue:Q', scale=alt.Scale(scheme='greens'), legend=None),
            tooltip=[
                'source',
                alt.Tooltip('revenue:Q', title='Revenue', format=',.0f')
            ]
        ).properties(height=300)
        
        st.altair_chart(source_rev_chart, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Revenue by Country")
        
        country_revenue = df_filtered.groupby('country')['revenue'].sum().reset_index()
        country_revenue = country_revenue.nlargest(6, 'revenue')
        
        country_rev_chart = alt.Chart(country_revenue).mark_bar().encode(
            x=alt.X('revenue:Q', title='Total Revenue (Rp)'),
            y=alt.Y('country:N', title='Country', sort='-x'),
            color=alt.Color('revenue:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=[
                'country',
                alt.Tooltip('revenue:Q', title='Revenue', format=',.0f')
            ]
        ).properties(height=300)
        
        st.altair_chart(country_rev_chart, use_container_width=True)

# ========== TAB 6: AI INSIGHTS ==========
with tab6:
    st.markdown('<div class="section-header">🤖 AI-Powered Insights & Predictions</div>', unsafe_allow_html=True)
    
    st.subheader("🔮 Predictive Analytics")
    
    # Simple trend prediction
    daily_sessions = df_filtered.groupby('date').size().reset_index(name='sessions')
    daily_sessions['day_num'] = range(len(daily_sessions))
    
    # Linear regression for trend
    from numpy.polynomial import Polynomial
    p = Polynomial.fit(daily_sessions['day_num'], daily_sessions['sessions'], 1)
    
    # Predict next 7 days
    future_days = range(len(daily_sessions), len(daily_sessions) + 7)
    predictions = [p(day) for day in future_days]
    
    future_dates = [daily_sessions['date'].max() + timedelta(days=i+1) for i in range(7)]
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'predicted_sessions': predictions,
        'type': 'Forecast'
    })
    
    historical_df = daily_sessions[['date', 'sessions']].copy()
    historical_df['type'] = 'Historical'
    historical_df.columns = ['date', 'predicted_sessions', 'type']
    
    combined_df = pd.concat([historical_df, forecast_df])
    
    forecast_chart = alt.Chart(combined_df).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('predicted_sessions:Q', title='Sessions'),
        color=alt.Color('type:N', scale=alt.Scale(domain=['Historical', 'Forecast'], range=['#3b82f6', '#f59e0b'])),
        strokeDash=alt.StrokeDash('type:N', scale=alt.Scale(domain=['Historical', 'Forecast'], range=[[1,0], [5,5]])),
        tooltip=[
            alt.Tooltip('date:T', format='%Y-%m-%d'),
            alt.Tooltip('predicted_sessions:Q', title='Sessions', format='.0f'),
            'type'
        ]
    ).properties(height=350, title='7-Day Traffic Forecast')
    
    st.altair_chart(forecast_chart, use_container_width=True)
    
    # AI Insights
    st.subheader("💡 Automated Insights")
    
    insights = []
    
    # Top performing source
    top_source = df_filtered.groupby('source')['conversion'].sum().idxmax()
    top_source_conv = df_filtered[df_filtered['source'] == top_source]['conversion'].mean()
    insights.append(f"🎯 **Best Performer**: {top_source} has the highest conversion rate at {top_source_conv:.1%}")
    
    # Peak hour
    peak_hour = df_filtered.groupby('hour').size().idxmax()
    insights.append(f"⏰ **Peak Traffic**: Most visitors arrive at {peak_hour}:00 - optimize campaigns for this time")
    
    # High-value users
    high_value = len(df_filtered[df_filtered['revenue'] > avg_order_value * 1.5])
    insights.append(f"💎 **High-Value Users**: {high_value} users ({high_value/len(df_filtered)*100:.1f}%) spent 50%+ above average")
    
    # Mobile trend
    mobile_pct = len(df_filtered[df_filtered['device'] == 'Mobile']) / len(df_filtered) * 100
    insights.append(f"📱 **Mobile Traffic**: {mobile_pct:.1f}% of traffic is mobile - ensure mobile optimization")
    
    for insight in insights:
        st.info(insight)

# ========== TAB 7: CUSTOM REPORTS ==========
with tab7:
    st.markdown('<div class="section-header">📊 Custom Reports & Data Export</div>', unsafe_allow_html=True)
    
    st.subheader("🔧 Build Your Custom Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metrics_to_show = st.multiselect(
            "Select Metrics",
            ['sessions', 'users', 'pageviews', 'duration', 'bounce_rate', 'conversion_rate', 'revenue'],
            default=['sessions', 'conversion_rate', 'revenue']
        )
    
    with col2:
        dimensions = st.multiselect(
            "Group By",
            ['source', 'device', 'country', 'landing_page'],
            default=['source']
        )
    
    if dimensions and metrics_to_show:
        # Build custom report
        agg_dict = {}
        if 'sessions' in metrics_to_show:
            agg_dict['session_id'] = 'count'
        if 'users' in metrics_to_show:
            agg_dict['user_id'] = 'nunique'
        if 'pageviews' in metrics_to_show:
            agg_dict['pageviews'] = 'sum'
        if 'duration' in metrics_to_show:
            agg_dict['duration'] = 'mean'
        if 'bounce_rate' in metrics_to_show:
            agg_dict['bounce'] = 'mean'
        if 'conversion_rate' in metrics_to_show:
            agg_dict['conversion'] = 'mean'
        if 'revenue' in metrics_to_show:
            agg_dict['revenue'] = 'sum'
        
        custom_report = df_filtered.groupby(dimensions).agg(agg_dict).reset_index()
        
        # Rename columns
        col_rename = {
            'session_id': 'Sessions',
            'user_id': 'Users',
            'pageviews': 'Pageviews',
            'duration': 'Avg Duration (s)',
            'bounce': 'Bounce Rate',
            'conversion': 'Conversion Rate',
            'revenue': 'Revenue (Rp)'
        }
        custom_report.rename(columns=col_rename, inplace=True)
        
        st.dataframe(custom_report.style.format({
            'Avg Duration (s)': '{:.0f}',
            'Bounce Rate': '{:.1%}',
            'Conversion Rate': '{:.2%}',
            'Revenue (Rp)': '{:,.0f}'
        }), use_container_width=True)
        
        # Export options
        st.divider()
        st.subheader("📥 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = custom_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.button("📊 Export to Excel", disabled=True, help="Excel export coming soon")
        
        with col3:
            st.button("📧 Email Report", disabled=True, help="Email feature coming soon")

# ========== FOOTER ==========
st.divider()
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    <p style='font-size: 1.1em; font-weight: 600;'>Advanced Web Analytics Dashboard v2.0</p>
    <p>Powered by AI • Real-time Insights • Predictive Analytics</p>
    <p style='font-size: 0.9em; margin-top: 10px;'>
        Built with ❤️ using Streamlit & Altair | Enterprise-grade Analytics Platform
    </p>
</div>
""", unsafe_allow_html=True)
