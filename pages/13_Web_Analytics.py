import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

st.set_page_config(page_title="Web Analytics | Google Analytics Style", page_icon="📊", layout="wide")

st.title("📊 Web Analytics Dashboard")
st.markdown("**Google Analytics-Style** comprehensive web analytics with real-time insights")

# ========== SIDEBAR CONFIGURATION ==========
st.sidebar.header("⚙️ Data Configuration")

data_mode = st.sidebar.radio(
    "Data Source",
    ["Synthetic Data (Auto)", "Manual Input", "Upload CSV"],
    help="Choose how to input your analytics data"
)

# Default values
default_sessions = 28337
default_users = 90
default_pageviews = 114103
default_duration = 180
default_bounce_rate = 0.144
default_conversion_rate = 0.0521
default_conversion_value = 500000

if data_mode == "Manual Input":
    st.sidebar.subheader("📊 Set Your Metrics")
    
    manual_sessions = st.sidebar.number_input(
        "Total Sessions",
        min_value=0,
        value=default_sessions,
        step=100
    )
    
    manual_users = st.sidebar.number_input(
        "Total Users",
        min_value=0,
        value=default_users,
        step=10
    )
    
    manual_pageviews = st.sidebar.number_input(
        "Total Pageviews",
        min_value=0,
        value=default_pageviews,
        step=100
    )
    
    manual_duration = st.sidebar.number_input(
        "Avg Duration (seconds)",
        min_value=0,
        value=default_duration,
        step=10
    )
    
    manual_bounce_rate = st.sidebar.slider(
        "Bounce Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=default_bounce_rate * 100,
        step=0.1
    ) / 100
    
    st.sidebar.divider()
    st.sidebar.subheader("💰 Conversion Settings")
    
    manual_conversion_rate = st.sidebar.slider(
        "Conversion Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=default_conversion_rate * 100,
        step=0.1
    ) / 100
    
    manual_conversion_value = st.sidebar.number_input(
        "Value per Conversion (Rp)",
        min_value=0,
        value=default_conversion_value,
        step=10000,
        format="%d"
    )

st.sidebar.divider()
st.sidebar.header("📅 Date Range")
date_range = st.sidebar.selectbox(
    "Select Period",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days"],
    index=1
)

days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
selected_days = days_map[date_range]

# ========== DATA GENERATION ==========
@st.cache_data
def generate_analytics_data(days=30, sessions_target=None, users_target=None, 
                           pageviews_target=None, duration_target=None, bounce_target=None,
                           conversion_rate=0.05):
    """Generate realistic web analytics data"""
    np.random.seed(42)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Traffic sources
    sources = ['Organic Search', 'Direct', 'Social Media', 'Referral', 'Paid Ads', 'Email']
    devices = ['Desktop', 'Mobile', 'Tablet']
    countries = ['Indonesia', 'USA', 'Singapore', 'Malaysia', 'Thailand', 'Philippines']
    
    # Calculate base multiplier if targets provided
    if sessions_target:
        base_multiplier = sessions_target / (len(dates) * 50)
    else:
        base_multiplier = 1.0
    
    data = []
    for date in dates:
        # Hourly pattern (more traffic during business hours)
        hour = date.hour
        hour_multiplier = 1.5 if 9 <= hour <= 17 else 0.5
        
        # Weekly pattern (less traffic on weekends)
        day_multiplier = 0.7 if date.weekday() >= 5 else 1.0
        
        base_sessions = int(np.random.poisson(50) * hour_multiplier * day_multiplier * base_multiplier)
        
        for _ in range(base_sessions):
            source = np.random.choice(sources, p=[0.35, 0.25, 0.15, 0.12, 0.08, 0.05])
            device = np.random.choice(devices, p=[0.5, 0.4, 0.1])
            country = np.random.choice(countries, p=[0.4, 0.2, 0.15, 0.1, 0.08, 0.07])
            
            # Session metrics - adjust based on targets
            if pageviews_target and sessions_target:
                avg_pages = pageviews_target / sessions_target
                pageviews = max(1, int(np.random.poisson(avg_pages)))
            else:
                pageviews = np.random.randint(1, 8)
            
            if duration_target:
                duration = np.random.exponential(duration_target)
            else:
                duration = np.random.exponential(180)
            
            if bounce_target:
                bounce = 1 if np.random.random() < bounce_target else 0
            else:
                bounce = 1 if pageviews == 1 else 0
            
            conversion = 1 if np.random.random() < conversion_rate else 0
            
            data.append({
                'timestamp': date,
                'date': date.date(),
                'hour': hour,
                'source': source,
                'device': device,
                'country': country,
                'pageviews': pageviews,
                'duration': duration,
                'bounce': bounce,
                'conversion': conversion,
                'new_user': 1 if np.random.random() < 0.3 else 0
            })
    
    return pd.DataFrame(data)

# Generate or load data based on mode
if data_mode == "Manual Input":
    df = generate_analytics_data(
        days=selected_days,
        sessions_target=manual_sessions,
        users_target=manual_users,
        pageviews_target=manual_pageviews,
        duration_target=manual_duration,
        bounce_target=manual_bounce_rate,
        conversion_rate=manual_conversion_rate
    )
    conversion_value = manual_conversion_value
elif data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Analytics CSV",
        type=['csv'],
        help="CSV must have columns: timestamp, source, device, pageviews, duration, bounce, conversion"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            st.sidebar.success(f"✅ Loaded {len(df)} sessions")
            conversion_value = default_conversion_value
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            df = generate_analytics_data(days=selected_days, conversion_rate=default_conversion_rate)
            conversion_value = default_conversion_value
    else:
        st.sidebar.info("👆 Upload a CSV file to use real data")
        df = generate_analytics_data(days=selected_days, conversion_rate=default_conversion_rate)
        conversion_value = default_conversion_value
else:
    # Synthetic data with defaults
    df = generate_analytics_data(
        days=selected_days,
        sessions_target=default_sessions,
        users_target=default_users,
        pageviews_target=default_pageviews,
        duration_target=default_duration,
        bounce_target=default_bounce_rate,
        conversion_rate=default_conversion_rate
    )
    conversion_value = default_conversion_value

# Filter data
cutoff_date = df['date'].max() - timedelta(days=selected_days)
df_filtered = df[df['date'] > cutoff_date].copy()

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview",
    "👥 Audience", 
    "🔍 Acquisition",
    "🎯 Behavior",
    "💰 Conversions"
])

# ========== TAB 1: OVERVIEW ==========
with tab1:
    st.header("Real-Time Overview")
    
    # Key Metrics
    total_sessions = len(df_filtered)
    total_users = df_filtered.groupby(['date', 'device']).size().count()
    total_pageviews = df_filtered['pageviews'].sum()
    avg_duration = df_filtered['duration'].mean()
    bounce_rate = df_filtered['bounce'].mean()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Sessions", f"{total_sessions:,}")
    col2.metric("Users", f"{total_users:,}")
    col3.metric("Pageviews", f"{total_pageviews:,}")
    col4.metric("Avg Duration", f"{int(avg_duration)}s")
    col5.metric("Bounce Rate", f"{bounce_rate:.1%}")
    
    st.divider()
    
    # Sessions over time
    st.subheader("📊 Sessions Trend")
    daily_sessions = df_filtered.groupby('date').size().reset_index(name='sessions')
    
    line_chart = alt.Chart(daily_sessions).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('sessions:Q', title='Sessions'),
        tooltip=[
            alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
            alt.Tooltip('sessions:Q', title='Sessions', format=',')
        ]
    ).properties(
        height=400,
        title='Daily Sessions'
    ).configure_mark(
        color='#1f77b4'
    )
    
    st.altair_chart(line_chart, use_container_width=True)
    
    # Traffic sources breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Traffic Sources")
        source_data = df_filtered.groupby('source').size().reset_index(name='sessions')
        
        pie_chart = alt.Chart(source_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta('sessions:Q'),
            color=alt.Color('source:N', legend=alt.Legend(title="Source")),
            tooltip=[
                alt.Tooltip('source:N', title='Source'),
                alt.Tooltip('sessions:Q', title='Sessions', format=',')
            ]
        ).properties(
            height=350
        )
        
        st.altair_chart(pie_chart, use_container_width=True)
    
    with col2:
        st.subheader("📱 Device Breakdown")
        device_data = df_filtered.groupby('device').size().reset_index(name='sessions')
        
        bar_chart = alt.Chart(device_data).mark_bar().encode(
            x=alt.X('sessions:Q', title='Sessions'),
            y=alt.Y('device:N', title='Device', sort='-x'),
            color=alt.Color('device:N', legend=None),
            tooltip=[
                alt.Tooltip('device:N', title='Device'),
                alt.Tooltip('sessions:Q', title='Sessions', format=',')
            ]
        ).properties(
            height=350
        )
        
        st.altair_chart(bar_chart, use_container_width=True)

# ========== TAB 2: AUDIENCE ==========
with tab2:
    st.header("👥 Audience Analytics")
    
    # New vs Returning
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🆕 New vs Returning Users")
        user_type_data = df_filtered.groupby('new_user').size().reset_index(name='count')
        user_type_data['type'] = user_type_data['new_user'].map({1: 'New Users', 0: 'Returning Users'})
        
        donut = alt.Chart(user_type_data).mark_arc(innerRadius=60).encode(
            theta=alt.Theta('count:Q'),
            color=alt.Color('type:N', scale=alt.Scale(scheme='category10')),
            tooltip=[
                alt.Tooltip('type:N', title='User Type'),
                alt.Tooltip('count:Q', title='Count', format=',')
            ]
        ).properties(height=300)
        
        st.altair_chart(donut, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Top Countries")
        country_data = df_filtered.groupby('country').size().reset_index(name='sessions').nlargest(6, 'sessions')
        
        country_bar = alt.Chart(country_data).mark_bar().encode(
            x=alt.X('sessions:Q', title='Sessions'),
            y=alt.Y('country:N', title='Country', sort='-x'),
            color=alt.Color('sessions:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=[
                alt.Tooltip('country:N', title='Country'),
                alt.Tooltip('sessions:Q', title='Sessions', format=',')
            ]
        ).properties(height=300)
        
        st.altair_chart(country_bar, use_container_width=True)
    
    # Hourly pattern heatmap
    st.subheader("⏰ Traffic by Hour of Day")
    hourly_data = df_filtered.groupby(['hour', df_filtered['timestamp'].dt.day_name()]).size().reset_index(name='sessions')
    hourly_data.columns = ['hour', 'day', 'sessions']
    
    # Define day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    heatmap = alt.Chart(hourly_data).mark_rect().encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('day:N', title='Day of Week', sort=day_order),
        color=alt.Color('sessions:Q', scale=alt.Scale(scheme='blues'), title='Sessions'),
        tooltip=[
            alt.Tooltip('day:N', title='Day'),
            alt.Tooltip('hour:O', title='Hour'),
            alt.Tooltip('sessions:Q', title='Sessions', format=',')
        ]
    ).properties(
        height=300
    )
    
    st.altair_chart(heatmap, use_container_width=True)

# ========== TAB 3: ACQUISITION ==========
with tab3:
    st.header("🔍 Traffic Acquisition")
    
    # Source performance
    st.subheader("📊 Source Performance Comparison")
    
    source_metrics = df_filtered.groupby('source').agg({
        'pageviews': 'sum',
        'duration': 'mean',
        'bounce': 'mean',
        'conversion': 'sum'
    }).reset_index()
    
    source_metrics.columns = ['source', 'pageviews', 'avg_duration', 'bounce_rate', 'conversions']
    
    # Multi-metric chart
    col1, col2 = st.columns(2)
    
    with col1:
        pageview_chart = alt.Chart(source_metrics).mark_bar().encode(
            x=alt.X('pageviews:Q', title='Total Pageviews'),
            y=alt.Y('source:N', title='Source', sort='-x'),
            color=alt.Color('source:N', legend=None),
            tooltip=[
                alt.Tooltip('source:N', title='Source'),
                alt.Tooltip('pageviews:Q', title='Pageviews', format=',')
            ]
        ).properties(
            height=350,
            title='Pageviews by Source'
        )
        
        st.altair_chart(pageview_chart, use_container_width=True)
    
    with col2:
        duration_chart = alt.Chart(source_metrics).mark_bar().encode(
            x=alt.X('avg_duration:Q', title='Avg Duration (seconds)'),
            y=alt.Y('source:N', title='Source', sort='-x'),
            color=alt.Color('avg_duration:Q', scale=alt.Scale(scheme='greens'), legend=None),
            tooltip=[
                alt.Tooltip('source:N', title='Source'),
                alt.Tooltip('avg_duration:Q', title='Avg Duration', format='.1f')
            ]
        ).properties(
            height=350,
            title='Average Session Duration'
        )
        
        st.altair_chart(duration_chart, use_container_width=True)
    
    # Source trend over time
    st.subheader("📈 Source Trends Over Time")
    source_daily = df_filtered.groupby(['date', 'source']).size().reset_index(name='sessions')
    
    trend_chart = alt.Chart(source_daily).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('sessions:Q', title='Sessions'),
        color=alt.Color('source:N', legend=alt.Legend(title='Source')),
        tooltip=[
            alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
            alt.Tooltip('source:N', title='Source'),
            alt.Tooltip('sessions:Q', title='Sessions', format=',')
        ]
    ).properties(
        height=400
    )
    
    st.altair_chart(trend_chart, use_container_width=True)

# ========== TAB 4: BEHAVIOR ==========
with tab4:
    st.header("🎯 User Behavior")
    
    # Engagement metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Pages/Session", f"{df_filtered['pageviews'].mean():.2f}")
    with col2:
        st.metric("Avg Session Duration", f"{int(df_filtered['duration'].mean())}s")
    with col3:
        st.metric("Bounce Rate", f"{df_filtered['bounce'].mean():.1%}")
    
    st.divider()
    
    # Session duration distribution
    st.subheader("⏱️ Session Duration Distribution")
    
    # Create bins for duration
    df_filtered['duration_bin'] = pd.cut(
        df_filtered['duration'], 
        bins=[0, 30, 60, 120, 300, 600, float('inf')],
        labels=['0-30s', '30-60s', '1-2min', '2-5min', '5-10min', '10min+']
    )
    
    duration_dist = df_filtered.groupby('duration_bin').size().reset_index(name='count')
    
    duration_chart = alt.Chart(duration_dist).mark_bar().encode(
        x=alt.X('duration_bin:N', title='Duration Range', sort=['0-30s', '30-60s', '1-2min', '2-5min', '5-10min', '10min+']),
        y=alt.Y('count:Q', title='Number of Sessions'),
        color=alt.Color('count:Q', scale=alt.Scale(scheme='viridis'), legend=None),
        tooltip=[
            alt.Tooltip('duration_bin:N', title='Duration'),
            alt.Tooltip('count:Q', title='Sessions', format=',')
        ]
    ).properties(
        height=350
    )
    
    st.altair_chart(duration_chart, use_container_width=True)
    
    # Pageviews distribution
    st.subheader("📄 Pages per Session")
    
    pageview_dist = df_filtered.groupby('pageviews').size().reset_index(name='count')
    
    pageview_chart = alt.Chart(pageview_dist).mark_bar().encode(
        x=alt.X('pageviews:O', title='Pages Viewed'),
        y=alt.Y('count:Q', title='Number of Sessions'),
        tooltip=[
            alt.Tooltip('pageviews:O', title='Pages'),
            alt.Tooltip('count:Q', title='Sessions', format=',')
        ]
    ).properties(
        height=350
    )
    
    st.altair_chart(pageview_chart, use_container_width=True)

# ========== TAB 5: CONVERSIONS ==========
with tab5:
    st.header("💰 Conversion Analytics")
    
    # Conversion metrics
    total_conversions = df_filtered['conversion'].sum()
    conversion_rate = df_filtered['conversion'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Conversions", f"{total_conversions:,}")
    col2.metric("Conversion Rate", f"{conversion_rate:.2%}")
    col3.metric("Value per Conversion", f"Rp {conversion_value:,}")
    
    st.divider()
    
    # Conversions by source
    st.subheader("🎯 Conversions by Traffic Source")
    
    conversion_by_source = df_filtered.groupby('source').agg({
        'conversion': ['sum', 'mean']
    }).reset_index()
    conversion_by_source.columns = ['source', 'conversions', 'conv_rate']
    
    col1, col2 = st.columns(2)
    
    with col1:
        conv_bar = alt.Chart(conversion_by_source).mark_bar().encode(
            x=alt.X('conversions:Q', title='Total Conversions'),
            y=alt.Y('source:N', title='Source', sort='-x'),
            color=alt.Color('conversions:Q', scale=alt.Scale(scheme='oranges'), legend=None),
            tooltip=[
                alt.Tooltip('source:N', title='Source'),
                alt.Tooltip('conversions:Q', title='Conversions', format=',')
            ]
        ).properties(
            height=300,
            title='Total Conversions'
        )
        
        st.altair_chart(conv_bar, use_container_width=True)
    
    with col2:
        rate_bar = alt.Chart(conversion_by_source).mark_bar().encode(
            x=alt.X('conv_rate:Q', title='Conversion Rate', axis=alt.Axis(format='%')),
            y=alt.Y('source:N', title='Source', sort='-x'),
            color=alt.Color('conv_rate:Q', scale=alt.Scale(scheme='greens'), legend=None),
            tooltip=[
                alt.Tooltip('source:N', title='Source'),
                alt.Tooltip('conv_rate:Q', title='Conv Rate', format='.2%')
            ]
        ).properties(
            height=300,
            title='Conversion Rate'
        )
        
        st.altair_chart(rate_bar, use_container_width=True)
    
    # Conversion trend
    st.subheader("📈 Conversion Trend")
    
    daily_conv = df_filtered.groupby('date').agg({
        'conversion': ['sum', 'mean']
    }).reset_index()
    daily_conv.columns = ['date', 'conversions', 'conv_rate']
    
    conv_trend = alt.Chart(daily_conv).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('conversions:Q', title='Conversions'),
        tooltip=[
            alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
            alt.Tooltip('conversions:Q', title='Conversions', format=','),
            alt.Tooltip('conv_rate:Q', title='Conv Rate', format='.2%')
        ]
    ).properties(
        height=350
    ).configure_mark(
        color='#ff7f0e'
    )
    
    st.altair_chart(conv_trend, use_container_width=True)

# ========== FOOTER ==========
st.divider()
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p><strong>Web Analytics Dashboard v1.0</strong> | Powered by Altair & Streamlit</p>
    <p>Real-time insights • Traffic analysis • Conversion tracking</p>
</div>
""", unsafe_allow_html=True)
