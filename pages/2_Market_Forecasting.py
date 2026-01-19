import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import scipy.stats as stats
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_sales_data

st.set_page_config(page_title="Advanced Demand Forecasting | Marketing Analytics", page_icon="📈", layout="wide")

st.title("📈 Advanced Market Demand Forecasting")
st.markdown("Enterprise-grade time series analytics with **multiple algorithms**, **seasonality analysis**, and **inventory optimization**.")

# ========== HELPER FUNCTIONS ==========

def calculate_accuracy_metrics(actual, predicted):
    """Calculate forecast accuracy metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }

def moving_average_forecast(data, window=7, periods=30):
    """Simple moving average forecast"""
    ma = data.rolling(window=window).mean()
    last_ma = ma.iloc[-1]
    
    forecast = [last_ma] * periods
    return np.array(forecast)

def exponential_smoothing_forecast(data, periods=30, seasonal_periods=7):
    """Exponential Smoothing (Holt-Winters) forecast"""
    try:
        model = ExponentialSmoothing(
            data,
            seasonal='add',
            seasonal_periods=seasonal_periods,
            trend='add'
        )
        fitted = model.fit()
        forecast = fitted.forecast(periods)
        return forecast.values
    except:
        # Fallback to simple exponential smoothing
        alpha = 0.3
        forecast = [data.iloc[-1]]
        for _ in range(periods - 1):
            forecast.append(alpha * forecast[-1] + (1 - alpha) * data.iloc[-1])
        return np.array(forecast)

def linear_trend_forecast(data, periods=30):
    """Linear regression trend forecast"""
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values
    
    # Simple linear regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast
    future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    return forecast

def calculate_inventory_metrics(avg_demand, std_demand, lead_time_days, service_level=0.95):
    """Calculate inventory optimization metrics"""
    # Z-score for service level
    z_score = stats.norm.ppf(service_level)
    
    # Safety Stock
    safety_stock = z_score * std_demand * np.sqrt(lead_time_days)
    
    # Reorder Point
    reorder_point = (avg_demand * lead_time_days) + safety_stock
    
    # Economic Order Quantity (EOQ) - simplified
    annual_demand = avg_demand * 365
    order_cost = 100  # Assumed
    holding_cost_per_unit = 5  # Assumed
    
    eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
    
    return {
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'eoq': eoq,
        'z_score': z_score
    }

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Forecast Configuration")

forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 180, 60)
algorithm = st.sidebar.selectbox(
    "Primary Algorithm",
    ["Prophet", "Exponential Smoothing", "Moving Average", "Linear Trend", "Ensemble"]
)

st.sidebar.divider()

st.sidebar.subheader("Inventory Settings")
lead_time = st.sidebar.slider("Lead Time (Days)", 1, 30, 7)
service_level = st.sidebar.slider("Service Level (%)", 80, 99, 95) / 100

st.sidebar.divider()

# ========== DATA LOADING ==========
df = generate_sales_data(days=365*2)
df.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

# Calculate basic statistics
avg_daily_demand = df['y'].mean()
std_daily_demand = df['y'].std()

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔮 Forecasting",
    "📈 Seasonality",
    "📦 Inventory",
    "🎯 Demand Drivers",
    "🎲 Scenarios"
])

# ========== TAB 1: OVERVIEW ==========
with tab1:
    st.subheader("📊 Historical Sales Overview")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg Daily Sales", f"{avg_daily_demand:,.0f}")
    col2.metric("Total Sales (2Y)", f"{df['y'].sum():,.0f}")
    col3.metric("Peak Sales", f"{df['y'].max():,.0f}")
    col4.metric("Min Sales", f"{df['y'].min():,.0f}")
    col5.metric("Std Deviation", f"{std_daily_demand:,.0f}")
    
    st.divider()
    
    # Historical Sales Chart
    st.markdown("### Sales History (Past 2 Years)")
    
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Daily Sales',
        line=dict(color='#3498DB', width=2)
    ))
    
    # Add moving average
    df['MA_7'] = df['y'].rolling(window=7).mean()
    df['MA_30'] = df['y'].rolling(window=30).mean()
    
    fig_hist.add_trace(go.Scatter(
        x=df['ds'],
        y=df['MA_7'],
        mode='lines',
        name='7-Day MA',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    fig_hist.add_trace(go.Scatter(
        x=df['ds'],
        y=df['MA_30'],
        mode='lines',
        name='30-Day MA',
        line=dict(color='red', width=1, dash='dot')
    ))
    
    fig_hist.update_layout(
        title="Historical Sales with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Sales Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sales Distribution")
        
        fig_dist = px.histogram(
            df,
            x='y',
            nbins=50,
            title="Sales Frequency Distribution",
            labels={'y': 'Sales'},
            color_discrete_sequence=['#2ECC71']
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown("### Monthly Sales Trend")
        
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly['ds'].dt.to_period('M')
        monthly_sales = df_monthly.groupby('Month')['y'].sum().reset_index()
        monthly_sales['Month'] = monthly_sales['Month'].astype(str)
        
        fig_monthly = px.bar(
            monthly_sales,
            x='Month',
            y='y',
            title="Monthly Total Sales",
            labels={'y': 'Sales'},
            color='y',
            color_continuous_scale='Blues'
        )
        fig_monthly.update_xaxes(tickangle=45)
        st.plotly_chart(fig_monthly, use_container_width=True)

# ========== TAB 2: FORECASTING ==========
with tab2:
    st.subheader("🔮 Multi-Algorithm Forecasting")
    
    # Prepare forecasts
    forecasts = {}
    
    # 1. Moving Average
    ma_forecast = moving_average_forecast(df['y'], window=7, periods=forecast_days)
    
    # 2. Exponential Smoothing
    exp_forecast = exponential_smoothing_forecast(df['y'], periods=forecast_days)
    
    # 3. Linear Trend
    linear_forecast = linear_trend_forecast(df['y'], periods=forecast_days)
    
    # 4. Prophet (if available)
    try:
        from prophet import Prophet
        
        df_prophet = df[['ds', 'y']].copy()
        m = Prophet(seasonality_mode='additive')
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=forecast_days)
        prophet_forecast_full = m.predict(future)
        prophet_forecast = prophet_forecast_full[prophet_forecast_full['ds'] > df['ds'].iloc[-1]]['yhat'].values
        
        forecasts['Prophet'] = prophet_forecast
        prophet_available = True
    except:
        prophet_available = False
        st.warning("⚠️ Prophet library not available. Using alternative algorithms.")
    
    forecasts['Moving Average'] = ma_forecast
    forecasts['Exponential Smoothing'] = exp_forecast
    forecasts['Linear Trend'] = linear_forecast
    
    # Ensemble (average of all)
    ensemble_forecast = np.mean([forecasts[k] for k in forecasts.keys()], axis=0)
    forecasts['Ensemble'] = ensemble_forecast
    
    # Create future dates
    last_date = df['ds'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    # Visualization
    st.markdown("### Forecast Comparison")
    
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Historical',
        line=dict(color='gray', width=1)
    ))
    
    # Forecasts
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
    for i, (name, forecast) in enumerate(forecasts.items()):
        fig_forecast.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2, dash='dash' if name != algorithm else 'solid')
        ))
    
    fig_forecast.update_layout(
        title="Multi-Algorithm Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Accuracy Metrics (on historical data - last 30 days)
    st.markdown("### Forecast Accuracy (Last 30 Days Validation)")
    
    # Split data for validation
    train_data = df.iloc[:-30]
    test_data = df.iloc[-30:]
    
    accuracy_results = []
    
    for name in ['Moving Average', 'Exponential Smoothing', 'Linear Trend']:
        if name == 'Moving Average':
            pred = moving_average_forecast(train_data['y'], window=7, periods=30)
        elif name == 'Exponential Smoothing':
            pred = exponential_smoothing_forecast(train_data['y'], periods=30)
        else:
            pred = linear_trend_forecast(train_data['y'], periods=30)
        
        metrics = calculate_accuracy_metrics(test_data['y'].values, pred)
        metrics['Algorithm'] = name
        accuracy_results.append(metrics)
    
    if prophet_available:
        # Prophet validation
        m_val = Prophet()
        m_val.fit(train_data[['ds', 'y']])
        future_val = m_val.make_future_dataframe(periods=30)
        forecast_val = m_val.predict(future_val)
        prophet_pred = forecast_val[forecast_val['ds'] > train_data['ds'].iloc[-1]]['yhat'].values[:30]
        
        metrics = calculate_accuracy_metrics(test_data['y'].values, prophet_pred)
        metrics['Algorithm'] = 'Prophet'
        accuracy_results.append(metrics)
    
    accuracy_df = pd.DataFrame(accuracy_results)
    
    # Display metrics
    st.dataframe(accuracy_df.style.format({
        'MAE': '{:.2f}',
        'RMSE': '{:.2f}',
        'MAPE': '{:.2f}%',
        'R²': '{:.4f}'
    }).background_gradient(subset=['MAE', 'RMSE', 'MAPE'], cmap='RdYlGn_r'), use_container_width=True)
    
    st.info("""
    **Metrics Explanation:**
    - **MAE** (Mean Absolute Error): Average absolute difference. Lower is better.
    - **RMSE** (Root Mean Squared Error): Penalizes large errors. Lower is better.
    - **MAPE** (Mean Absolute Percentage Error): Error as percentage. Lower is better.
    - **R²** (R-squared): Goodness of fit. Higher is better (max 1.0).
    """)
    
    # Forecast Summary Table
    st.markdown("### Forecast Summary (Next 30 Days)")
    
    summary_data = []
    for name, forecast in forecasts.items():
        summary_data.append({
            'Algorithm': name,
            'Avg Forecast': forecast[:30].mean(),
            'Total Forecast': forecast[:30].sum(),
            'Min': forecast[:30].min(),
            'Max': forecast[:30].max()
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    st.dataframe(summary_df.style.format({
        'Avg Forecast': '{:,.0f}',
        'Total Forecast': '{:,.0f}',
        'Min': '{:,.0f}',
        'Max': '{:,.0f}'
    }), use_container_width=True)

# ========== TAB 3: SEASONALITY ==========
with tab3:
    st.subheader("📈 Seasonality & Trend Analysis")
    
    # Seasonal Decomposition
    st.markdown("### Seasonal Decomposition (STL)")
    
    try:
        # Perform decomposition
        decomposition = seasonal_decompose(df['y'], model='additive', period=7)
        
        # Create subplots
        fig_decomp = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        # Original
        fig_decomp.add_trace(
            go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Original', line=dict(color='#3498DB')),
            row=1, col=1
        )
        
        # Trend
        fig_decomp.add_trace(
            go.Scatter(x=df['ds'], y=decomposition.trend, mode='lines', name='Trend', line=dict(color='#E74C3C')),
            row=2, col=1
        )
        
        # Seasonal
        fig_decomp.add_trace(
            go.Scatter(x=df['ds'], y=decomposition.seasonal, mode='lines', name='Seasonal', line=dict(color='#2ECC71')),
            row=3, col=1
        )
        
        # Residual
        fig_decomp.add_trace(
            go.Scatter(x=df['ds'], y=decomposition.resid, mode='lines', name='Residual', line=dict(color='#F39C12')),
            row=4, col=1
        )
        
        fig_decomp.update_layout(height=800, showlegend=False, template="plotly_white")
        st.plotly_chart(fig_decomp, use_container_width=True)
        
        # Seasonality Strength
        col1, col2, col3 = st.columns(3)
        
        trend_strength = 1 - (decomposition.resid.var() / (decomposition.trend + decomposition.resid).var())
        seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var())
        
        col1.metric("Trend Strength", f"{trend_strength:.2%}")
        col2.metric("Seasonal Strength", f"{seasonal_strength:.2%}")
        col3.metric("Residual Variance", f"{decomposition.resid.var():,.0f}")
        
    except Exception as e:
        st.error(f"Could not perform decomposition: {e}")
    
    st.divider()
    
    # Day of Week Analysis
    st.markdown("### Day of Week Patterns")
    
    df_dow = df.copy()
    df_dow['DayOfWeek'] = df_dow['ds'].dt.day_name()
    dow_avg = df_dow.groupby('DayOfWeek')['y'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    fig_dow = px.bar(
        x=dow_avg.index,
        y=dow_avg.values,
        title="Average Sales by Day of Week",
        labels={'x': 'Day', 'y': 'Avg Sales'},
        color=dow_avg.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_dow, use_container_width=True)
    
    # Month of Year Analysis
    st.markdown("### Monthly Patterns")
    
    df_month = df.copy()
    df_month['Month'] = df_month['ds'].dt.month_name()
    month_avg = df_month.groupby('Month')['y'].mean().reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    
    fig_month = px.line(
        x=month_avg.index,
        y=month_avg.values,
        title="Average Sales by Month",
        labels={'x': 'Month', 'y': 'Avg Sales'},
        markers=True
    )
    st.plotly_chart(fig_month, use_container_width=True)

# ========== TAB 4: INVENTORY ==========
with tab4:
    st.subheader("📦 Inventory Optimization")
    
    # Calculate inventory metrics
    inventory_metrics = calculate_inventory_metrics(
        avg_daily_demand,
        std_daily_demand,
        lead_time,
        service_level
    )
    
    # Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Safety Stock", f"{inventory_metrics['safety_stock']:,.0f} units")
    col2.metric("Reorder Point", f"{inventory_metrics['reorder_point']:,.0f} units")
    col3.metric("Economic Order Qty", f"{inventory_metrics['eoq']:,.0f} units")
    col4.metric("Service Level", f"{service_level:.0%}")
    
    st.divider()
    
    # Inventory Visualization
    st.markdown("### Inventory Level Simulation")
    
    # Simulate inventory levels
    days_sim = 90
    inventory_levels = []
    current_inventory = inventory_metrics['reorder_point'] + inventory_metrics['eoq']
    
    for day in range(days_sim):
        # Daily demand (with some randomness)
        daily_demand = np.random.normal(avg_daily_demand, std_daily_demand)
        current_inventory -= daily_demand
        
        # Reorder if below reorder point
        if current_inventory < inventory_metrics['reorder_point']:
            current_inventory += inventory_metrics['eoq']
        
        inventory_levels.append(current_inventory)
    
    fig_inventory = go.Figure()
    
    fig_inventory.add_trace(go.Scatter(
        x=list(range(days_sim)),
        y=inventory_levels,
        mode='lines',
        name='Inventory Level',
        line=dict(color='#3498DB', width=2)
    ))
    
    # Add reorder point line
    fig_inventory.add_hline(
        y=inventory_metrics['reorder_point'],
        line_dash="dash",
        line_color="red",
        annotation_text="Reorder Point"
    )
    
    # Add safety stock line
    fig_inventory.add_hline(
        y=inventory_metrics['safety_stock'],
        line_dash="dot",
        line_color="orange",
        annotation_text="Safety Stock"
    )
    
    fig_inventory.update_layout(
        title="Inventory Level Simulation (90 Days)",
        xaxis_title="Day",
        yaxis_title="Inventory (units)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_inventory, use_container_width=True)
    
    # Formulas Explanation
    with st.expander("📐 Inventory Formulas Explained"):
        st.markdown(f"""
        **Safety Stock Formula:**
        ```
        Safety Stock = Z-score × Std Dev × √Lead Time
                    = {inventory_metrics['z_score']:.2f} × {std_daily_demand:.0f} × √{lead_time}
                    = {inventory_metrics['safety_stock']:.0f} units
        ```
        
        **Reorder Point Formula:**
        ```
        Reorder Point = (Avg Daily Demand × Lead Time) + Safety Stock
                      = ({avg_daily_demand:.0f} × {lead_time}) + {inventory_metrics['safety_stock']:.0f}
                      = {inventory_metrics['reorder_point']:.0f} units
        ```
        
        **Economic Order Quantity (EOQ):**
        ```
        EOQ = √(2 × Annual Demand × Order Cost / Holding Cost)
            = {inventory_metrics['eoq']:.0f} units
        ```
        
        **Service Level:** {service_level:.0%} (Z-score: {inventory_metrics['z_score']:.2f})
        """)
    
    # Stock-out Risk
    st.markdown("### Stock-out Risk Analysis")
    
    # Calculate probability of stock-out
    stockout_prob = 1 - service_level
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Stock-out Probability", f"{stockout_prob:.2%}", delta="Risk Level", delta_color="inverse")
        st.metric("Expected Stock-outs/Year", f"{stockout_prob * 365:.1f} days")
    
    with col2:
        # Risk assessment
        if stockout_prob < 0.01:
            st.success("🟢 **Low Risk** - Excellent inventory management")
        elif stockout_prob < 0.05:
            st.info("🟡 **Moderate Risk** - Acceptable stock-out risk")
        else:
            st.warning("🔴 **High Risk** - Consider increasing safety stock")

# ========== TAB 5: DEMAND DRIVERS ==========
with tab5:
    st.subheader("🎯 Demand Drivers Analysis")
    
    # Add synthetic external factors
    df_drivers = df.copy()
    df_drivers['DayOfWeek'] = df_drivers['ds'].dt.dayofweek
    df_drivers['Month'] = df_drivers['ds'].dt.month
    df_drivers['IsWeekend'] = (df_drivers['DayOfWeek'] >= 5).astype(int)
    
    # Synthetic promotional data
    np.random.seed(42)
    df_drivers['Promotion'] = np.random.choice([0, 1], size=len(df_drivers), p=[0.8, 0.2])
    df_drivers['Price_Index'] = 100 + np.random.normal(0, 10, len(df_drivers))
    
    # Correlation Analysis
    st.markdown("### Correlation with External Factors")
    
    corr_data = df_drivers[['y', 'DayOfWeek', 'Month', 'IsWeekend', 'Promotion', 'Price_Index']].corr()
    
    fig_corr = px.imshow(
        corr_data,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Correlation Matrix"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Promotional Impact
    st.markdown("### Promotional Impact Analysis")
    
    promo_impact = df_drivers.groupby('Promotion')['y'].mean()
    promo_lift = ((promo_impact[1] - promo_impact[0]) / promo_impact[0]) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Sales (No Promo)", f"{promo_impact[0]:,.0f}")
    col2.metric("Avg Sales (With Promo)", f"{promo_impact[1]:,.0f}")
    col3.metric("Promotional Lift", f"{promo_lift:+.1f}%", delta="Increase")
    
    fig_promo = px.box(
        df_drivers,
        x='Promotion',
        y='y',
        title="Sales Distribution: Promotional vs Non-Promotional Days",
        labels={'Promotion': 'Promotion Active', 'y': 'Sales'},
        color='Promotion',
        color_discrete_map={0: '#E74C3C', 1: '#2ECC71'}
    )
    st.plotly_chart(fig_promo, use_container_width=True)
    
    # Weekend Effect
    st.markdown("### Weekend Effect")
    
    weekend_impact = df_drivers.groupby('IsWeekend')['y'].mean()
    weekend_lift = ((weekend_impact[1] - weekend_impact[0]) / weekend_impact[0]) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Sales (Weekday)", f"{weekend_impact[0]:,.0f}")
    col2.metric("Avg Sales (Weekend)", f"{weekend_impact[1]:,.0f}")
    col3.metric("Weekend Lift", f"{weekend_lift:+.1f}%")

# ========== TAB 6: SCENARIOS ==========
with tab6:
    st.subheader("🎲 Advanced Scenario Planning")
    
    # Scenario Inputs
    st.markdown("### What-If Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        promo_lift_pct = st.slider("Promotional Lift (%)", 0, 50, 15)
    with col2:
        price_impact_pct = st.slider("Price Change Impact (%)", -30, 30, -10)
    with col3:
        market_growth_pct = st.slider("Market Growth (%)", -20, 30, 5)
    
    # Calculate scenarios
    base_forecast = forecasts.get(algorithm, ensemble_forecast)
    
    # Best case
    best_case = base_forecast * (1 + (promo_lift_pct + abs(price_impact_pct) + market_growth_pct) / 100)
    
    # Worst case
    worst_case = base_forecast * (1 - (promo_lift_pct + abs(price_impact_pct) + market_growth_pct) / 100)
    
    # Adjusted case
    net_impact = 1 + (promo_lift_pct + price_impact_pct + market_growth_pct) / 100
    adjusted_forecast = base_forecast * net_impact
    
    # Visualization
    st.markdown("### Scenario Comparison")
    
    fig_scenario = go.Figure()
    
    # Historical
    fig_scenario.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Historical',
        line=dict(color='gray', width=1)
    ))
    
    # Base forecast
    fig_scenario.add_trace(go.Scatter(
        x=future_dates,
        y=base_forecast,
        mode='lines',
        name='Base Case',
        line=dict(color='#3498DB', width=2, dash='dash')
    ))
    
    # Best case
    fig_scenario.add_trace(go.Scatter(
        x=future_dates,
        y=best_case,
        mode='lines',
        name='Best Case',
        line=dict(color='#2ECC71', width=2, dash='dot')
    ))
    
    # Worst case
    fig_scenario.add_trace(go.Scatter(
        x=future_dates,
        y=worst_case,
        mode='lines',
        name='Worst Case',
        line=dict(color='#E74C3C', width=2, dash='dot')
    ))
    
    # Adjusted
    fig_scenario.add_trace(go.Scatter(
        x=future_dates,
        y=adjusted_forecast,
        mode='lines',
        name='Adjusted Forecast',
        line=dict(color='#F39C12', width=3)
    ))
    
    fig_scenario.update_layout(
        title="Multi-Scenario Forecast Analysis",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig_scenario, use_container_width=True)
    
    # Scenario Summary
    st.markdown("### Scenario Summary (Next 30 Days)")
    
    scenario_summary = pd.DataFrame({
        'Scenario': ['Best Case', 'Base Case', 'Adjusted', 'Worst Case'],
        'Total Sales': [
            best_case[:30].sum(),
            base_forecast[:30].sum(),
            adjusted_forecast[:30].sum(),
            worst_case[:30].sum()
        ],
        'Avg Daily': [
            best_case[:30].mean(),
            base_forecast[:30].mean(),
            adjusted_forecast[:30].mean(),
            worst_case[:30].mean()
        ],
        'vs Base': [
            ((best_case[:30].sum() - base_forecast[:30].sum()) / base_forecast[:30].sum()) * 100,
            0,
            ((adjusted_forecast[:30].sum() - base_forecast[:30].sum()) / base_forecast[:30].sum()) * 100,
            ((worst_case[:30].sum() - base_forecast[:30].sum()) / base_forecast[:30].sum()) * 100
        ]
    })
    
    st.dataframe(scenario_summary.style.format({
        'Total Sales': '{:,.0f}',
        'Avg Daily': '{:,.0f}',
        'vs Base': '{:+.1f}%'
    }).background_gradient(subset=['vs Base'], cmap='RdYlGn'), use_container_width=True)
    
    # Monte Carlo Simulation
    st.markdown("### Monte Carlo Simulation")
    
    n_simulations = st.slider("Number of Simulations", 100, 1000, 500)
    
    if st.button("🎲 Run Monte Carlo Simulation"):
        with st.spinner("Running simulations..."):
            simulations = []
            
            for _ in range(n_simulations):
                # Random factors
                random_promo = np.random.uniform(0, promo_lift_pct)
                random_price = np.random.uniform(price_impact_pct, 0)
                random_growth = np.random.uniform(-5, market_growth_pct)
                
                random_impact = 1 + (random_promo + random_price + random_growth) / 100
                sim_forecast = base_forecast * random_impact
                
                simulations.append(sim_forecast[:30].sum())
            
            # Results
            simulations = np.array(simulations)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Forecast", f"{simulations.mean():,.0f}")
            col2.metric("Median Forecast", f"{np.median(simulations):,.0f}")
            col3.metric("Std Deviation", f"{simulations.std():,.0f}")
            col4.metric("95% Confidence", f"{np.percentile(simulations, 95):,.0f}")
            
            # Distribution
            fig_mc = px.histogram(
                x=simulations,
                nbins=50,
                title="Monte Carlo Simulation Results (30-Day Total Sales)",
                labels={'x': 'Total Sales'},
                color_discrete_sequence=['#9B59B6']
            )
            
            # Add percentile lines
            fig_mc.add_vline(x=np.percentile(simulations, 5), line_dash="dash", line_color="red", annotation_text="5th %ile")
            fig_mc.add_vline(x=np.percentile(simulations, 50), line_dash="solid", line_color="green", annotation_text="Median")
            fig_mc.add_vline(x=np.percentile(simulations, 95), line_dash="dash", line_color="red", annotation_text="95th %ile")
            
            st.plotly_chart(fig_mc, use_container_width=True)
            
            st.success(f"✅ Completed {n_simulations} simulations")

# ========== FOOTER ==========
st.divider()
st.caption("💡 **Pro Tip:** Use multiple algorithms and compare their accuracy to choose the best forecast for your business!")
