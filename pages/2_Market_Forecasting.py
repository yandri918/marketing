import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_sales_data

st.set_page_config(page_title="Demand Forecasting | Marketing Analytics", page_icon="📈", layout="wide")

st.title("📈 Market Demand Forecasting")
st.markdown("Predict future sales trends to optimize inventory and marketing spend.")

# Sidebar Config
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 90, 60)
seasonality_mode = st.sidebar.radio("Seasonality Mode", ["Additive", "Multiplicative"])

# Load Data
df = generate_sales_data(days=365*2)
df.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)

st.subheader("Historical Sales Data")
fig_hist = px.line(df, x='ds', y='y', title='Past 2 Years Sales History', template="plotly_white")
st.plotly_chart(fig_hist, use_container_width=True)

# Forecasting Logic (Using Prophet if available, else simple simulation for demo stability)
st.divider()
st.subheader("🔮 Sales Forecast")

try:
    from prophet import Prophet
    
    with st.spinner('Training Prophet Model...'):
        m = Prophet(seasonality_mode=seasonality_mode.lower())
        m.fit(df)
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        
        # Plotting with Plotly
        fig_forecast = go.Figure()
        
        # Actual Data
        fig_forecast.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual Sales', line=dict(color='gray', width=1)))
        
        # Forecast
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Sales', line=dict(color='blue', width=2)))
        
        # Confidence Interval
        fig_forecast.add_trace(go.Scatter(
            x=list(forecast['ds']) + list(forecast['ds'])[::-1],
            y=list(forecast['yhat_upper']) + list(forecast['yhat_lower'])[::-1],
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig_forecast.update_layout(title="Sales Forecast with Confidence Intervals", template="plotly_white")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.success("✅ Model Trained Successfully using Facebook Prophet")

except ImportError:
    st.warning("⚠️ 'Prophet' library not found. Showing simulated forecast for demonstration.")
    # Simulated Forecast
    last_date = df['ds'].iloc[-1]
    last_val = df['y'].iloc[-1]
    
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days+1)]
    future_vals = [last_val + (i * 2) + np.random.normal(0, 50) for i in range(1, forecast_days+1)]
    
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': future_vals})
    
    fig_sim = px.line(forecast_df, x='ds', y='yhat', title='Projected Sales (Simulated)', template="plotly_white")
    fig_sim.add_scatter(x=df['ds'], y=df['y'], mode='lines', name='History', line=dict(color='gray'))
    st.plotly_chart(fig_sim, use_container_width=True)

# Sceario Analysis
st.divider()
st.subheader("⚡ Scenario Planning")

c1, c2, c3 = st.columns(3)
with c1:
    promo_lift = st.slider("Promotional Lift (%)", 0, 50, 10, help="Expected sales increase from marketing campaign")
with c2:
    price_impact = st.slider("Price Increase Impact (%)", -30, 0, -10, help="Expected drop due to price hike")
with c3:
    market_growth = st.slider("Organic Market Growth (%)", -10, 20, 5)

net_impact = 1 + (promo_lift + price_impact + market_growth) / 100

st.metric("Net Predicted Impact", f"{net_impact:.2%}", delta=f"{(promo_lift + price_impact + market_growth)}%")

if 'forecast' in locals():
    adjusted_forecast = forecast[forecast['ds'] > df['ds'].iloc[-1]].copy()
    adjusted_forecast['adjusted_yhat'] = adjusted_forecast['yhat'] * net_impact
    
    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Scatter(x=adjusted_forecast['ds'], y=adjusted_forecast['yhat'], name='Baseline', line=dict(dash='dot')))
    fig_scenario.add_trace(go.Scatter(x=adjusted_forecast['ds'], y=adjusted_forecast['adjusted_yhat'], name='Scenario Adjusted', line=dict(color='green', width=3)))
    
    fig_scenario.update_layout(title="Scenario Impact Analysis", template="plotly_white")
    st.plotly_chart(fig_scenario, use_container_width=True)
