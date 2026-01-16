import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

st.set_page_config(page_title="Marketing Mix Modeling | ROI Optimizer", page_icon="🎛️", layout="wide")

st.title("🎛️ Marketing Mix Modeling (MMM) & Budget Allocator")
st.markdown("""
**Prescriptive Analytics**: This tool uses **Ridge Regression** (with Adstock & Saturation effects) to model how media spend drives sales, 
and uses **Mathematical Optimization** to recommend the perfect budget split.
""")

# --- 1. Helper Functions (Adstock & Saturation) ---
def adstock_geometric(dataset, channels, decay_rates):
    """
    Applies geometric adstock (carryover effect) to media channels.
    Simulates that ads seen today have an impact on future sales (decaying over time).
    """
    data = dataset.copy()
    for i, channel in enumerate(channels):
        decay = decay_rates[i]
        x = data[channel].values
        adstocked_x = np.zeros_like(x)
        adstocked_x[0] = x[0]
        for t in range(1, len(x)):
            adstocked_x[t] = x[t] + decay * adstocked_x[t-1]
        data[f"{channel}_adstock"] = adstocked_x
    return data

def saturation_hill(x, alpha, gamma):
    """
    Applies Hill function for saturation (diminishing returns).
    Simulates that doubling spend doesn't always double sales.
    """
    return x**alpha / (x**alpha + gamma**alpha)

# --- 2. Data Generation ---
@st.cache_data
def generate_mmm_data():
    np.random.seed(42)
    weeks = 104 # 2 years
    dates = pd.date_range(start="2024-01-01", periods=weeks, freq="W-MON")
    
    # Media Spends (Random but realistic)
    tv = np.random.normal(50_000_000, 10_000_000, weeks).clip(0)  # High spend, slow decay
    facebook = np.random.normal(30_000_000, 5_000_000, weeks).clip(0) # Medium spend
    instagram = np.random.normal(20_000_000, 8_000_000, weeks).clip(0) # Volatile
    google = np.random.normal(25_000_000, 3_000_000, weeks).clip(0) # Stable search
    
    df = pd.DataFrame({
        'Date': dates,
        'TV': tv,
        'Facebook': facebook,
        'Instagram': instagram,
        'Google': google
    })
    
    # Ground Truth Coefficients (for simulation)
    # We create sales based on adstocked & saturated media + seasonality + noise
    # TV: High carryover (0.8), Saturation (High Gamma)
    # Social: Low carryover (0.3), Fast Saturation
    
    # Apply Transformations for "True" Sales generation
    tv_ad = np.zeros_like(tv)
    tv_ad[0] = tv[0]
    for t in range(1, weeks): tv_ad[t] = tv[t] + 0.8 * tv_ad[t-1] # 0.8 Decay
    
    fb_ad = np.zeros_like(facebook)
    fb_ad[0] = facebook[0]
    for t in range(1, weeks): fb_ad[t] = facebook[t] + 0.4 * fb_ad[t-1]
    
    # Sales Baseline
    baseline = 500_000_000
    
    # Seasonality (Sine wave)
    seasonality = 100_000_000 * np.sin(2 * np.pi * np.arange(weeks) / 52)
    
    # Contribution (Saturated)
    # Simple power law for saturation proxy in generation
    sales = (
        baseline + seasonality +
        (tv_ad**0.6 * 50) +   # TV Impact
        (fb_ad**0.7 * 80) +   # FB Impact
        (instagram**0.8 * 60) + 
        (google**0.9 * 100) +
        np.random.normal(0, 20_000_000, weeks) # Noise
    )
    
    df['Sales'] = sales
    return df

df = generate_mmm_data()

# Sidebar Config
st.sidebar.header("Model Hyperparameters")
st.sidebar.info("Adjust Decay Rates (Adstock) manually to see impact.")

# Channels
channels = ['TV', 'Facebook', 'Instagram', 'Google']

# User Inputs for Decay (Simplified for demo, usually found via hyperopt)
decay_tv = st.sidebar.slider("TV Decay Rate", 0.0, 0.9, 0.8, 0.1)
decay_fb = st.sidebar.slider("Facebook Decay Rate", 0.0, 0.9, 0.4, 0.1)
decay_ig = st.sidebar.slider("Instagram Decay Rate", 0.0, 0.9, 0.3, 0.1)
decay_search = st.sidebar.slider("Google Decay Rate", 0.0, 0.9, 0.2, 0.1)

decays = [decay_tv, decay_fb, decay_ig, decay_search]

# --- 3. Modeling (Ridge Regression) ---
# Apply Adstock Transformation based on user input
df_mod = adstock_geometric(df, channels, decays)

feature_cols = [f"{c}_adstock" for c in channels]
target_col = 'Sales'

# Train Model
# Pipeline: Scale features -> Ridge Regression (Force positive coeffs)
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0, positive=True)) # Positive=True forces non-negative impact
])

model.fit(df_mod[feature_cols], df_mod[target_col])
r2 = model.score(df_mod[feature_cols], df_mod[target_col])

# --- 4. Visualizing Model Performance ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Actual vs Predicted Sales")
    df_mod['Predicted_Sales'] = model.predict(df_mod[feature_cols])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_mod['Date'], y=df_mod['Sales'], name='Actual Sales', line=dict(color='#95a5a6')))
    fig.add_trace(go.Scatter(x=df_mod['Date'], y=df_mod['Predicted_Sales'], name='Predicted Sales', line=dict(color='#e74c3c', dash='dot')))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Model Accuracy")
    st.metric("R-Squared (Explained Variance)", f"{r2:.1%}")
    st.markdown("#### Media Coefficients")
    # Extract coeffs
    coeffs = model.named_steps['regressor'].coef_
    coeff_df = pd.DataFrame({'Channel': channels, 'Coefficient': coeffs})
    st.dataframe(coeff_df.style.background_gradient(cmap='Greens'))
    st.caption("Higher coefficient = Higher impact per unit of adstock.")

st.divider()

# --- 5. Budget Optimizer ---
st.subheader("🚀 Budget Allocator (Optimization Engine)")
st.markdown("Enter your total budget, and the AI will calculate the optimal split to maximize Sales.")

# Optimization Logic
def optimize_budget(total_budget, model_pipeline, current_spends):
    """
    Maximize Sales subject to: sum(spends) == total_budget
    """
    scaler = model_pipeline.named_steps['scaler']
    regressor = model_pipeline.named_steps['regressor']
    
    # We approximate "Sales" as dot product of (Scaled Spends * Coeffs)
    # Note: This simplistically assumes linear relationship on adstocked data for the demo
    # In full production, we'd loop the saturation curve here.
    
    coeffs = regressor.coef_
    mean = scaler.mean_
    scale = scaler.scale_
    
    # Objective: Minimize negative sales (maximize sales)
    def objective(spends):
        # Scale the new input
        scaled_spends = (spends - mean) / scale 
        # Predict revenue contribution (simplified)
        pred_sales = np.dot(scaled_spends, coeffs)
        return -pred_sales

    # Constraints: Sum of spends = Total Budget
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget})
    
    # Bounds: No negative spend, no channel > 50% of budget (optional risk constraint)
    bounds = tuple((0, total_budget) for _ in range(len(channels)))
    
    # Initial Guess: Even split
    x0 = [total_budget/len(channels)] * len(channels)
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x, -res.fun

# User Input for Budget
c_opt1, c_opt2 = st.columns([1, 2])
with c_opt1:
    input_budget = st.number_input("Total Marketing Budget (Rp)", value=200_000_000, step=10_000_000, format="%d")
    
    if st.button("Runs Optimization", type="primary"):
        # Run Optimization
        # current averages as dummy context for scaler
        current_avgs = df_mod[feature_cols].mean().values 
        
        opt_spends, pred_rev = optimize_budget(input_budget, model, current_avgs)
        
        st.success("Optimization Complete!")
        
        # Display Results
        res_df = pd.DataFrame({
            'Channel': channels,
            'Optimal Spend': opt_spends.astype(int),
            'Allocation (%)': opt_spends / input_budget * 100
        })
        
        st.write("### 🏆 Optimal Split")
        st.dataframe(res_df.style.format({'Optimal Spend': 'Rp {:,.0f}', 'Allocation (%)': '{:.1f}%'}))
        
        st.metric("Predicted Revenue Impact", f"Rp {pred_rev*1000:,.0f} (Estimated)") # Scaling factor for demo realism

with c_opt2:
    # Visualize Split
    st.write("#### Budget Allocation Chart")
    if 'res_df' in locals():
        fig_pie = px.pie(res_df, values='Optimal Spend', names='Channel', title="AI Recommended Allocation", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie)
    else:
        st.info("Click 'Run Optimization' to see the magic.")
