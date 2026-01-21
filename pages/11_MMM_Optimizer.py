import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import our advanced MMM utilities
from utils import (
    # Adstock
    apply_adstock_to_dataframe,
    get_adstock_curve,
    # Saturation
    apply_saturation_to_dataframe,
    fit_hill_saturation,
    calculate_optimal_spend,
    # Optimization
    single_objective_optimizer,
    multi_objective_optimizer,
    scenario_analysis,
    sensitivity_analysis,
    # Decomposition
    decompose_sales,
    calculate_channel_roas,
    calculate_incremental_roas,
    create_waterfall_data,
    calculate_marginal_roas,
    # Visualization
    plot_adstock_curves,
    plot_saturation_curves,
    plot_contribution_waterfall,
    plot_actual_vs_predicted,
    plot_channel_contributions_over_time,
    plot_pareto_frontier,
    plot_tornado_chart,
    plot_coefficient_importance
)

st.set_page_config(
    page_title="Advanced Marketing Mix Modeling | AgriSensa",
    page_icon="🎛️",
    layout="wide"
)

# ========== HEADER ==========
st.title("🎛️ Advanced Marketing Mix Modeling (MMM) & Budget Allocator")
st.markdown("""
**Enterprise-Grade Prescriptive Analytics** powered by:
- 🔬 **Advanced Adstock** (Geometric, Weibull, Delayed, Carryover)
- 📈 **Saturation Modeling** (Hill, Logistic, Michaelis-Menten)
- 🎯 **Multi-Objective Optimization** (Pareto Frontier Analysis)
- 💡 **Contribution Decomposition** (Waterfall Charts, ROAS, iROAS)
- 🔮 **Scenario Planning** (What-If Analysis, Sensitivity Testing)
""")

# ========== DATA GENERATION ==========
@st.cache_data
def generate_mmm_data():
    """Generate realistic synthetic MMM data"""
    np.random.seed(42)
    weeks = 104  # 2 years
    dates = pd.date_range(start="2024-01-01", periods=weeks, freq="W-MON")
    
    # Media Spends (Random but realistic)
    tv = np.random.normal(50_000_000, 10_000_000, weeks).clip(0)
    facebook = np.random.normal(30_000_000, 5_000_000, weeks).clip(0)
    instagram = np.random.normal(20_000_000, 8_000_000, weeks).clip(0)
    google = np.random.normal(25_000_000, 3_000_000, weeks).clip(0)
    
    df = pd.DataFrame({
        'Date': dates,
        'TV': tv,
        'Facebook': facebook,
        'Instagram': instagram,
        'Google': google
    })
    
    # Generate "true" sales with adstock + saturation + seasonality
    # TV: High carryover (0.8)
    tv_ad = np.zeros_like(tv)
    tv_ad[0] = tv[0]
    for t in range(1, weeks):
        tv_ad[t] = tv[t] + 0.8 * tv_ad[t-1]
    
    # Facebook: Medium carryover (0.4)
    fb_ad = np.zeros_like(facebook)
    fb_ad[0] = facebook[0]
    for t in range(1, weeks):
        fb_ad[t] = facebook[t] + 0.4 * fb_ad[t-1]
    
    # Baseline + Seasonality
    baseline = 500_000_000
    seasonality = 100_000_000 * np.sin(2 * np.pi * np.arange(weeks) / 52)
    
    # Sales (with saturation via power law)
    sales = (
        baseline + seasonality +
        (tv_ad**0.6 * 50) +
        (fb_ad**0.7 * 80) +
        (instagram**0.8 * 60) +
        (google**0.9 * 100) +
        np.random.normal(0, 20_000_000, weeks)
    )
    
    df['Sales'] = sales
    return df

df = generate_mmm_data()

# ========== SIDEBAR CONFIGURATION ==========
st.sidebar.header("⚙️ Model Configuration")

# Model Type Selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Ridge Regression (Current)", "Bayesian MMM (Coming Soon)", "Hierarchical MMM (Coming Soon)"],
    help="Select the modeling approach"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Adstock Configuration")

channels = ['TV', 'Facebook', 'Instagram', 'Google']

# Adstock type selection
adstock_type = st.sidebar.selectbox(
    "Adstock Type",
    ["geometric", "weibull", "delayed", "carryover"],
    help="Type of carryover effect modeling"
)

# Adstock parameters (simplified - per channel in advanced mode)
if adstock_type == "geometric":
    decay_tv = st.sidebar.slider("TV Decay Rate", 0.0, 0.9, 0.8, 0.1)
    decay_fb = st.sidebar.slider("Facebook Decay Rate", 0.0, 0.9, 0.4, 0.1)
    decay_ig = st.sidebar.slider("Instagram Decay Rate", 0.0, 0.9, 0.3, 0.1)
    decay_google = st.sidebar.slider("Google Decay Rate", 0.0, 0.9, 0.2, 0.1)
    
    adstock_params = {
        'TV': {'decay': decay_tv},
        'Facebook': {'decay': decay_fb},
        'Instagram': {'decay': decay_ig},
        'Google': {'decay': decay_google}
    }
elif adstock_type == "weibull":
    st.sidebar.info("Weibull allows delayed peak effects (e.g., TV campaigns)")
    shape = st.sidebar.slider("Shape Parameter", 0.5, 3.0, 1.5, 0.1)
    scale = st.sidebar.slider("Scale Parameter", 1.0, 10.0, 3.0, 0.5)
    
    adstock_params = {ch: {'shape': shape, 'scale': scale, 'peak_delay': 0} for ch in channels}
else:
    # Simplified for demo
    adstock_params = {ch: {'decay': 0.5} for ch in channels}

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Saturation Configuration")

saturation_type = st.sidebar.selectbox(
    "Saturation Type",
    ["hill", "logistic", "michaelis_menten"],
    help="Diminishing returns curve type"
)

auto_fit_saturation = st.sidebar.checkbox(
    "Auto-Fit Saturation Parameters",
    value=True,
    help="Automatically calibrate saturation curves from data"
)

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Model Results",
    "🎨 Adstock & Saturation",
    "💰 Contribution Analysis",
    "🎯 Budget Optimizer",
    "🔮 Scenario Planning",
    "📈 Advanced Analytics"
])

# ========== MODELING ==========
# Apply transformations
df_transformed = apply_adstock_to_dataframe(df, channels, adstock_type, adstock_params)

adstock_cols = [f"{ch}_adstock" for ch in channels]

# Apply saturation
if auto_fit_saturation:
    df_transformed = apply_saturation_to_dataframe(
        df_transformed,
        adstock_cols,
        saturation_type,
        auto_fit=True,
        target_col='Sales'
    )
else:
    # Use default parameters
    sat_params = {col: {'alpha': 1.0, 'gamma': df_transformed[col].median()} for col in adstock_cols}
    df_transformed = apply_saturation_to_dataframe(
        df_transformed,
        adstock_cols,
        saturation_type,
        params=sat_params
    )

feature_cols = [f"{ch}_adstock_saturated" for ch in channels]

# Train Model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0, positive=True))
])

model.fit(df_transformed[feature_cols], df_transformed['Sales'])
r2 = model.score(df_transformed[feature_cols], df_transformed['Sales'])

df_transformed['Predicted_Sales'] = model.predict(df_transformed[feature_cols])

# ========== TAB 1: MODEL RESULTS ==========
with tab1:
    st.header("📊 Model Performance & Diagnostics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.1%}", help="Explained variance")
    
    with col2:
        mape = np.mean(np.abs((df_transformed['Sales'] - df_transformed['Predicted_Sales']) / df_transformed['Sales'])) * 100
        st.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error")
    
    with col3:
        rmse = np.sqrt(np.mean((df_transformed['Sales'] - df_transformed['Predicted_Sales'])**2))
        st.metric("RMSE", f"Rp {rmse/1e9:.1f}B", help="Root Mean Squared Error")
    
    with col4:
        total_sales = df_transformed['Sales'].sum()
        st.metric("Total Sales", f"Rp {total_sales/1e9:.1f}B")
    
    st.markdown("---")
    
    # Actual vs Predicted
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Actual vs Predicted Sales")
        fig = plot_actual_vs_predicted(
            df_transformed,
            'Date',
            'Sales',
            'Predicted_Sales'
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_actual_vs_predicted")
    
    with col2:
        st.subheader("Channel Coefficients")
        coeffs = model.named_steps['regressor'].coef_
        coeff_dict = {ch: coeffs[i] for i, ch in enumerate(channels)}
        
        fig_coeff = plot_coefficient_importance(coeff_dict)
        st.plotly_chart(fig_coeff, use_container_width=True, key="chart_coefficients")

# ========== TAB 2: ADSTOCK & SATURATION ==========
with tab2:
    st.header("🎨 Adstock & Saturation Curves")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Adstock Decay Curves")
        st.markdown("*How advertising effects decay over time*")
        
        adstock_configs = {
            ch: {'type': adstock_type, 'params': adstock_params[ch]}
            for ch in channels
        }
        
        fig_adstock = plot_adstock_curves(adstock_configs, periods=20)
        st.plotly_chart(fig_adstock, use_container_width=True, key="chart_adstock_curves")
        
        st.info("💡 **Insight**: Higher decay = longer lasting effect. TV typically has highest decay (0.7-0.9).")
    
    with col2:
        st.subheader("Saturation Curves")
        st.markdown("*Diminishing returns visualization*")
        
        # Get saturation params from fitted model
        sat_configs = {}
        for ch in channels:
            adstock_col = f"{ch}_adstock"
            median_spend = df_transformed[adstock_col].median()
            
            # Create params based on saturation type
            if saturation_type == 'hill':
                params = {'alpha': 1.0, 'gamma': median_spend}
            elif saturation_type == 'logistic':
                params = {'k': 0.01, 'x0': median_spend, 'L': 1.0}
            elif saturation_type == 'michaelis_menten':
                params = {'vmax': 1.0, 'km': median_spend}
            else:
                # Default to hill
                params = {'alpha': 1.0, 'gamma': median_spend}
            
            sat_configs[ch] = {
                'type': saturation_type,
                'params': params
            }
        
        fig_sat = plot_saturation_curves(
            sat_configs,
            x_range=(0, df_transformed[[f"{ch}_adstock" for ch in channels]].max().max()),
            show_optimal=True
        )
        st.plotly_chart(fig_sat, use_container_width=True, key="chart_saturation_curves")
        
        st.info("💡 **Insight**: Stars mark optimal spend points (70% efficiency threshold).")

# ========== TAB 3: CONTRIBUTION ANALYSIS ==========
with tab3:
    st.header("💰 Contribution Decomposition & ROAS Analysis")
    
    # Decompose sales
    df_decomposed = decompose_sales(
        df_transformed,
        'Date',
        'Sales',
        feature_cols,
        model,
        include_seasonality=True,
        include_trend=True
    )
    
    # Calculate ROAS
    roas_df = calculate_channel_roas(
        df_decomposed,
        channels,
        [f"{ch}_contribution" for ch in channels]
    )
    
    # Calculate iROAS
    iroas_df = calculate_incremental_roas(
        df_decomposed,
        channels,
        [f"{ch}_contribution" for ch in channels]
    )
    
    # Calculate Marginal ROAS
    current_avg_allocation = df_transformed[channels].mean().values
    marginal_roas_df = calculate_marginal_roas(
        model,
        current_avg_allocation,
        channels
    )
    
    # Waterfall chart
    st.subheader("Sales Decomposition Waterfall")
    
    waterfall_data = create_waterfall_data(
        df_decomposed,
        'baseline',
        'seasonality',
        'trend',
        [f"{ch}_contribution" for ch in channels],
        'residual'
    )
    
    fig_waterfall = plot_contribution_waterfall(
        waterfall_data,
        df_decomposed['baseline'].mean(),
        df_decomposed['Sales'].sum()
    )
    st.plotly_chart(fig_waterfall, use_container_width=True, key="chart_waterfall")
    
    st.markdown("---")
    
    # ROAS Tables
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 ROAS by Channel")
        st.dataframe(
            roas_df.style.format({
                'Total_Spend': 'Rp {:,.0f}',
                'Total_Contribution': 'Rp {:,.0f}',
                'ROAS': '{:.2f}',
                'Contribution_Pct': '{:.1%}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("🎯 Incremental ROAS")
        st.dataframe(
            iroas_df.style.format({
                'Total_Spend': 'Rp {:,.0f}',
                'Incremental_Sales': 'Rp {:,.0f}',
                'iROAS': '{:.2f}',
                'Lift_Pct': '{:.1%}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with col3:
        st.subheader("📈 Marginal ROAS")
        st.dataframe(
            marginal_roas_df.style.format({
                'Current_Spend': 'Rp {:,.0f}',
                'Marginal_Sales': 'Rp {:,.0f}',
                'Marginal_ROAS': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Contribution over time
    st.subheader("Channel Contributions Over Time")
    fig_contrib_time = plot_channel_contributions_over_time(
        df_decomposed,
        'Date',
        [f"{ch}_contribution" for ch in channels]
    )
    st.plotly_chart(fig_contrib_time, use_container_width=True, key="chart_contrib_time")

# ========== TAB 4: BUDGET OPTIMIZER ==========
with tab4:
    st.header("🎯 Budget Allocation Optimizer")
    
    optimization_mode = st.radio(
        "Optimization Mode",
        ["Single-Objective", "Multi-Objective (Pareto)"],
        horizontal=True
    )
    
    if optimization_mode == "Single-Objective":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuration")
            
            total_budget = st.number_input(
                "Total Marketing Budget (Rp)",
                value=200_000_000,
                step=10_000_000,
                format="%d"
            )
            
            objective = st.selectbox(
                "Objective",
                ["sales", "roi", "efficiency"],
                format_func=lambda x: {
                    "sales": "Maximize Sales",
                    "roi": "Maximize ROI",
                    "efficiency": "Maximize Efficiency (Sales/Spend)"
                }[x]
            )
            
            st.markdown("**Constraints (Optional)**")
            
            use_constraints = st.checkbox("Enable Channel Constraints")
            
            constraints = None
            if use_constraints:
                constraints = {'min_pct': {}, 'max_pct': {}}
                
                for ch in channels:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        min_pct = st.number_input(
                            f"{ch} Min %",
                            0, 100, 0, 5,
                            key=f"min_{ch}"
                        ) / 100
                        if min_pct > 0:
                            constraints['min_pct'][ch] = min_pct
                    
                    with col_b:
                        max_pct = st.number_input(
                            f"{ch} Max %",
                            0, 100, 100, 5,
                            key=f"max_{ch}"
                        ) / 100
                        if max_pct < 1.0:
                            constraints['max_pct'][ch] = max_pct
            
            if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
                with st.spinner("Optimizing..."):
                    optimal_allocation, predicted_outcome = single_objective_optimizer(
                        model,
                        total_budget,
                        objective,
                        constraints,
                        channels
                    )
                    
                    st.session_state['optimal_allocation'] = optimal_allocation
                    st.session_state['predicted_outcome'] = predicted_outcome
        
        with col2:
            st.subheader("Optimization Results")
            
            if 'optimal_allocation' in st.session_state:
                optimal_allocation = st.session_state['optimal_allocation']
                predicted_outcome = st.session_state['predicted_outcome']
                
                # Results table
                results_df = pd.DataFrame({
                    'Channel': channels,
                    'Optimal Spend': optimal_allocation.astype(int),
                    'Allocation (%)': optimal_allocation / total_budget * 100
                })
                
                st.dataframe(
                    results_df.style.format({
                        'Optimal Spend': 'Rp {:,.0f}',
                        'Allocation (%)': '{:.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Pie chart
                import plotly.express as px
                fig_pie = px.pie(
                    results_df,
                    values='Optimal Spend',
                    names='Channel',
                    title="Optimal Budget Allocation",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="chart_budget_pie")
                
                st.success(f"✅ **Predicted {objective.title()}**: Rp {predicted_outcome/1e9:.2f}B")
            else:
                st.info("👈 Configure and run optimization to see results")
    
    else:  # Multi-Objective
        st.subheader("Multi-Objective Optimization (Pareto Frontier)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            total_budget = st.number_input(
                "Total Budget (Rp)",
                value=200_000_000,
                step=10_000_000,
                format="%d",
                key="mo_budget"
            )
            
            obj1 = st.selectbox("Objective 1", ["sales", "roi", "efficiency"], key="obj1")
            obj2 = st.selectbox("Objective 2", ["sales", "roi", "efficiency"], index=1, key="obj2")
            
            n_solutions = st.slider("Number of Solutions", 10, 50, 20)
            
            if st.button("🔬 Find Pareto Frontier", type="primary", use_container_width=True):
                with st.spinner("Running multi-objective optimization..."):
                    solutions = multi_objective_optimizer(
                        model,
                        total_budget,
                        [obj1, obj2],
                        n_solutions,
                        channel_names=channels
                    )
                    
                    st.session_state['pareto_solutions'] = solutions
        
        with col2:
            if 'pareto_solutions' in st.session_state:
                solutions = st.session_state['pareto_solutions']
                
                fig_pareto = plot_pareto_frontier(
                    solutions,
                    obj1,
                    obj2,
                    obj1.title(),
                    obj2.title()
                )
                st.plotly_chart(fig_pareto, use_container_width=True, key="chart_pareto_frontier")
                
                st.info("💡 Each point is a Pareto-optimal solution. Pick based on your priority!")
                
                # Show top 5 solutions
                st.subheader("Top 5 Solutions")
                st.dataframe(solutions.head(5), use_container_width=True)
            else:
                st.info("👈 Run optimization to see Pareto frontier")

# ========== TAB 5: SCENARIO PLANNING ==========
with tab5:
    st.header("🔮 Scenario Planning & Sensitivity Analysis")
    
    analysis_type = st.radio(
        "Analysis Type",
        ["What-If Scenarios", "Sensitivity Analysis"],
        horizontal=True
    )
    
    current_allocation = df_transformed[channels].mean().values
    
    if analysis_type == "What-If Scenarios":
        st.subheader("What-If Scenario Builder")
        
        # Predefined scenarios
        predefined_scenarios = {
            "Cut TV 50%": {'TV': -0.5},
            "Double Social Media": {'Facebook': 1.0, 'Instagram': 1.0},
            "Shift 20M from TV to Digital": {
                'TV': -20_000_000,
                'Facebook': 10_000_000,
                'Instagram': 10_000_000
            },
            "Increase Google 30%": {'Google': 0.3}
        }
        
        selected_scenarios = st.multiselect(
            "Select Scenarios to Test",
            list(predefined_scenarios.keys()),
            default=list(predefined_scenarios.keys())[:2]
        )
        
        if st.button("📊 Run Scenario Analysis", type="primary"):
            scenarios_to_test = {k: predefined_scenarios[k] for k in selected_scenarios}
            
            results = scenario_analysis(
                model,
                current_allocation,
                scenarios_to_test,
                channels
            )
            
            st.dataframe(
                results.style.format({
                    **{ch: 'Rp {:,.0f}' for ch in channels},
                    'Predicted_Sales': 'Rp {:,.0f}',
                    'Change_vs_Baseline': 'Rp {:+,.0f}',
                    'Change_Pct': '{:+.1%}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualize
            import plotly.graph_objects as go
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=results['Scenario'],
                y=results['Change_vs_Baseline'],
                marker_color=['gray' if x == 'Baseline (Current)' else ('#2ecc71' if x > 0 else '#e74c3c') 
                              for x in results['Change_vs_Baseline']],
                text=[f"Rp {x/1e9:+.1f}B" for x in results['Change_vs_Baseline']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Impact vs Baseline",
                xaxis_title="Scenario",
                yaxis_title="Change in Sales (Rp)",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="chart_scenario_impact")
    
    else:  # Sensitivity Analysis
        st.subheader("Sensitivity Analysis")
        
        perturbation = st.slider(
            "Perturbation Percentage",
            5, 30, 10,
            help="How much to change each channel's budget"
        ) / 100
        
        if st.button("🔍 Run Sensitivity Analysis", type="primary"):
            sensitivities = sensitivity_analysis(
                model,
                current_allocation,
                channels,
                perturbation
            )
            
            fig_tornado = plot_tornado_chart(sensitivities, 0)
            st.plotly_chart(fig_tornado, use_container_width=True, key="chart_sensitivity_tornado")
            
            st.info(f"💡 **Insight**: Shows impact of ±{perturbation:.0%} change in each channel's budget")

# ========== TAB 6: ADVANCED ANALYTICS ==========
with tab6:
    st.header("📈 Advanced Analytics & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Diagnostics")
        
        # Residual plot
        residuals = df_transformed['Sales'] - df_transformed['Predicted_Sales']
        
        import plotly.graph_objects as go
        fig_resid = go.Figure()
        
        fig_resid.add_trace(go.Scatter(
            x=df_transformed['Predicted_Sales'],
            y=residuals,
            mode='markers',
            marker=dict(color='#3498db', opacity=0.6),
            name='Residuals'
        ))
        
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig_resid.update_layout(
            title="Residual Plot",
            xaxis_title="Predicted Sales",
            yaxis_title="Residuals",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_resid, use_container_width=True, key="chart_residuals")
    
    with col2:
        st.subheader("Feature Importance")
        
        # Calculate feature importance (absolute coefficient values)
        coeffs = model.named_steps['regressor'].coef_
        importance = {ch: abs(coeffs[i]) for i, ch in enumerate(channels)}
        
        fig_importance = plot_coefficient_importance(importance)
        st.plotly_chart(fig_importance, use_container_width=True, key="chart_feature_importance")
    
    st.markdown("---")
    
    # Download section
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export decomposed data
        csv_decomposed = df_decomposed.to_csv(index=False)
        st.download_button(
            "Download Decomposed Data",
            csv_decomposed,
            "mmm_decomposed_data.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export ROAS data
        csv_roas = roas_df.to_csv(index=False)
        st.download_button(
            "Download ROAS Analysis",
            csv_roas,
            "mmm_roas_analysis.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col3:
        # Export model summary
        summary = f"""
Marketing Mix Modeling Summary
==============================

Model Performance:
- R² Score: {r2:.2%}
- MAPE: {mape:.1f}%
- RMSE: Rp {rmse/1e9:.1f}B

Channel Coefficients:
{chr(10).join([f"- {ch}: {coeffs[i]:.6f}" for i, ch in enumerate(channels)])}

ROAS by Channel:
{roas_df.to_string(index=False)}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        st.download_button(
            "Download Model Summary",
            summary,
            "mmm_summary.txt",
            "text/plain",
            use_container_width=True
        )

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p><strong>Advanced Marketing Mix Modeling v2.0</strong> | Powered by AgriSensa Analytics</p>
    <p>Built with: Ridge Regression, Hill Saturation, Geometric/Weibull Adstock, SLSQP Optimization</p>
</div>
""", unsafe_allow_html=True)
