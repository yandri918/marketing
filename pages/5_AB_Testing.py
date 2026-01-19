import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, t, chi2, mannwhitneyu, beta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Advanced A/B Testing | Experimentation", page_icon="🧪", layout="wide")

st.title("🧪 Advanced A/B Testing Simulator")
st.markdown("Enterprise-grade experimentation platform with **multiple statistical tests**, **Bayesian analysis**, and **multi-variant testing**.")

# ========== HELPER FUNCTIONS ==========

def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """Calculate required sample size per variant"""
    # Effect size
    effect_size = mde
    
    # Z-scores
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # Sample size calculation
    p1 = baseline_rate
    p2 = baseline_rate * (1 + effect_size)
    
    n = ((z_alpha + z_beta)**2 * (p1*(1-p1) + p2*(1-p2))) / ((p1 - p2)**2)
    
    return int(np.ceil(n))

def z_test(conversions_a, visitors_a, conversions_b, visitors_b):
    """Perform Z-test for proportions"""
    rate_a = conversions_a / visitors_a
    rate_b = conversions_b / visitors_b
    
    # Pooled proportion
    p_pool = (conversions_a + conversions_b) / (visitors_a + visitors_b)
    
    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/visitors_a + 1/visitors_b))
    
    # Z-score
    z_score = (rate_b - rate_a) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    # Confidence interval
    se_diff = np.sqrt(rate_a*(1-rate_a)/visitors_a + rate_b*(1-rate_b)/visitors_b)
    ci_lower = (rate_b - rate_a) - 1.96 * se_diff
    ci_upper = (rate_b - rate_a) + 1.96 * se_diff
    
    return {
        'z_score': z_score,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'rate_a': rate_a,
        'rate_b': rate_b
    }

def chi_square_test(conversions_a, visitors_a, conversions_b, visitors_b):
    """Perform Chi-square test"""
    # Contingency table
    observed = np.array([
        [conversions_a, visitors_a - conversions_a],
        [conversions_b, visitors_b - conversions_b]
    ])
    
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
    
    return {
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'dof': dof
    }

def calculate_effect_size(rate_a, rate_b, visitors_a, visitors_b):
    """Calculate Cohen's h effect size for proportions"""
    # Arcsine transformation
    phi_a = 2 * np.arcsin(np.sqrt(rate_a))
    phi_b = 2 * np.arcsin(np.sqrt(rate_b))
    
    cohens_h = phi_b - phi_a
    
    return cohens_h

def bayesian_ab_test(conversions_a, visitors_a, conversions_b, visitors_b, prior_alpha=1, prior_beta=1):
    """Perform Bayesian A/B test"""
    # Posterior parameters
    alpha_a = prior_alpha + conversions_a
    beta_a = prior_beta + (visitors_a - conversions_a)
    
    alpha_b = prior_alpha + conversions_b
    beta_b = prior_beta + (visitors_b - conversions_b)
    
    # Monte Carlo simulation
    samples = 100000
    samples_a = np.random.beta(alpha_a, beta_a, samples)
    samples_b = np.random.beta(alpha_b, beta_b, samples)
    
    # Probability B > A
    prob_b_better = (samples_b > samples_a).mean()
    
    # Expected loss
    expected_loss_a = (np.maximum(samples_b - samples_a, 0)).mean()
    expected_loss_b = (np.maximum(samples_a - samples_b, 0)).mean()
    
    # Credible intervals
    ci_a = (np.percentile(samples_a, 2.5), np.percentile(samples_a, 97.5))
    ci_b = (np.percentile(samples_b, 2.5), np.percentile(samples_b, 97.5))
    
    return {
        'prob_b_better': prob_b_better,
        'expected_loss_a': expected_loss_a,
        'expected_loss_b': expected_loss_b,
        'ci_a': ci_a,
        'ci_b': ci_b,
        'samples_a': samples_a,
        'samples_b': samples_b
    }

# ========== SESSION STATE ==========
if 'experiment_history' not in st.session_state:
    st.session_state.experiment_history = []

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Configuration")

test_type = st.sidebar.selectbox(
    "Test Type",
    ["A/B Test (2 variants)", "Multi-Variant Test (3+ variants)"]
)

significance_level = st.sidebar.select_slider(
    "Significance Level (α)",
    options=[0.01, 0.05, 0.10],
    value=0.05,
    format_func=lambda x: f"{x} ({(1-x)*100:.0f}% confidence)"
)

st.sidebar.divider()

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Test Setup",
    "📈 Results",
    "🔢 Sample Size",
    "🔄 Sequential",
    "🎲 Bayesian",
    "📋 History"
])

# ========== TAB 1: TEST SETUP ==========
with tab1:
    st.subheader("📊 Experiment Setup")
    
    if test_type == "A/B Test (2 variants)":
        st.markdown("### Input Your Test Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🅰️ Control Group (A)")
            visitors_a = st.number_input("Visitors (Control)", min_value=1, value=1000, step=100, key="visitors_a")
            conversions_a = st.number_input("Conversions (Control)", min_value=0, max_value=visitors_a, value=120, step=10, key="conv_a")
            
            rate_a = conversions_a / visitors_a if visitors_a > 0 else 0
            st.metric("Conversion Rate A", f"{rate_a:.2%}")
        
        with col2:
            st.markdown("#### 🅱️ Variant Group (B)")
            visitors_b = st.number_input("Visitors (Variant)", min_value=1, value=1000, step=100, key="visitors_b")
            conversions_b = st.number_input("Conversions (Variant)", min_value=0, max_value=visitors_b, value=150, step=10, key="conv_b")
            
            rate_b = conversions_b / visitors_b if visitors_b > 0 else 0
            st.metric("Conversion Rate B", f"{rate_b:.2%}")
        
        # Store in session state
        st.session_state.test_data = {
            'type': 'AB',
            'variants': ['A', 'B'],
            'visitors': [visitors_a, visitors_b],
            'conversions': [conversions_a, conversions_b],
            'rates': [rate_a, rate_b]
        }
    
    else:  # Multi-variant
        st.markdown("### Multi-Variant Test Setup")
        
        num_variants = st.slider("Number of Variants", 3, 5, 3)
        
        variants_data = []
        cols = st.columns(num_variants)
        
        for i, col in enumerate(cols):
            with col:
                variant_name = chr(65 + i)  # A, B, C, D, E
                st.markdown(f"#### Variant {variant_name}")
                
                visitors = st.number_input(f"Visitors {variant_name}", min_value=1, value=1000, step=100, key=f"vis_{i}")
                conversions = st.number_input(f"Conversions {variant_name}", min_value=0, max_value=visitors, value=100 + i*20, step=10, key=f"conv_{i}")
                
                rate = conversions / visitors if visitors > 0 else 0
                st.metric(f"Rate {variant_name}", f"{rate:.2%}")
                
                variants_data.append({
                    'name': variant_name,
                    'visitors': visitors,
                    'conversions': conversions,
                    'rate': rate
                })
        
        # Store in session state
        st.session_state.test_data = {
            'type': 'MVT',
            'variants': [v['name'] for v in variants_data],
            'visitors': [v['visitors'] for v in variants_data],
            'conversions': [v['conversions'] for v in variants_data],
            'rates': [v['rate'] for v in variants_data]
        }

# ========== TAB 2: RESULTS ==========
with tab2:
    st.subheader("📈 Statistical Analysis Results")
    
    if 'test_data' not in st.session_state:
        st.warning("⚠️ Please set up your test in the 'Test Setup' tab first.")
    else:
        test_data = st.session_state.test_data
        
        if test_data['type'] == 'AB':
            # A/B Test Analysis
            visitors_a, visitors_b = test_data['visitors']
            conversions_a, conversions_b = test_data['conversions']
            rate_a, rate_b = test_data['rates']
            
            # Calculate uplift
            uplift = (rate_b - rate_a) / rate_a if rate_a > 0 else 0
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Control Rate (A)", f"{rate_a:.2%}")
            col2.metric("Variant Rate (B)", f"{rate_b:.2%}")
            col3.metric("Absolute Lift", f"{(rate_b - rate_a):.2%}")
            col4.metric("Relative Uplift", f"{uplift:.2%}", delta=f"{uplift:.2%}")
            
            st.divider()
            
            # Statistical Tests
            st.markdown("### Statistical Test Results")
            
            # Z-Test
            z_result = z_test(conversions_a, visitors_a, conversions_b, visitors_b)
            
            # Chi-Square
            chi_result = chi_square_test(conversions_a, visitors_a, conversions_b, visitors_b)
            
            # Effect Size
            effect_size = calculate_effect_size(rate_a, rate_b, visitors_a, visitors_b)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Z-Test")
                st.metric("Z-Score", f"{z_result['z_score']:.4f}")
                st.metric("P-Value", f"{z_result['p_value']:.6f}")
                
                if z_result['p_value'] < significance_level:
                    st.success("✅ Statistically Significant")
                else:
                    st.warning("⚠️ Not Significant")
            
            with col2:
                st.markdown("#### Chi-Square Test")
                st.metric("χ² Statistic", f"{chi_result['chi2_stat']:.4f}")
                st.metric("P-Value", f"{chi_result['p_value']:.6f}")
                st.metric("Degrees of Freedom", chi_result['dof'])
            
            with col3:
                st.markdown("#### Effect Size")
                st.metric("Cohen's h", f"{effect_size:.4f}")
                
                if abs(effect_size) < 0.2:
                    effect_label = "Small"
                elif abs(effect_size) < 0.5:
                    effect_label = "Medium"
                else:
                    effect_label = "Large"
                
                st.info(f"Effect Size: **{effect_label}**")
            
            st.divider()
            
            # Confidence Interval
            st.markdown("### 95% Confidence Interval")
            
            ci_lower = z_result['ci_lower']
            ci_upper = z_result['ci_upper']
            
            st.write(f"**Difference in conversion rates:** [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            if ci_lower > 0:
                st.success("✅ **Variant B is significantly better** (entire CI is above zero)")
            elif ci_upper < 0:
                st.error("❌ **Control A is significantly better** (entire CI is below zero)")
            else:
                st.warning("⚠️ **No clear winner** (CI includes zero)")
            
            # Visualization
            st.markdown("### Distribution Comparison")
            
            # Create distribution plot
            se_a = np.sqrt(rate_a * (1 - rate_a) / visitors_a)
            se_b = np.sqrt(rate_b * (1 - rate_b) / visitors_b)
            
            x = np.linspace(min(rate_a - 3*se_a, rate_b - 3*se_b), max(rate_a + 3*se_a, rate_b + 3*se_b), 1000)
            y_a = norm.pdf(x, rate_a, se_a)
            y_b = norm.pdf(x, rate_b, se_b)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x, y=y_a,
                mode='lines',
                name='Control A',
                fill='tozeroy',
                line=dict(color='#E74C3C', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=x, y=y_b,
                mode='lines',
                name='Variant B',
                fill='tozeroy',
                line=dict(color='#3498DB', width=2),
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Probability Density Functions of Conversion Rates",
                xaxis_title="Conversion Rate",
                yaxis_title="Probability Density",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # MVT
            # Multi-Variant Analysis
            st.markdown("### Multi-Variant Test Results")
            
            # Display all variants
            variants_df = pd.DataFrame({
                'Variant': test_data['variants'],
                'Visitors': test_data['visitors'],
                'Conversions': test_data['conversions'],
                'Conv. Rate': test_data['rates']
            })
            
            st.dataframe(variants_df.style.format({
                'Visitors': '{:,.0f}',
                'Conversions': '{:,.0f}',
                'Conv. Rate': '{:.2%}'
            }).background_gradient(subset=['Conv. Rate'], cmap='RdYlGn'), use_container_width=True, hide_index=True)
            
            # ANOVA
            st.markdown("### ANOVA Test")
            
            # Prepare data for ANOVA
            groups = []
            for i, (visitors, conversions) in enumerate(zip(test_data['visitors'], test_data['conversions'])):
                # Create binary outcomes
                successes = [1] * conversions
                failures = [0] * (visitors - conversions)
                groups.append(successes + failures)
            
            # Perform ANOVA
            f_stat, p_value_anova = stats.f_oneway(*groups)
            
            col1, col2 = st.columns(2)
            col1.metric("F-Statistic", f"{f_stat:.4f}")
            col2.metric("P-Value", f"{p_value_anova:.6f}")
            
            if p_value_anova < significance_level:
                st.success("✅ **Significant difference detected** among variants")
            else:
                st.warning("⚠️ **No significant difference** among variants")
            
            # Visualization
            fig_mvt = px.bar(
                variants_df,
                x='Variant',
                y='Conv. Rate',
                title="Conversion Rate by Variant",
                color='Conv. Rate',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig_mvt, use_container_width=True)

# ========== TAB 3: SAMPLE SIZE ==========
with tab3:
    st.subheader("🔢 Sample Size Calculator")
    
    st.markdown("""
    Calculate the required sample size for your A/B test before running it.
    This helps ensure your test has enough statistical power to detect meaningful differences.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Test Parameters")
        
        baseline_rate = st.slider("Baseline Conversion Rate (%)", 0.1, 50.0, 5.0, 0.1) / 100
        mde = st.slider("Minimum Detectable Effect (MDE) %", 1.0, 100.0, 20.0, 1.0) / 100
        alpha_calc = st.select_slider("Significance Level (α)", options=[0.01, 0.05, 0.10], value=0.05)
        power = st.select_slider("Statistical Power (1-β)", options=[0.80, 0.90, 0.95], value=0.80)
    
    with col2:
        st.markdown("### Required Sample Size")
        
        # Calculate sample size
        sample_size = calculate_sample_size(baseline_rate, mde, alpha_calc, power)
        
        st.metric("Sample Size per Variant", f"{sample_size:,}")
        st.metric("Total Sample Size", f"{sample_size * 2:,}")
        
        # Duration estimation
        st.markdown("### Duration Estimation")
        
        daily_visitors = st.number_input("Expected Daily Visitors", min_value=1, value=1000, step=100)
        
        days_needed = (sample_size * 2) / daily_visitors
        weeks_needed = days_needed / 7
        
        st.metric("Days Needed", f"{days_needed:.1f}")
        st.metric("Weeks Needed", f"{weeks_needed:.1f}")
    
    st.divider()
    
    # Power Analysis Visualization
    st.markdown("### Power Analysis")
    
    # Calculate power for different sample sizes
    sample_sizes = np.linspace(100, sample_size * 2, 50)
    powers = []
    
    for n in sample_sizes:
        # Simplified power calculation
        effect = mde
        z_alpha = norm.ppf(1 - alpha_calc/2)
        z_beta = (effect * np.sqrt(n/2)) / np.sqrt(baseline_rate * (1-baseline_rate) * 2) - z_alpha
        power_calc = norm.cdf(z_beta)
        powers.append(power_calc)
    
    fig_power = go.Figure()
    
    fig_power.add_trace(go.Scatter(
        x=sample_sizes,
        y=powers,
        mode='lines',
        name='Statistical Power',
        line=dict(color='#3498DB', width=3)
    ))
    
    fig_power.add_hline(y=0.80, line_dash="dash", line_color="green", annotation_text="80% Power")
    fig_power.add_vline(x=sample_size, line_dash="dash", line_color="red", annotation_text=f"Required: {sample_size:,}")
    
    fig_power.update_layout(
        title="Statistical Power vs Sample Size",
        xaxis_title="Sample Size per Variant",
        yaxis_title="Statistical Power",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_power, use_container_width=True)

# ========== TAB 4: SEQUENTIAL ==========
with tab4:
    st.subheader("🔄 Sequential Testing")
    
    st.markdown("""
    Monitor your test in real-time and make decisions with early stopping rules.
    Sequential testing allows you to stop tests early when results are conclusive.
    """)
    
    if 'test_data' not in st.session_state or st.session_state.test_data['type'] != 'AB':
        st.warning("⚠️ Sequential testing is only available for A/B tests. Please set up an A/B test first.")
    else:
        test_data = st.session_state.test_data
        visitors_a, visitors_b = test_data['visitors']
        conversions_a, conversions_b = test_data['conversions']
        
        # Calculate current metrics
        current_sample = visitors_a + visitors_b
        z_result = z_test(conversions_a, visitors_a, conversions_b, visitors_b)
        
        # Sequential testing threshold (adjusted for multiple looks)
        # Using O'Brien-Fleming spending function
        looks = st.slider("Number of Planned Looks", 1, 10, 5)
        current_look = st.slider("Current Look Number", 1, looks, 1)
        
        # Adjusted alpha for current look
        t = current_look / looks
        alpha_adjusted = 2 * (1 - norm.cdf(norm.ppf(1 - significance_level/2) / np.sqrt(t)))
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Current Sample Size", f"{current_sample:,}")
        col2.metric("Current P-Value", f"{z_result['p_value']:.6f}")
        col3.metric("Adjusted Threshold", f"{alpha_adjusted:.6f}")
        
        st.divider()
        
        # Decision
        st.markdown("### Sequential Testing Decision")
        
        if z_result['p_value'] < alpha_adjusted:
            st.success(f"""
            ✅ **STOP THE TEST - Significant Result!**
            
            The p-value ({z_result['p_value']:.6f}) is below the adjusted threshold ({alpha_adjusted:.6f}).
            You can confidently stop the test and implement the winning variant.
            """)
        else:
            st.info(f"""
            ⏳ **CONTINUE THE TEST**
            
            The p-value ({z_result['p_value']:.6f}) is above the adjusted threshold ({alpha_adjusted:.6f}).
            Continue collecting data or wait for the next planned look.
            """)
        
        # Visualization of sequential boundaries
        st.markdown("### Sequential Testing Boundaries")
        
        look_numbers = np.arange(1, looks + 1)
        thresholds = []
        
        for look in look_numbers:
            t = look / looks
            threshold = 2 * (1 - norm.cdf(norm.ppf(1 - significance_level/2) / np.sqrt(t)))
            thresholds.append(threshold)
        
        fig_seq = go.Figure()
        
        fig_seq.add_trace(go.Scatter(
            x=look_numbers,
            y=thresholds,
            mode='lines+markers',
            name='Stopping Boundary',
            line=dict(color='red', width=2)
        ))
        
        fig_seq.add_hline(y=significance_level, line_dash="dash", line_color="gray", annotation_text=f"Fixed α = {significance_level}")
        
        # Add current position
        if current_look <= looks:
            fig_seq.add_trace(go.Scatter(
                x=[current_look],
                y=[z_result['p_value']],
                mode='markers',
                name='Current Position',
                marker=dict(size=15, color='blue', symbol='star')
            ))
        
        fig_seq.update_layout(
            title="O'Brien-Fleming Sequential Boundaries",
            xaxis_title="Look Number",
            yaxis_title="P-Value Threshold",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_seq, use_container_width=True)

# ========== TAB 5: BAYESIAN ==========
with tab5:
    st.subheader("🎲 Bayesian A/B Testing")
    
    st.markdown("""
    Bayesian analysis provides intuitive probabilities and doesn't require fixed sample sizes.
    """)
    
    if 'test_data' not in st.session_state or st.session_state.test_data['type'] != 'AB':
        st.warning("⚠️ Bayesian analysis is only available for A/B tests. Please set up an A/B test first.")
    else:
        test_data = st.session_state.test_data
        visitors_a, visitors_b = test_data['visitors']
        conversions_a, conversions_b = test_data['conversions']
        
        # Prior selection
        st.markdown("### Prior Distribution")
        
        col1, col2 = st.columns(2)
        with col1:
            prior_alpha = st.number_input("Prior α (successes)", min_value=0.1, value=1.0, step=0.1)
        with col2:
            prior_beta = st.number_input("Prior β (failures)", min_value=0.1, value=1.0, step=0.1)
        
        # Perform Bayesian analysis
        bayes_result = bayesian_ab_test(conversions_a, visitors_a, conversions_b, visitors_b, prior_alpha, prior_beta)
        
        st.divider()
        
        # Results
        st.markdown("### Bayesian Results")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("P(B > A)", f"{bayes_result['prob_b_better']:.2%}")
        col2.metric("Expected Loss (A)", f"{bayes_result['expected_loss_a']:.4f}")
        col3.metric("Expected Loss (B)", f"{bayes_result['expected_loss_b']:.4f}")
        
        # Decision
        if bayes_result['prob_b_better'] > 0.95:
            st.success("✅ **Strong evidence that B is better than A**")
        elif bayes_result['prob_b_better'] < 0.05:
            st.error("❌ **Strong evidence that A is better than B**")
        else:
            st.warning("⚠️ **Inconclusive - need more data**")
        
        st.divider()
        
        # Posterior distributions
        st.markdown("### Posterior Distributions")
        
        # Create histogram
        fig_bayes = go.Figure()
        
        fig_bayes.add_trace(go.Histogram(
            x=bayes_result['samples_a'],
            name='Control A',
            opacity=0.7,
            nbinsx=50,
            marker_color='#E74C3C'
        ))
        
        fig_bayes.add_trace(go.Histogram(
            x=bayes_result['samples_b'],
            name='Variant B',
            opacity=0.7,
            nbinsx=50,
            marker_color='#3498DB'
        ))
        
        fig_bayes.update_layout(
            title="Posterior Distributions of Conversion Rates",
            xaxis_title="Conversion Rate",
            yaxis_title="Frequency",
            barmode='overlay',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_bayes, use_container_width=True)
        
        # Credible intervals
        st.markdown("### 95% Credible Intervals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Control A:** [{bayes_result['ci_a'][0]:.4f}, {bayes_result['ci_a'][1]:.4f}]")
        
        with col2:
            st.write(f"**Variant B:** [{bayes_result['ci_b'][0]:.4f}, {bayes_result['ci_b'][1]:.4f}]")

# ========== TAB 6: HISTORY ==========
with tab6:
    st.subheader("📋 Experiment History")
    
    st.markdown("Track all your experiments and compare results over time.")
    
    # Save current experiment
    if st.button("💾 Save Current Experiment"):
        if 'test_data' in st.session_state:
            experiment = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': st.session_state.test_data['type'],
                'variants': st.session_state.test_data['variants'],
                'visitors': st.session_state.test_data['visitors'],
                'conversions': st.session_state.test_data['conversions'],
                'rates': st.session_state.test_data['rates']
            }
            
            st.session_state.experiment_history.append(experiment)
            st.success("✅ Experiment saved!")
        else:
            st.warning("⚠️ No experiment data to save. Please set up a test first.")
    
    st.divider()
    
    # Display history
    if st.session_state.experiment_history:
        st.markdown("### Saved Experiments")
        
        for i, exp in enumerate(reversed(st.session_state.experiment_history)):
            with st.expander(f"Experiment {len(st.session_state.experiment_history) - i} - {exp['timestamp']}"):
                st.write(f"**Type:** {exp['type']}")
                st.write(f"**Variants:** {', '.join(exp['variants'])}")
                
                exp_df = pd.DataFrame({
                    'Variant': exp['variants'],
                    'Visitors': exp['visitors'],
                    'Conversions': exp['conversions'],
                    'Conv. Rate': exp['rates']
                })
                
                st.dataframe(exp_df.style.format({
                    'Visitors': '{:,.0f}',
                    'Conversions': '{:,.0f}',
                    'Conv. Rate': '{:.2%}'
                }), use_container_width=True, hide_index=True)
        
        # Clear history
        if st.button("🗑️ Clear All History"):
            st.session_state.experiment_history = []
            st.rerun()
    else:
        st.info("No experiments saved yet. Run a test and click 'Save Current Experiment' to track your results.")

# ========== FOOTER ==========
st.divider()
st.caption("💡 **Pro Tip:** Always run your tests for at least one full business cycle to account for day-of-week effects!")
