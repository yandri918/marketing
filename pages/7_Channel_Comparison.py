import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as optimize

st.set_page_config(page_title="Advanced Channel Effectiveness | ROI Analysis", page_icon="⚔️", layout="wide")

st.title("⚔️ Advanced Channel Effectiveness Analysis")
st.markdown("Enterprise-grade **ROI & Efficiency Analysis** comparing **Digital Marketing** vs **Field Officer (Offline Acquisition)**.")

# ========== HELPER FUNCTIONS ==========

def calculate_ltv_cac_ratio(ltv, cac):
    """Calculate LTV:CAC ratio"""
    return ltv / cac if cac > 0 else 0

def calculate_payback_period(cac, monthly_revenue):
    """Calculate months to recover CAC"""
    return cac / monthly_revenue if monthly_revenue > 0 else 0

def optimize_budget_allocation(channels_data, total_budget):
    """Optimize budget allocation across channels"""
    # Simplified optimization - allocate based on ROI
    rois = [ch['roi'] for ch in channels_data]
    total_roi = sum(rois)
    
    if total_roi > 0:
        allocations = [(roi / total_roi) * total_budget for roi in rois]
    else:
        allocations = [total_budget / len(channels_data)] * len(channels_data)
    
    return allocations

# ========== SESSION STATE ==========
if 'channel_history' not in st.session_state:
    st.session_state.channel_history = []

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Configuration")

analysis_period = st.sidebar.selectbox(
    "Analysis Period",
    ["Monthly", "Quarterly", "Annual"]
)

currency = st.sidebar.selectbox("Currency", ["Rp", "USD", "EUR"])

st.sidebar.divider()

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Channel Input",
    "💰 ROI Analysis",
    "⚡ Efficiency Metrics",
    "🎯 Attribution",
    "📈 Optimization",
    "🔮 Scenarios"
])

# ========== TAB 1: CHANNEL INPUT ==========
with tab1:
    st.subheader("📊 Channel Performance Input")
    
    st.markdown("### Digital Marketing Channels")
    
    col1, col2, col3 = st.columns(3)
    
    # Digital channels
    digital_channels = {}
    
    with col1:
        st.markdown("#### 📱 Social Media")
        social_spend = st.number_input("Social Ad Spend (Rp)", min_value=0, value=20000000, step=1000000, key="social_spend")
        social_leads = st.number_input("Social Leads", min_value=0, value=800, step=50, key="social_leads")
        social_conv = st.slider("Social Conv. Rate (%)", 0.0, 20.0, 4.0, 0.1, key="social_conv")
        
        digital_channels['Social Media'] = {
            'spend': social_spend,
            'leads': social_leads,
            'conv_rate': social_conv / 100,
            'customers': int(social_leads * social_conv / 100)
        }
    
    with col2:
        st.markdown("#### 🔍 Search (Google/SEO)")
        search_spend = st.number_input("Search Ad Spend (Rp)", min_value=0, value=25000000, step=1000000, key="search_spend")
        search_leads = st.number_input("Search Leads", min_value=0, value=1000, step=50, key="search_leads")
        search_conv = st.slider("Search Conv. Rate (%)", 0.0, 20.0, 6.0, 0.1, key="search_conv")
        
        digital_channels['Search'] = {
            'spend': search_spend,
            'leads': search_leads,
            'conv_rate': search_conv / 100,
            'customers': int(search_leads * search_conv / 100)
        }
    
    with col3:
        st.markdown("#### 📧 Email Marketing")
        email_spend = st.number_input("Email Costs (Rp)", min_value=0, value=5000000, step=500000, key="email_spend")
        email_leads = st.number_input("Email Leads", min_value=0, value=400, step=50, key="email_leads")
        email_conv = st.slider("Email Conv. Rate (%)", 0.0, 20.0, 8.0, 0.1, key="email_conv")
        
        digital_channels['Email'] = {
            'spend': email_spend,
            'leads': email_leads,
            'conv_rate': email_conv / 100,
            'customers': int(email_leads * email_conv / 100)
        }
    
    st.divider()
    
    st.markdown("### Offline Channels")
    
    col1, col2 = st.columns(2)
    
    offline_channels = {}
    
    with col1:
        st.markdown("#### 👔 Field Officers")
        field_officers = st.number_input("Number of Field Officers", min_value=1, value=5, step=1)
        field_salary = st.number_input("Monthly Salary per Officer (Rp)", min_value=0, value=5000000, step=500000)
        field_transport = st.number_input("Transport & Logistics (Rp)", min_value=0, value=1500000, step=100000)
        field_commission = st.number_input("Commission per Customer (Rp)", min_value=0, value=100000, step=10000)
        field_customers_per = st.number_input("Customers per Officer/Month", min_value=1, value=30, step=1)
        
        field_customers = field_officers * field_customers_per
        field_base_cost = (field_salary + field_transport) * field_officers
        field_comm_cost = field_commission * field_customers
        field_total_spend = field_base_cost + field_comm_cost
        
        offline_channels['Field Officers'] = {
            'spend': field_total_spend,
            'leads': field_customers,
            'conv_rate': 1.0,
            'customers': field_customers
        }
    
    with col2:
        st.markdown("#### 🎪 Events & Trade Shows")
        events_spend = st.number_input("Events Budget (Rp)", min_value=0, value=15000000, step=1000000)
        events_leads = st.number_input("Event Leads", min_value=0, value=200, step=10)
        events_conv = st.slider("Event Conv. Rate (%)", 0.0, 50.0, 15.0, 0.5)
        
        offline_channels['Events'] = {
            'spend': events_spend,
            'leads': events_leads,
            'conv_rate': events_conv / 100,
            'customers': int(events_leads * events_conv / 100)
        }
    
    # Combine all channels
    all_channels = {**digital_channels, **offline_channels}
    
    # Store in session state
    st.session_state.all_channels = all_channels
    
    st.divider()
    
    # Summary table
    st.markdown("### Channel Summary")
    
    summary_data = []
    for channel, data in all_channels.items():
        cac = data['spend'] / data['customers'] if data['customers'] > 0 else 0
        summary_data.append({
            'Channel': channel,
            'Spend': data['spend'],
            'Leads': data['leads'],
            'Customers': data['customers'],
            'Conv. Rate': f"{data['conv_rate']*100:.1f}%",
            'CAC': cac
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    st.dataframe(summary_df.style.format({
        'Spend': 'Rp {:,.0f}',
        'Leads': '{:,.0f}',
        'Customers': '{:,.0f}',
        'CAC': 'Rp {:,.0f}'
    }).background_gradient(subset=['Customers'], cmap='Greens'), use_container_width=True, hide_index=True)

# ========== TAB 2: ROI ANALYSIS ==========
with tab2:
    st.subheader("💰 ROI & Profitability Analysis")
    
    if 'all_channels' not in st.session_state:
        st.warning("⚠️ Please input channel data in the 'Channel Input' tab first.")
    else:
        # Customer value inputs
        st.markdown("### Customer Economics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_order_value = st.number_input("Average Order Value (Rp)", min_value=0, value=500000, step=10000)
            orders_per_year = st.number_input("Orders per Customer/Year", min_value=1, value=4, step=1)
        
        with col2:
            gross_margin = st.slider("Gross Margin (%)", 0, 100, 40)
            customer_lifespan = st.slider("Customer Lifespan (years)", 1, 10, 3)
        
        with col3:
            retention_rate = st.slider("Annual Retention Rate (%)", 0, 100, 70)
        
        # Calculate LTV
        annual_revenue = avg_order_value * orders_per_year
        annual_profit = annual_revenue * (gross_margin / 100)
        
        # Simple LTV calculation
        ltv = annual_profit * customer_lifespan * (retention_rate / 100)
        
        st.metric("Customer Lifetime Value (LTV)", f"Rp {ltv:,.0f}")
        
        st.divider()
        
        # Calculate ROI metrics for each channel
        st.markdown("### Channel ROI Metrics")
        
        roi_data = []
        for channel, data in st.session_state.all_channels.items():
            cac = data['spend'] / data['customers'] if data['customers'] > 0 else 0
            ltv_cac_ratio = calculate_ltv_cac_ratio(ltv, cac)
            roi = ((ltv - cac) / cac * 100) if cac > 0 else 0
            payback = calculate_payback_period(cac, annual_profit / 12)
            
            roi_data.append({
                'Channel': channel,
                'CAC': cac,
                'LTV': ltv,
                'LTV:CAC': ltv_cac_ratio,
                'ROI': roi,
                'Payback (months)': payback,
                'Total Customers': data['customers'],
                'Total Profit': (ltv - cac) * data['customers']
            })
        
        roi_df = pd.DataFrame(roi_data)
        roi_df = roi_df.sort_values('ROI', ascending=False)
        
        st.dataframe(roi_df.style.format({
            'CAC': 'Rp {:,.0f}',
            'LTV': 'Rp {:,.0f}',
            'LTV:CAC': '{:.2f}x',
            'ROI': '{:.1f}%',
            'Payback (months)': '{:.1f}',
            'Total Customers': '{:,.0f}',
            'Total Profit': 'Rp {:,.0f}'
        }).background_gradient(subset=['ROI'], cmap='RdYlGn'), use_container_width=True, hide_index=True)
        
        # Interpretation
        st.markdown("### 📊 ROI Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**LTV:CAC Ratio Guidelines:**")
            st.info("""
            - **< 1.0** 🔴 Losing money
            - **1.0 - 3.0** 🟡 Breakeven to acceptable
            - **> 3.0** 🟢 Excellent
            - **> 5.0** 🟢 Outstanding
            """)
        
        with col2:
            st.markdown("**Payback Period Guidelines:**")
            st.info("""
            - **< 6 months** 🟢 Excellent
            - **6-12 months** 🟡 Good
            - **12-18 months** 🟠 Acceptable
            - **> 18 months** 🔴 Concerning
            """)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ROI by Channel")
            
            fig_roi = px.bar(
                roi_df,
                x='Channel',
                y='ROI',
                color='ROI',
                color_continuous_scale='RdYlGn',
                title="Return on Investment (%)"
            )
            
            fig_roi.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Breakeven")
            fig_roi.update_layout(template="plotly_white")
            
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with col2:
            st.markdown("### LTV:CAC Ratio")
            
            fig_ltv = px.bar(
                roi_df,
                x='Channel',
                y='LTV:CAC',
                color='LTV:CAC',
                color_continuous_scale='Viridis',
                title="LTV to CAC Ratio"
            )
            
            fig_ltv.add_hline(y=3, line_dash="dash", line_color="green", annotation_text="Target: 3x")
            fig_ltv.update_layout(template="plotly_white")
            
            st.plotly_chart(fig_ltv, use_container_width=True)

# ========== TAB 3: EFFICIENCY METRICS ==========
with tab3:
    st.subheader("⚡ Channel Efficiency Metrics")
    
    if 'all_channels' not in st.session_state:
        st.warning("⚠️ Please input channel data in the 'Channel Input' tab first.")
    else:
        # Calculate efficiency metrics
        efficiency_data = []
        
        for channel, data in st.session_state.all_channels.items():
            cpl = data['spend'] / data['leads'] if data['leads'] > 0 else 0
            cac = data['spend'] / data['customers'] if data['customers'] > 0 else 0
            
            efficiency_data.append({
                'Channel': channel,
                'Spend': data['spend'],
                'Leads': data['leads'],
                'Customers': data['customers'],
                'Cost per Lead': cpl,
                'Cost per Customer': cac,
                'Conversion Rate': data['conv_rate'] * 100,
                'Efficiency Score': (data['customers'] / (data['spend'] / 1000000)) if data['spend'] > 0 else 0
            })
        
        eff_df = pd.DataFrame(efficiency_data)
        
        st.markdown("### Efficiency Comparison (Editable)")
        
        st.info("💡 **Edit the data below** - Modify Spend, Leads, or Customers to see efficiency metrics update in real-time.")
        
        # Make data editable
        edited_eff_df = st.data_editor(
            eff_df[['Channel', 'Spend', 'Leads', 'Customers']],
            key="efficiency_editor",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Channel": st.column_config.TextColumn("Channel", disabled=True),
                "Spend": st.column_config.NumberColumn("Spend (Rp)", min_value=0, step=100000),
                "Leads": st.column_config.NumberColumn("Leads", min_value=0, step=10),
                "Customers": st.column_config.NumberColumn("Customers", min_value=0, step=5)
            }
        )
        
        # Recalculate metrics with edited data
        updated_efficiency_data = []
        
        for idx, row in edited_eff_df.iterrows():
            cpl = row['Spend'] / row['Leads'] if row['Leads'] > 0 else 0
            cac = row['Spend'] / row['Customers'] if row['Customers'] > 0 else 0
            conv_rate = (row['Customers'] / row['Leads'] * 100) if row['Leads'] > 0 else 0
            efficiency_score = (row['Customers'] / (row['Spend'] / 1000000)) if row['Spend'] > 0 else 0
            
            updated_efficiency_data.append({
                'Channel': row['Channel'],
                'Spend': row['Spend'],
                'Leads': row['Leads'],
                'Customers': row['Customers'],
                'Cost per Lead': cpl,
                'Cost per Customer': cac,
                'Conversion Rate': conv_rate,
                'Efficiency Score': efficiency_score
            })
        
        updated_eff_df = pd.DataFrame(updated_efficiency_data)
        
        st.divider()
        
        # Display calculated metrics
        st.markdown("### Calculated Efficiency Metrics")
        
        st.dataframe(updated_eff_df.style.format({
            'Spend': 'Rp {:,.0f}',
            'Leads': '{:,.0f}',
            'Customers': '{:,.0f}',
            'Cost per Lead': 'Rp {:,.0f}',
            'Cost per Customer': 'Rp {:,.0f}',
            'Conversion Rate': '{:.1f}%',
            'Efficiency Score': '{:.2f}'
        }).background_gradient(subset=['Efficiency Score'], cmap='Greens'), use_container_width=True, hide_index=True)
        
        st.caption("**Efficiency Score** = Customers acquired per Rp 1M spent (higher is better)")
        
        st.divider()
        
        # Visualizations with updated data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Cost per Lead vs Cost per Customer")
            
            fig_cost = go.Figure()
            
            fig_cost.add_trace(go.Bar(
                name='Cost per Lead',
                x=updated_eff_df['Channel'],
                y=updated_eff_df['Cost per Lead'],
                marker_color='#3498DB'
            ))
            
            fig_cost.add_trace(go.Bar(
                name='Cost per Customer',
                x=updated_eff_df['Channel'],
                y=updated_eff_df['Cost per Customer'],
                marker_color='#E74C3C'
            ))
            
            fig_cost.update_layout(
                barmode='group',
                template="plotly_white",
                yaxis_title="Cost (Rp)"
            )
            
            st.plotly_chart(fig_cost, use_container_width=True)
        
        with col2:
            st.markdown("### Conversion Rate Comparison")
            
            fig_conv = px.bar(
                updated_eff_df,
                x='Channel',
                y='Conversion Rate',
                color='Conversion Rate',
                color_continuous_scale='Greens',
                title="Lead to Customer Conversion Rate (%)"
            )
            
            fig_conv.update_layout(template="plotly_white")
            
            st.plotly_chart(fig_conv, use_container_width=True)

# ========== TAB 4: ATTRIBUTION ==========
with tab4:
    st.subheader("🎯 Multi-Touch Attribution")
    
    st.markdown("""
    Understand how different channels work together in the customer journey.
    """)
    
    # Attribution model selection
    attribution_model = st.selectbox(
        "Attribution Model",
        ["Last-Touch", "First-Touch", "Linear", "Time-Decay", "Position-Based"]
    )
    
    st.info(f"**{attribution_model} Attribution:** " + {
        "Last-Touch": "100% credit to the final touchpoint before conversion",
        "First-Touch": "100% credit to the first touchpoint",
        "Linear": "Equal credit to all touchpoints",
        "Time-Decay": "More credit to recent touchpoints",
        "Position-Based": "40% to first, 40% to last, 20% to middle touchpoints"
    }[attribution_model])
    
    st.divider()
    
    # Simulated customer journey
    st.markdown("### Sample Customer Journey")
    
    journey_data = pd.DataFrame({
        'Step': [1, 2, 3, 4],
        'Touchpoint': ['Social Media Ad', 'Search (Google)', 'Email Campaign', 'Field Officer Visit'],
        'Channel Type': ['Digital', 'Digital', 'Digital', 'Offline'],
        'Days from First Touch': [0, 3, 7, 14]
    })
    
    st.dataframe(journey_data, use_container_width=True, hide_index=True)
    
    # Calculate attribution
    if attribution_model == "Last-Touch":
        credits = [0, 0, 0, 100]
    elif attribution_model == "First-Touch":
        credits = [100, 0, 0, 0]
    elif attribution_model == "Linear":
        credits = [25, 25, 25, 25]
    elif attribution_model == "Time-Decay":
        credits = [10, 20, 30, 40]
    else:  # Position-Based
        credits = [40, 10, 10, 40]
    
    journey_data['Attribution Credit (%)'] = credits
    
    # Visualization
    fig_attr = px.bar(
        journey_data,
        x='Touchpoint',
        y='Attribution Credit (%)',
        color='Channel Type',
        title=f"{attribution_model} Attribution Model",
        color_discrete_map={'Digital': '#3498DB', 'Offline': '#2ECC71'}
    )
    
    fig_attr.update_layout(template="plotly_white")
    
    st.plotly_chart(fig_attr, use_container_width=True)
    
    # Channel contribution
    st.markdown("### Channel Contribution Summary")
    
    digital_credit = sum([credits[i] for i in range(len(credits)) if journey_data.iloc[i]['Channel Type'] == 'Digital'])
    offline_credit = sum([credits[i] for i in range(len(credits)) if journey_data.iloc[i]['Channel Type'] == 'Offline'])
    
    col1, col2 = st.columns(2)
    col1.metric("Digital Contribution", f"{digital_credit}%")
    col2.metric("Offline Contribution", f"{offline_credit}%")

# ========== TAB 5: OPTIMIZATION ==========
with tab5:
    st.subheader("📈 Budget Optimization")
    
    if 'all_channels' not in st.session_state:
        st.warning("⚠️ Please input channel data in the 'Channel Input' tab first.")
    else:
        st.markdown("### Current Budget Allocation")
        
        # Current allocation
        total_spend = sum([data['spend'] for data in st.session_state.all_channels.values()])
        
        current_allocation = []
        for channel, data in st.session_state.all_channels.items():
            current_allocation.append({
                'Channel': channel,
                'Current Budget': data['spend'],
                'Allocation %': (data['spend'] / total_spend * 100) if total_spend > 0 else 0,
                'Customers': data['customers']
            })
        
        current_df = pd.DataFrame(current_allocation)
        
        st.dataframe(current_df.style.format({
            'Current Budget': 'Rp {:,.0f}',
            'Allocation %': '{:.1f}%',
            'Customers': '{:,.0f}'
        }), use_container_width=True, hide_index=True)
        
        # Pie chart
        fig_pie = px.pie(
            current_df,
            values='Current Budget',
            names='Channel',
            title="Current Budget Distribution"
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.divider()
        
        # Optimization recommendation
        st.markdown("### 🎯 Optimization Recommendations")
        
        # Calculate ROI for each channel
        roi_scores = []
        for channel, data in st.session_state.all_channels.items():
            cac = data['spend'] / data['customers'] if data['customers'] > 0 else float('inf')
            roi = 1 / cac if cac > 0 else 0
            roi_scores.append({'channel': channel, 'roi': roi, 'current_spend': data['spend']})
        
        # Optimize allocation
        optimized_allocations = optimize_budget_allocation(roi_scores, total_spend)
        
        optimization_data = []
        for i, channel_data in enumerate(roi_scores):
            channel = channel_data['channel']
            current = channel_data['current_spend']
            optimized = optimized_allocations[i]
            change = optimized - current
            change_pct = (change / current * 100) if current > 0 else 0
            
            optimization_data.append({
                'Channel': channel,
                'Current': current,
                'Optimized': optimized,
                'Change': change,
                'Change %': change_pct
            })
        
        opt_df = pd.DataFrame(optimization_data)
        
        st.dataframe(opt_df.style.format({
            'Current': 'Rp {:,.0f}',
            'Optimized': 'Rp {:,.0f}',
            'Change': 'Rp {:+,.0f}',
            'Change %': '{:+.1f}%'
        }).background_gradient(subset=['Change %'], cmap='RdYlGn'), use_container_width=True, hide_index=True)
        
        # Recommendations
        st.success("""
        **💡 Optimization Strategy:**
        - Increase budget for high-ROI channels
        - Reduce budget for low-performing channels
        - Test new channels with small budgets
        - Monitor performance weekly
        """)

# ========== TAB 6: SCENARIOS ==========
with tab6:
    st.subheader("🔮 Scenario Planning")
    
    st.markdown("### What-If Analysis")
    
    st.markdown("Simulate different budget scenarios to predict outcomes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scenario_budget = st.number_input(
            "Total Budget (Rp)",
            min_value=0,
            value=100000000,
            step=5000000
        )
        
        digital_allocation = st.slider(
            "Digital Allocation (%)",
            0, 100, 60
        )
    
    with col2:
        st.metric("Digital Budget", f"Rp {scenario_budget * digital_allocation / 100:,.0f}")
        st.metric("Offline Budget", f"Rp {scenario_budget * (100 - digital_allocation) / 100:,.0f}")
    
    # Simulate outcomes
    st.divider()
    
    st.markdown("### Projected Outcomes")
    
    # Simplified projection
    digital_budget = scenario_budget * digital_allocation / 100
    offline_budget = scenario_budget * (100 - digital_allocation) / 100
    
    # Assume average CAC
    avg_digital_cac = 500000
    avg_offline_cac = 650000
    
    projected_digital_customers = int(digital_budget / avg_digital_cac)
    projected_offline_customers = int(offline_budget / avg_offline_cac)
    total_customers = projected_digital_customers + projected_offline_customers
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Digital Customers", f"{projected_digital_customers:,}")
    col2.metric("Offline Customers", f"{projected_offline_customers:,}")
    col3.metric("Total Customers", f"{total_customers:,}")
    
    # Comparison chart
    scenarios = []
    for digital_pct in range(0, 101, 10):
        dig_budget = scenario_budget * digital_pct / 100
        off_budget = scenario_budget * (100 - digital_pct) / 100
        
        dig_customers = int(dig_budget / avg_digital_cac)
        off_customers = int(off_budget / avg_offline_cac)
        
        scenarios.append({
            'Digital %': digital_pct,
            'Total Customers': dig_customers + off_customers
        })
    
    scenario_df = pd.DataFrame(scenarios)
    
    fig_scenario = px.line(
        scenario_df,
        x='Digital %',
        y='Total Customers',
        title="Total Customer Acquisition by Digital Allocation",
        markers=True
    )
    
    # Mark current selection
    fig_scenario.add_vline(
        x=digital_allocation,
        line_dash="dash",
        line_color="red",
        annotation_text="Current Selection"
    )
    
    fig_scenario.update_layout(template="plotly_white")
    
    st.plotly_chart(fig_scenario, use_container_width=True)

# ========== FOOTER ==========
st.divider()
st.caption("💡 **Pro Tip:** Optimize your channel mix quarterly based on performance data and market conditions!")
