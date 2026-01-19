import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as optimize

st.set_page_config(page_title="Advanced Competitive Analysis | Economics", page_icon="⚖️", layout="wide")

st.title("⚖️ Advanced Competitive Market Analysis")
st.markdown("MBA-level economic modeling with **Game Theory**, **Porter's Five Forces**, and **Strategic Positioning**.")

# ========== HELPER FUNCTIONS ==========

def calculate_hhi(market_shares):
    """Calculate Herfindahl-Hirschman Index"""
    return sum([s**2 for s in market_shares])

def calculate_nash_equilibrium_prices(mc_a, mc_b, a, b):
    """Calculate Nash Equilibrium for Bertrand competition"""
    # Simplified Bertrand model
    p_a = (a + 2*mc_a + mc_b) / 3
    p_b = (a + mc_a + 2*mc_b) / 3
    return p_a, p_b

def optimal_price_elasticity(mc, elasticity):
    """Calculate optimal price given marginal cost and elasticity"""
    if elasticity >= -1:
        return None  # No optimal price for inelastic demand
    return mc / (1 + 1/elasticity)

def porter_score_to_attractiveness(score):
    """Convert Porter's score to industry attractiveness"""
    if score <= 10:
        return "Very Unattractive", "🔴"
    elif score <= 15:
        return "Unattractive", "🟠"
    elif score <= 20:
        return "Moderate", "🟡"
    elif score <= 23:
        return "Attractive", "🟢"
    else:
        return "Very Attractive", "🟢"

# ========== SESSION STATE ==========
if 'competitors_data' not in st.session_state:
    st.session_state.competitors_data = pd.DataFrame({
        'Company': ['Your Brand', 'Competitor A', 'Competitor B', 'Competitor C'],
        'Market Share': [25.0, 30.0, 20.0, 25.0],
        'Price': [150000, 140000, 160000, 145000],
        'Quality Score': [8, 7, 9, 7],
        'Innovation Score': [7, 6, 8, 6]
    })

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Configuration")

analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Price Elasticity", "Game Theory", "Porter's Forces", "All"]
)

st.sidebar.divider()

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Economic Models",
    "🎮 Game Theory",
    "🏭 Porter's Forces",
    "📍 Positioning",
    "📈 Market Structure",
    "🎯 Strategy & SWOT"
])

# ========== TAB 1: ECONOMIC MODELS ==========
with tab1:
    st.subheader("📊 Advanced Price Elasticity & Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Price Elasticity Model")
        
        base_price = st.number_input("Your Base Price (Rp)", min_value=10000, value=150000, step=5000)
        own_elasticity = st.slider("Own-Price Elasticity", -3.0, -0.1, -1.5, 0.1)
        
        st.info(f"**Interpretation:** {abs(own_elasticity):.1f} means a 1% price increase causes {abs(own_elasticity):.1f}% demand decrease")
        
        # Cross-price elasticity
        st.markdown("### Cross-Price Elasticity")
        competitor_price = st.number_input("Competitor Price (Rp)", min_value=10000, value=140000, step=5000)
        cross_elasticity = st.slider("Cross-Price Elasticity", 0.0, 3.0, 0.8, 0.1)
        
        st.info(f"**Interpretation:** {cross_elasticity:.1f} means a 1% competitor price increase causes {cross_elasticity:.1f}% increase in your demand")
    
    with col2:
        st.markdown("### Demand Simulation")
        
        # Generate demand curves
        prices = np.linspace(base_price * 0.5, base_price * 1.5, 100)
        base_demand = 1000
        
        # Own demand (affected by own price and competitor price)
        demands = base_demand * (prices / base_price) ** own_elasticity * (competitor_price / base_price) ** cross_elasticity
        revenues = prices * demands
        
        # Find optimal price
        optimal_idx = revenues.argmax()
        optimal_price = prices[optimal_idx]
        optimal_revenue = revenues[optimal_idx]
        
        st.metric("Optimal Price", f"Rp {optimal_price:,.0f}")
        st.metric("Expected Revenue", f"Rp {optimal_revenue:,.0f}")
        st.metric("Expected Demand", f"{demands[optimal_idx]:.0f} units")
    
    st.divider()
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Demand Curve")
        
        fig_demand = go.Figure()
        
        fig_demand.add_trace(go.Scatter(
            x=prices, y=demands,
            mode='lines',
            name='Demand',
            line=dict(color='#3498DB', width=2)
        ))
        
        fig_demand.add_trace(go.Scatter(
            x=[base_price], y=[base_demand],
            mode='markers',
            name='Current',
            marker=dict(size=12, color='red')
        ))
        
        fig_demand.add_trace(go.Scatter(
            x=[optimal_price], y=[demands[optimal_idx]],
            mode='markers',
            name='Optimal',
            marker=dict(size=12, color='green', symbol='star')
        ))
        
        fig_demand.update_layout(
            xaxis_title="Price (Rp)",
            yaxis_title="Demand (units)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_demand, use_container_width=True)
    
    with col2:
        st.markdown("### Revenue Optimization")
        
        fig_revenue = go.Figure()
        
        fig_revenue.add_trace(go.Scatter(
            x=prices, y=revenues,
            mode='lines',
            name='Revenue',
            line=dict(color='#2ECC71', width=2),
            fill='tozeroy'
        ))
        
        fig_revenue.add_trace(go.Scatter(
            x=[optimal_price], y=[optimal_revenue],
            mode='markers+text',
            name='Maximum',
            marker=dict(size=15, color='gold', symbol='star'),
            text=[f"Max: Rp {optimal_revenue/1e6:.1f}M"],
            textposition="top center"
        ))
        
        fig_revenue.update_layout(
            xaxis_title="Price (Rp)",
            yaxis_title="Revenue (Rp)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_revenue, use_container_width=True)

# ========== TAB 2: GAME THEORY ==========
with tab2:
    st.subheader("🎮 Game Theory & Strategic Pricing")
    
    st.markdown("""
    Analyze strategic interactions with competitors using game theory models.
    """)
    
    # Nash Equilibrium
    st.markdown("### Nash Equilibrium - Bertrand Competition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Your Company")
        your_mc = st.number_input("Your Marginal Cost (Rp)", min_value=10000, value=80000, step=5000)
        
    with col2:
        st.markdown("#### Competitor")
        comp_mc = st.number_input("Competitor Marginal Cost (Rp)", min_value=10000, value=75000, step=5000)
    
    # Market parameters
    market_size = st.slider("Market Size Parameter (a)", 100000, 500000, 300000, 10000)
    
    # Calculate Nash Equilibrium
    nash_price_you, nash_price_comp = calculate_nash_equilibrium_prices(your_mc, comp_mc, market_size, 1)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Your Nash Price", f"Rp {nash_price_you:,.0f}")
    col2.metric("Competitor Nash Price", f"Rp {nash_price_comp:,.0f}")
    col3.metric("Price Difference", f"Rp {abs(nash_price_you - nash_price_comp):,.0f}")
    
    if nash_price_you < nash_price_comp:
        st.success("✅ **You have cost advantage!** You can price lower and capture more market share.")
    elif nash_price_you > nash_price_comp:
        st.warning("⚠️ **Competitor has cost advantage.** Consider differentiation or cost reduction.")
    else:
        st.info("ℹ️ **Equal equilibrium prices.** Market will be split based on other factors.")
    
    st.divider()
    
    # Payoff Matrix
    st.markdown("### Strategic Payoff Matrix")
    
    st.markdown("**Scenario:** Price War vs Premium Pricing")
    
    # Create payoff matrix
    payoff_data = {
        'Your Strategy': ['Low Price', 'High Price'],
        'Comp Low → Your Profit': ['Rp 10M', 'Rp 5M'],
        'Comp Low → Comp Profit': ['Rp 10M', 'Rp 30M'],
        'Comp High → Your Profit': ['Rp 30M', 'Rp 20M'],
        'Comp High → Comp Profit': ['Rp 5M', 'Rp 20M']
    }
    
    payoff_df = pd.DataFrame(payoff_data)
    
    st.dataframe(payoff_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Analysis:**
    - **(Low, Low)** = Nash Equilibrium (Prisoner's Dilemma)
    - Both would be better at (High, High) but Low Price is dominant strategy
    - **Recommendation:** Seek differentiation to escape price competition
    """)
    
    # Visualization
    fig_payoff = go.Figure()
    
    strategies = ['Low Price', 'High Price']
    your_payoffs_comp_low = [10, 5]
    your_payoffs_comp_high = [30, 20]
    
    fig_payoff.add_trace(go.Bar(
        name='Competitor Prices Low',
        x=strategies,
        y=your_payoffs_comp_low,
        marker_color='#E74C3C'
    ))
    
    fig_payoff.add_trace(go.Bar(
        name='Competitor Prices High',
        x=strategies,
        y=your_payoffs_comp_high,
        marker_color='#2ECC71'
    ))
    
    fig_payoff.update_layout(
        title="Your Profit by Strategy (Rp Millions)",
        xaxis_title="Your Strategy",
        yaxis_title="Your Profit (Rp M)",
        barmode='group',
        template="plotly_white"
    )
    
    st.plotly_chart(fig_payoff, use_container_width=True)

# ========== TAB 3: PORTER'S FORCES ==========
with tab3:
    st.subheader("🏭 Porter's Five Forces Analysis")
    
    st.markdown("""
    Evaluate industry attractiveness and competitive intensity using Porter's framework.
    Rate each force from 1 (Low) to 5 (High).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Rate Each Force")
        
        threat_new = st.slider("1️⃣ Threat of New Entrants", 1, 5, 3,
                               help="Barriers to entry: capital requirements, regulations, brand loyalty")
        
        supplier_power = st.slider("2️⃣ Bargaining Power of Suppliers", 1, 5, 3,
                                   help="Supplier concentration, switching costs, uniqueness")
        
        buyer_power = st.slider("3️⃣ Bargaining Power of Buyers", 1, 5, 3,
                                help="Buyer concentration, price sensitivity, switching costs")
        
        threat_substitutes = st.slider("4️⃣ Threat of Substitutes", 1, 5, 3,
                                       help="Alternative products, price-performance, switching ease")
        
        competitive_rivalry = st.slider("5️⃣ Competitive Rivalry", 1, 5, 4,
                                       help="Number of competitors, industry growth, exit barriers")
    
    with col2:
        st.markdown("### Industry Attractiveness")
        
        # Calculate total score (inverse for attractiveness)
        total_threat = threat_new + supplier_power + buyer_power + threat_substitutes + competitive_rivalry
        attractiveness_score = 30 - total_threat  # Max 25, Min 5
        
        attractiveness, emoji = porter_score_to_attractiveness(attractiveness_score)
        
        st.metric("Total Force Score", total_threat, help="Lower is better for profitability")
        st.metric("Industry Attractiveness", f"{attractiveness} {emoji}")
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attractiveness_score,
            title={'text': "Attractiveness Score"},
            gauge={
                'axis': {'range': [0, 25]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 10], 'color': "#E74C3C"},
                    {'range': [10, 15], 'color': "#F39C12"},
                    {'range': [15, 20], 'color': "#F1C40F"},
                    {'range': [20, 25], 'color': "#2ECC71"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.divider()
    
    # Radar chart
    st.markdown("### Forces Visualization")
    
    forces_data = pd.DataFrame({
        'Force': ['New Entrants', 'Suppliers', 'Buyers', 'Substitutes', 'Rivalry'],
        'Score': [threat_new, supplier_power, buyer_power, threat_substitutes, competitive_rivalry]
    })
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=forces_data['Score'].tolist() + [forces_data['Score'].iloc[0]],
        theta=forces_data['Force'].tolist() + [forces_data['Force'].iloc[0]],
        fill='toself',
        name='Force Intensity',
        line_color='#3498DB'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5])
        ),
        showlegend=False,
        title="Porter's Five Forces Radar",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Strategic recommendations
    st.markdown("### Strategic Recommendations")
    
    if attractiveness_score >= 20:
        st.success("""
        ✅ **Highly Attractive Industry**
        - Invest aggressively in market share
        - Focus on differentiation
        - Build barriers to entry
        """)
    elif attractiveness_score >= 15:
        st.info("""
        ℹ️ **Moderately Attractive Industry**
        - Selective investment
        - Focus on cost leadership or niche
        - Monitor competitive dynamics
        """)
    else:
        st.warning("""
        ⚠️ **Unattractive Industry**
        - Consider exit or repositioning
        - Focus on cash generation
        - Explore adjacent markets
        """)

# ========== TAB 4: POSITIONING ==========
with tab4:
    st.subheader("📍 Competitive Positioning Map")
    
    st.markdown("Visualize your position relative to competitors on key dimensions.")
    
    # Editable competitors data
    st.markdown("### Competitor Data (Editable)")
    
    edited_competitors = st.data_editor(
        st.session_state.competitors_data,
        key="competitors_editor",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Market Share": st.column_config.NumberColumn("Market Share (%)", min_value=0, max_value=100, step=0.1),
            "Price": st.column_config.NumberColumn("Price (Rp)", min_value=0, step=1000),
            "Quality Score": st.column_config.NumberColumn("Quality (1-10)", min_value=1, max_value=10),
            "Innovation Score": st.column_config.NumberColumn("Innovation (1-10)", min_value=1, max_value=10)
        }
    )
    
    # Update session state
    if not edited_competitors.equals(st.session_state.competitors_data):
        st.session_state.competitors_data = edited_competitors
    
    st.divider()
    
    # Positioning map
    st.markdown("### Price vs Quality Positioning")
    
    fig_position = px.scatter(
        edited_competitors,
        x='Price',
        y='Quality Score',
        size='Market Share',
        color='Company',
        text='Company',
        title="Competitive Positioning Map",
        labels={'Price': 'Price (Rp)', 'Quality Score': 'Quality Score (1-10)'},
        size_max=60
    )
    
    fig_position.update_traces(textposition='top center')
    fig_position.update_layout(template="plotly_white", height=500)
    
    st.plotly_chart(fig_position, use_container_width=True)
    
    # Strategic quadrants
    avg_price = edited_competitors['Price'].mean()
    avg_quality = edited_competitors['Quality Score'].mean()
    
    fig_position.add_hline(y=avg_quality, line_dash="dash", line_color="gray", annotation_text="Avg Quality")
    fig_position.add_vline(x=avg_price, line_dash="dash", line_color="gray", annotation_text="Avg Price")
    
    st.plotly_chart(fig_position, use_container_width=True)
    
    # Gap analysis
    st.markdown("### Market Gap Analysis")
    
    your_data = edited_competitors[edited_competitors['Company'] == 'Your Brand'].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Your Position", "Current")
    col2.metric("Price vs Avg", f"{((your_data['Price'] / avg_price - 1) * 100):+.1f}%")
    col3.metric("Quality vs Avg", f"{(your_data['Quality Score'] - avg_quality):+.1f}")
    
    # Recommendations
    if your_data['Price'] > avg_price and your_data['Quality Score'] > avg_quality:
        st.success("✅ **Premium Positioning** - High price, high quality. Maintain differentiation.")
    elif your_data['Price'] < avg_price and your_data['Quality Score'] < avg_quality:
        st.info("ℹ️ **Budget Positioning** - Low price, acceptable quality. Focus on cost leadership.")
    elif your_data['Price'] > avg_price and your_data['Quality Score'] < avg_quality:
        st.error("❌ **Unfavorable Position** - High price, low quality. Urgent repositioning needed!")
    else:
        st.success("✅ **Value Positioning** - Good quality, competitive price. Strong position!")

# ========== TAB 5: MARKET STRUCTURE ==========
with tab5:
    st.subheader("📈 Market Structure Analysis")
    
    st.markdown("Analyze market concentration and competitive dynamics.")
    
    # Calculate metrics
    market_shares = edited_competitors['Market Share'].values
    
    # HHI
    hhi = calculate_hhi(market_shares)
    
    # CR4 (top 4 concentration)
    top_4_shares = sorted(market_shares, reverse=True)[:4]
    cr4 = sum(top_4_shares)
    
    # Lerner Index (simplified)
    avg_price = edited_competitors['Price'].mean()
    assumed_mc = avg_price * 0.6  # Assume 40% margin
    lerner = (avg_price - assumed_mc) / avg_price
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("HHI", f"{hhi:.0f}", help="Herfindahl-Hirschman Index")
    col2.metric("CR4", f"{cr4:.1f}%", help="Top 4 Concentration Ratio")
    col3.metric("Lerner Index", f"{lerner:.2f}", help="Market Power (0-1)")
    
    st.divider()
    
    # Interpret HHI
    st.markdown("### Market Structure Classification")
    
    if hhi < 1500:
        structure = "Competitive"
        color = "🟢"
        desc = "Low concentration, many competitors, high competition"
    elif hhi < 2500:
        structure = "Moderately Concentrated"
        color = "🟡"
        desc = "Medium concentration, oligopolistic tendencies"
    else:
        structure = "Highly Concentrated"
        color = "🔴"
        desc = "High concentration, potential for market power"
    
    st.info(f"{color} **{structure}** - {desc}")
    
    # Market share pie chart
    st.markdown("### Market Share Distribution")
    
    fig_share = px.pie(
        edited_competitors,
        values='Market Share',
        names='Company',
        title="Market Share Breakdown",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig_share, use_container_width=True)
    
    # Concentration curve
    st.markdown("### Concentration Curve")
    
    sorted_shares = sorted(market_shares, reverse=True)
    cumulative_shares = np.cumsum(sorted_shares)
    
    fig_conc = go.Figure()
    
    fig_conc.add_trace(go.Scatter(
        x=list(range(1, len(sorted_shares) + 1)),
        y=cumulative_shares,
        mode='lines+markers',
        name='Cumulative Share',
        line=dict(color='#3498DB', width=3)
    ))
    
    fig_conc.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% Market")
    
    fig_conc.update_layout(
        title="Cumulative Market Share Concentration",
        xaxis_title="Number of Firms",
        yaxis_title="Cumulative Market Share (%)",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_conc, use_container_width=True)

# ========== TAB 6: STRATEGY & SWOT ==========
with tab6:
    st.subheader("🎯 Strategic Analysis & SWOT")
    
    st.markdown("### SWOT Analysis Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Strengths (Internal Positive)")
        strengths = st.text_area("List your strengths", 
                                 "- Strong brand recognition\n- Cost advantage\n- Innovative products",
                                 height=150)
        
        st.markdown("#### ⚠️ Weaknesses (Internal Negative)")
        weaknesses = st.text_area("List your weaknesses",
                                  "- Limited distribution\n- High employee turnover\n- Outdated technology",
                                  height=150)
    
    with col2:
        st.markdown("#### 🎯 Opportunities (External Positive)")
        opportunities = st.text_area("List market opportunities",
                                     "- Growing market demand\n- New technology adoption\n- Competitor weakness",
                                     height=150)
        
        st.markdown("#### 🚨 Threats (External Negative)")
        threats = st.text_area("List external threats",
                              "- New competitors\n- Regulatory changes\n- Economic downturn",
                              height=150)
    
    st.divider()
    
    # Strategic recommendations
    st.markdown("### Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **SO Strategies (Strength-Opportunity)**
        - Leverage strengths to capitalize on opportunities
        - Aggressive growth strategies
        - Market expansion
        """)
        
        st.warning("""
        **ST Strategies (Strength-Threat)**
        - Use strengths to mitigate threats
        - Defensive strategies
        - Competitive positioning
        """)
    
    with col2:
        st.info("""
        **WO Strategies (Weakness-Opportunity)**
        - Overcome weaknesses to pursue opportunities
        - Development strategies
        - Capability building
        """)
        
        st.error("""
        **WT Strategies (Weakness-Threat)**
        - Minimize weaknesses and avoid threats
        - Defensive/retrenchment strategies
        - Risk mitigation
        """)
    
    st.divider()
    
    # Summary metrics
    st.markdown("### Strategic Position Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Market Position", f"#{edited_competitors[edited_competitors['Company']=='Your Brand'].index[0] + 1}")
    col2.metric("Market Share", f"{your_data['Market Share']:.1f}%")
    col3.metric("Price Position", "Premium" if your_data['Price'] > avg_price else "Value")
    col4.metric("Quality Position", "High" if your_data['Quality Score'] > avg_quality else "Standard")

# ========== FOOTER ==========
st.divider()
st.caption("💡 **Pro Tip:** Use game theory to anticipate competitor moves and Porter's Forces to assess long-term industry profitability!")
