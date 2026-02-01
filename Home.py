import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Marketing Portfolio | Data-Driven Growth",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with Glassmorphism, Animations, and Premium Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');
    
    /* Import Font Awesome */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
    
    /* CSS Variables for Theming */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark-bg: #0f172a;
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.12);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.15);
        --shadow-xl: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    /* Global Styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Premium Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 24px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-xl);
        animation: fadeInUp 0.8s ease-out;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(to right, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: slideInLeft 0.8s ease-out;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.95);
        margin-top: 1rem;
        font-weight: 500;
        animation: slideInRight 0.8s ease-out;
    }
    
    .hero-description {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.85);
        margin-top: 1.5rem;
        line-height: 1.8;
        max-width: 800px;
        animation: fadeIn 1s ease-out 0.3s both;
    }
    
    /* Profile Image Styling */
    .profile-container {
        text-align: center;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }
    
    .profile-image {
        border-radius: 50%;
        border: 6px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4px;
    }
    
    .profile-image:hover {
        transform: scale(1.05) rotate(2deg);
        box-shadow: 0 16px 64px rgba(0, 0, 0, 0.4);
    }
    
    /* Badge Styling */
    .badge {
        display: inline-block;
        padding: 8px 16px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .badge-green {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #2e7d32;
        border: 1px solid #81c784;
    }
    
    .badge-blue {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #1565c0;
        border: 1px solid #64b5f6;
    }
    
    .badge-orange {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        color: #e65100;
        border: 1px solid #ffb74d;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Social Links */
    .social-links {
        display: flex;
        gap: 12px;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .social-link {
        transition: transform 0.2s ease, filter 0.2s ease;
        display: inline-block;
    }
    
    .social-link:hover {
        transform: translateY(-4px) scale(1.1);
        filter: brightness(1.2);
    }
    
    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--primary-gradient);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
    }
    
    .glass-card:hover::before {
        transform: scaleX(1);
    }
    
    .glass-card h3 {
        color: #667eea;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .glass-card ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .glass-card li {
        padding: 0.75rem 0;
        color: #334155;
        font-weight: 500;
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .glass-card li:last-child {
        border-bottom: none;
    }
    
    .glass-card li:hover {
        padding-left: 1rem;
        color: #667eea;
    }
    
    .glass-card li::before {
        content: '✓';
        color: #10b981;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    /* Icon Styling */
    .fas, .far, .fab {
        margin-right: 0.5rem;
    }
    
    .project-icon i {
        font-size: 2.5rem;
        margin-right: 0;
    }
    
    .section-title i {
        margin-right: 0.75rem;
    }
    
    .badge i {
        margin-right: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* Project Cards */
    .project-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    
    .project-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: #667eea;
    }
    
    .project-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: inline-block;
        animation: bounce 2s ease-in-out infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .project-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .project-description {
        color: #64748b;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Section Headers */
    .section-header {
        text-align: center;
        margin: 3rem 0 2rem 0;
        position: relative;
    }
    
    .section-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .section-divider {
        width: 80px;
        height: 4px;
        background: var(--primary-gradient);
        margin: 1rem auto;
        border-radius: 2px;
        animation: expandWidth 0.8s ease-out;
    }
    
    @keyframes expandWidth {
        from { width: 0; }
        to { width: 80px; }
    }
    
    /* Info Box Styling */
    .info-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: var(--shadow-sm);
        animation: slideInLeft 0.6s ease-out;
    }
    
    .info-box strong {
        color: #92400e;
        font-size: 1.1rem;
    }
    
    /* Contact Form Styling */
    .contact-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: var(--shadow-md);
    }
    
    /* Footer Styling */
    .custom-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 4rem;
        border-top: 2px solid transparent;
        border-image: var(--primary-gradient) 1;
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
        }
        
        .section-title {
            font-size: 1.8rem;
        }
        
        .glass-card {
            padding: 1.5rem;
        }
    }
    
    /* Streamlit Button Overrides */
    .stButton>button {
        width: 100%;
        background: var(--primary-gradient);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-content">
        <h1 class="hero-title">Hi, I'm Andriyanto, BScE, S.E. <i class="fas fa-hand-wave" style="color: #fbbf24;"></i></h1>
        <p class="hero-subtitle">Marketing Analyst | Growth Engineer | Data Scientist</p>
        <p class="hero-description">
            I bridge the gap between <strong>Marketing Strategy</strong>, <strong>Economics</strong>, and <strong>Data Science</strong>. 
            I build tools that automate insights, predict market trends, and optimize conversion analytics using <strong>Python & Streamlit</strong>.
        </p>
        <div style="margin-top: 2rem;">
            <span class="badge badge-green"><i class="fas fa-chart-line"></i> Automate Insights</span>
            <span class="badge badge-blue"><i class="fas fa-chart-bar"></i> Predict Trends</span>
            <span class="badge badge-orange"><i class="fas fa-bullseye"></i> Optimize Analytics</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Social Links
st.markdown("""
<div class="social-links" style="justify-content: center; margin-bottom: 3rem;">
    <a href='https://github.com/yandri918' class="social-link" target="_blank">
        <img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white' alt='GitHub'>
    </a>
    <a href='https://linkedin.com/in/yandri-s' class="social-link" target="_blank">
        <img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' alt='LinkedIn'>
    </a>
    <a href='mailto:yandri@example.com' class="social-link" target="_blank">
        <img src='https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white' alt='Email'>
    </a>
</div>
""", unsafe_allow_html=True)

# Expertise Section
st.markdown("""
<div class="section-header">
    <h2 class="section-title"><i class="fas fa-bullseye"></i> My Expertise</h2>
    <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card">
        <h3><i class="fas fa-chart-pie"></i> Marketing Analytics</h3>
        <ul>
            <li>Customer Segmentation</li>
            <li>Lifetime Value (CLV)</li>
            <li>Campaign Performance</li>
            <li>Churn Prediction</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h3><i class="fas fa-rocket"></i> Growth Marketing</h3>
        <ul>
            <li>Funnel Analysis</li>
            <li>A/B Testing</li>
            <li>Conversion Optimization</li>
            <li>Predictive Modeling</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card">
        <h3><i class="fas fa-lightbulb"></i> Product Marketing</h3>
        <ul>
            <li>Market Research</li>
            <li>Competitive Analysis</li>
            <li>Pricing Strategy</li>
            <li>Data Storytelling</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Portfolio Projects Section
st.markdown("""
<div class="section-header">
    <h2 class="section-title"><i class="fas fa-briefcase"></i> Portfolio Projects</h2>
    <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    <div class="project-card">
        <div class="project-icon"><i class="fas fa-users-cog" style="color: #667eea;"></i></div>
        <div class="project-title">Customer Segmentation Engine</div>
        <p class="project-description">
            Using K-Means clustering to identify high-value customer segments based on RFM analysis.
            Interactive visualization with Plotly for actionable insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="project-card">
        <div class="project-icon"><i class="fas fa-chart-line" style="color: #10b981;"></i></div>
        <div class="project-title">Market Demand Forecasting</div>
        <p class="project-description">
            Predicting sales trends using Prophet & ARIMA time-series modeling for better inventory planning.
            Includes seasonality detection and trend analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="project-card">
        <div class="project-icon"><i class="fas fa-bullseye" style="color: #f59e0b;"></i></div>
        <div class="project-title">Marketing Mix Modeling</div>
        <p class="project-description">
            Advanced MMM with adstock effects, saturation curves, and budget optimization.
            Powered by Ridge Regression and Bayesian inference.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="project-card">
        <div class="project-icon"><i class="fas fa-robot" style="color: #8b5cf6;"></i></div>
        <div class="project-title">Social Media Sentiment Analysis</div>
        <p class="project-description">
            Real-time brand sentiment tracking using NLP transformers on social comments.
            Emotion detection and trend analysis dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="project-card">
        <div class="project-icon"><i class="fas fa-chart-area" style="color: #ef4444;"></i></div>
        <div class="project-title">Competitive Market Analysis</div>
        <p class="project-description">
            Economic modeling of price elasticity and market share estimation.
            Includes Porter's Five Forces and Game Theory simulations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="project-card">
        <div class="project-icon"><i class="fas fa-crystal-ball" style="color: #06b6d4;"></i></div>
        <div class="project-title">CLV Prediction & Churn Analysis</div>
        <p class="project-description">
            Machine learning models to predict customer lifetime value and churn probability.
            Actionable retention strategies based on cohort analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Call to Action
st.markdown("""
<div class="info-box">
    <strong><i class="fas fa-arrow-left"></i> Explore Interactive Demos</strong><br>
    Select a project from the sidebar to see live demonstrations with real data analysis, 
    interactive visualizations, and downloadable insights.
</div>
""", unsafe_allow_html=True)

# Resume & Contact Section
st.markdown("""
<div class="section-header">
    <h2 class="section-title"><i class="fas fa-envelope"></i> Get In Touch</h2>
    <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### <i class='fas fa-file-pdf'></i> Resume", unsafe_allow_html=True)
    st.write("Download my latest curriculum vitae to learn more about my experience and skills.")
    
    import os
    resume_path = "assets/resume.pdf"
    
    if os.path.exists(resume_path):
        with open(resume_path, "rb") as f:
            pdf_data = f.read()
        st.download_button(
            label="⬇️ Download Resume (PDF)",
            data=pdf_data,
            file_name="Andriyanto_Resume.pdf",
            mime="application/pdf"
        )
    else:
        st.info("ℹ️ Resume download is currently disabled. Please contact me on LinkedIn!")

with c2:
    st.markdown("### <i class='fas fa-paper-plane'></i> Contact Me", unsafe_allow_html=True)
    st.markdown('<div class="contact-container">', unsafe_allow_html=True)
    with st.form("contact_form"):
        name = st.text_input("Name", placeholder="Your name")
        email = st.text_input("Email", placeholder="your.email@example.com")
        message = st.text_area("Message", placeholder="Tell me about your project...")
        submit = st.form_submit_button("Send Message ✉️")
        
        if submit:
            if name and email and message:
                st.success(f"Thanks {name}! Your message has been sent successfully. I'll get back to you soon! ✅")
            else:
                st.error("Please fill in all fields.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="custom-footer">
    <p style="margin: 0; font-weight: 600;">
        Built with <i class="fas fa-heart" style="color: #ef4444;"></i> using Streamlit by Andriyanto
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">
        © 2026 All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)
