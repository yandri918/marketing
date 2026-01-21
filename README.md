# 📊 Marketing Analytics Portfolio

**Enterprise-Grade Marketing Analytics Platform** built with Streamlit, featuring advanced MMM, AI-powered insights, and comprehensive web analytics.

---

## 🚀 Features Overview

### 📈 **Module 1-7: Core Marketing Analytics**
- Customer Segmentation (RFM Analysis)
- Market Basket Analysis
- Churn Prediction
- Sentiment Analysis
- Campaign Performance Tracking
- A/B Testing Framework
- Competitive Analysis

### 🎯 **Module 8: Advanced Marketing Mix Modeling (MMM)**
Enterprise-grade MMM with:
- **Bayesian Hierarchical Modeling** (PyMC)
- **Multi-objective Optimization** (Pymoo)
- **SHAP Explainability**
- **Prophet Seasonality** decomposition
- **Saturation & Adstock** effects
- **Budget Allocation** optimizer
- **What-If Scenarios**

### 👥 **Module 9: Cohort Analysis**
- Retention heatmaps
- User lifecycle tracking
- Engagement metrics
- Altair visualizations

### 📄 **Module 10: Executive Report Generator**
- PDF report generation (fpdf2)
- Automated insights
- KPI summaries
- Visual dashboards

### 🎯 **Module 11: MMM Optimizer**
- Advanced saturation curves
- Adstock modeling
- Budget optimization
- ROI maximization

### 🤖 **Module 12: Product Recommender**
- Collaborative filtering
- User similarity matrix
- Interactive Altair heatmaps
- Personalized recommendations

### 📊 **Module 13: Web Analytics Dashboard**
**Google Analytics-Style** comprehensive analytics:
- **5 Main Tabs:**
  - 📈 Overview - Real-time metrics & trends
  - 👥 Audience - Demographics & behavior
  - 🔍 Acquisition - Traffic sources
  - 🎯 Behavior - User engagement
  - 💰 Conversions - Goal tracking

- **Data Input Options:**
  - 🤖 AI-Generated (synthetic data)
  - ✍️ Manual Input (customizable metrics)
  - 📤 CSV Upload (real data)

- **Configurable Metrics:**
  - Sessions, Users, Pageviews
  - Avg Duration, Bounce Rate
  - Conversion Rate & Value
  - Revenue tracking

### 🚀 **Module 14: Advanced Analytics Dashboard**
**Enterprise-Grade GA4 Style** with AI insights:

#### **7 Professional Tabs:**

1. **📈 Real-Time Analytics**
   - Active users (last hour)
   - Pageviews per minute
   - Live traffic visualization
   - Top active pages
   - Real-time revenue

2. **🎯 User Journey & Behavior Flow**
   - Landing page performance
   - Conversion by page
   - Engagement distribution (Low/Medium/High/Very High)
   - Session duration analysis

3. **🔄 Funnel Analysis**
   - 4-stage conversion funnel
   - Drop-off rate calculation
   - Time to conversion
   - Funnel visualization

4. **👥 Cohort Retention**
   - Retention heatmap (14 days)
   - Average retention curve
   - User behavior tracking
   - Cohort comparison

5. **💰 Revenue & E-commerce Analytics**
   - Total revenue, orders, AOV
   - Revenue per session
   - Revenue by source & country
   - Revenue trend analysis

6. **🤖 AI-Powered Insights**
   - 7-day traffic forecast
   - Predictive analytics
   - Automated insights:
     - Best performing sources
     - Peak traffic hours
     - High-value user identification
     - Mobile traffic trends

7. **📊 Custom Reports**
   - Build custom reports
   - Select metrics & dimensions
   - CSV export
   - Formatted data tables

#### **Advanced Features:**
- ✅ Period comparison (vs previous period)
- ✅ Advanced segmentation filters
- ✅ Traffic source filtering
- ✅ Gradient UI with professional styling
- ✅ Engagement scoring (0-100)
- ✅ Event tracking
- ✅ Revenue analytics
- ✅ Predictive forecasting

---

## 🛠️ Technology Stack

### **Core Framework**
- **Streamlit** - Interactive web applications
- **Python 3.13** - Latest Python features

### **Data Science & ML**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
- **lifetimes** - CLV prediction

### **Visualization**
- **Altair** - Interactive charts (primary)
- **Plotly** - 3D visualizations
- **matplotlib** - Statistical plots

### **Advanced Analytics**
- **PyMC** - Bayesian modeling
- **Arviz** - Bayesian visualization
- **Pymoo** - Multi-objective optimization
- **SHAP** - Model explainability
- **Prophet** - Time series forecasting

### **Utilities**
- **fpdf2** - PDF generation
- **joblib** - Model persistence

---

## 📦 Installation

### **1. Clone Repository**
```bash
git clone https://github.com/yandri918/marketing.git
cd marketing
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run Application**
```bash
streamlit run Home.py
```

---

## 📋 Requirements

```txt
# Core
streamlit>=1.31.0
pandas>=2.1.0
numpy>=1.24.0
plotly>=5.18.0

# Machine Learning
scikit-learn>=1.4.0
scipy>=1.12.0

# Bayesian Modeling
pymc>=5.10.0
arviz>=0.17.0
pytensor>=2.18.0

# Optimization
pymoo>=0.6.1.1
optuna>=3.5.0

# Explainability
shap>=0.44.0

# Time Series
prophet>=1.1.5

# Utilities
joblib>=1.3.0
altair>=5.0.0
lifetimes>=0.11.0
fpdf2>=2.7.0
matplotlib>=3.7.0
```

---

## 🎯 Use Cases

### **For Marketing Teams**
- Optimize marketing budget allocation
- Identify high-value customer segments
- Track campaign performance
- Predict customer churn
- Analyze customer lifetime value

### **For Data Scientists**
- Advanced MMM with Bayesian inference
- Multi-objective optimization
- SHAP-based model interpretation
- Time series forecasting
- A/B testing analysis

### **For Business Analysts**
- Executive dashboards
- KPI tracking
- Cohort analysis
- Revenue analytics
- Funnel optimization

### **For Product Managers**
- User behavior analysis
- Feature adoption tracking
- Conversion optimization
- Retention analysis
- Product recommendations

---

## 📊 Key Metrics Tracked

### **Acquisition Metrics**
- Traffic sources
- Campaign performance
- Cost per acquisition (CPA)
- Return on ad spend (ROAS)

### **Engagement Metrics**
- Session duration
- Pages per session
- Bounce rate
- Engagement score (0-100)

### **Retention Metrics**
- Cohort retention rates
- Churn prediction
- Customer lifetime value (CLV)
- Repeat purchase rate

### **Revenue Metrics**
- Total revenue
- Average order value (AOV)
- Revenue per session
- Conversion rate
- Revenue by source/country

---

## 🔧 Configuration

### **Data Input Modes**

#### **1. AI-Generated (Default)**
- Realistic synthetic data
- Configurable parameters
- Instant visualization

#### **2. Manual Input**
- Custom metric values
- Session, user, pageview counts
- Conversion & revenue settings

#### **3. CSV Upload**
Required columns:
```csv
timestamp,source,device,pageviews,duration,bounce,conversion,revenue
```

---

## 🎨 UI/UX Features

### **Professional Design**
- Gradient backgrounds
- Custom CSS styling
- Responsive layouts
- Interactive tooltips

### **Data Visualization**
- Altair interactive charts
- Plotly 3D visualizations
- Heatmaps & matrices
- Trend lines & forecasts

### **User Experience**
- Period comparison
- Advanced filters
- Segment selection
- Export capabilities

---

## 📈 Analytics Capabilities

### **Descriptive Analytics**
- What happened? (Historical data)
- KPI dashboards
- Trend analysis

### **Diagnostic Analytics**
- Why did it happen? (Root cause)
- Cohort analysis
- Funnel analysis

### **Predictive Analytics**
- What will happen? (Forecasting)
- 7-day traffic forecast
- Churn prediction
- CLV prediction

### **Prescriptive Analytics**
- What should we do? (Optimization)
- Budget allocation
- Campaign optimization
- A/B test recommendations

---

## 🚀 Deployment

### **Streamlit Cloud**
1. Push to GitHub: `yandri918/marketing`
2. Connect to Streamlit Cloud
3. Deploy from `main` branch
4. Set main file: `Home.py`

### **Local Development**
```bash
streamlit run Home.py --server.port 8501
```

---

## 📝 Version History

### **v2.0** (Current)
- ✅ Advanced Analytics Dashboard (GA4 style)
- ✅ AI-powered insights & forecasting
- ✅ Cohort retention analysis
- ✅ Revenue analytics
- ✅ Custom report builder

### **v1.5**
- ✅ Web Analytics Dashboard
- ✅ Manual input & CSV upload
- ✅ Dynamic conversion metrics
- ✅ Altair visualizations

### **v1.0**
- ✅ Advanced MMM with Bayesian modeling
- ✅ Product recommender system
- ✅ Executive report generator
- ✅ Core marketing analytics

---

## 🤝 Contributing

This is a portfolio project showcasing enterprise-grade marketing analytics capabilities.

---

## 📧 Contact

**Developer**: Yandri
**Repository**: [github.com/yandri918/marketing](https://github.com/yandri918/marketing)

---

## 📄 License

This project is part of a professional portfolio demonstrating advanced data science and marketing analytics capabilities.

---

## 🎓 Skills Demonstrated

- ✅ **Advanced Python** - OOP, async, type hints
- ✅ **Data Science** - pandas, numpy, scikit-learn
- ✅ **Machine Learning** - Bayesian models, optimization
- ✅ **Visualization** - Altair, Plotly, interactive charts
- ✅ **Web Development** - Streamlit, responsive UI
- ✅ **Analytics** - Descriptive, diagnostic, predictive, prescriptive
- ✅ **Marketing** - MMM, attribution, optimization
- ✅ **Statistics** - Bayesian inference, A/B testing
- ✅ **Product** - User analytics, funnel optimization

---

**Built with ❤️ using Streamlit, Altair, and Python**
