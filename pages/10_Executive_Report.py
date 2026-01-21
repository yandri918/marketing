import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import sys
import os

# Try to import FPDF, gracefully handle if not installed
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    st.error("⚠️ **PDF Generation Unavailable**: The `fpdf2` library is not installed. Please contact the administrator to add it to requirements.txt")
    st.info("💡 The platform is rebuilding with updated dependencies. Please refresh in a few minutes.")
    st.stop()

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_sales_data, generate_customer_data

st.set_page_config(page_title="Executive Report | PDF", page_icon="📄", layout="wide")

st.title("📄 Automated Executive Reporting")
st.markdown("Generate and download a professional PDF strategy report for stakeholders.")

# 1. Inputs for the Report
with st.container():
    st.subheader("📝 Report Configuration")
    col1, col2 = st.columns(2)
    with col1:
        report_title = st.text_input("Report Title", "Monthly Marketing Strategy Update")
        analyst_name = st.text_input("Analyst Name", "Senior Marketing Manager")
    with col2:
        report_month = st.selectbox("Reporting Period", ["January 2026", "February 2026", "March 2026"])
        key_insight = st.text_area("Key Strategic Insight (Executive Summary)", "Overall performance is strong. CLV analysis indicates a need to focus on retention for the 'High Income, Low Spending' segment. Recommended shifting 15% of budget to loyalty programs.")

# 2. Generate Data & Charts for the PDF
st.divider()
st.subheader("📊 Report Preview")

# Data
sales_df = generate_sales_data()
customer_df = generate_customer_data()

# Charts (Matplotlib for PDF compatibility)
def create_sales_chart():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sales_df['Date'], sales_df['Sales'], color='#2980b9')
    ax.set_title('Sales Trend (Last 12 Months)')
    ax.set_ylabel('Revenue (IDR)')
    ax.grid(True, alpha=0.3)
    return fig

def create_segment_chart():
    fig, ax = plt.subplots(figsize=(6, 6))
    # Simple cluster visualization (Income vs Score)
    ax.scatter(customer_df['Income'], customer_df['SpendingScore'], alpha=0.6, c='#27ae60')
    ax.set_title('Customer Segmentation Distribution')
    ax.set_xlabel('Income (IDR)')
    ax.set_ylabel('Spending Score')
    return fig

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Sales Trend**")
    fig1 = create_sales_chart()
    st.pyplot(fig1)

with c2:
    st.markdown("**Segmentation Overview**")
    fig2 = create_segment_chart()
    st.pyplot(fig2)

# 3. PDF Generation Class
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Confident Marketing Analytics', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(title, analyst, period, insight, fig1, fig2):
    pdf = PDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 20, title, 0, 1, 'C')
    
    # Meta
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Period: {period}", 0, 1, 'C')
    pdf.cell(0, 10, f"Analyst: {analyst}", 0, 1, 'C')
    pdf.ln(10)
    
    # Executive Summary
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, insight)
    pdf.ln(10)
    
    # Save charts to temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile1:
        fig1.savefig(tmpfile1.name)
        img1_path = tmpfile1.name
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile2:
        fig2.savefig(tmpfile2.name)
        img2_path = tmpfile2.name

    # Add Charts
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '1. Sales Performance', 0, 1, 'L')
    pdf.image(img1_path, x=10, w=190)
    pdf.ln(5)
    
    pdf.add_page()
    pdf.cell(0, 10, '2. Customer Segmentation Analysis', 0, 1, 'L')
    pdf.image(img2_path, x=30, w=150)
    
    # Recommendations
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Strategic Recommendations', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, "- Focus on 'High Value' users identified in the Cohort Analysis.\n- Increase budget for digital channels by 10% based on ROI analysis.\n- Launch 'Win-Back' campaign for churned users detected by the CLV model.")
    
    # Cleanup temp files
    # os.remove(img1_path) # Streamlit might lock this, usually fine in tmp
    return pdf.output(dest='S').encode('latin-1')

# 4. Download Button
st.divider()
st.subheader("📥 Download Report")

if st.button("Generate PDF Report"):
    with st.spinner("Generating PDF..."):
        pdf_bytes = create_pdf(report_title, analyst_name, report_month, key_insight, fig1, fig2)
        
    st.success("Report Generated Successfully!")
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="Marketing_Strategy_Report.pdf",
        mime="application/pdf"
    )
