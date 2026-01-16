import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Add parent directory to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_sentiment_data

# Downloader for NLTK (quietly)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Sentiment Analysis | AI Marketing", page_icon="💬", layout="wide")

st.title("💬 Social Media Sentiment Analysis")
st.markdown("Analyze brand perception from social media comments using **NLP**.")

# Input Section
st.sidebar.header("Data Source")
source = st.sidebar.radio("Choose Source", ["Demo Data", "Direct Input"])

if source == "Demo Data":
    df = generate_sentiment_data()
else:
    user_input = st.text_area("Paste social media comments (one per line)", 
                              "I love this product!\nTerrible service.\nGreat value for money.")
    if user_input:
        df = pd.DataFrame({'Comment': user_input.split('\n')})
    else:
        df = pd.DataFrame(columns=['Comment'])

if not df.empty:
    # Sentiment Analysis Logic
    sia = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        if not text: return "Neutral", 0
        score = sia.polarity_scores(str(text))['compound']
        if score >= 0.05: return "Positive", score
        elif score <= -0.05: return "Negative", score
        else: return "Neutral", score

    df[['Sentiment', 'Score']] = df['Comment'].apply(lambda x: pd.Series(get_sentiment(x)))

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments", len(df))
    col2.metric("Positive Sentiment", f"{(df['Sentiment']=='Positive').mean():.1%}")
    col3.metric("Negative Sentiment", f"{(df['Sentiment']=='Negative').mean():.1%}")

    # Charts
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Sentiment Distribution")
        fig_dist = px.pie(df, names='Sentiment', color='Sentiment', 
                          color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#95a5a6'},
                          hole=0.4)
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.subheader("Data Preview")
        st.dataframe(df[['Comment', 'Sentiment', 'Score']], height=300)

    # Word Cloud
    st.divider()
    st.subheader("☁️ Word Cloud Analysis")
    
    all_text = " ".join(df['Comment'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
else:
    st.info("Please provide input data to run the analysis.")
