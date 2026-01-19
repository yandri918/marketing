import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import sys
import os

# NLP Libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Add parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_sentiment_data

# Download NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

st.set_page_config(page_title="Advanced Sentiment Analysis | AI Marketing", page_icon="💬", layout="wide")

st.title("💬 Advanced Social Media Sentiment Analysis")
st.markdown("Enterprise-grade NLP platform with **emotion detection**, **topic modeling**, and **competitive intelligence**.")

# ========== HELPER FUNCTIONS ==========

def analyze_sentiment_multilevel(text):
    """Multi-level sentiment analysis"""
    if not text or pd.isna(text):
        return {
            'polarity': 0,
            'subjectivity': 0,
            'vader_compound': 0,
            'sentiment': 'Neutral',
            'intensity': 'Neutral',
            'emotion': 'Neutral'
        }
    
    text = str(text)
    
    # TextBlob analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # VADER analysis
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(text)
    compound = vader_scores['compound']
    
    # Sentiment classification
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    # Intensity classification
    if compound >= 0.5:
        intensity = 'Very Positive'
    elif compound >= 0.05:
        intensity = 'Positive'
    elif compound <= -0.5:
        intensity = 'Very Negative'
    elif compound <= -0.05:
        intensity = 'Negative'
    else:
        intensity = 'Neutral'
    
    # Simple emotion detection (keyword-based)
    emotion = detect_emotion(text)
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'vader_compound': compound,
        'sentiment': sentiment,
        'intensity': intensity,
        'emotion': emotion
    }

def detect_emotion(text):
    """Detect primary emotion from text"""
    text_lower = text.lower()
    
    # Emotion keywords
    emotions = {
        'Joy': ['happy', 'joy', 'love', 'excited', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome'],
        'Anger': ['angry', 'hate', 'furious', 'terrible', 'worst', 'horrible', 'awful', 'disgusting', 'mad'],
        'Sadness': ['sad', 'disappointed', 'unhappy', 'depressed', 'miserable', 'sorry', 'regret'],
        'Fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'concerned', 'fear'],
        'Surprise': ['surprised', 'shocked', 'unexpected', 'wow', 'amazing', 'incredible'],
        'Disgust': ['disgusting', 'gross', 'nasty', 'revolting', 'sick']
    }
    
    emotion_scores = {}
    for emotion, keywords in emotions.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        emotion_scores[emotion] = score
    
    if max(emotion_scores.values()) == 0:
        return 'Neutral'
    
    return max(emotion_scores, key=emotion_scores.get)

def extract_aspects(text):
    """Extract product aspects from text"""
    text_lower = text.lower()
    
    aspects = {
        'Product': ['product', 'quality', 'item', 'goods'],
        'Price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money'],
        'Service': ['service', 'support', 'help', 'customer', 'staff'],
        'Delivery': ['delivery', 'shipping', 'ship', 'arrived', 'package'],
        'Packaging': ['packaging', 'box', 'wrapped', 'package']
    }
    
    found_aspects = []
    for aspect, keywords in aspects.items():
        if any(keyword in text_lower for keyword in keywords):
            found_aspects.append(aspect)
    
    return found_aspects if found_aspects else ['General']

def perform_topic_modeling(texts, n_topics=5):
    """Perform LDA topic modeling"""
    # Vectorize
    vectorizer = CountVectorizer(max_features=100, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Get topics
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'topic_id': topic_idx,
            'keywords': ', '.join(top_words)
        })
    
    return topics, lda, vectorizer

def extract_keywords_tfidf(texts, n_keywords=10):
    """Extract keywords using TF-IDF"""
    tfidf = TfidfVectorizer(max_features=n_keywords, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    
    feature_names = tfidf.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    
    keyword_scores = list(zip(feature_names, scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return keyword_scores

def extract_hashtags(text):
    """Extract hashtags from text"""
    return re.findall(r'#\w+', text)

def calculate_engagement_rate(df):
    """Calculate engagement metrics"""
    if 'Likes' in df.columns and 'Comments' in df.columns:
        df['Engagement'] = df['Likes'] + df['Comments']
        return df['Engagement'].mean()
    return 0

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Configuration")

data_source = st.sidebar.radio("Data Source", ["Demo Data", "Custom Input"])

if data_source == "Custom Input":
    user_input = st.sidebar.text_area(
        "Paste comments (one per line)",
        "I love this product!\nTerrible customer service.\nGreat value for money.\nShipping was too slow."
    )

st.sidebar.divider()

st.sidebar.subheader("Analysis Settings")
n_topics = st.sidebar.slider("Number of Topics (LDA)", 2, 10, 5)
show_advanced = st.sidebar.checkbox("Show Advanced Analytics", value=True)

# ========== DATA LOADING ==========
if data_source == "Demo Data":
    df = generate_sentiment_data()
else:
    if user_input:
        comments = [c.strip() for c in user_input.split('\n') if c.strip()]
        df = pd.DataFrame({'Comment': comments})
    else:
        df = pd.DataFrame(columns=['Comment'])

if not df.empty:
    # Perform sentiment analysis
    sentiment_results = df['Comment'].apply(analyze_sentiment_multilevel)
    
    df['Polarity'] = sentiment_results.apply(lambda x: x['polarity'])
    df['Subjectivity'] = sentiment_results.apply(lambda x: x['subjectivity'])
    df['VADER_Score'] = sentiment_results.apply(lambda x: x['vader_compound'])
    df['Sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
    df['Intensity'] = sentiment_results.apply(lambda x: x['intensity'])
    df['Emotion'] = sentiment_results.apply(lambda x: x['emotion'])
    
    # Extract aspects
    df['Aspects'] = df['Comment'].apply(extract_aspects)
    
    # Extract hashtags
    df['Hashtags'] = df['Comment'].apply(extract_hashtags)
    
    # ========== TABS ==========
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "😊 Emotion Analysis",
        "🎯 Topics & Aspects",
        "📈 Trends",
        "🏆 Competitive",
        "🚨 Insights"
    ])
    
    # ========== TAB 1: OVERVIEW ==========
    with tab1:
        st.subheader("📊 Sentiment Overview")
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Comments", len(df))
        col2.metric("Positive", f"{(df['Sentiment']=='Positive').sum()}")
        col3.metric("Negative", f"{(df['Sentiment']=='Negative').sum()}")
        col4.metric("Avg Polarity", f"{df['Polarity'].mean():.2f}")
        col5.metric("Avg Subjectivity", f"{df['Subjectivity'].mean():.2f}")
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sentiment Distribution")
            
            sentiment_counts = df['Sentiment'].value_counts()
            
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Breakdown",
                color=sentiment_counts.index,
                color_discrete_map={'Positive': '#2ECC71', 'Negative': '#E74C3C', 'Neutral': '#95A5A6'},
                hole=0.4
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            st.markdown("### Sentiment Intensity")
            
            intensity_counts = df['Intensity'].value_counts().reindex([
                'Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative'
            ], fill_value=0)
            
            fig_intensity = px.bar(
                x=intensity_counts.index,
                y=intensity_counts.values,
                title="Sentiment Intensity Distribution",
                labels={'x': 'Intensity', 'y': 'Count'},
                color=intensity_counts.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_intensity, use_container_width=True)
        
        # Polarity vs Subjectivity
        st.markdown("### Polarity vs Subjectivity Analysis")
        
        fig_scatter = px.scatter(
            df,
            x='Polarity',
            y='Subjectivity',
            color='Sentiment',
            title="Sentiment Positioning",
            labels={'Polarity': 'Polarity (Negative ← → Positive)', 'Subjectivity': 'Subjectivity (Objective ← → Subjective)'},
            color_discrete_map={'Positive': '#2ECC71', 'Negative': '#E74C3C', 'Neutral': '#95A5A6'},
            hover_data=['Comment']
        )
        
        # Add quadrant lines
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Word Cloud
        st.markdown("### Word Cloud")
        
        all_text = " ".join(df['Comment'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)
        
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc)
    
    # ========== TAB 2: EMOTION ANALYSIS ==========
    with tab2:
        st.subheader("😊 Emotion Detection")
        
        # Emotion Distribution
        emotion_counts = df['Emotion'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Emotion Breakdown")
            
            # Emotion metrics
            for emotion in ['Joy', 'Anger', 'Sadness', 'Fear', 'Surprise', 'Disgust', 'Neutral']:
                count = emotion_counts.get(emotion, 0)
                pct = (count / len(df)) * 100 if len(df) > 0 else 0
                
                emoji_map = {
                    'Joy': '😊',
                    'Anger': '😠',
                    'Sadness': '😢',
                    'Fear': '😨',
                    'Surprise': '😲',
                    'Disgust': '🤢',
                    'Neutral': '😐'
                }
                
                st.metric(f"{emoji_map.get(emotion, '')} {emotion}", f"{count} ({pct:.1f}%)")
        
        with col2:
            st.markdown("### Emotion Distribution")
            
            fig_emotion = px.bar(
                x=emotion_counts.index,
                y=emotion_counts.values,
                title="Emotions Detected in Comments",
                labels={'x': 'Emotion', 'y': 'Count'},
                color=emotion_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_emotion, use_container_width=True)
        
        # Emotion-Sentiment Correlation
        st.markdown("### Emotion vs Sentiment")
        
        emotion_sentiment = df.groupby(['Emotion', 'Sentiment']).size().reset_index(name='Count')
        
        fig_emotion_sent = px.bar(
            emotion_sentiment,
            x='Emotion',
            y='Count',
            color='Sentiment',
            title="Emotion-Sentiment Correlation",
            color_discrete_map={'Positive': '#2ECC71', 'Negative': '#E74C3C', 'Neutral': '#95A5A6'},
            barmode='stack'
        )
        st.plotly_chart(fig_emotion_sent, use_container_width=True)
        
        # Sample comments by emotion
        st.markdown("### Sample Comments by Emotion")
        
        selected_emotion = st.selectbox("Select Emotion", emotion_counts.index.tolist())
        
        emotion_samples = df[df['Emotion'] == selected_emotion][['Comment', 'Sentiment', 'VADER_Score']].head(5)
        st.dataframe(emotion_samples, use_container_width=True, hide_index=True)
    
    # ========== TAB 3: TOPICS & ASPECTS ==========
    with tab3:
        st.subheader("🎯 Topic Modeling & Aspect Analysis")
        
        # Topic Modeling
        st.markdown("### LDA Topic Modeling")
        
        if len(df) >= n_topics:
            topics, lda_model, vectorizer = perform_topic_modeling(df['Comment'].tolist(), n_topics=n_topics)
            
            topics_df = pd.DataFrame(topics)
            st.dataframe(topics_df, use_container_width=True, hide_index=True)
            
            # Topic word clouds
            st.markdown("### Topic Word Clouds")
            
            cols = st.columns(min(3, n_topics))
            
            for i, topic in enumerate(topics[:3]):
                with cols[i]:
                    keywords = topic['keywords']
                    wc = WordCloud(width=300, height=200, background_color='white').generate(keywords)
                    
                    fig_topic, ax = plt.subplots(figsize=(4, 3))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f"Topic {i+1}")
                    st.pyplot(fig_topic)
        else:
            st.warning(f"Need at least {n_topics} comments for topic modeling. Current: {len(df)}")
        
        st.divider()
        
        # Aspect-Based Sentiment
        st.markdown("### Aspect-Based Sentiment Analysis")
        
        # Flatten aspects
        all_aspects = []
        for aspects_list in df['Aspects']:
            all_aspects.extend(aspects_list)
        
        aspect_counts = Counter(all_aspects)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Aspect Frequency")
            
            aspect_df = pd.DataFrame(aspect_counts.items(), columns=['Aspect', 'Count'])
            aspect_df = aspect_df.sort_values('Count', ascending=False)
            
            fig_aspects = px.bar(
                aspect_df,
                x='Aspect',
                y='Count',
                title="Most Mentioned Aspects",
                color='Count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_aspects, use_container_width=True)
        
        with col2:
            st.markdown("#### Aspect Sentiment")
            
            # Calculate sentiment per aspect
            aspect_sentiment = []
            
            for aspect in aspect_counts.keys():
                aspect_comments = df[df['Aspects'].apply(lambda x: aspect in x)]
                if len(aspect_comments) > 0:
                    avg_sentiment = aspect_comments['VADER_Score'].mean()
                    aspect_sentiment.append({
                        'Aspect': aspect,
                        'Avg Sentiment': avg_sentiment,
                        'Count': len(aspect_comments)
                    })
            
            aspect_sent_df = pd.DataFrame(aspect_sentiment)
            
            if not aspect_sent_df.empty:
                fig_aspect_sent = px.bar(
                    aspect_sent_df,
                    x='Aspect',
                    y='Avg Sentiment',
                    title="Average Sentiment by Aspect",
                    color='Avg Sentiment',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0
                )
                st.plotly_chart(fig_aspect_sent, use_container_width=True)
        
        # Keyword Extraction
        st.markdown("### Top Keywords (TF-IDF)")
        
        keywords = extract_keywords_tfidf(df['Comment'].tolist(), n_keywords=15)
        
        keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Score'])
        
        fig_keywords = px.bar(
            keywords_df,
            x='Score',
            y='Keyword',
            orientation='h',
            title="Top 15 Keywords by TF-IDF",
            color='Score',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_keywords, use_container_width=True)
    
    # ========== TAB 4: TRENDS ==========
    with tab4:
        st.subheader("📈 Sentiment Trends")
        
        # Add synthetic timestamp if not present
        if 'Date' not in df.columns:
            df['Date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='H')
        
        # Time series
        df_sorted = df.sort_values('Date')
        
        st.markdown("### Sentiment Over Time")
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=df_sorted['Date'],
            y=df_sorted['VADER_Score'],
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='#3498DB', width=2)
        ))
        
        # Add moving average
        df_sorted['MA_5'] = df_sorted['VADER_Score'].rolling(window=min(5, len(df))).mean()
        
        fig_trend.add_trace(go.Scatter(
            x=df_sorted['Date'],
            y=df_sorted['MA_5'],
            mode='lines',
            name='Moving Average (5)',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig_trend.add_hline(y=0, line_dash="dot", line_color="gray")
        
        fig_trend.update_layout(
            title="Sentiment Score Timeline",
            xaxis_title="Time",
            yaxis_title="Sentiment Score",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Volume trends
        st.markdown("### Comment Volume Trends")
        
        df_sorted['Hour'] = df_sorted['Date'].dt.hour
        hourly_volume = df_sorted.groupby('Hour').size().reset_index(name='Count')
        
        fig_volume = px.bar(
            hourly_volume,
            x='Hour',
            y='Count',
            title="Comments by Hour of Day",
            labels={'Hour': 'Hour', 'Count': 'Number of Comments'},
            color='Count',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Hashtag trends
        st.markdown("### Hashtag Analysis")
        
        all_hashtags = []
        for hashtags in df['Hashtags']:
            all_hashtags.extend(hashtags)
        
        if all_hashtags:
            hashtag_counts = Counter(all_hashtags).most_common(10)
            
            hashtag_df = pd.DataFrame(hashtag_counts, columns=['Hashtag', 'Count'])
            
            fig_hashtags = px.bar(
                hashtag_df,
                x='Count',
                y='Hashtag',
                orientation='h',
                title="Top 10 Hashtags",
                color='Count',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_hashtags, use_container_width=True)
        else:
            st.info("No hashtags found in comments")
    
    # ========== TAB 5: COMPETITIVE ==========
    with tab5:
        st.subheader("🏆 Competitive Sentiment Comparison")
        
        st.info("💡 **Demo Mode:** In production, this would compare sentiment across multiple brands/competitors.")
        
        # Simulated competitive data
        brands = ['Your Brand', 'Competitor A', 'Competitor B', 'Competitor C']
        
        competitive_data = pd.DataFrame({
            'Brand': brands,
            'Positive %': [60, 45, 55, 40],
            'Negative %': [20, 35, 25, 40],
            'Neutral %': [20, 20, 20, 20],
            'Avg Sentiment': [0.45, 0.15, 0.30, 0.05],
            'Volume': [len(df), 150, 120, 180]
        })
        
        # Competitive comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sentiment Comparison")
            
            fig_comp = px.bar(
                competitive_data,
                x='Brand',
                y=['Positive %', 'Neutral %', 'Negative %'],
                title="Sentiment Distribution by Brand",
                barmode='stack',
                color_discrete_map={'Positive %': '#2ECC71', 'Neutral %': '#95A5A6', 'Negative %': '#E74C3C'}
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with col2:
            st.markdown("### Share of Voice")
            
            fig_sov = px.pie(
                competitive_data,
                values='Volume',
                names='Brand',
                title="Comment Volume Share",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_sov, use_container_width=True)
        
        # Net Sentiment Score
        st.markdown("### Net Sentiment Score (NSS)")
        
        competitive_data['NSS'] = competitive_data['Positive %'] - competitive_data['Negative %']
        
        fig_nss = px.bar(
            competitive_data,
            x='Brand',
            y='NSS',
            title="Net Sentiment Score (Positive % - Negative %)",
            color='NSS',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        st.plotly_chart(fig_nss, use_container_width=True)
        
        # Competitive table
        st.dataframe(competitive_data.style.format({
            'Positive %': '{:.0f}%',
            'Negative %': '{:.0f}%',
            'Neutral %': '{:.0f}%',
            'Avg Sentiment': '{:.2f}',
            'Volume': '{:,.0f}',
            'NSS': '{:+.0f}%'
        }).background_gradient(subset=['NSS'], cmap='RdYlGn'), use_container_width=True, hide_index=True)
    
    # ========== TAB 6: INSIGHTS & ALERTS ==========
    with tab6:
        st.subheader("🚨 Actionable Insights & Alerts")
        
        # Calculate metrics
        positive_pct = (df['Sentiment'] == 'Positive').mean() * 100
        negative_pct = (df['Sentiment'] == 'Negative').mean() * 100
        avg_sentiment = df['VADER_Score'].mean()
        
        # Crisis Detection
        st.markdown("### Crisis Detection")
        
        if negative_pct > 50:
            st.error(f"""
            🚨 **CRISIS ALERT: High Negative Sentiment**
            
            - Negative comments: {negative_pct:.1f}%
            - Average sentiment: {avg_sentiment:.2f}
            
            **Recommended Actions:**
            1. 🔍 Investigate root cause immediately
            2. 📞 Activate crisis response team
            3. 📧 Prepare public statement
            4. 🤝 Reach out to affected customers
            """)
        elif negative_pct > 30:
            st.warning(f"""
            ⚠️ **WARNING: Elevated Negative Sentiment**
            
            - Negative comments: {negative_pct:.1f}%
            - Average sentiment: {avg_sentiment:.2f}
            
            **Recommended Actions:**
            1. 📊 Monitor closely for next 24-48 hours
            2. 🔍 Identify common complaints
            3. 📝 Prepare response strategy
            """)
        else:
            st.success(f"""
            ✅ **HEALTHY: Sentiment Within Normal Range**
            
            - Negative comments: {negative_pct:.1f}%
            - Average sentiment: {avg_sentiment:.2f}
            
            **Continue monitoring and maintain engagement.**
            """)
        
        st.divider()
        
        # Opportunity Detection
        st.markdown("### Opportunity Detection")
        
        if positive_pct > 60:
            st.success(f"""
            🎯 **OPPORTUNITY: High Positive Sentiment**
            
            - Positive comments: {positive_pct:.1f}%
            
            **Recommended Actions:**
            1. 📣 Amplify positive testimonials
            2. 🎁 Launch referral program
            3. 📸 Collect user-generated content
            4. 🌟 Feature customer success stories
            """)
        
        # Top Issues
        st.markdown("### Top Issues to Address")
        
        negative_comments = df[df['Sentiment'] == 'Negative']
        
        if len(negative_comments) > 0:
            # Extract aspects from negative comments
            negative_aspects = []
            for aspects in negative_comments['Aspects']:
                negative_aspects.extend(aspects)
            
            issue_counts = Counter(negative_aspects).most_common(5)
            
            st.markdown("**Most Complained Aspects:**")
            for aspect, count in issue_counts:
                st.markdown(f"- **{aspect}**: {count} mentions")
        else:
            st.info("No negative comments to analyze")
        
        st.divider()
        
        # Response Priority
        st.markdown("### Response Priority Queue")
        
        # Priority scoring: negative sentiment + recency
        df_priority = df.copy()
        df_priority['Priority_Score'] = (1 - df_priority['VADER_Score']) * 100
        df_priority = df_priority.sort_values('Priority_Score', ascending=False)
        
        priority_comments = df_priority[['Comment', 'Sentiment', 'VADER_Score', 'Emotion', 'Priority_Score']].head(5)
        
        st.dataframe(priority_comments.style.format({
            'VADER_Score': '{:.2f}',
            'Priority_Score': '{:.0f}'
        }).background_gradient(subset=['Priority_Score'], cmap='Reds'), use_container_width=True, hide_index=True)
        
        st.info("💡 **Tip:** Respond to high-priority negative comments within 1-2 hours to prevent escalation.")

else:
    st.info("Please provide input data to run the analysis.")

# ========== FOOTER ==========
st.divider()
st.caption("💡 **Pro Tip:** Use this advanced NLP platform to gain deep insights into customer sentiment, emotions, and trending topics!")
