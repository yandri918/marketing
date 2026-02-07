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

# Add parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_sentiment_data

# Import advanced NLP functions
try:
    from utils.advanced_nlp import (
        analyze_sentiment_comprehensive,
        perform_bertopic_modeling,
        extract_aspects_advanced,
        detect_language
    )
    ADVANCED_NLP_AVAILABLE = True
except ImportError as e:
    st.warning(f"Advanced NLP not available: {e}. Using basic models.")
    ADVANCED_NLP_AVAILABLE = False

# Fallback imports
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download NLTK data
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Advanced Sentiment Analysis | AI Marketing", page_icon="💬", layout="wide")

# ========== HEADER ==========
st.title("💬 Advanced Social Media Sentiment Analysis")

if ADVANCED_NLP_AVAILABLE:
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h3 style='color: white; margin: 0;'>🚀 Enterprise-Grade AI Platform</h3>
        <p style='color: white; margin: 0.5rem 0 0 0;'>
            Powered by <b>Deep Learning</b> • 28 Emotions • Sarcasm Detection • BERTopic • Multi-Language
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("Enterprise-grade NLP platform with **emotion detection**, **topic modeling**, and **competitive intelligence**.")

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
n_topics = st.sidebar.slider("Number of Topics", 2, 10, 5)
use_advanced_models = st.sidebar.checkbox("Use Deep Learning Models", value=ADVANCED_NLP_AVAILABLE)
show_sarcasm = st.sidebar.checkbox("Detect Sarcasm", value=True)
show_language = st.sidebar.checkbox("Detect Language", value=True)

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
    with st.spinner("🔬 Analyzing sentiment with AI models..."):
        if use_advanced_models and ADVANCED_NLP_AVAILABLE:
            # Use advanced comprehensive analysis
            results = df['Comment'].apply(analyze_sentiment_comprehensive)
            
            df['Polarity'] = results.apply(lambda x: x['polarity'])
            df['Subjectivity'] = results.apply(lambda x: x['subjectivity'])
            df['VADER_Score'] = results.apply(lambda x: x['vader_compound'])
            df['Sentiment'] = results.apply(lambda x: x['sentiment'])
            df['Emotion'] = results.apply(lambda x: x['emotion'])
            df['Emotion_Confidence'] = results.apply(lambda x: x['emotion_confidence'])
            df['All_Emotions'] = results.apply(lambda x: x['all_emotions'])
            
            if show_sarcasm:
                df['Is_Sarcastic'] = results.apply(lambda x: x['is_sarcastic'])
                df['Sarcasm_Confidence'] = results.apply(lambda x: x['sarcasm_confidence'])
            
            if show_language:
                df['Language'] = results.apply(lambda x: x['language'])
            
            df['Aspects'] = results.apply(lambda x: x['aspects'])
            
        else:
            # Fallback to basic analysis
            from textblob import TextBlob
            
            def basic_analysis(text):
                blob = TextBlob(str(text))
                sia = SentimentIntensityAnalyzer()
                vader = sia.polarity_scores(str(text))
                
                compound = vader['compound']
                sentiment = 'Positive' if compound >= 0.05 else ('Negative' if compound <= -0.05 else 'Neutral')
                
                return {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity,
                    'vader_compound': compound,
                    'sentiment': sentiment
                }
            
            results = df['Comment'].apply(basic_analysis)
            df['Polarity'] = results.apply(lambda x: x['polarity'])
            df['Subjectivity'] = results.apply(lambda x: x['subjectivity'])
            df['VADER_Score'] = results.apply(lambda x: x['vader_compound'])
            df['Sentiment'] = results.apply(lambda x: x['sentiment'])
            df['Emotion'] = 'Neutral'
            df['Aspects'] = [['General']] * len(df)
    
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
            st.markdown("### Sentiment Score Distribution")
            
            fig_hist = px.histogram(
                df,
                x='VADER_Score',
                nbins=30,
                title="VADER Score Distribution",
                labels={'VADER_Score': 'Sentiment Score'},
                color_discrete_sequence=['#3498DB']
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_hist, use_container_width=True)
        
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
        
        # Sarcasm Detection (if enabled)
        if show_sarcasm and 'Is_Sarcastic' in df.columns:
            st.divider()
            st.markdown("### 😏 Sarcasm Detection")
            
            sarcastic_count = df['Is_Sarcastic'].sum()
            sarcastic_pct = (sarcastic_count / len(df)) * 100
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Sarcastic Comments", f"{sarcastic_count} ({sarcastic_pct:.1f}%)")
                
                if sarcastic_count > 0:
                    st.info("💡 Sarcasm detected! Sentiment scores have been adjusted.")
            
            with col2:
                if sarcastic_count > 0:
                    sarcastic_samples = df[df['Is_Sarcastic']][['Comment', 'Sentiment', 'Sarcasm_Confidence']].head(3)
                    st.dataframe(sarcastic_samples, use_container_width=True, hide_index=True)
    
    # ========== TAB 2: EMOTION ANALYSIS ==========
    with tab2:
        st.subheader("😊 Deep Learning Emotion Detection")
        
        if 'Emotion' in df.columns and df['Emotion'].notna().any():
            # Emotion Distribution
            emotion_counts = df['Emotion'].value_counts()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Emotion Breakdown")
                
                # Emotion metrics
                for emotion in emotion_counts.index[:7]:
                    count = emotion_counts.get(emotion, 0)
                    pct = (count / len(df)) * 100 if len(df) > 0 else 0
                    
                    emoji_map = {
                        'joy': '😊', 'anger': '😠', 'sadness': '😢',
                        'fear': '😨', 'surprise': '😲', 'disgust': '🤢',
                        'neutral': '😐', 'love': '❤️', 'optimism': '🌟'
                    }
                    
                    emoji = emoji_map.get(emotion.lower(), '💭')
                    st.metric(f"{emoji} {emotion.title()}", f"{count} ({pct:.1f}%)")
            
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
            
            # Emotion Confidence
            if 'Emotion_Confidence' in df.columns:
                st.markdown("### Emotion Detection Confidence")
                
                fig_conf = px.box(
                    df,
                    x='Emotion',
                    y='Emotion_Confidence',
                    title="Confidence Scores by Emotion",
                    labels={'Emotion_Confidence': 'Confidence'},
                    color='Emotion',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_conf, use_container_width=True)
                
                avg_confidence = df['Emotion_Confidence'].mean()
                st.info(f"📊 Average emotion detection confidence: **{avg_confidence:.1%}**")
            
            # Sample comments by emotion
            st.markdown("### Sample Comments by Emotion")
            
            selected_emotion = st.selectbox("Select Emotion", emotion_counts.index.tolist())
            
            emotion_samples = df[df['Emotion'] == selected_emotion][['Comment', 'Sentiment', 'VADER_Score']].head(5)
            st.dataframe(emotion_samples, use_container_width=True, hide_index=True)
        else:
            st.warning("Emotion detection not available. Enable deep learning models in sidebar.")
    
    # ========== TAB 3: TOPICS & ASPECTS ==========
    with tab3:
        st.subheader("🎯 Topic Modeling & Aspect Analysis")
        
        # Topic Modeling
        st.markdown("### Semantic Topic Modeling")
        
        if len(df) >= n_topics:
            with st.spinner("🔬 Extracting topics with BERTopic..."):
                try:
                    if use_advanced_models and ADVANCED_NLP_AVAILABLE:
                        topics_df, topic_model, topic_assignments = perform_bertopic_modeling(
                            df['Comment'].tolist(),
                            n_topics=n_topics
                        )
                        
                        st.dataframe(topics_df, use_container_width=True, hide_index=True)
                        
                        st.success("✅ Using BERTopic (semantic topic modeling)")
                    else:
                        # Fallback to LDA
                        from sklearn.feature_extraction.text import CountVectorizer
                        from sklearn.decomposition import LatentDirichletAllocation
                        
                        vectorizer = CountVectorizer(max_features=100, stop_words='english')
                        doc_term_matrix = vectorizer.fit_transform(df['Comment'])
                        
                        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                        lda.fit(doc_term_matrix)
                        
                        feature_names = vectorizer.get_feature_names_out()
                        topics = []
                        
                        for topic_idx, topic in enumerate(lda.components_):
                            top_words_idx = topic.argsort()[-5:][::-1]
                            top_words = [feature_names[i] for i in top_words_idx]
                            topics.append({
                                'topic_id': topic_idx,
                                'keywords': ', '.join(top_words)
                            })
                        
                        topics_df = pd.DataFrame(topics)
                        st.dataframe(topics_df, use_container_width=True, hide_index=True)
                        
                        st.info("ℹ️ Using LDA (basic topic modeling). Enable deep learning for better results.")
                        
                except Exception as e:
                    st.error(f"Topic modeling failed: {e}")
        else:
            st.warning(f"Need at least {n_topics} comments for topic modeling. Current: {len(df)}")
        
        st.divider()
        
        # Aspect-Based Sentiment
        st.markdown("### Aspect-Based Sentiment Analysis")
        
        # Flatten aspects
        all_aspects = []
        for aspects_list in df['Aspects']:
            if isinstance(aspects_list, list):
                all_aspects.extend(aspects_list)
            else:
                all_aspects.append(str(aspects_list))
        
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
                aspect_comments = df[df['Aspects'].apply(lambda x: aspect in x if isinstance(x, list) else aspect == str(x))]
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
        
        # Language Distribution (if available)
        if show_language and 'Language' in df.columns:
            st.markdown("### 🌍 Language Distribution")
            
            lang_counts = df['Language'].value_counts()
            
            lang_names = {
                'en': 'English',
                'id': 'Indonesian',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'zh-cn': 'Chinese'
            }
            
            lang_display = pd.DataFrame({
                'Language': [lang_names.get(lang, lang.upper()) for lang in lang_counts.index],
                'Count': lang_counts.values
            })
            
            fig_lang = px.pie(
                lang_display,
                values='Count',
                names='Language',
                title="Comments by Language",
                hole=0.4
            )
            st.plotly_chart(fig_lang, use_container_width=True)
    
    # ========== TAB 5: COMPETITIVE ==========
    with tab5:
        st.subheader("🏆 Competitive Sentiment Comparison")
        
        st.info("💡 **Demo Mode:** In production, this would compare sentiment across multiple brands/competitors using real-time social media data.")
        
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
    
    # ========== TAB 6: INSIGHTS ==========
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
        
        # Response Priority
        st.markdown("### Response Priority Queue")
        
        # Priority scoring: negative sentiment + recency
        df_priority = df.copy()
        df_priority['Priority_Score'] = (1 - df_priority['VADER_Score']) * 100
        df_priority = df_priority.sort_values('Priority_Score', ascending=False)
        
        priority_cols = ['Comment', 'Sentiment', 'VADER_Score', 'Priority_Score']
        if 'Emotion' in df_priority.columns:
            priority_cols.insert(3, 'Emotion')
        
        priority_comments = df_priority[priority_cols].head(5)
        
        st.dataframe(priority_comments.style.format({
            'VADER_Score': '{:.2f}',
            'Priority_Score': '{:.0f}'
        }).background_gradient(subset=['Priority_Score'], cmap='Reds'), use_container_width=True, hide_index=True)
        
        st.info("💡 **Tip:** Respond to high-priority negative comments within 1-2 hours to prevent escalation.")

else:
    st.info("Please provide input data to run the analysis.")

# ========== FOOTER ==========
st.divider()

if ADVANCED_NLP_AVAILABLE:
    st.success("✅ **Advanced AI Models Active** - Using deep learning for superior accuracy")
else:
    st.warning("⚠️ **Basic Models Active** - Install transformers, bertopic, and spacy for advanced features")

st.caption("💡 **Pro Tip:** Use this advanced NLP platform to gain deep insights into customer sentiment, emotions, and trending topics!")
