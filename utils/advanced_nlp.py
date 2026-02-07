"""
Advanced NLP Helper Functions for Enterprise Sentiment Analysis
Uses deep learning models for emotion detection, topic modeling, and sarcasm detection
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Lazy imports for heavy models (only load when needed)
_emotion_model = None
_sarcasm_model = None
_topic_model = None
_embedding_model = None

@st.cache_resource
def load_emotion_model():
    """Load deep learning emotion detection model (DistilRoBERTa)"""
    global _emotion_model
    if _emotion_model is None:
        try:
            from transformers import pipeline
            _emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=-1  # CPU
            )
        except Exception as e:
            st.warning(f"Could not load emotion model: {e}. Falling back to keyword-based.")
            _emotion_model = "fallback"
    return _emotion_model

@st.cache_resource
def load_sarcasm_model():
    """Load sarcasm detection model"""
    global _sarcasm_model
    if _sarcasm_model is None:
        try:
            from transformers import pipeline
            _sarcasm_model = pipeline(
                "text-classification",
                model="helinivan/english-sarcasm-detector",
                device=-1
            )
        except Exception as e:
            st.warning(f"Could not load sarcasm model: {e}")
            _sarcasm_model = "fallback"
    return _sarcasm_model

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer for embeddings"""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.warning(f"Could not load embedding model: {e}")
            _embedding_model = "fallback"
    return _embedding_model

def detect_emotion_advanced(text: str) -> Dict:
    """
    Advanced emotion detection using deep learning
    Returns top 3 emotions with confidence scores
    """
    model = load_emotion_model()
    
    if model == "fallback":
        # Fallback to keyword-based
        return detect_emotion_keywords(text)
    
    try:
        results = model(text)[0]
        
        # Get top 3 emotions
        top_emotions = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
        
        return {
            'primary_emotion': top_emotions[0]['label'],
            'confidence': top_emotions[0]['score'],
            'all_emotions': {e['label']: e['score'] for e in top_emotions}
        }
    except Exception as e:
        st.warning(f"Emotion detection error: {e}")
        return detect_emotion_keywords(text)

def detect_emotion_keywords(text: str) -> Dict:
    """Fallback keyword-based emotion detection"""
    text_lower = text.lower()
    
    emotions = {
        'joy': ['happy', 'joy', 'love', 'excited', 'great', 'excellent', 'amazing'],
        'anger': ['angry', 'hate', 'furious', 'terrible', 'worst', 'horrible'],
        'sadness': ['sad', 'disappointed', 'unhappy', 'depressed', 'miserable'],
        'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous'],
        'surprise': ['surprised', 'shocked', 'unexpected', 'wow'],
        'disgust': ['disgusting', 'gross', 'nasty', 'revolting']
    }
    
    emotion_scores = {}
    for emotion, keywords in emotions.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if not emotion_scores:
        return {'primary_emotion': 'neutral', 'confidence': 0.5, 'all_emotions': {'neutral': 1.0}}
    
    primary = max(emotion_scores, key=emotion_scores.get)
    total = sum(emotion_scores.values())
    
    return {
        'primary_emotion': primary,
        'confidence': emotion_scores[primary] / total,
        'all_emotions': {k: v/total for k, v in emotion_scores.items()}
    }

def detect_sarcasm(text: str) -> Dict:
    """Detect sarcasm in text"""
    model = load_sarcasm_model()
    
    if model == "fallback":
        return {'is_sarcastic': False, 'confidence': 0.0}
    
    try:
        result = model(text)[0]
        is_sarcastic = result['label'] == 'SARCASM'
        
        return {
            'is_sarcastic': is_sarcastic,
            'confidence': result['score']
        }
    except Exception as e:
        return {'is_sarcastic': False, 'confidence': 0.0}

def perform_bertopic_modeling(texts: List[str], n_topics: int = 5) -> Tuple:
    """
    Perform BERTopic modeling for semantic topic extraction
    Returns: (topics_df, topic_model, embeddings)
    """
    try:
        from bertopic import BERTopic
        
        embedding_model = load_embedding_model()
        
        if embedding_model == "fallback":
            raise Exception("Embedding model not available")
        
        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=n_topics,
            verbose=False
        )
        
        # Fit model
        topics, probs = topic_model.fit_transform(texts)
        
        # Get topic info
        topic_info = topic_model.get_topic_info()
        
        return topic_info, topic_model, topics
        
    except Exception as e:
        st.warning(f"BERTopic failed: {e}. Falling back to LDA.")
        return perform_lda_modeling(texts, n_topics)

def perform_lda_modeling(texts: List[str], n_topics: int = 5) -> Tuple:
    """Fallback LDA topic modeling"""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    
    vectorizer = CountVectorizer(max_features=100, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'Topic': topic_idx,
            'Count': int(np.sum(lda.transform(doc_term_matrix).argmax(axis=1) == topic_idx)),
            'Name': f"Topic {topic_idx}",
            'Representation': top_words
        })
    
    topics_df = pd.DataFrame(topics)
    return topics_df, lda, None

def extract_aspects_advanced(text: str) -> List[Tuple[str, str, float]]:
    """
    Advanced aspect-based sentiment extraction using spaCy
    Returns: List of (aspect, opinion, sentiment_score) tuples
    """
    try:
        import spacy
        from textblob import TextBlob
        
        # Try to load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            st.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            return extract_aspects_simple(text)
        
        doc = nlp(text)
        aspects = []
        
        # Extract noun-adjective pairs
        for token in doc:
            if token.pos_ == "NOUN":
                for child in token.children:
                    if child.pos_ == "ADJ":
                        # Get sentiment of the adjective
                        blob = TextBlob(child.text)
                        sentiment = blob.sentiment.polarity
                        
                        aspects.append((token.text, child.text, sentiment))
        
        return aspects if aspects else [("General", "neutral", 0.0)]
        
    except Exception as e:
        return extract_aspects_simple(text)

def extract_aspects_simple(text: str) -> List[str]:
    """Simple keyword-based aspect extraction (fallback)"""
    text_lower = text.lower()
    
    aspects_map = {
        'Product': ['product', 'quality', 'item', 'goods'],
        'Price': ['price', 'cost', 'expensive', 'cheap', 'value'],
        'Service': ['service', 'support', 'help', 'customer'],
        'Delivery': ['delivery', 'shipping', 'ship', 'arrived'],
        'Packaging': ['packaging', 'box', 'wrapped']
    }
    
    found = []
    for aspect, keywords in aspects_map.items():
        if any(kw in text_lower for kw in keywords):
            found.append(aspect)
    
    return found if found else ['General']

def detect_language(text: str) -> str:
    """Detect language of text"""
    try:
        from langdetect import detect
        return detect(text)
    except:
        return 'en'  # Default to English

def analyze_sentiment_comprehensive(text: str) -> Dict:
    """
    Comprehensive sentiment analysis combining multiple models
    Returns all analysis results in one dict
    """
    from textblob import TextBlob
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Basic sentiment
    blob = TextBlob(text)
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(text)
    
    # Advanced emotion
    emotion_result = detect_emotion_advanced(text)
    
    # Sarcasm
    sarcasm_result = detect_sarcasm(text)
    
    # Language
    language = detect_language(text)
    
    # Aspects
    aspects = extract_aspects_simple(text)
    
    # Determine final sentiment (adjusted for sarcasm)
    compound = vader_scores['compound']
    if sarcasm_result['is_sarcastic'] and sarcasm_result['confidence'] > 0.7:
        compound = -compound  # Flip sentiment if sarcastic
    
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return {
        'text': text,
        'language': language,
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'vader_compound': compound,
        'sentiment': sentiment,
        'emotion': emotion_result['primary_emotion'],
        'emotion_confidence': emotion_result['confidence'],
        'all_emotions': emotion_result['all_emotions'],
        'is_sarcastic': sarcasm_result['is_sarcastic'],
        'sarcasm_confidence': sarcasm_result['confidence'],
        'aspects': aspects
    }
