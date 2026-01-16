import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Recommender System", page_icon="🛍️", layout="wide")

st.title("🛍️ AI Product Recommendation Engine")
st.markdown("""
**Collaborative Filtering**: This system uses **Cosine Similarity** to find users with similar purchase history and recommends products they haven't bought yet.
It answers: *"Users like you also bought..."*
""")

# --- 1. Data Generation (User-Item Matrix) ---
@st.cache_data
def generate_user_item_matrix():
    np.random.seed(42)
    n_users = 20
    n_products = 10
    
    products = [
        "Premium NPK Fertilizer", "Organic Pesticide", "Growth Booster Hormone", 
        "Hydroponic Kit A", "Hydroponic Kit B", "Drip Irrigation Hose", 
        "Sprayer 5L", "Soil pH Meter", "Greenhouse Plastic", "Coco Peat Slab"
    ]
    
    # 0 = No purchase, 1-5 = Rating/Purchase Count
    # We create a sparse matrix where people buy related things (clusters)
    data = np.random.randint(0, 6, size=(n_users, n_products))
    
    # Inject patterns: 
    # Users 0-9 like Fertilizers/Chemicals (Cols 0-2)
    data[:10, 0:3] = np.random.randint(3, 6, size=(10, 3))
    # Users 10-19 like Hydroponics (Cols 3-5)
    data[10:, 3:6] = np.random.randint(3, 6, size=(10, 3))
    
    # Random sparsity
    mask = np.random.choice([0, 1], size=data.shape, p=[0.3, 0.7])
    data = data * mask
    
    df = pd.DataFrame(data, columns=products, index=[f"User {i+1}" for i in range(n_users)])
    return df

df_matrix = generate_user_item_matrix()

# --- 2. Compute Similarity ---
# Calculate Cosine Similarity between USERS
user_similarity = cosine_similarity(df_matrix)
df_similarity = pd.DataFrame(user_similarity, index=df_matrix.index, columns=df_matrix.index)

# --- 3. UI & Logic ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Select Profile")
    selected_user = st.selectbox("Choose a Customer", df_matrix.index)
    
    # Show their history
    st.markdown("##### Purchase History:")
    user_history = df_matrix.loc[selected_user]
    purchased = user_history[user_history > 0].sort_values(ascending=False)
    
    if not purchased.empty:
        for product, rating in purchased.items():
            st.write(f"- {product} (Rating: {rating})")
    else:
        st.write("No purchases yet.")

with col2:
    st.subheader("💡 AI Recommendations")
    
    # Recommendation Logic
    # 1. Get similarity scores for this user
    sim_scores = df_similarity[selected_user]
    
    # 2. Find most similar user (excluding self)
    most_similar_user = sim_scores.drop(selected_user).idxmax()
    similarity_score = sim_scores[most_similar_user]
    
    st.info(f"**Most Similar Profile:** {most_similar_user} (Similarity: {similarity_score:.2f})")
    
    # 3. Recommend items that 'most_similar_user' bought BUT 'selected_user' hasn't
    similar_user_history = df_matrix.loc[most_similar_user]
    recommendations = []
    
    for product, rating in similar_user_history.items():
        if rating > 0 and user_history[product] == 0:
            recommendations.append((product, rating))
    
    # Sort by rating of similar user
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    if recommendations:
        st.success("🔥 Top Recommendations:")
        c_rec1, c_rec2, c_rec3 = st.columns(3)
        
        # Display up to 3 recs
        for i, (prod, rate) in enumerate(recommendations[:3]):
            with [c_rec1, c_rec2, c_rec3][i]:
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;">
                    <h4 style="margin:0; color: #2c3e50;">{prod}</h4>
                    <p style="margin:5px 0; color: #7f8c8d;">Rated {rate}/5 by similar users</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No new recommendations found (User has bought everything similar users bought).")

# --- 4. Visualizations ---
st.divider()
st.subheader("🧠 Under the Hood: The AI Brain")

tab1, tab2 = st.tabs(["User-Item Matrix (Raw Data)", "Similarity Heatmap (The AI Model)"])

with tab1:
    st.caption("Rows = Users, Columns = Products, Values = Rating/Purchase")
    st.dataframe(df_matrix.style.background_gradient(cmap='Blues'))

with tab2:
    st.caption("How close is User A to User B? (1.0 = Identical, 0.0 = Different)")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_similarity, cmap="RdBu_r", center=0, ax=ax)
    st.pyplot(fig)
