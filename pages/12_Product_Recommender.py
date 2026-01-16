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

# --- 1. Dynamic Data Loading ---
st.sidebar.header("🎛️ Data Configuration")
data_source = st.sidebar.radio("Data Source", ["Generate Synthetic Data", "Upload CSV File"])

@st.cache_data
def generate_user_item_matrix(n_users, n_products):
    np.random.seed(42)
    
    # Generic Tech Products
    base_products = [
        "Wireless Noise-Canceling Headphones", "Smart Watch Series 7", "Mechanical Gaming Keyboard", 
        "Ergonomic Office Chair", "4K Monitor 27-inch", "USB-C Docking Station", 
        "Portable Power Bank 20000mAh", "Bluetooth Smart Speaker", "Laptop Stand", "HD Webcam 1080p",
        "Tablet Pro 11", "Stylus Pen 2nd Gen", "Wireless Mouse RGB", "External SSD 1TB", "Robot Vacuum Cleaner"
    ]
    
    # Slice or extend products list
    if n_products <= len(base_products):
        products = base_products[:n_products]
    else:
        products = base_products + [f"Product {i+dummy_start}" for i in range(n_products - len(base_products))]
    
    # 0 = No purchase, 1-5 = Rating/Purchase Count
    data = np.random.randint(0, 6, size=(n_users, n_products))
    
    # Inject patterns (Simplified for variable size)
    # create some clusters if size permits
    if n_products >= 3:
         data[:int(n_users/2), :3] = np.random.randint(3, 6, size=(int(n_users/2), 3))
    
    # Random sparsity
    mask = np.random.choice([0, 1], size=data.shape, p=[0.3, 0.7])
    data = data * mask
    
    df = pd.DataFrame(data, columns=products, index=[f"User {i+1}" for i in range(n_users)])
    return df

df_matrix = None

if data_source == "Generate Synthetic Data":
    n_users = st.sidebar.slider("Number of Users", 10, 100, 20)
    n_products = st.sidebar.slider("Number of Products", 5, 20, 10)
    
    if st.sidebar.button("Generate New Data"):
        st.cache_data.clear()
        
    df_matrix = generate_user_item_matrix(n_users, n_products)

else:
    uploaded_file = st.sidebar.file_uploader("Upload User-Item Matrix (CSV)", type=["csv"])
    st.sidebar.info("CSV Format: Rows = Users, Columns = Products, Values = Ratings/Counts (0 for empty).")
    
    if uploaded_file:
        try:
            df_matrix = pd.read_csv(uploaded_file, index_col=0)
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

# --- 2. Main Logic ---
if df_matrix is not None:
    # Compute Similarity
    user_similarity = cosine_similarity(df_matrix)
    df_similarity = pd.DataFrame(user_similarity, index=df_matrix.index, columns=df_matrix.index)

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
        sim_scores = df_similarity[selected_user]
        
        # Find most similar user (excluding self)
        # Handle case where user has no similar neighbors (all 0 sim)
        try:
             most_similar_user = sim_scores.drop(selected_user).idxmax()
             similarity_score = sim_scores[most_similar_user]
        except:
             most_similar_user = None
             similarity_score = 0
        
        if similarity_score > 0:
            st.info(f"**Most Similar Profile:** {most_similar_user} (Similarity: {similarity_score:.2f})")
            
            # Recommend items
            similar_user_history = df_matrix.loc[most_similar_user]
            recommendations = []
            
            for product, rating in similar_user_history.items():
                if rating > 0 and user_history[product] == 0:
                    recommendations.append((product, rating))
            
            # Sort by rating of similar user
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            if recommendations:
                st.success("🔥 Top Recommendations:")
                
                # Dynamic columns for variable number of recs
                cols = st.columns(3)
                for i, (prod, rate) in enumerate(recommendations[:3]):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;">
                            <h4 style="margin:0; color: #2c3e50;">{prod}</h4>
                            <p style="margin:5px 0; color: #7f8c8d;">Rated {rate:.0f}/5 by similar users</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No new recommendations found (User has bought everything similar users bought).")
        else:
            st.warning("No similar users found yet (cold start).")

    # --- 3. Visualizations ---
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

else:
    st.info("👈 Please upload a CSV file or use synthetic data to begin.")
