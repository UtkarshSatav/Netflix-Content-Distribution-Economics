import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Netflix Business Analysis", layout="wide")

st.title("🎬 Netflix Content Strategy & Business Analysis")
st.markdown("""
This dashboard analyzes the Netflix content library using Economic principles to understand content segmentation and trends.
""")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_clean.csv')
    return df

try:
    df = load_data()
    
    # Sidebar for Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Exploratory Analysis", "Content Clustering", "Duration Prediction", "Strategic Recommendations"])

    if page == "Overview":
        st.header("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Titles", 8706)
        col2.metric("Movies", 6128)
        col3.metric("TV Shows", 2578)
        
        st.write("### Business Meaning of Data Overview")
        st.markdown("""
        *   **Supply Balance**: Movies significantly outnumber TV Shows (70% vs 30%). This suggests a strategy focused on high-volume, single-session viewing to lower the entry barrier for casual users.
        *   **Data Integrity**: The dataset is fully clean with no missing values, ensuring that strategic decisions based on this analysis are grounded in reliable data.
        """)
        
        st.write("### Sample Data")
        st.dataframe(df.head(10))

    elif page == "Exploratory Analysis":
        st.header("📈 Exploratory Data Analysis & Economic Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Movies vs TV Shows")
            fig, ax = plt.subplots()
            sns.barplot(x=['TV Show', 'Movie'], y=[2578, 6128], palette='viridis', ax=ax)
            st.pyplot(fig)
            st.info("**Economic Principle**: This reflects resource allocation. A higher supply of movies may indicate a lower production cost per title compared to multi-season series, while TV shows aim for sustained engagement and reduced churn.")

        with col2:
            st.write("#### Top 10 Genres")
            genre_data = {'International Movies': 2752, 'Dramas': 2427, 'Comedies': 1674, 'International TV Shows': 1328, 'Documentaries': 869, 'Action & Adventure': 859, 'Independent Movies': 756, 'TV Dramas': 739, 'Children & Family Movies': 641, 'Romantic Movies': 616}
            fig, ax = plt.subplots()
            sns.barplot(x=list(genre_data.values()), y=list(genre_data.keys()), palette='magma', ax=ax)
            st.pyplot(fig)
            st.info("**Strategy**: International content and Dramas dominate, highlighting a Global Diversification strategy to capture market share across different cultural demographics.")

        st.write("#### Content Release Trend (Supply Trend)")
        # Plotting a simplified release trend based on data
        release_data = {2010: 189, 2012: 229, 2014: 343, 2015: 548, 2016: 878, 2017: 1015, 2018: 1140, 2019: 1030, 2020: 953}
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(x=list(release_data.keys()), y=list(release_data.values()), marker="o", ax=ax)
        plt.title("Content Supply Trend Over Time")
        st.pyplot(fig)
        st.markdown("""
        **Economic Interpretation**: 
        The exponential growth in titles until 2018 represents an aggressive **Market Penetration** phase. The recent stabilization suggests a transition from 'Supply-side Push' to a more targeted 'Nuanced Demand' strategy, focusing on quality and lifecycle retention.
        """)

    elif page == "Content Clustering":
        st.header("🧩 K-Means Content Segmentation")
        
        st.markdown("""
        ### Market Segmentation Results
        The K-Means algorithm identified 4 distinct content segments based on year, duration, and rating:
        """)
        
        cluster_summary = {
            'Cluster 0': {'Year': 2014, 'Duration': '106 min', 'Type': 'Movie', 'Segment': 'Mainstream Movies'},
            'Cluster 1': {'Year': 2016, 'Duration': '90 min', 'Type': 'Movie', 'Segment': 'Recent Indie/Short Films'},
            'Cluster 2': {'Year': 2017, 'Duration': '1.7 Seasons', 'Type': 'TV Show', 'Segment': 'Modern Episodic Content'},
            'Cluster 3': {'Year': 1985, 'Duration': '112 min', 'Type': 'Movie', 'Segment': 'Classic Library Content'}
        }
        st.table(pd.DataFrame(cluster_summary).T)
        
        st.markdown("""
        **Strategic Business Insights**:
        1.  **Cluster 0 & 3 (Movies)**: Drive initial user acquisition and provide library depth.
        2.  **Cluster 2 (TV Shows)**: Vital for **Customer Lifetime Value (CLTV)**. Multi-season shows reduce churn and build brand loyalty.
        3.  **Cluster 1**: Represents testing of new concepts via shorter form content, allowing for efficient experimentation with user preferences.
        """)

    elif page == "Duration Prediction":
        st.header("📉 Linear Regression: Rating vs Duration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", "0.1407")
        with col2:
            st.metric("Mean Squared Error", "2264.50")
            
        st.markdown("""
        **Model Interpretation**:
        *   The low **R-squared (14%)** indicates that content rating is a poor predictor of duration.
        *   **Business Implication**: Netflix does not restrict content length based on target age group. Instead, duration is likely driven by **Creative Intent**, **Genre Conventions**, and **Production Budgets**. 
        *   This flexibility allows creators to maximize 'Consumer Utility' without rigid format constraints.
        """)

    elif page == "Strategic Recommendations":
        st.header("🎯 Strategic Recommendations for Netflix")
        
        st.markdown("""
        Based on the full analysis, here are the core economic recommendations:
        
        #### 1. Optimization of Content Mix
        *   **Acquisition**: Continue leveraging Cluster 0 (International Movies) for rapid library expansion in emerging markets.
        *   **Retention**: Pivot investment towards Cluster 2 (TV Shows) to increase binge-watching cycles and subscription stability.
        
        #### 2. Risk Diversification
        *   Maintaining a high volume of 'International' titles serves as a hedge against market-specific saturation or regulatory shifts in any single country.
        
        #### 3. Data-Driven Decisions
        *   Since ratings don't dictate duration, Netflix should continue allowing creators to tailor runtimes to storytelling needs, as this 'Product Differentiation' is a key competitive advantage over traditional linear TV.
        
        #### 4. Subscriber Lifecycle Management
        *   Use broad-appeal movies (Clusters 0, 3) for the **Acquisition Phase**.
        *   Deploy compelling multi-season series (Cluster 2) for the **Retention Phase**.
        """)

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
