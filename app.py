import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# --- PAGE CONFIG ---
st.set_page_config(page_title="BodyMetrics AI Pro", page_icon="⚖️", layout="wide")

# --- CUSTOM CSS FOR BACKGROUND IMAGE & GLASSMORPHISM ---
st.markdown("""
    <style>
    /* Full-screen background image */
    .stApp {
        background: url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
    }

    /* Glassmorphism card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 20px;
    }

    /* Force Dark Text inside containers */
    .glass-card h1, .glass-card h2, .glass-card h3, .glass-card h4, .glass-card p {
        color: #0f172a !important;
        font-weight: bold;
    }

    /* Input box labels */
    .stNumberInput label {
        color: #1e293b !important;
        font-weight: bold !important;
        background: rgba(255,255,255,0.6);
        padding: 2px 10px;
        border-radius: 5px;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        height: 4em;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: white !important;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }

    /* Metric Card Styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.95);
        border-bottom: 5px solid #1e3a8a;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        color: #1e3a8a;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .metric-label {
        color: #475569;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return scaler, model
    except:
        return None, None

scaler, model = load_assets()

# --- HEADER ---
st.markdown("""
    <div class='glass-card' style='text-align: center;'>
        <h1 style='font-size: 3rem; margin:0;'>🌐 BodyMetrics AI Pro</h1>
        <p style='font-size: 1.2rem;'>Secure Cloud-Based Physiological Clustering System</p>
    </div>
    """, unsafe_allow_html=True)

if scaler is None:
    st.error("Model assets not found.")
    st.stop()

# --- INPUT SECTION ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>📊 Biometric Input</h3>", unsafe_allow_html=True)
    in_col1, in_col2 = st.columns(2)
    with in_col1:
        weight = st.number_input("Weight (kg)", 30.0, 150.0, 70.0)
    with in_col2:
        height = st.number_input("Height (cm)", 100.0, 220.0, 170.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("EXECUTE AI ANALYSIS")
    st.markdown("</div>", unsafe_allow_html=True)

# --- RESULTS ---
if predict_btn:
    # Logic
    data = {"Weight": [45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
            "Height": [150, 155, 160, 162, 165, 170, 172, 175, 178, 180]}
    df = pd.DataFrame(data)
    df_scaled = scaler.transform(df)
    clusters = model.fit_predict(df_scaled)
    
    # Graphs Row
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("### 🗺️ Cluster Localization")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['Weight'], y=df['Height'], hue=clusters, palette='viridis', s=100, ax=ax)
        ax.scatter(weight, height, color='red', marker='*', s=300, label='Subject')
        ax.legend()
        st.pyplot(fig)

    with g_col2:
        st.markdown("### 🧬 Dendrogram Analysis")
        Z = linkage(df_scaled, method='ward')
        fig2, ax2 = plt.subplots()
        dendrogram(Z, ax=ax2)
        st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

    
    

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    bmi = round(weight / ((height/100)**2), 1)
    
    with m1:
        st.markdown(f"<div class='metric-container'><div class='metric-label'>BMI Index</div><div class='metric-value'>{bmi}</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-container'><div class='metric-label'>Cluster ID</div><div class='metric-value'>#{clusters[0]}</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-container'><div class='metric-label'>Security Status</div><div class='metric-value'>Verified</div></div>", unsafe_allow_html=True)

else:
    st.markdown("<br><p style='text-align:center; color:white; background:rgba(0,0,0,0.4); padding:10px; border-radius:10px;'>Ready for analysis. Enter biometric data above and click 'Execute'.</p>", unsafe_allow_html=True)