import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from streamlit_option_menu import option_menu

# ===== Setup UI Page =====
st.set_page_config(page_title="Dashboard Prediksi Curah Hujan Surabaya", layout="wide")

# ===== Custom Style: Sidebar Purple =====
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #6a0dad;
        color: white;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] .css-1cpxqw2 {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Sidebar Menu =====
with st.sidebar:
    st.image("https://img.icons8.com/external-flat-juicy-fish/64/rain.png", width=40)
    st.title("Prediksi Curah Hujan")
    menu = option_menu("Navigasi", ["Dataset", "Visualisasi", "Model"],
        icons=["cloud-upload", "bar-chart", "cpu"],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#6a0dad"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "16px"},
            "nav-link-selected": {"background-color": "#a76ef4"},
        }
    )

# ===== Session State (simpan data) =====
if 'df' not in st.session_state:
    st.session_state.df = None

# ===== PAGE: Dataset =====
if menu == "Dataset":
    st.header("📂 Upload Dataset Curah Hujan Surabaya")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Dataset berhasil diunggah")
        st.dataframe(df.head())
    else:
        st.info("Silakan unggah dataset dengan kolom curah hujan.")

# ===== PAGE: Visualisasi =====
elif menu == "Visualisasi":
    st.header("📈 Visualisasi Data Curah Hujan")

    if st.session_state.df is not None:
        df = st.session_state.df

        col_date = st.selectbox("Pilih kolom tanggal", df.columns, index=0)
        col_rain = st.selectbox("Pilih kolom curah hujan", df.columns, index=1)

        fig, ax = plt.subplots()
        ax.plot(df[col_date], df[col_rain], color="#6a0dad", label="Curah Hujan")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Curah Hujan (mm)")
        ax.set_title("Grafik Curah Hujan Harian")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("⚠️ Belum ada dataset yang diunggah.")

# ===== PAGE: Model GA-LSTM =====
elif menu == "Model":
    st.header("🤖 Evaluasi Model GA-LSTM")

    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("📊 Metode: Genetic Algorithm + LSTM")
        st.markdown
