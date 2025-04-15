import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# ==== Page Setup ====
st.set_page_config(page_title="Prediksi Curah Hujan Surabaya", layout="wide")

# ==== Custom CSS untuk Sidebar Ungu ====
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
    [data-testid="stSidebar"] .css-1cpxqw2,
    [data-testid="stSidebar"] .css-10trblm {
        color: white !important;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==== Sidebar Menu ====
with st.sidebar:
    st.image("https://img.icons8.com/external-flat-juicy-fish/64/rain.png", width=40)
    st.title("Dashboard Hujan")
    menu = option_menu("Menu Utama", ["Upload Dataset", "Visualisasi", "Hasil Peramalan"],
                       icons=["cloud-upload", "bar-chart-line", "activity"],
                       menu_icon="list", default_index=0,
                       styles={
                           "container": {"padding": "5px", "background-color": "#6a0dad"},
                           "icon": {"color": "white", "font-size": "20px"},
                           "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px"},
                           "nav-link-selected": {"background-color": "#a76ef4"},
                       })

# ==== Halaman: Upload Dataset ====
if menu == "Upload Dataset":
    st.header("📂 Upload Dataset Curah Hujan")
    file = st.file_uploader("Upload file CSV berisi data curah hujan", type=["csv"])
    
    if file:
        df = pd.read_csv(file)
        st.success("✅ Dataset berhasil diunggah")
        st.dataframe(df.head())
    else:
        st.info("Silakan unggah file .csv yang berisi data waktu dan curah hujan.")

# ==== Halaman: Visualisasi ====
elif menu == "Visualisasi":
    st.header("📈 Visualisasi Curah Hujan")
    if "df" in locals():
        fig, ax = plt.subplots()
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], label="Curah Hujan", color="#6a0dad")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Curah Hujan (mm)")
        ax.set_title("Grafik Curah Hujan")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("❗ Harap unggah dataset terlebih dahulu di menu 'Upload Dataset'.")

# ==== Halaman: Hasil Peramalan ====
elif menu == "Hasil Peramalan":
    st.header("🔮 Hasil Peramalan Curah Hujan")
    st.markdown("Fitur ini akan menampilkan hasil prediksi dari model (misal LSTM).")
    
    # Contoh Dummy Output
    prediksi_dummy = np.random.uniform(10, 100, size=7)
    tanggal_dummy = pd.date_range(start=pd.Timestamp.today(), periods=7).strftime("%d %b %Y")

    pred_df = pd.DataFrame({
        "Tanggal": tanggal_dummy,
        "Prediksi Curah Hujan (mm)": np.round(prediksi_dummy, 2)
    })

    st.table(pred_df)

# ==== Footer ====
st.markdown("---")
st.caption("🎓 Dibuat Dienna Eries Linggarsari ")
