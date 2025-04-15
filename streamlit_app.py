import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ===== UI SETUP =====
st.set_page_config(page_title="Prediksi Curah Hujan Surabaya", layout="wide")

# ===== HEADER =====
st.markdown("""
<style>
h1 {
    font-family: 'Helvetica Neue', sans-serif;
    color: #0077b6;
}
body {
    background-color: #f5f5f5;
}
.sidebar .sidebar-content {
    background-color: #e0f7fa;
}
</style>
""", unsafe_allow_html=True)

st.title("🌧️ Prediksi Curah Hujan di Surabaya")
st.caption("Sumber data: Simulasi atau API BMKG (dummy data)")

# ===== DATA SIMULASI =====
# Prediksi curah hujan jam-jam
today = datetime.now().date()
jam = pd.date_range(start="00:00", end="23:00", freq="1H").time
curah_hujan = [round(max(0, 10 + 10*i%5 - 3*(i%3)), 1) for i in range(24)]
df = pd.DataFrame({"Jam": jam, "Curah Hujan (mm)": curah_hujan})

# ===== TAB LAYOUT =====
tab1, tab2, tab3 = st.tabs(["📊 Grafik Harian", "🗺️ Peta Curah Hujan", "📚 Info Cuaca"])

# ===== TAB 1: Grafik Harian =====
with tab1:
    st.subheader(f"Prediksi Curah Hujan Hari Ini ({today.strftime('%d %B %Y')})")
    fig = px.area(df, x="Jam", y="Curah Hujan (mm)",
                  labels={"Jam": "Jam", "Curah Hujan (mm)": "Curah Hujan (mm)"},
                  color_discrete_sequence=["#4FC3F7"])
    fig.update_layout(xaxis_tickangle=-45, plot_bgcolor="#f5f5f5", paper_bgcolor="#f5f5f5")
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 Tips: Waspadai curah hujan tinggi antara jam 14:00 - 18:00 di wilayah Surabaya Timur.")

# ===== TAB 2: Peta Curah Hujan (Mock) =====
with tab2:
    st.subheader("Peta Intensitas Curah Hujan Surabaya (Simulasi)")

    st.markdown("📍 Lokasi: Surabaya\n\n⚠️ *Peta interaktif dalam pengembangan. Fitur ini akan menggunakan GeoData Surabaya.*")

    st.image("https://i.ibb.co/ZVLkTxs/rainmap.jpg", caption="Peta Curah Hujan Surabaya (Ilustrasi)", use_column_width=True)

# ===== TAB 3: Info Edukatif =====
with tab3:
    st.subheader("Apa Itu Curah Hujan (mm)?")
    st.write("""
    Curah hujan diukur dalam milimeter (mm), yang menunjukkan tinggi air hujan jika ditampung dalam wadah rata:
    
    - **0-5 mm**: Hujan ringan  
    - **5-20 mm**: Hujan sedang  
    - **20-50 mm**: Hujan lebat  
    - **>50 mm**: Hujan sangat lebat
    
    Sumber: BMKG
    """)

    st.success("💧 Curah hujan tinggi secara berturut-turut dapat menyebabkan banjir. Perhatikan wilayah rawan seperti Rungkut, Wonokromo, dan Tambaksari.")

# ===== FOOTER =====
st.markdown("---")
st.markdown("Made with ❤️ untuk warga Surabaya | Desain oleh ahli UI/UX Amerika 🇺🇸, Jepang 🇯🇵, Inggris 🇬🇧")

