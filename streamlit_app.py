import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

st.set_page_config(page_title="GA-LSTM Rainfall Forecasting", layout="wide")

# Sidebar Navigation
menu = st.sidebar.selectbox("Navigasi", ["Beranda", "Preprocessing", "Hasil Pemodelan", "RMFE"])

# Global Variables
if "df" not in st.session_state:
    st.session_state.df = None
if "scaled_data" not in st.session_state:
    st.session_state.scaled_data = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "train_test" not in st.session_state:
    st.session_state.train_test = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "risk_df" not in st.session_state:
    st.session_state.risk_df = None

# ========== Beranda ==========
if menu == "Beranda":
    st.title("Prediksi Curah Hujan Harian Menggunakan GA-LSTM")
    st.markdown("""
    **GA-LSTM** (Genetic Algorithm - Long Short-Term Memory) adalah metode hybrid yang memanfaatkan kekuatan optimasi Genetic Algorithm dan kemampuan prediktif LSTM untuk menghasilkan prediksi curah hujan yang lebih akurat.
    
    Silakan unggah data Anda dengan format Excel (.xlsx).
    """)
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("File berhasil diupload!")
        st.write("5 data pertama:")
        st.dataframe(df.head())

# ========== Preprocessing ==========
elif menu == "Preprocessing":
    st.title("Preprocessing Data")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        df['RR'] = df['RR'].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
        df['RR'] = pd.to_numeric(df['RR'], errors='coerce').fillna(0)
        df['Tanggal'] = pd.to_datetime(df['Tanggal']) if 'Tanggal' in df.columns else pd.date_range(start='2025-01-01', periods=len(df))

        data_asli = df['RR'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_asli)
        df_scaled = pd.DataFrame({
            'Tanggal': df['Tanggal'],
            'Curah Hujan Asli': data_asli.flatten(),
            'Curah Hujan Ternormalisasi': scaled_data.flatten()
        })

        look_back = 7
        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i + look_back])
            y.append(scaled_data[i + look_back])
        X, y = np.array(X), np.array(y)
        split = int(0.8 * len(X))
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

        st.session_state.scaled_data = scaled_data
        st.session_state.scaler = scaler
        st.session_state.train_test = (X_train, X_test, y_train, y_test)

        st.subheader("Data Setelah Normalisasi")
        st.dataframe(df_scaled.head())

        st.subheader("Data Training (Ternormalisasi)")
        st.dataframe(pd.DataFrame(X_train.reshape(X_train.shape[0], -1)))

        st.subheader("Data Testing (Ternormalisasi)")
        st.dataframe(pd.DataFrame(X_test.reshape(X_test.shape[0], -1)))
    else:
        st.warning("Silakan upload data terlebih dahulu di halaman Beranda.")

# ========== Hasil Pemodelan ==========
elif menu == "Hasil Pemodelan":
    st.title("Hasil Pemodelan GA-LSTM")
    if st.session_state.train_test:
        X_train, X_test, y_train, y_test = st.session_state.train_test
        scaler = st.session_state.scaler

        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test)

        st.subheader("Evaluasi Model")
        st.write("**RMSE:**", np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
        st.write("**MAE:**", mean_absolute_error(y_test_inv, y_pred_inv))
        st.write("**MSE:**", mean_squared_error(y_test_inv, y_pred_inv))

        st.subheader("Plot Data Aktual vs Prediksi")
        fig, ax = plt.subplots()
        ax.plot(y_test_inv, label='Aktual')
        ax.plot(y_pred_inv, label='Prediksi')
        ax.legend()
        st.pyplot(fig)

        # Prediksi 14 hari ke depan
        input_data = st.session_state.scaled_data[-7:].reshape(1, 7, 1)
        preds = []
        for _ in range(14):
            pred = model.predict(input_data)
            preds.append(pred[0][0])
            input_data = np.append(input_data[:,1:,:], [[[pred[0][0]]]], axis=1)

        future = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        last_date = st.session_state.df['Tanggal'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1,15)]
        future_df = pd.DataFrame({'Tanggal': future_dates, 'Prediksi Curah Hujan (mm)': future})


        st.subheader("Prediksi Curah Hujan 14 Hari Kedepan")
        st.dataframe(future_df)

        fig2, ax2 = plt.subplots()
        ax2.plot(future_dates, future, marker='o')
        ax2.set_title("Prediksi 14 Hari Kedepan")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Curah Hujan (mm)")
        st.pyplot(fig2)

        st.session_state.prediction = future_df
    else:
        st.warning("Data belum diproses. Silakan lakukan preprocessing terlebih dahulu.")

# ========== Halaman RMFE ==========
elif menu == "RMFE":
    st.title("Risk Matrix Flood Estimation")
    if st.session_state.prediction is not None:
        df_pred = st.session_state.prediction

        def get_impact_details(r):
            if r < 30:
                return "Very Low", 1, 0, 0
            elif r < 60:
                return "Low", 2, 30, 10
            elif r < 100:
                return "Medium", 3, 60, 50
            elif r < 150:
                return "High", 4, 100, 200
            else:
                return "Very High", 5, 150, 1000

        def get_probability(r):
            if r < 30:
                return 0.5
            elif r < 60:
                return 0.2
            elif r < 100:
                return 0.1
            elif r < 150:
                return 0.04
            else:
                return 0.02

        df_pred['impact_level'], df_pred['impact_score'], df_pred['estimated_height_cm'], df_pred['estimated_affected_buildings'] = zip(*df_pred['Prediksi Curah Hujan (mm)'].map(get_impact_details))
        df_pred['probability_per_year'] = df_pred['Prediksi Curah Hujan (mm)'].map(get_probability)
        df_pred['risk_score_buildings_per_year'] = df_pred['probability_per_year'] * df_pred['estimated_affected_buildings']

        st.dataframe(df_pred)
        st.session_state.risk_df = df_pred
    else:
        st.warning("Prediksi belum tersedia. Silakan jalankan pemodelan terlebih dahulu.")
