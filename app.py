import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from pathlib import Path

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Energy Predictor Pro", layout="centered")

# --- LOAD MODEL & SCALER ---
# Menggunakan Path agar aman saat di-deploy di server Linux/Cloud
base_path = Path(__file__).parent
model_path = base_path / 'model_listrik_best.pkl'
scaler_path = base_path / 'scaler.pkl'

@st.cache_resource
def load_assets():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan. Pastikan file .pkl sudah di-upload ke GitHub.")
    st.stop()

# --- INTERFACE APLIKASI ---
st.title("⚡ Household Energy Predictor")
st.markdown(f"**Model Performance:** R² Score 0.999 (Random Forest)")
st.write("Masukkan parameter di bawah ini untuk memprediksi Global Active Power.")

# --- FORM INPUT ---
with st.container():
    st.subheader("1. Parameter Listrik (Teknis)")
    col1, col2, col3 = st.columns(3)
    with col1:
        reactive = st.number_input("Reactive Power (kW)", min_value=0.0, value=0.1, step=0.01)
    with col2:
        voltage = st.number_input("Voltage (V)", min_value=200.0, value=240.0, step=0.1)
    with col3:
        intensity = st.number_input("Intensity (A)", min_value=0.0, value=5.0, step=0.1)

    st.subheader("2. Konsumsi Sub-Metering (Wh)")
    c1, c2, c3 = st.columns(3)
    with c1:
        sub1 = st.number_input("Kitchen", min_value=0.0, value=0.0)
    with c2:
        sub2 = st.number_input("Laundry", min_value=0.0, value=0.0)
    with c3:
        sub3 = st.number_input("AC & Heater", min_value=0.0, value=15.0)

    st.subheader("3. Waktu & Kalender")
    ca, cb = st.columns(2)
    with ca:
        # User cukup pilih tanggal, hari dan bulan akan dihitung otomatis
        input_date = st.date_input("Pilih Tanggal", datetime.date(2006, 12, 17))
        day = input_date.weekday() # Senin=0, Minggu=6
        month = input_date.month
        
        nama_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        st.caption(f"Terdeteksi: Hari **{nama_hari[day]}**, Bulan **{month}**")
        
    with cb:
        hour = st.slider("Pilih Jam (Hour)", 0, 23, 19)

# --- TOMBOL PREDIKSI ---
st.markdown("---")
if st.button("Hitung Prediksi Daya", use_container_width=True):
    # MENYELESAIKAN VALUE ERROR:
    # Membuat DataFrame dengan nama kolom yang sama persis saat training
    input_df = pd.DataFrame({
        'Global_reactive_power': [reactive],
        'Voltage': [voltage],
        'Global_intensity': [intensity],
        'Sub_metering_1': [sub1],
        'Sub_metering_2': [sub2],
        'Sub_metering_3': [sub3],
        'hour': [hour],
        'day': [day],
        'month': [month]
    })

    # 1. Transformasi data menggunakan scaler
    input_scaled = scaler.transform(input_df)

    # 2. Prediksi menggunakan model
    prediction = model.predict(input_scaled)

    # 3. Tampilkan Hasil
    st.balloons()
    st.metric(label="Estimasi Global Active Power", value=f"{prediction[0]:.4f} kW")
    
    st.info("Hasil ini didasarkan pada pola data historis dengan akurasi model 99.9%.")
