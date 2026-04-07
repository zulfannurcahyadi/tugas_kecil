import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Energy Monitor Pro", page_icon="⚡", layout="centered")

# --- LOAD MODEL DAN SCALER ---
@st.cache_resource
def load_assets():
    with open('model_xgb.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# --- CUSTOM CSS UNTUK TAMPILAN KEREN ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("⚡ Energy Monitor Pro")
st.markdown("---")

# --- INPUT SECTION ---
st.subheader("📊 Input Parameter Sensor")
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        jam = st.number_input("🕒 Jam (0-23)", 0, 23, 12)
    with c2:
        tegangan = st.number_input("🔌 Tegangan (V)", 0.0, 300.0, 230.0)
    with c3:
        intensitas = st.number_input("📉 Arus (A)", 0.0, 50.0, 0.5)

    c4, c5 = st.columns(2)
    with c4:
        # Konversi ke kW untuk tampilan (Watt / 1000)
        daya_watt = tegangan * intensitas
        daya_kw = daya_watt / 1000
        st.write(f"**Estimasi Daya:** {daya_kw:.4f} kW")
    with c5:
        rolling_mean = st.number_input("🔄 Rata-rata Arus (5m)", 0.0, 50.0, 0.5)

# --- PREDIKSI ---
st.markdown("###")
if st.button("🚀 ANALISIS BEBAN LISTRIK", use_container_width=True):
    # Data untuk model tetap dalam Watt (Daya_Semu) sesuai training
    input_data = pd.DataFrame([[jam, tegangan, intensitas, daya_watt, rolling_mean]], 
                              columns=['Jam', 'Tegangan', 'Intensitas_Arus', 'Daya_Semu', 'Rata_Rata_Bergerak_5'])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    # Mapping Label
    kategori = {0: "EFISIEN (RENDAH)", 1: "NORMAL (SEDANG)", 2: "WARNING (TINGGI)"}
    warna = {0: "normal", 1: "off", 2: "inverse"}
    
    st.divider()
    
    # --- HASIL ---
    st.subheader("📋 Hasil Diagnosa Sistem")
    
    # Tampilan kW yang lebih keren
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="Konsumsi Daya Saat Ini", value=f"{daya_kw:.3f} kW")
    with col_res2:
        # Menampilkan kategori dengan gaya badge
        if prediction == 0:
            st.success(f"Status: {kategori[0]}")
        elif prediction == 1:
            st.warning(f"Status: {kategori[1]}")
        else:
            st.error(f"Status: {kategori[2]}")

    # Tips Interaktif
    messages = [
        "Sistem berjalan sangat efisien. Tidak ada pemborosan terdeteksi.",
        "Pemakaian stabil. Pertahankan pola penggunaan ini untuk menghemat biaya.",
        "Beban puncak terdeteksi! Segera matikan perangkat yang tidak digunakan."
    ]
    st.info(messages[prediction])
