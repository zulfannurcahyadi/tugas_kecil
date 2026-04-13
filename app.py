import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Energy Predictor", page_icon="⚡", layout="centered")

# --- LOAD ASSETS ---
base_path = Path(__file__).parent
model_path = base_path / 'model_listrik_best.pkl'
scaler_path = base_path / 'scaler.pkl'

@st.cache_resource
def load_assets():
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model/scaler: {e}")
        return None, None

model, scaler = load_assets()

# --- UI HEADER ---
st.title("⚡ Household Energy Predictor")
st.markdown("""
Aplikasi ini memprediksi konsumsi daya listrik rumah tangga berdasarkan parameter teknis 
dan pola waktu (jam). Fitur tanggal telah dihilangkan untuk efisiensi model.
""")

if model is not None and scaler is not None:
    # --- INPUT SECTION ---
    st.subheader("📊 Input Parameter Listrik")
    
    col1, col2 = st.columns(2)
    with col1:
        reactive = st.number_input("Global Reactive Power (kW)", value=0.1, format="%.3f")
        voltage = st.number_input("Voltage (Volt)", value=240.0, format="%.1f")
        intensity = st.number_input("Global Intensity (Ampere)", value=5.0, format="%.2f")

    with col2:
        sub1 = st.number_input("Sub Metering 1 (Wh) - Dapur", value=0.0)
        sub2 = st.number_input("Sub Metering 2 (Wh) - Laundry", value=0.0)
        sub3 = st.number_input("Sub Metering 3 (Wh) - AC/Pemanas", value=15.0)

    st.divider()
    
    # --- INPUT WAKTU (JAM SAJA) ---
    st.subheader("🕒 Konteks Waktu")
    hour = st.select_slider("Pilih Jam Operasional (0-23)", options=list(range(24)), value=19)

    # --- PROSES PREDIKSI ---
    if st.button("🚀 Hitung Prediksi Daya", use_container_width=True):
        # Menyusun data sesuai urutan 7 fitur yang dilatih pada model
        data_input = {
            'Global_reactive_power': [reactive],
            'Voltage': [voltage],
            'Global_intensity': [intensity],
            'Sub_metering_1': [sub1],
            'Sub_metering_2': [sub2],
            'Sub_metering_3': [sub3],
            'hour': [hour]
        }
        
        input_df = pd.DataFrame(data_input)
        
        try:
            # 1. Transformasi data menggunakan Scaler yang sudah dilatih
            input_scaled = scaler.transform(input_df)
            
            # 2. Prediksi menggunakan model Random Forest
            prediction = model.predict(input_scaled)
            
            # --- TAMPILAN HASIL ---
            st.divider()
            st.markdown("### 💡 Hasil Estimasi")
            
            # Menampilkan hasil dengan komponen metric agar lebih menarik
            st.metric(label="Global Active Power", value=f"{prediction[0]:.4f} kW")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Terjadi kesalahan teknis saat prediksi: {e}")

# --- FOOTER ---
st.divider()
st.caption("Dikembangkan untuk Case Project Machine Learning - Informatika")
