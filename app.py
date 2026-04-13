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
    except FileNotFoundError:
        st.error("File model atau scaler tidak ditemukan. Pastikan file .pkl ada di folder yang sama.")
        return None, None

model, scaler = load_assets()

# --- TAMPILAN UTAMA ---
st.title("⚡ Household Energy Predictor")
st.markdown("""
Aplikasi ini memprediksi konsumsi daya listrik berdasarkan parameter elektrikal dan pola waktu harian.
""")

if model and scaler:
    # Menggunakan Tabs agar tampilan lebih rapi
    tab1, tab2 = st.tabs(["📝 Input Data", "📊 Hasil Analisis"])

    with tab1:
        st.subheader("Parameter Elektrikal")
        col1, col2 = st.columns(2)
        
        with col1:
            intensity = st.number_input("Global Intensity (Ampere)", value=5.0, step=0.1, help="Arus rata-rata dalam satu menit")
            voltage = st.number_input("Voltage (Volt)", value=240.0, step=0.1)
            reactive = st.number_input("Global Reactive Power (kW)", value=0.1, step=0.01)
        
        with col2:
            sub1 = st.number_input("Sub Metering 1 (Wh) - Dapur", value=0.0)
            sub2 = st.number_input("Sub Metering 2 (Wh) - Laundry", value=0.0)
            sub3 = st.number_input("Sub Metering 3 (Wh) - AC/Pemanas", value=15.0)

        st.divider()
        st.subheader("Konteks Waktu")
        # Hanya menyisakan jam sesuai logika model kita
        hour = st.select_slider("Pilih Jam Operasional", options=list(range(24)), value=19)
        st.info(f"Model akan memprediksi konsumsi daya pada pukul {hour}:00")

    with tab2:
        if st.button("🚀 Jalankan Prediksi", use_container_width=True):
            # Menyelaraskan 7 Fitur (Tanpa Date/Day/Month)
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
                # 1. Scaling
                input_scaled = scaler.transform(input_df)
                
                # 2. Predict
                prediction = model.predict(input_scaled)
                
                # Tampilan Hasil yang Menarik
                st.markdown("### Estimasi Penggunaan Daya")
                st.metric(label="Global Active Power", value=f"{prediction[0]:.4f} kW")
                
                # Penjelasan singkat
                if prediction[0] > 2.0:
                    st.warning("Konsumsi daya terdeteksi tinggi.")
                else:
                    st.success("Konsumsi daya terpantau normal.")
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")
        else:
            st.write("Silakan klik tombol di atas untuk melihat hasil.")

# --- FOOTER ---
st.divider()
st.caption("Case Project Machine Learning - Informatika")
