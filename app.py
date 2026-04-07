import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# --- LOAD MODEL DAN SCALER ---
def load_model():
    with open('model_xgb.pkl', 'rb') as f:
        model = pickle.dump = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# --- HEADER APLIKASI ---
st.set_page_config(page_title="Prediksi Beban Listrik", layout="centered")
st.title("⚡ Sistem Klasifikasi Beban Listrik")
st.write("Aplikasi ini memprediksi kategori penggunaan listrik berdasarkan parameter sensor menggunakan model XGBoost.")

st.divider()

# --- INPUT FORM ---
st.subheader("Input Parameter Listrik")
col1, col2 = st.columns(2)

with col1:
    jam = st.number_input("Jam (0-23)", min_value=0, max_value=23, value=12)
    tegangan = st.number_input("Tegangan (Volt)", min_value=0.0, value=230.0)
    intensitas = st.number_input("Intensitas Arus (Ampere)", min_value=0.0, value=0.5)

with col2:
    daya_semu = tegangan * intensitas
    st.text_input("Daya Semu (VA) - Auto", value=f"{daya_semu:.2f}", disabled=True)
    
    rolling_mean = st.number_input("Rata-rata Arus (Rolling Mean 5 Menit)", min_value=0.0, value=0.5)
    
# --- PROSES PREDIKSI ---
if st.button("Prediksi Kategori Beban"):
    # 1. Susun data input menjadi DataFrame sesuai urutan saat training
    input_data = pd.DataFrame([[jam, tegangan, intensitas, daya_semu, rolling_mean]], 
                              columns=['Jam', 'Tegangan', 'Intensitas_Arus', 'Daya_Semu', 'Rata_Rata_Bergerak_5'])
    
    # 2. Scaling data input
    input_scaled = scaler.transform(input_data)
    
    # 3. Prediksi
    prediction = model.predict(input_scaled)[0]
    
    # 4. Mapping Label
    kategori = {0: "RENDAH", 1: "SEDANG", 2: "TINGGI"}
    warna = {0: "green", 1: "orange", 2: "red"}
    
    st.divider()
    st.subheader("Hasil Analisis:")
    st.markdown(f"### Kategori Beban: :{warna[prediction]}[**{kategori[prediction]}**]")
    
    # Penjelasan Tambahan
    if prediction == 0:
        st.info("Penggunaan energi sangat efisien. Beban pada perangkat elektronik minimal.")
    elif prediction == 1:
        st.warning("Penggunaan energi dalam batas wajar namun perlu diperhatikan.")
    else:
        st.error("Beban energi sangat tinggi! Segera periksa perangkat elektronik Anda untuk menghindari pemborosan.")
