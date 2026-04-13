import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from pathlib import Path

# --- LOAD ASSETS ---
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

model, scaler = load_assets()

# --- UI ---
st.title("⚡ Household Energy Predictor")

# --- INPUT ---
col1, col2 = st.columns(2)
with col1:
    reactive = st.number_input("Global Reactive Power (kW)", value=0.1)
    voltage = st.number_input("Voltage (V)", value=240.0)
    intensity = st.number_input("Global Intensity (A)", value=5.0)
    sub1 = st.number_input("Sub Metering 1 (Wh)", value=0.0)

with col2:
    sub2 = st.number_input("Sub Metering 2 (Wh)", value=0.0)
    sub3 = st.number_input("Sub Metering 3 (Wh)", value=15.0)
    hour = st.slider("Pilih Jam", 0, 23, 19)

# --- PROSES PREDIKSI ---
if st.button("Hitung Prediksi"):
    # Kita hapus 'day' dan 'month' karena Scaler kamu tidak mengenalnya saat training
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
        # 1. Scaling (Sekarang jumlah fitur sudah pas yaitu 7 fitur)
        input_scaled = scaler.transform(input_df)
        
        # 2. Predict
        prediction = model.predict(input_scaled)
        
        st.success(f"Hasil Prediksi: {prediction[0]:.4f} kW")
        st.balloons()
        
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
