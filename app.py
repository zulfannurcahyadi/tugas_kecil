import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load Model dan Scaler
model = pickle.load(open('model_regresi.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Prediksi Penggunaan Listrik Rumah Tangga")
st.write("Masukkan data di bawah ini untuk memprediksi Global Active Power (kW)")

# 2. Input Form (Sesuaikan dengan fitur yang kamu pakai)
# Sesuai pembahasan kita, fitur: Reactive, Voltage, Intensity, Sub1, Sub2, Sub3, Hour, Day, Month
col1, col2 = st.columns(2)

with col1:
    reactive = st.number_input("Global Reactive Power (kW)", value=0.1)
    voltage = st.number_input("Voltage (V)", value=240.0)
    intensity = st.number_input("Global Intensity (A)", value=5.0)
    sub1 = st.number_input("Sub Metering 1 (Kitchen)", value=0.0)

with col2:
    sub2 = st.number_input("Sub Metering 2 (Laundry)", value=0.0)
    sub3 = st.number_input("Sub Metering 3 (AC/Heater)", value=0.0)
    hour = st.slider("Hour", 0, 23, 12)
    day = st.selectbox("Day of Week (0=Mon, 6=Sun)", [0, 1, 2, 3, 4, 5, 6])
    month = st.slider("Month", 1, 12, 1)

# 3. Tombol Prediksi
if st.button("Prediksi Sekarang"):
    # Gabungkan input menjadi array
    input_data = np.array([[reactive, voltage, intensity, sub1, sub2, sub3, hour, day, month]])
    
    # --- PROSES PENTING: SCALING ---
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)
    
    st.success(f"Estimasi Penggunaan Daya (Global Active Power): {prediction[0]:.4f} kW")
