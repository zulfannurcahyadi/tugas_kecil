import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model dan Scaler
model = pickle.load(open('model_listrik_best.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("⚡ Smart Energy Predictor (Random Forest)")
st.write(f"Model Accuracy (R2 Score): 0.999")

# Input Form
col1, col2 = st.columns(2)
with col1:
    reactive = st.number_input("Global Reactive Power (kW)", value=0.1)
    voltage = st.number_input("Voltage (V)", value=240.0)
    intensity = st.number_input("Global Intensity (A)", value=5.0)
    sub1 = st.number_input("Sub Metering 1 (Dapur)", value=0.0)

with col2:
    sub2 = st.number_input("Sub Metering 2 (Cuci)", value=0.0)
    sub3 = st.number_input("Sub Metering 3 (AC/Heater)", value=0.0)
    hour = st.slider("Hour", 0, 23, 12)
    day = st.selectbox("Day (0=Mon, 6=Sun)", [0, 1, 2, 3, 4, 5, 6])
    month = st.slider("Month", 1, 12, 1)

if st.button("Prediksi"):
    # Input harus sesuai urutan fitur saat training: 
    # [Reactive, Voltage, Intensity, Sub1, Sub2, Sub3, Hour, Day, Month]
    features = np.array([[reactive, voltage, intensity, sub1, sub2, sub3, hour, day, month]])
    
    # Transformasi dengan Scaler
    features_scaled = scaler.transform(features)
    
    # Prediksi
    prediction = model.predict(features_scaled)
    
    st.success(f"Prediksi Daya Aktif: {prediction[0]:.4f} kW")
