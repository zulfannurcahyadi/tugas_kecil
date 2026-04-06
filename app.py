import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Konfigurasi Halaman (Harus diletakkan paling atas)
st.set_page_config(
    page_title="Prediksi Konsumsi Energi",
    page_icon="⚡",
    layout="centered"
)

# Fungsi untuk memuat model dan mencegah error jika file hilang
@st.cache_resource
def load_components():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_rf.pkl', 'rb') as f:
            model = pickle.load(f)
        return scaler, model
    except FileNotFoundError:
        st.error("❌ File 'scaler.pkl' atau 'model_rf.pkl' tidak ditemukan! Pastikan file berada di folder yang sama dengan app.py.")
        st.stop()

scaler, model = load_components()

# Judul Aplikasi
st.title("⚡ Klasifikasi Konsumsi Energi Listrik")
st.markdown("""
Aplikasi ini memprediksi beban penggunaan energi rumah tangga (Rendah, Sedang, Tinggi) 
berdasarkan input parameter kelistrikan secara *real-time*.
""")
st.divider()

# Layout Input menggunakan dua kolom agar rapi
col1, col2 = st.columns(2)

with col1:
    st.subheader("Waktu & Arus")
    jam = st.slider("Jam Penggunaan (0 - 23)", min_value=0, max_value=23, value=12, step=1)
    intensitas_arus = st.number_input("Intensitas Arus (Ampere)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)

with col2:
    st.subheader("Tegangan & Tren Daya")
    tegangan = st.number_input("Tegangan (Volt)", min_value=200.0, max_value=250.0, value=240.0, step=0.5)
    # Rata-rata bergerak disimulasikan sebagai input dari pengguna untuk kebutuhan deployment sederhana
    rata_rata_5 = st.number_input("Tren Daya 5 Menit Terakhir (kW)", min_value=0.0, max_value=15.0, value=1.5, step=0.1)

# Proses Prediksi di balik layar
if st.button("🚀 Lakukan Prediksi", use_container_width=True):
    
    # 1. Feature Engineering: Menghitung Daya Semu secara otomatis
    daya_semu = tegangan * intensitas_arus
    
    # 2. Menyusun data input sesuai urutan fitur saat pelatihan:
    # ['Jam', 'Tegangan', 'Intensitas_Arus', 'Daya_Semu', 'Rata_Rata_Bergerak_5']
    input_data = np.array([[jam, tegangan, intensitas_arus, daya_semu, rata_rata_5]])
    
    # 3. Menerapkan Scaler pada data input
    input_scaled = scaler.transform(input_data)
    
    # 4. Melakukan Prediksi
    prediksi = model.predict(input_scaled)[0]
    probabilitas = model.predict_proba(input_scaled)
    akurasi_prediksi = np.max(probabilitas) * 100
    
    # 5. Menampilkan Output yang profesional dan terwarna
    st.divider()
    st.subheader("Hasil Klasifikasi:")
    
    if prediksi == 0:
        st.success("🟢 **Kelas 0 : KONSUMSI RENDAH**")
        st.info(f"Tingkat Keyakinan Model: **{akurasi_prediksi:.2f}%**\n\nSistem kelistrikan berada pada beban dasar yang aman dan stabil.")
    elif prediksi == 1:
        st.warning("🟡 **Kelas 1 : KONSUMSI SEDANG**")
        st.info(f"Tingkat Keyakinan Model: **{akurasi_prediksi:.2f}%**\n\nPemakaian listrik terpantau aktif. Beberapa alat elektronik utama sedang beroperasi.")
    else:
        st.error("🔴 **Kelas 2 : KONSUMSI TINGGI**")
        st.info(f"Tingkat Keyakinan Model: **{akurasi_prediksi:.2f}%**\n\nPeringatan: Beban arus tinggi terdeteksi. Alat berat seperti AC atau pemanas kemungkinan besar sedang aktif.")
        
    st.caption(f"*Info Teknis: Daya Semu terhitung pada sistem adalah {daya_semu:.2f} VA*")
