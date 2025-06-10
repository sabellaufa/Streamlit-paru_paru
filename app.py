import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# === 1. Load Model dan Scaler ===
# Pastikan file-file ini berada di direktori yang sama dengan app.py
try:
    model = tf.keras.models.load_model('model_paru_paru.h5')
    scaler = joblib.load('scaler.save')
    st.success("Model dan Scaler berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model atau scaler. Pastikan file ada: {e}")
    st.stop() # Hentikan aplikasi jika gagal memuat

# === 2. Judul Aplikasi ===
st.set_page_config(page_title="Prediksi Penyakit Paru-paru")
st.title("Aplikasi Prediksi Penyakit Paru-paru")
st.write("Isi data berikut untuk memprediksi risiko penyakit paru-paru.")

# === 3. Input Fitur dari Pengguna ===

# Menggunakan expander untuk tata letak yang lebih rapi
with st.expander("Informasi Pribadi"):
    # Asumsi mapping: Perempuan=0, Laki-laki=1
    jenis_kelamin = st.selectbox("Jenis Kelamin", ("Perempuan", "Laki-laki"))
    usia = st.number_input("Usia", min_value=0, max_value=120, value=30, step=1)

with st.expander("Gaya Hidup & Kebiasaan"):
    # Asumsi mapping: Tidak=0, Ya=1
    bekerja = st.selectbox("Status Pekerjaan", ("Tidak", "Ya"))
    rumah_tangga = st.selectbox("Status Rumah Tangga (Apakah mengurus rumah tangga)", ("Tidak", "Ya"))
    aktivitas_begadang = st.selectbox("Kebiasaan Begadang", ("Tidak", "Ya"))
    aktivitas_olahraga = st.selectbox("Rutin Berolahraga", ("Tidak", "Ya"))
    merokok = st.number_input("Jumlah Batang Rokok per Hari (jika merokok)", min_value=0, max_value=100, value=0, step=1)

with st.expander("Riwayat Medis"):
    # Asumsi mapping: Tidak=0, Ya=1
    penyakit_bawaan = st.selectbox("Memiliki Penyakit Bawaan", ("Tidak", "Ya"))
    asuransi = st.selectbox("Memiliki Asuransi Kesehatan", ("Tidak", "Ya"))

# === 4. Tombol Prediksi ===
if st.button("Prediksi Risiko Penyakit Paru-paru"):
    # === 5. Preprocessing Input Pengguna ===
    # Penting: Mapping ini HARUS sesuai dengan bagaimana LabelEncoder bekerja pada data training
    # Jika di training 'Tidak' -> 0 dan 'Ya' -> 1, maka di sini juga harus sama.
    # Saya telah mengubah urutan di selectbox dan mapping di sini agar konsisten dengan asumsi 'Tidak'/'Perempuan' = 0 dan 'Ya'/'Laki-laki' = 1.
    # Mohon pastikan kembali bagaimana LabelEncoder Anda mengkodekan pada training.

    # Encoding untuk fitur kategorikal (sesuai urutan kolom df.info() Anda)
    jk_encoded = 1 if jenis_kelamin == "Laki-laki" else 0
    bekerja_encoded = 1 if bekerja == "Ya" else 0
    rumah_tangga_encoded = 1 if rumah_tangga == "Ya" else 0
    aktivitas_begadang_encoded = 1 if aktivitas_begadang == "Ya" else 0
    aktivitas_olahraga_encoded = 1 if aktivitas_olahraga == "Ya" else 0
    asuransi_encoded = 1 if asuransi == "Ya" else 0
    penyakit_bawaan_encoded = 1 if penyakit_bawaan == "Ya" else 0

    # Mengumpulkan semua fitur dalam urutan yang sama seperti saat training
    # Urutan kolom dari df.info() Anda adalah:
    # Jenis_Kelamin, Bekerja, Rumah_Tangga, Aktivitas_Begadang, Aktivitas_Olahraga,
    # Asuransi, Penyakit_Bawaan, Merokok, Usia
    input_data = pd.DataFrame([[
        jk_encoded,
        bekerja_encoded,
        rumah_tangga_encoded,
        aktivitas_begadang_encoded,
        aktivitas_olahraga_encoded,
        asuransi_encoded,
        penyakit_bawaan_encoded,
        merokok,  # Numerik, akan diskalakan
        usia      # Numerik, akan diskalakan
    ]], columns=[
        'Jenis_Kelamin', 'Bekerja', 'Rumah_Tangga', 'Aktivitas_Begadang',
        'Aktivitas_Olahraga', 'Asuransi', 'Penyakit_Bawaan', 'Merokok', 'Usia'
    ])

    # Mengidentifikasi kolom numerik untuk standarisasi
    numerical_cols = ['Merokok', 'Usia']

    # Membuat salinan DataFrame untuk preprocessing
    processed_input = input_data.copy()

    # Standarisasi hanya pada kolom numerik
    processed_input[numerical_cols] = scaler.transform(processed_input[numerical_cols])

    # === 6. Membuat Prediksi ===
    prediction_proba = model.predict(processed_input)[0][0]
    prediction_class = 1 if prediction_proba >= 0.5 else 0

    # === 7. Menampilkan Hasil ===
    st.subheader("Hasil Prediksi:")
    if prediction_class == 1:
        st.error(f"**BERISIKO TINGGI Mengidap Penyakit Paru-paru** (Probabilitas: {prediction_proba:.2f})")
        st.write("Disarankan untuk segera berkonsultasi dengan tenaga medis.")
    else:
        st.success(f"**TIDAK BERISIKO TINGGI Mengidap Penyakit Paru-paru** (Probabilitas: {prediction_proba:.2f})")
        st.write("Tetap jaga gaya hidup sehat.")

    st.info("Catatan: Prediksi ini adalah hasil dari model Machine Learning dan bukan diagnosis medis. Selalu konsultasikan dengan dokter untuk diagnosis dan penanganan yang akurat.")