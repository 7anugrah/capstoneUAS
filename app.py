import streamlit as st
import pandas as pd
import pickle

# Konfigurasi halaman harus menjadi perintah Streamlit pertama
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    layout="wide"
)

# --- FUNGSI UNTUK MEMUAT MODEL DAN ARTEFAK LAINNYA ---
@st.cache_data
def load_artifacts():
    """Memuat semua artefak yang dibutuhkan dari file .pkl"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('transformer.pkl', 'rb') as f:
            transformer = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, transformer, scaler, encoder, feature_columns
    except FileNotFoundError:
        st.error("File model (.pkl) tidak ditemukan. Pastikan semua file berada di folder yang sama dengan app.py.")
        return None, None, None, None, None

# --- MEMUAT ARTEFAK ---
model, transformer, scaler, encoder, feature_columns = load_artifacts()


# --- ANTARMUKA APLIKASI ---
st.title("Prediksi Tingkat Obesitas")
st.markdown("Aplikasi ini bertujuan untuk memprediksi kategori obesitas berdasarkan kebiasaan gaya hidup dan parameter fisik.")
st.markdown("---")

# --- INPUT PENGGUNA DI SIDEBAR ---
with st.sidebar:
    st.header("Formulir Input Data Pengguna")

    # --- Mapping untuk input yang diterjemahkan ---
    # Ini penting agar model menerima data sesuai format training (Bahasa Inggris)
    gender_map = {"Laki-laki": "Male", "Perempuan": "Female"}
    caec_map = {"Tidak": "no", "Kadang-kadang": "Sometimes", "Sering": "Frequently", "Selalu": "Always"}
    calc_map = {"Tidak": "no", "Kadang-kadang": "Sometimes", "Sering": "Frequently", "Selalu": "Always"}
    mtrans_map = {"Mobil": "Automobile", "Motor": "Motorbike", "Sepeda": "Bike", "Transportasi Umum": "Public_Transportation", "Jalan Kaki": "Walking"}

    # Input data kategorikal dengan teks Bahasa Indonesia
    gender_id = st.selectbox("Jenis Kelamin", list(gender_map.keys()))
    family_history_id = st.radio("Memiliki riwayat obesitas dalam keluarga?", ["yes", "no"])
    favc_id = st.radio("Sering mengonsumsi makanan tinggi kalori (FAVC)?", ["yes", "no"])
    caec_id = st.selectbox("Konsumsi makanan di antara waktu makan utama (CAEC)?", list(caec_map.keys()))
    smoke_id = st.radio("Apakah Anda merokok?", ["yes", "no"])
    scc_id = st.radio("Apakah Anda memantau asupan kalori harian?", ["yes", "no"])
    calc_id = st.selectbox("Frekuensi konsumsi alkohol (CALC)", list(calc_map.keys()))
    mtrans_id = st.selectbox("Transportasi utama yang digunakan (MTRANS)", list(mtrans_map.keys()))

    st.markdown("---")

    # Input data numerik
    age_val = st.slider("Umur", 14, 65, 25)
    fcvc_val = st.slider("Frekuensi konsumsi sayur (FCVC)", 1, 3, 2, help="1: Tidak Pernah, 2: Kadang-kadang, 3: Selalu")
    ncp_val = st.slider("Jumlah porsi makan utama per hari (NCP)", 1, 4, 3)
    ch2o_val = st.slider("Konsumsi air per hari (liter) (CH2O)", 1, 3, 2)
    faf_val = st.slider("Frekuensi aktivitas fisik per minggu (FAF)", 0, 3, 1, help="0: Tidak ada, 1: 1-2 hari, 2: 2-4 hari, 3: 4-5 hari")
    tue_val = st.slider("Waktu penggunaan gawai per hari (jam) (TUE)", 0, 2, 1, help="0: 0-2 jam, 1: 3-5 jam, 2: >5 jam")

# --- TOMBOL PREDIKSI ---
if st.button("Jalankan Prediksi"):
    if model is None:
        st.warning("Model tidak berhasil dimuat, proses tidak dapat dilanjutkan.")
        st.stop()
        
    # 1. Konversi input Bahasa Indonesia kembali ke format asli (Bahasa Inggris)
    input_data = {
        'Gender': gender_map[gender_id],
        'family_history_with_overweight': 1 if family_history_id == 'yes' else 0,
        'FAVC': 1 if favc_id == 'yes' else 0,
        'FCVC': fcvc_val,
        'NCP': ncp_val,
        'CAEC': caec_map[caec_id],
        'SMOKE': 1 if smoke_id == 'yes' else 0,
        'CH2O': ch2o_val,
        'SCC': 1 if scc_id == 'yes' else 0,
        'FAF': faf_val,
        'TUE': tue_val,
        'CALC': calc_map[calc_id],
        'MTRANS': mtrans_map[mtrans_id]
    }
    input_df = pd.DataFrame([input_data])
    
    # Pastikan urutan kolom sesuai dengan saat training
    input_df = input_df[feature_columns]

    # 2. Terapkan preprocessing pada data input
    transformed_input = transformer.transform(input_df)
    scaled_input = scaler.transform(transformed_input)
    
    # 3. Lakukan prediksi menggunakan model
    prediction_encoded = model.predict(scaled_input)
    
    # 4. Ubah hasil prediksi (angka) kembali ke label kategori asli
    prediction_label = encoder.inverse_transform(prediction_encoded)
    
    # --- TAMPILKAN HASIL PREDIKSI ---
    st.markdown("---")
    st.subheader("Hasil Analisis Prediktif")
    
    result_text = prediction_label[0].replace("_", " ")
    if "Insufficient" in result_text:
        st.success(f"Berdasarkan analisis, Anda tergolong dalam kategori: **{result_text}** (Berat Badan Kurang).")
    elif "Normal" in result_text:
        st.success(f"Berdasarkan analisis, Anda tergolong dalam kategori: **{result_text}** (Berat Badan Ideal).")
    elif "Overweight" in result_text:
        st.warning(f"Berdasarkan analisis, Anda tergolong dalam kategori: **{result_text}** (Kelebihan Berat Badan).")
    else: # Kategori Obesitas
        st.error(f"Berdasarkan analisis, Anda tergolong dalam kategori: **{result_text}** (Obesitas). Disarankan untuk menjaga pola hidup yang lebih sehat.")