import streamlit as st
import numpy as np
import joblib
import pandas as pd # Untuk mendapatkan nama fitur dan rentang data

# --- KONFIGURASI DAN PEMUATAN MODEL ---

st.set_page_config(page_title="Prediksi Harga Rumah Boston", layout="wide")

# Muat model pipeline Random Forest yang sudah dilatih
try:
    model = joblib.load('boston_random_forest_model.pkl')
    # Catatan: Pastikan Anda telah menjalankan script pelatihan model Random Forest
    # yang menyimpan model ini sebagai 'boston_random_forest_model.pkl'
except FileNotFoundError:
    st.error("File model 'boston_random_forest_model.pkl' tidak ditemukan. Pastikan Anda telah melatih dan menyimpan model tersebut.")
    st.stop() # Menghentikan eksekusi jika model tidak ada

# Dapatkan nama fitur dan rentang nilai dari file csv untuk label form dan validasi
try:
    boston_df = pd.read_csv('boston.csv')
    feature_names = boston_df.drop('MEDV', axis=1).columns
    
    # Simpan rentang fitur untuk validasi input
    feature_ranges = {}
    for col in feature_names:
        feature_ranges[col] = {'min': boston_df[col].min(), 'max': boston_df[col].max(), 'mean': boston_df[col].mean()}
    
except FileNotFoundError:
    st.error("File 'boston.csv' tidak ditemukan. File ini diperlukan untuk label form dan rentang nilai.")
    st.stop()

# --- INTERFACE APLIKASI STREAMLIT ---

st.title("🏠 Prediksi Harga Rumah Boston")
st.write("Selamat datang di aplikasi prediksi harga rumah! Masukkan 13 fitur di bawah ini untuk mendapatkan estimasi harga rumah.")
st.markdown("---")

# Form input
with st.form("prediction_form"):
    st.header("Form Input Fitur")
    
    # Membuat dictionary untuk menampung semua input
    inputs = {}
    
    # Menggunakan kolom untuk tata letak yang lebih baik
    col1, col2, col3 = st.columns(3)
    
    # Loop untuk membuat input field secara dinamis
    for i, feature in enumerate(feature_names):
        # Tentukan kolom target
        if i < 5:
            target_col = col1
        elif i < 10:
            target_col = col2
        else:
            target_col = col3
        
        # Ambil rentang min, max, dan nilai rata-rata sebagai default dari data asli
        min_val = float(feature_ranges[feature]['min'])
        max_val = float(feature_ranges[feature]['max'])
        default_val = float(feature_ranges[feature]['mean'])

        with target_col:
            # Gunakan st.number_input dengan rentang min/max dan nilai default
            inputs[feature] = st.number_input(
                f'Masukkan nilai untuk {feature}',
                min_value=min_val,
                max_value=max_val,
                value=default_val, # Nilai default dari rata-rata fitur
                format="%.2f",
                help=f"Rentang nilai yang umum untuk {feature}: {min_val:.2f} - {max_val:.2f}"
            )

    submit = st.form_submit_button("Prediksi Harga")

# Ketika tombol submit ditekan
if submit:
    # Mengumpulkan input dalam urutan yang benar sesuai feature_names
    feature_values = [inputs[feature] for feature in feature_names]
    
    # Mengubahnya menjadi format array 2D yang dibutuhkan model (1 sampel, 13 fitur)
    features_array = np.array(feature_values).reshape(1, -1)
    
    # Melakukan prediksi menggunakan pipeline
    # Pipeline akan secara otomatis menerapkan scaler yang sudah dilatih
    # sebelum meneruskan data ke Random Forest Regressor.
    prediction = model.predict(features_array)[0]
    
    # Harga prediksi biasanya dalam ribuan dolar untuk dataset Boston Housing
    # Jika model Anda menghasilkan harga dalam satuan asli, tidak perlu dikalikan 1000.
    # Dataset Boston Housing MEDV sudah dalam $1000s, jadi langsung pakai.
    
    # Penanganan hasil prediksi negatif (Random Forest cenderung tidak menghasilkan negatif,
    # tetapi ini adalah praktik terbaik jika terjadi ekstrapolasi ekstrem)
    if prediction < 0:
        st.warning(f"Prediksi menghasilkan harga negatif (${prediction:,.2f}). Ini mungkin karena kombinasi fitur input yang sangat tidak biasa dan berada jauh di luar rentang data pelatihan model.")
        predicted_price = 0.0 # Setel ke 0 jika prediksi negatif
    else:
        predicted_price = prediction
    
    # Menampilkan hasil
    st.subheader("Hasil Prediksi")
    st.success(f"Estimasi Harga Rumah adalah: ${predicted_price:,.2f}")
    st.markdown("---")
    st.info("Catatan: Hasil prediksi dalam ribuan dolar. Misalnya, $20.00 berarti $20,000.")
