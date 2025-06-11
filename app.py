import streamlit as st
import numpy as np
import joblib
import pandas as pd #untuk mendapatkan nama fitur

#KONFIGURASI DAN PEMUATAN MODEL

st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")

#Muat model yang sudah dilatih
try:
    model = joblib.load('boston_model.pkl')
except FileNotFoundError:
    st.error("File model 'boston_model.pkl' tidak ditemukan. Jalankan 'train_model.py' terlebih dahulu.")
    st.stop() #Menghentikan eksekusi jika model tidak ada

#Dapatkan nama fitur dari file csv untuk label form (hanya untuk referensi)
try:
    feature_names = pd.read_csv('boston.csv').drop('MEDV', axis=1).columns
except FileNotFoundError:
    st.error("File 'boston.csv' tidak ditemukan. File ini diperlukan untuk label form.")
    st.stop()

#INTERFACE APLIKASI STREAMLIT

st.title("🏠 Prediksi Harga Rumah Boston")
st.write("Masukkan 13 fitur rumah untuk mendapatkan estimasi harga.")

#Form input
with st.form("prediction_form"):
    st.header("Form Input Fitur")
    
    #Membuat dictionary untuk menampung semua input
    inputs = {}
    
    #Menggunakan kolom untuk tata letak yang lebih baik
    col1, col2, col3 = st.columns(3)
    
    #Loop untuk membuat input field secara dinamis
    for i, feature in enumerate(feature_names):
        if i < 5:
            target_col = col1
        elif i < 10:
            target_col = col2
        else:
            target_col = col3
        
        with target_col:
             inputs[feature] = st.number_input(f'Masukkan nilai untuk {feature}', format="%.2f")

    submit = st.form_submit_button("Prediksi Harga")

#Ketika tombol submit ditekan
if submit:
    #Mengumpulkan input dalam urutan yang benar
    feature_values = [inputs[feature] for feature in feature_names]
    
    #Mengubahnya menjadi format array 2D yang dibutuhkan model
    features_array = np.array(feature_values).reshape(1, -1)
    
    #Melakukan prediksi
    #Pipeline akan secara otomatis menerapkan scaler sebelum regressor
    prediction = model.predict(features_array)[0]
    
    #Hasil prediksi dalam ribuan dolar, jadi kita kalikan 1000
    predicted_price = prediction * 1000
    
    # Menampilkan hasil
    st.subheader("Hasil Prediksi")
    st.success(f"Estimasi Harga Rumah adalah: ${predicted_price:,.2f}")

