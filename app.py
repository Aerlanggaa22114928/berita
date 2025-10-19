import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import pandas as pd # Diperlukan untuk memastikan semua dependencies scikit-learn terload

# =========================================================
# 1. Pemuatan Model dan Komponen (Caching)
# =========================================================

# @st.cache_resource memastikan model hanya dimuat sekali
@st.cache_resource
def load_model_components():
    try:
        # Tentukan Nama File
        MODEL_FILE = 'naive_bayes_hoax_model.pkl'
        VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
        ENCODER_FILE = 'label_encoder.pkl'

        # Muat Model Naive Bayes
        with open(MODEL_FILE, 'rb') as file:
            loaded_model = pickle.load(file)
        
        # Muat TF-IDF Vectorizer
        with open(VECTORIZER_FILE, 'rb') as file:
            loaded_vectorizer = pickle.load(file)
            
        # Muat Label Encoder
        with open(ENCODER_FILE, 'rb') as file:
            loaded_encoder = pickle.load(file)

        return loaded_model, loaded_vectorizer, loaded_encoder
    
    except FileNotFoundError:
        st.error("Gagal memuat file model. Pastikan file .pkl berada di direktori yang sama.")
        return None, None, None

loaded_model, loaded_vectorizer, loaded_encoder = load_model_components()


# =========================================================
# 2. Fungsi Pra-pemrosesan (Wajib sama saat pelatihan)
# =========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'^"|"$', '', text) 
    return text


# =========================================================
# 3. Fungsi Prediksi Inti
# =========================================================
def predict_hoax_status(berita_text):
    if not loaded_model:
        return "Model Tidak Tersedia"

    # 1. Bersihkan teks
    cleaned_text = clean_text(berita_text)
    
    # 2. Transformasi ke fitur TF-IDF
    X_new = loaded_vectorizer.transform([cleaned_text])
    
    # 3. Prediksi
    prediction_numeric = loaded_model.predict(X_new)
    
    # 4. Konversi hasil numerik ke label
    prediction_label = loaded_encoder.inverse_transform(prediction_numeric)
    
    return prediction_label[0].upper()


# =========================================================
# 4. Antarmuka Streamlit
# =========================================================
st.set_page_config(page_title="Detektor Berita Hoax", layout="centered")

st.title("Aplikasi Deteksi Berita Hoax 📰")
st.markdown("Model: Naive Bayes dengan Ekstraksi Fitur TF-IDF")

# Area input teks
berita_input = st.text_area(
    "Masukkan Teks Berita di Bawah Ini:", 
    height=250, 
    placeholder="Contoh: 'Semua aplikasi media sosial akan dipantau mulai besok...'")

# Tombol untuk prediksi
if st.button("Deteksi Status Berita"):
    if berita_input:
        with st.spinner('Menganalisis berita...'):
            hasil_prediksi = predict_hoax_status(berita_input)
            
            # Menampilkan hasil
            st.markdown("---")
            st.subheader("Hasil Analisis:")
            
            if hasil_prediksi == "HOAX":
                st.error(f"⚠️ STATUS: **{hasil_prediksi}**")
                st.markdown("Hasil prediksi menunjukkan berita ini cenderung **TIDAK VALID**.")
            elif hasil_prediksi == "VALID":
                st.success(f"✅ STATUS: **{hasil_prediksi}**")
                st.markdown("Hasil prediksi menunjukkan berita ini cenderung **VALID**.")
            else:
                 st.warning("Hasil prediksi tidak terdefinisi.")
    else:
        st.warning("Mohon masukkan teks berita terlebih dahulu.")