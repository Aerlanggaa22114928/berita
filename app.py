import streamlit as st

import pickle

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import MultinomialNB

import pandas as pd # Import ini membantu memastikan dependency sklearn terload



# =========================================================

# 1. Pemuatan Model dan Komponen (Caching)

#    Menggunakan @st.cache_resource agar model dimuat hanya sekali

# =========================================================



# @st.cache_resource adalah pengganti @st.singleton/st.cache(allow_output_mutation=True)

@st.cache_resource 

def load_model_components():

    """Memuat model Naive Bayes, Vectorizer, dan Encoder dari file .pkl."""

    

    # Tentukan Nama File (Pastikan sesuai dengan nama file Anda)

    MODEL_FILE = 'naive_bayes_hoax_model.pkl'

    VECTORIZER_FILE = 'tfidf_vectorizer.pkl'

    ENCODER_FILE = 'label_encoder.pkl'



    try:

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

        # Menampilkan pesan error yang jelas jika file tidak ditemukan

        st.error("Gagal memuat file model. Pastikan file .pkl berada di direktori yang sama dengan app.py.")

        return None, None, None

    except Exception as e:

        st.error(f"Terjadi kesalahan saat memuat model: {e}")

        return None, None, None



loaded_model, loaded_vectorizer, loaded_encoder = load_model_components()





# =========================================================

# 2. Fungsi Pra-pemrosesan (Harus sama persis saat pelatihan)

# =========================================================

def clean_text(text):

    """Fungsi untuk membersihkan teks, mengubahnya ke lowercase."""

    text = str(text).lower()

    # Menghapus tanda kutip ganda yang mungkin ada

    text = re.sub(r'^"|"$', '', text) 

    return text





# =========================================================

# 3. Fungsi Prediksi Inti

# =========================================================

def predict_hoax_status(berita_text):

    """Melakukan prediksi status hoax/valid dari teks berita."""

    if loaded_model is None:

        return "MODEL ERROR"



    # 1. Bersihkan teks

    cleaned_text = clean_text(berita_text)

    

    # 2. Transformasi ke fitur TF-IDF (Penting: menggunakan .transform)

    X_new = loaded_vectorizer.transform([cleaned_text])

    

    # 3. Prediksi

    prediction_numeric = loaded_model.predict(X_new)

    

    # 4. Konversi hasil numerik ke label teks

    prediction_label = loaded_encoder.inverse_transform(prediction_numeric)

    

    return prediction_label[0].upper()





# =========================================================

# 4. Antarmuka Streamlit (Frontend)

# =========================================================

st.set_page_config(page_title="Detektor Berita Hoax", layout="centered")



st.title("Aplikasi Deteksi Berita Hoax 📰")

st.markdown("---")

st.subheader("Model: Naive Bayes dengan Ekstraksi Fitur TF-IDF")

st.markdown("Aplikasi ini menganalisis teks berita dan memprediksi statusnya (HOAX atau VALID).")

st.markdown("---")



# Area input teks

berita_input = st.text_area(

    "Masukkan Teks Berita yang Ingin Anda Analisis:", 

    height=250, 

    placeholder="Contoh: 'Pemerintah akan menerapkan jam malam total di seluruh Indonesia mulai besok...'")



# Tombol untuk prediksi

if st.button("Deteksi Status Berita", type="primary"):

    if loaded_model is None:

        st.error("Model tidak dapat dimuat. Periksa file .pkl dan dependensi.")

    elif berita_input:

        with st.spinner('⏳ Menganalisis berita...'):

            hasil_prediksi = predict_hoax_status(berita_input)

            

            # Menampilkan hasil

            st.markdown("## Hasil Analisis")

            

            if hasil_prediksi == "HOAX":

                st.error(f"⚠️ STATUS PREDIKSI: **{hasil_prediksi}**")

                st.markdown("**Kesimpulan:** Berita ini memiliki kecenderungan tinggi untuk menjadi **TIDAK VALID** berdasarkan pola bahasa pada data pelatihan.")

            elif hasil_prediksi == "VALID":

                st.success(f"✅ STATUS PREDIKSI: **{hasil_prediksi}**")

                st.markdown("**Kesimpulan:** Berita ini memiliki kecenderungan tinggi untuk menjadi **VALID** berdasarkan pola bahasa pada data pelatihan.")

            else:

                 st.warning("Hasil prediksi tidak terdefinisi.")

                 

    else:

        st.warning("⚠️ Mohon masukkan teks berita terlebih dahulu untuk memulai analisis.")
