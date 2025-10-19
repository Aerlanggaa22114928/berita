import streamlit as st
import pickle
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# =========================================================
# 1. Pemuatan Model dan Komponen (Caching)
# =========================================================

# @st.cache_resource memastikan model hanya dimuat sekali
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
        st.error("Gagal memuat file model. Pastikan file .pkl berada di direktori yang sama dengan app.py.")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None, None

loaded_model, loaded_vectorizer, loaded_encoder = load_model_components()


# =========================================================
# 2. Fungsi Ekstraksi Teks dari URL
# =========================================================
def fetch_article_text(url):
    """Mengambil dan membersihkan teks artikel dari URL menggunakan BeautifulSoup."""
    try:
        # Mengirim permintaan HTTP dengan user-agent dasar
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        # Gagal jika status kode bukan 200
        if response.status_code != 200:
            return None, f"Gagal mengambil URL. Status: {response.status_code}"

        # Parsing HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Mencoba mencari elemen yang mungkin berisi konten artikel
        # Ini adalah heuristik dan mungkin perlu disesuaikan untuk situs tertentu
        
        # Cari elemen paragraf (p) di dalam tag umum artikel
        paragraphs = soup.find_all(['p', 'div'], class_=['content-body', 'article-content', 'isi-berita'])
        
        if not paragraphs:
             # Fallback: ambil semua teks dari body dan hapus skrip/style
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            text = soup.get_text()
            # Membersihkan baris ekstra
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            if len(text) < 200:
                 return None, "Gagal menemukan konten artikel yang cukup, mungkin situs memblokir akses atau format tidak standar."
            return text, None
        
        # Gabungkan teks dari paragraf yang ditemukan
        article_text = ' '.join(p.get_text() for p in paragraphs)
        
        return article_text.strip(), None

    except requests.exceptions.RequestException as e:
        return None, f"Kesalahan koneksi: {e}"
    except Exception as e:
        return None, f"Kesalahan parsing atau lainnya: {e}"


# =========================================================
# 3. Fungsi Pra-pemrosesan (Harus sama persis saat pelatihan)
# =========================================================
def clean_text(text):
    """Fungsi untuk membersihkan teks, mengubahnya ke lowercase."""
    text = str(text).lower()
    # Menghapus tanda kutip ganda yang mungkin ada
    text = re.sub(r'^"|"$', '', text)  
    return text


# =========================================================
# 4. Fungsi Prediksi Inti
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
# 5. Antarmuka Streamlit (Frontend)
# =========================================================
st.set_page_config(page_title="Detektor Berita Hoax", layout="centered")

st.title("Aplikasi Deteksi Berita Hoax 📰")
st.markdown("---")
st.subheader("Model: Naive Bayes dengan Ekstraksi Fitur TF-IDF")
st.markdown("Analisis dapat dilakukan dengan **memasukkan URL artikel** atau **teks lengkap**.")
st.markdown("---")

# Area Input URL
url_input = st.text_input(
    "1. Masukkan URL Berita (Opsional)", 
    placeholder="Contoh: https://www.liputan6.com/...",
    key="url_input"
)

# Tombol untuk prediksi
if st.button("Deteksi Status Berita", type="primary"):
    
    if loaded_model is None:
        st.error("Model tidak dapat dimuat. Periksa file .pkl dan dependensi.")
    
    # --- Logika Utama ---
    final_text = ""
    source = ""
    
    # 1. Jika URL dimasukkan, ambil teks dari sana
    if url_input:
        with st.spinner('⏳ Mengambil teks dari URL...'):
            article_text, error_msg = fetch_article_text(url_input)
            
        if error_msg:
            st.error(f"Gagal mengambil teks dari URL: {error_msg}")
            # Lanjut ke pengecekan input manual (opsi 2)
        elif article_text:
            final_text = article_text
            source = f"dari URL: {url_input}"
    
    # 2. Jika ada teks yang berhasil diambil atau jika URL kosong, cek input manual
    if final_text or not url_input:
        # Jika URL gagal/kosong, minta pengguna masukkan teks manual (untuk demo kita anggap teks_area_manual adalah input utama)
        
        # Di Streamlit, kita tidak bisa secara reaktif mengambil dari text_area_manual
        # Jadi kita gunakan input URL sebagai prioritas, dan jika kosong, minta teks manual.
        
        if not final_text:
             st.warning("URL kosong atau gagal, Anda dapat menyalin dan memasukkan teks secara manual di bawah.")
             # Kita minta pengguna scroll ke bawah dan masukkan manual
             pass

    
    # --- Jalankan Prediksi ---
    if final_text and len(final_text.split()) > 10: # Minimal 10 kata
        with st.spinner('⏳ Menganalisis teks...'):
            hasil_prediksi = predict_hoax_status(final_text)
            
            # Menampilkan hasil
            st.markdown("---")
            st.markdown("## Hasil Analisis")
            
            # Tampilkan Ringkasan Teks yang dianalisis
            st.info(f"Teks yang Dianalisis {source} (Panjang: {len(final_text)} karakter):\n\n{final_text[:500]}...")
            
            if hasil_prediksi == "HOAX":
                st.error(f"⚠️ STATUS PREDIKSI: **{hasil_prediksi}**")
                st.markdown("**Kesimpulan:** Berita ini memiliki kecenderungan tinggi untuk menjadi **TIDAK VALID** berdasarkan pola bahasa pada model.")
            elif hasil_prediksi == "VALID":
                st.success(f"✅ STATUS PREDIKSI: **{hasil_prediksi}**")
                st.markdown("**Kesimpulan:** Berita ini memiliki kecenderungan tinggi untuk menjadi **VALID** berdasarkan pola bahasa pada model.")
            else:
                st.warning("Hasil prediksi tidak terdefinisi.")
                
    elif final_text and len(final_text.split()) <= 10:
        st.warning("Teks yang diambil terlalu pendek. Mohon masukkan artikel yang lebih panjang.")
    elif url_input and not final_text:
        # Pesan error sudah ditampilkan di atas
        pass
    else:
        st.warning("⚠️ Masukkan URL atau Teks untuk memulai analisis.")

# Area input Teks Manual (Jika URL gagal atau tidak digunakan)
st.markdown("---")
st.markdown("### 2. Input Teks Manual (Opsional)")
st.caption("Gunakan ini jika pengambilan teks dari URL gagal atau Anda memiliki teks siap salin.")

# Teks manual di simpan ke state
manual_text = st.text_area(
    "Salin & Tempel Teks Artikel Lengkap:", 
    height=250, 
    key="manual_text_area",
    label_visibility="collapsed"
)

if st.button("Deteksi Teks Manual", key="manual_button", type="secondary"):
    if loaded_model is None:
        st.error("Model tidak dapat dimuat. Periksa file .pkl dan dependensi.")
    elif manual_text:
        with st.spinner('⏳ Menganalisis teks...'):
            final_text = manual_text
            source = "dari Input Manual"
            hasil_prediksi = predict_hoax_status(final_text)
            
            st.markdown("## Hasil Analisis")
            st.info(f"Teks yang Dianalisis {source} (Panjang: {len(final_text)} karakter):\n\n{final_text[:500]}...")
            
            if hasil_prediksi == "HOAX":
                st.error(f"⚠️ STATUS PREDIKSI: **{hasil_prediksi}**")
                st.markdown("**Kesimpulan:** Berita ini memiliki kecenderungan tinggi untuk menjadi **TIDAK VALID**.")
            elif hasil_prediksi == "VALID":
                st.success(f"✅ STATUS PREDIKSI: **{hasil_prediksi}**")
                st.markdown("**Kesimpulan:** Berita ini memiliki kecenderungan tinggi untuk menjadi **VALID**.")
            else:
                st.warning("Hasil prediksi tidak terdefinisi.")
    else:
        st.warning("⚠️ Mohon masukkan teks di area manual.")
