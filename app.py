import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Veteriner Cerrahi Asistanı", layout="wide")

st.title("🩺 Veteriner Cerrahi Asistanı")
st.write("Yapay Zeka Destekli Cerrahi Alet Tanıma Sistemi")

# --- Model Yükleme ---
try:
    # Modelin klasörde olduğundan emin oluyoruz
    model = YOLO('best.pt')
    st.sidebar.success("Model (best.pt) Yüklendi! ✅")
except Exception as e:
    st.error(f"HATA: 'best.pt' dosyası bulunamadı! Lütfen dosyayı bu klasöre atın. Hata: {e}")

# --- Ayarlar ---
st.sidebar.header("Görüntü Ayarları")
confidence = st.sidebar.slider("Güven Eşiği (Hassasiyet)", 0.0, 1.0, 0.25)

# --- Video Yükleme ---
uploaded_file = st.file_uploader("Video Yükle (1 veya 2 numaralı videoyu seç)", type=['mp4', 'mov', 'avi', 'mkv'])

if uploaded_file is not None:
    # Videoyu ekranda göster
    st.video(uploaded_file)
    
    if st.button("Videoyu Analiz Et ve Aletleri Bul"):
        st.write("Analiz yapılıyor, lütfen bekleyin...")
        
        # Geçici dosya oluştur (Streamlit için gerekli)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty() # Videonun oynayacağı çerçeve
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- YAPAY ZEKA GÖRÜŞÜ ---
            results = model(frame, conf=confidence)
            
            # Kutucukları çiz
            res_plotted = results[0].plot()
            
            # Renkleri düzelt (OpenCV BGR -> Ekran RGB)
            frame_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Ekrana bas
            st_frame.image(frame_rgb, caption='Gerçek Zamanlı Analiz', use_column_width=True)
        
        cap.release()
        st.success("İşlem Tamamlandı.")