"""
Posture Guard — Real-time Posture Detection with YOLOv8
=======================================================
Jalankan dengan:  streamlit run app.py
Kebutuhan:        pip install streamlit ultralytics opencv-python-headless numpy
"""

import streamlit as st
import cv2
import numpy as np
import time
import base64
from pathlib import Path
from dataclasses import dataclass
from ultralytics import YOLO

# ── Load Model YOLOv8 ──────────────────────────────────────────────────────
MODEL_PATH = Path("best.pt")
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH))
    return None

model = load_model()

# ── Konstanta & Konfigurasi ────────────────────────────────────────────────
POSTURE_COLORS = {
    "Baik":    (29,  158, 117),  # hijau teal
    "Bungkuk": (216,  90,  48),  # oranye coral
}
POSTURE_COLORS_HEX = {
    "Baik":    "#1D9E75",
    "Bungkuk": "#D85A30",
}
POSTURE_ADVICE = {
    "Baik":    "Pertahankan posisi ini! Postur Anda sudah bagus.",
    "Bungkuk": "Tegakkan punggung Anda! Jangan membungkuk ke arah layar.",
}
ALERT_COOLDOWN_SEC = 8

@dataclass
class PostureResult:
    label:      str
    confidence: float
    bbox:       tuple
    advice:     str
    color_hex:  str
    color_bgr:  tuple

# # ─────────────────────────────────────────────────────────────────────────────
# # Audio Alert — generate beep sintetis
# # ─────────────────────────────────────────────────────────────────────────────
# def _make_beep_wav(freq=880, duration=0.35, volume=0.6, sample_rate=22050) -> bytes:
#     import struct, math
#     n_samples  = int(sample_rate * duration)
#     fade_len   = int(sample_rate * 0.04)
#     samples    = []
#     for i in range(n_samples):
#         t   = i / sample_rate
#         amp = math.sin(2 * math.pi * freq * t)
#         if i < fade_len: amp *= i / fade_len
#         elif i > n_samples - fade_len: amp *= (n_samples - i) / fade_len
#         samples.append(int(amp * volume * 32767))

#     data_size  = n_samples * 2
#     header = struct.pack(
#         "<4sI4s4sIHHIIHH4sI",
#         b"RIFF", 36 + data_size, b"WAVE",
#         b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16, b"data", data_size,
#     )
#     return header + struct.pack(f"<{n_samples}h", *samples)

# @st.cache_data
# def _get_alert_sounds() -> dict:
#     return {
#         "warning": base64.b64encode(_make_beep_wav(660, 0.3, 0.5)).decode(),
#         "danger":  base64.b64encode(_make_beep_wav(440, 0.2, 0.6) + _make_beep_wav(380, 0.25, 0.7)).decode(),
#     }

# def play_alert(kind: str = "warning"):
#     sounds = _get_alert_sounds()
#     wav_b64 = sounds.get(kind, sounds["warning"])
#     audio_html = f'<audio autoplay style="display:none"><source src="data:audio/wav;base64,{wav_b64}" type="audio/wav"></audio>'
#     st.markdown(audio_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Audio Alert — Memutar file MP3 Custom
# ─────────────────────────────────────────────────────────────────────────────
import os

@st.cache_data
def load_audio_b64(file_path: str) -> str:
    """Membaca file MP3 dan mengubahnya ke base64 agar bisa diputar browser (Di-cache agar cepat)"""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

def play_mp3_alert(file_path: str = "alert.mp3"):
    """Menyisipkan HTML Audio Autoplay ke dalam Streamlit"""
    b64_audio = load_audio_b64(file_path)
    
    if b64_audio:
        audio_html = f'''
            <audio autoplay style="display:none">
                <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
            </audio>
        '''
        # Gunakan st.empty() agar tidak menggeser layout UI ke bawah setiap kali bunyi
        st.empty().markdown(audio_html, unsafe_allow_html=True)
    else:
        st.error(f"⚠️ File audio '{file_path}' tidak ditemukan!")

# ─────────────────────────────────────────────────────────────────────────────
# Frame Processing YOLO
# ─────────────────────────────────────────────────────────────────────────────
def process_frame(frame: np.ndarray, yolo_model) -> PostureResult:
    # Inference YOLO
    results = yolo_model(frame, verbose=False)
    r = results[0]

    # Jika tidak ada orang/objek terdeteksi
    if len(r.boxes) == 0:
        return None

    # Ambil deteksi dengan confidence paling tinggi
    best_box = r.boxes[0]
    for box in r.boxes:
        if box.conf[0] > best_box.conf[0]:
            best_box = box

    # Mapping class dari YOLO ke UI
    cls_id   = int(best_box.cls[0])
    conf     = float(best_box.conf[0])
    cls_name = r.names[cls_id].lower()

    # Sesuaikan dengan nama kelas di dataset kamu (biasanya 'good' atau 'bad')
    is_good = "good" in cls_name
    label = "Baik" if is_good else "Bungkuk"
    
    # Kordinat Bounding Box
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    return PostureResult(
        label=label, confidence=conf, bbox=(x1, y1, x2, y2),
        advice=POSTURE_ADVICE[label],
        color_hex=POSTURE_COLORS_HEX[label],
        color_bgr=POSTURE_COLORS[label]
    )

def annotate_frame(frame: np.ndarray, result: PostureResult, show_box: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    bgr = result.color_bgr

    # Gambar Bounding Box YOLO
    if show_box and result.bbox:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, f"{result.label} {result.confidence*100:.0f}%", 
                    (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    # UI Banner Atas
    cv2.rectangle(frame, (0, 0), (w, 64), (20, 20, 28), -1)
    cv2.rectangle(frame, (0, 0), (w, 64), bgr, 2)
    cv2.putText(frame, f"Status: {result.label}", (16, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, bgr, 2, cv2.LINE_AA)

    # Confidence bar bawah
    bar_w = int(w * result.confidence)
    cv2.rectangle(frame, (0, h - 6), (w, h), (40, 40, 50), -1)
    cv2.rectangle(frame, (0, h - 6), (bar_w, h), bgr, -1)

    return frame

# ─────────────────────────────────────────────────────────────────────────────
# CSS & HTML UI
# ─────────────────────────────────────────────────────────────────────────────
def build_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .pg-header { background: #0d0f14; border-bottom: 1px solid #1e2230; padding: 1rem 1.5rem; display: flex; align-items: center; gap: 14px; border-radius: 12px; margin-bottom: 1rem; }
    .pg-title { font-size:18px; font-weight:600; color:#eef0f6; margin:0; }
    .status-card { border-radius: 12px; padding: 1.1rem; border: 1px solid; margin-top: 0.5rem; }
    .status-good { background:#051a12; border-color:#1D9E75; }
    .status-bad  { background:#1a0d07; border-color:#D85A30; }
    .alert-log { background:#0d0f14; border:1px solid #1e2230; border-radius:10px; padding:1rem; max-height:200px; overflow-y:auto; font-family:monospace; font-size:12px; }
    #MainMenu, footer, header { visibility:hidden; }
    </style>
    """

# ─────────────────────────────────────────────────────────────────────────────
# App Utama
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="YOLOv8 Posture", layout="wide", initial_sidebar_state="expanded")
    st.markdown(build_css(), unsafe_allow_html=True)
    st.markdown("""<div class="pg-header"><div class="pg-title">YOLOv8 Posture Guard</div></div>""", unsafe_allow_html=True)

    if model is None:
        st.error(f"❌ Model tidak ditemukan! Pastikan file `{MODEL_PATH}` berada di folder yang sama dengan `app.py`.")
        st.stop()

    with st.sidebar:
        st.markdown("### ⚙️ Kontrol Kamera")
        run_detection = st.toggle("▶ Aktifkan Kamera", value=False)
        show_box      = st.toggle("🔲 Tampilkan Bounding Box", value=True)
        alert_enabled = st.toggle("🔔 Alert Audio", value=True)

        st.markdown("---")
        bad_frame_threshold = st.slider("Sensitivitas (Frame Buruk sebelum Alert)", 5, 40, 15)
        st.success("✅ Model YOLOv8 Aktif")

    col_cam, col_info = st.columns([3, 2], gap="medium")
    with col_cam: frame_placeholder = st.empty()
    with col_info:
        status_placeholder = st.empty()
        log_placeholder    = st.empty()

    if "last_alert_time" not in st.session_state: st.session_state.last_alert_time = 0.0
    if "bad_streak"      not in st.session_state: st.session_state.bad_streak = 0
    if "alert_log"       not in st.session_state: st.session_state.alert_log = []

    if not run_detection:
        frame_placeholder.info("Pencet toggle 'Aktifkan Kamera' di sidebar untuk mulai.")
        return

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Kamera tidak ditemukan.")
        return

    while run_detection:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Mirror
        
        # Eksekusi YOLOv8
        result = process_frame(frame, model)

        if result:
            annotated = annotate_frame(frame.copy(), result, show_box)
            is_bad = result.label == "Bungkuk"
            
            if is_bad: st.session_state.bad_streak += 1
            else:      st.session_state.bad_streak = 0

            # Alert Logic
            now = time.time()
            if alert_enabled and is_bad and st.session_state.bad_streak >= bad_frame_threshold and (now - st.session_state.last_alert_time) >= ALERT_COOLDOWN_SEC:
                st.session_state.last_alert_time = now
                play_mp3_alert("alert.mp3")
                st.session_state.alert_log.insert(0, f"[{time.strftime('%H:%M:%S')}] Peringatan Postur Bungkuk ({result.confidence*100:.0f}%)")
                st.session_state.alert_log = st.session_state.alert_log[:15]

            # Update UI
            css_cls = "status-bad" if is_bad else "status-good"
            status_placeholder.markdown(f"""
            <div class="status-card {css_cls}">
                <h3 style="color:{result.color_hex};margin:0">{result.label} ({result.confidence*100:.0f}%)</h3>
                <p style="color:#aaa;font-size:14px">{result.advice}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            annotated = frame
            cv2.putText(annotated, "Mencari target...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        # Render Kamera
        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

        # Render Log
        if st.session_state.alert_log:
            logs = "<br>".join(st.session_state.alert_log)
            log_placeholder.markdown(f'<div class="alert-log" style="color:#D85A30">{logs}</div>', unsafe_allow_html=True)

        time.sleep(0.01)

    cap.release()

if __name__ == "__main__":
    main()