"""
Emotion Recognition — Text + Face (dual model)
"""
import sys, os
import numpy as np
import cv2
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import EMOTION_LABELS, IMAGE_SIZE
from src.inference import EmotionPredictor

# ==================== Design tokens ====================
EMOTION_COLORS = {
    "happy": "#f7c948", "sad": "#5b8def", "angry": "#e85545",
    "fearful": "#9b59b6", "disgust": "#5da85a", "surprised": "#f08c3c",
    "neutral": "#8e8e9a",
}
EMOTION_EMOJI = {
    "happy": "😊", "sad": "😢", "angry": "😠", "fearful": "😨",
    "disgust": "🤢", "surprised": "😲", "neutral": "😐",
}

st.set_page_config(page_title="Emotion Recognition", page_icon="🎭", layout="wide", initial_sidebar_state="collapsed")

# ==================== Custom CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Figtree', -apple-system, sans-serif; color: #e0dcd0; }
.stApp { background: radial-gradient(ellipse at 50% 0%, #14142a 0%, #0d0d1a 70%); }

.main-header { text-align: center; padding: 2rem 0 0.5rem 0; }
.main-header h1 { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.03em; color: #e0dcd0; margin: 0; }
.main-header .tagline { font-size: 0.95rem; color: #7a7690; }

.input-panel { background: rgba(26,26,48,0.7); border: 1px solid #2a2a40; border-radius: 18px; padding: 1.25rem; }
.result-card { background: rgba(26,26,48,0.6); border: 1px solid #2a2a40; border-radius: 16px; padding: 1.25rem; text-align: center; }

.emotion-result {
    border-radius: 20px; padding: 1.25rem; text-align: center; margin-bottom: 0.75rem;
}

div.stButton > button {
    background: linear-gradient(135deg, #c8a45c, #a07840); color: #0d0d1a;
    border: none; border-radius: 12px; padding: 0.65rem 2rem; font-weight: 600;
    font-size: 1rem; letter-spacing: 0.02em; transition: all 0.2s; width: 100%;
}
div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(200,164,92,0.3); }

[data-testid="stFileUploader"] { border: 2px dashed #3a3a50 !important; border-radius: 14px !important; background: rgba(20,20,38,0.5) !important; }
textarea { background: rgba(20,20,38,0.6) !important; border: 1px solid #2a2a40 !important; border-radius: 12px !important; color: #e0dcd0 !important; }

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 0.5rem 1.2rem; font-weight: 500; color: #7a7690; background: transparent; }
.stTabs [aria-selected="true"] { background: rgba(200,164,92,0.15); color: #c8a45c; }

.prob-row { display: flex; align-items: center; margin: 0.35rem 0; gap: 0.6rem; }
.prob-label { width: 80px; text-align: right; font-weight: 500; font-size: 0.82rem; }
.prob-bar-bg { flex: 1; height: 7px; background: #1a1a30; border-radius: 4px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
.prob-value { width: 42px; font-size: 0.78rem; font-weight: 600; text-align: left; }

.footer-spacer { height: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==================== Load models ====================
@st.cache_resource
def load_predictor():
    return EmotionPredictor()

predictor = load_predictor()

# ==================== Helpers ====================
def detect_all_faces(image_bgr):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = image_bgr if len(image_bgr.shape) == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
    results = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
            face = face.astype(np.float32) / 255.0
            face = (face - 0.5) / 0.5
            results.append((face, (x, y, w, h)))
    return results

def draw_face_boxes(image_bgr, faces):
    img = image_bgr.copy()
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for _, bbox in faces:
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 220, 100), 2)
            cv2.putText(img, "face", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def render_prob_bars(all_probs):
    html = ""
    for emotion in EMOTION_LABELS:
        prob = all_probs.get(emotion, 0)
        color = EMOTION_COLORS[emotion]
        html += f"""<div class="prob-row">
            <span class="prob-label" style="color:{color}">{emotion}</span>
            <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{prob*100}%;background:{color}"></div></div>
            <span class="prob-value" style="color:{color}">{prob:.1%}</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)

def render_emotion_card(prediction, confidence, modality_label):
    color = EMOTION_COLORS[prediction]
    emoji = EMOTION_EMOJI.get(prediction, "")
    st.markdown(f"""
    <div class="emotion-result" style="
        background: radial-gradient(circle at center, {color}22 0%, transparent 70%);
        border: 1px solid {color}44;
    ">
        <div style="font-size:2.5rem;">{emoji}</div>
        <div style="font-size:1.5rem;font-weight:700;color:{color};">{prediction.upper()}</div>
        <div style="font-size:0.9rem;color:{color}88;">{confidence:.1%}</div>
        <div style="font-size:0.7rem;color:#7a7690;margin-top:0.3rem;">{modality_label}</div>
    </div>""", unsafe_allow_html=True)

# ==================== UI ====================
st.markdown("""
<div class="main-header">
    <h1>Emotion Recognition</h1>
    <p class="tagline">Text &middot; Face &middot; Dual Model</p>
</div>
""", unsafe_allow_html=True)

tab_img, tab_cam = st.tabs(["Upload Image", "Camera"])

# ==================== TAB 1: Upload ====================
with tab_img:
    col_text, col_img = st.columns([1, 1])
    with col_text:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("### Text")
        text_input = st.text_area(
            "Describe how you feel...",
            value="I am feeling great today, everything is wonderful!",
            height=140, label_visibility="collapsed",
        )
        if st.button("Analyze Text", type="primary", key="btn_text"):
            result = predictor.predict_text(text_input.strip())
            if "error" not in result:
                col_card, col_bars = st.columns([1, 2])
                with col_card:
                    render_emotion_card(result["prediction"], result["confidence"], "Text Model · 94.8%")
                with col_bars:
                    st.markdown("#### Text probabilities")
                    render_prob_bars(result["all_probs"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_img:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("### Face Image")
        uploaded_file = st.file_uploader(
            "Upload a photo containing faces",
            type=["jpg","jpeg","png","bmp"], label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        img_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        faces = detect_all_faces(img_bgr)

        col_orig, col_detected = st.columns([1, 1])
        with col_orig:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with col_detected:
            if faces:
                st.image(draw_face_boxes(img_bgr, faces), caption=f"{len(faces)} face(s) detected", use_container_width=True)
            else:
                st.warning("No faces detected. Using whole image.")

        if st.button("Analyze Faces", type="primary", key="btn_img"):
            if not faces:
                st.error("No faces found.")
            else:
                st.markdown("---")
                st.markdown(f"### {len(faces)} face(s) analyzed — Image Model")
                face_cols = st.columns(min(len(faces), 3))
                for i, (face_np, bbox) in enumerate(faces):
                    result = predictor.predict_image(face_np)
                    col = face_cols[i % 3]
                    with col:
                        color = EMOTION_COLORS[result["prediction"]]
                        emoji = EMOTION_EMOJI.get(result["prediction"], "")
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="font-size:2rem;">{emoji}</div>
                            <div style="font-weight:600;color:{color};">{result['prediction'].upper()}</div>
                            <div style="font-size:0.85rem;color:#7a7690;">{result['confidence']:.1%}</div>
                        </div>""", unsafe_allow_html=True)

                # Detail for first face
                if faces:
                    result = predictor.predict_image(faces[0][0])
                    st.markdown("---")
                    st.markdown("### Face #1 Detail — Image Model")
                    col_card, col_bars = st.columns([1, 2])
                    with col_card:
                        render_emotion_card(result["prediction"], result["confidence"], "Image Model · 87%+")
                    with col_bars:
                        st.markdown("#### Image probabilities")
                        render_prob_bars(result["all_probs"])

# ==================== TAB 2: Camera ====================
with tab_cam:
    col_text_cam, col_cam = st.columns([1, 1])
    with col_text_cam:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("### Text")
        text_input_cam = st.text_area(
            "Describe how you feel...",
            value="I am feeling great today, everything is wonderful!",
            height=140, label_visibility="collapsed", key="text_cam",
        )
        if st.button("Analyze Text", type="primary", key="btn_text_cam"):
            result = predictor.predict_text(text_input_cam.strip())
            if "error" not in result:
                col_card, col_bars = st.columns([1, 2])
                with col_card:
                    render_emotion_card(result["prediction"], result["confidence"], "Text Model · 94.8%")
                with col_bars:
                    st.markdown("#### Text probabilities")
                    render_prob_bars(result["all_probs"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cam:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("### Camera")
        camera_img = st.camera_input("Take a picture", key="webcam", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    if camera_img:
        img_bytes = camera_img.getvalue()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        faces = detect_all_faces(img_bgr)

        col_orig, col_detected = st.columns([1, 1])
        with col_orig:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Captured", use_container_width=True)
        with col_detected:
            if faces:
                st.image(draw_face_boxes(img_bgr, faces), caption=f"{len(faces)} face(s) detected", use_container_width=True)
            else:
                st.warning("No faces detected.")

        if st.button("Analyze Faces", type="primary", key="btn_cam"):
            if not faces:
                st.error("No faces found.")
            else:
                st.markdown("---")
                st.markdown(f"### {len(faces)} face(s) analyzed — Image Model")
                face_cols = st.columns(min(len(faces), 3))
                for i, (face_np, bbox) in enumerate(faces):
                    result = predictor.predict_image(face_np)
                    col = face_cols[i % 3]
                    with col:
                        color = EMOTION_COLORS[result["prediction"]]
                        emoji = EMOTION_EMOJI.get(result["prediction"], "")
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="font-size:2rem;">{emoji}</div>
                            <div style="font-weight:600;color:{color};">{result['prediction'].upper()}</div>
                            <div style="font-size:0.85rem;color:#7a7690;">{result['confidence']:.1%}</div>
                        </div>""", unsafe_allow_html=True)

                if faces:
                    result = predictor.predict_image(faces[0][0])
                    st.markdown("---")
                    st.markdown("### Face #1 Detail — Image Model")
                    col_card, col_bars = st.columns([1, 2])
                    with col_card:
                        render_emotion_card(result["prediction"], result["confidence"], "Image Model · 87%+")
                    with col_bars:
                        st.markdown("#### Image probabilities")
                        render_prob_bars(result["all_probs"])

st.markdown('<div class="footer-spacer"></div>', unsafe_allow_html=True)
