"""
情感识别 — 文本 + 人脸 双模型
"""
import sys, os, time
import numpy as np

try:    import cv2; HAS_CV2 = True
except ImportError: HAS_CV2 = False

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import EMOTION_LABELS, IMAGE_SIZE

HAS_MODELS = False
try:
    from src.inference import EmotionPredictor
    HAS_MODELS = True
except Exception:
    pass

# ==================== 情感中文映射 ====================
EMOTION_CN = {
    "happy": "开心", "sad": "悲伤", "angry": "愤怒",
    "fearful": "恐惧", "disgust": "厌恶", "surprised": "惊讶", "neutral": "中性",
}
EMOTION_COLORS = {
    "happy": "#e8a020", "sad": "#4a8fd4", "angry": "#d4453b",
    "fearful": "#8b5ea8", "disgust": "#5a9a4b", "surprised": "#e87830",
    "neutral": "#8e8e9a",
}
EMOTION_EMOJI = {
    "happy": "😊", "sad": "😢", "angry": "😠", "fearful": "😨",
    "disgust": "🤢", "surprised": "😲", "neutral": "😐",
}

st.set_page_config(page_title="情感识别", page_icon="🎭", layout="wide", initial_sidebar_state="collapsed")

# ==================== 轻量明亮主题 CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', -apple-system, sans-serif;
    color: #2c2c34;
}
.stApp {
    background: linear-gradient(180deg, #f8f7f4 0%, #f0eeeb 100%);
}

.main-header { text-align: center; padding: 1.8rem 0 0.3rem 0; }
.main-header h1 { font-size: 2rem; font-weight: 700; color: #1a1a22; margin: 0; Letter-spacing: 0.04em; }
.main-header .tagline { font-size: 0.9rem; color: #8e8c95; margin-top: 0.15rem; }

.input-panel {
    background: #ffffff; border: 1px solid #e8e6e1; border-radius: 16px;
    padding: 1.25rem; box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.result-card {
    background: #ffffff; border: 1px solid #e8e6e1; border-radius: 14px;
    padding: 1rem; text-align: center; box-shadow: 0 1px 8px rgba(0,0,0,0.03);
}

.emotion-result {
    border-radius: 18px; padding: 1.1rem; text-align: center; margin-bottom: 0.6rem;
    background: #ffffff; border: 1px solid #e8e6e1;
}

div.stButton > button {
    background: #2c2c34; color: #ffffff;
    border: none; border-radius: 10px; padding: 0.6rem 1.8rem; font-weight: 600;
    font-size: 0.95rem; transition: all 0.15s; width: 100%;
    letter-spacing: 0.03em;
}
div.stButton > button:hover { background: #444450; transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0,0,0,0.1); }

[data-testid="stFileUploader"] {
    border: 2px dashed #d8d5d0 !important; border-radius: 14px !important; background: #fafaf9 !important;
}
textarea {
    background: #fafaf9 !important; border: 1px solid #e8e6e1 !important;
    border-radius: 10px !important; color: #2c2c34 !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 0.4rem 1rem; font-weight: 500;
    color: #8e8c95; background: transparent;
}
.stTabs [aria-selected="true"] { background: #2c2c3410; color: #2c2c34; }

.prob-row { display: flex; align-items: center; margin: 0.3rem 0; gap: 0.5rem; }
.prob-label { width: 55px; text-align: right; font-weight: 500; font-size: 0.8rem; }
.prob-bar-bg { flex: 1; height: 6px; background: #eeedeb; border-radius: 3px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
.prob-value { width: 38px; font-size: 0.75rem; font-weight: 600; text-align: left; }

.footer-spacer { height: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==================== 加载模型 ====================
@st.cache_resource
def load_predictor():
    if HAS_MODELS:
        return EmotionPredictor()
    return None

predictor = load_predictor()

# ==================== 人脸检测 ====================
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
            cv2.rectangle(img, (x, y), (x+w, y+h), (40, 180, 80), 2)
            cv2.putText(img, "人脸", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 180, 80), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==================== 概率条 ====================
def render_prob_bars(all_probs):
    html = ""
    for emotion in EMOTION_LABELS:
        prob = all_probs.get(emotion, 0)
        color = EMOTION_COLORS[emotion]
        cn = EMOTION_CN.get(emotion, emotion)
        html += f"""<div class="prob-row">
            <span class="prob-label" style="color:{color}">{cn}</span>
            <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{prob*100}%;background:{color}"></div></div>
            <span class="prob-value" style="color:{color}">{prob:.1%}</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)

def render_emotion_card(prediction, confidence, model_name):
    color = EMOTION_COLORS[prediction]
    emoji = EMOTION_EMOJI.get(prediction, "")
    cn = EMOTION_CN.get(prediction, prediction)
    st.markdown(f"""
    <div class="emotion-result" style="border-color: {color}33;">
        <div style="font-size:2.4rem;">{emoji}</div>
        <div style="font-size:1.4rem;font-weight:700;color:{color};">{cn}</div>
        <div style="font-size:0.85rem;color:#8e8c95;">置信度 {confidence:.1%}</div>
        <div style="font-size:0.7rem;color:#b8b6b2;margin-top:0.2rem;">{model_name}</div>
    </div>""", unsafe_allow_html=True)

# ==================== 摄像头重置 ====================
if "cam_counter" not in st.session_state:
    st.session_state.cam_counter = 0

# ==================== UI 头部 ====================
st.markdown("""
<div class="main-header">
    <h1>情感识别</h1>
    <p class="tagline">文本 &middot; 人脸 &middot; 双模型独立推理</p>
</div>
""", unsafe_allow_html=True)

tab_img, tab_cam = st.tabs(["📷 上传图片", "📸 摄像头"])

# ==================== Tab 1: 上传图片 ====================
with tab_img:
    col_text, col_img = st.columns([1, 1])
    with col_text:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("### 文本识别")
        text_input = st.text_area(
            "输入描述情绪的英文文本...",
            value="I am feeling great today, everything is wonderful!",
            height=140, label_visibility="collapsed",
        )
        if st.button("分析文本", type="primary", key="btn_text"):
            if predictor:
                result = predictor.predict_text(text_input.strip())
                if "error" not in result:
                    col_card, col_bars = st.columns([1, 2])
                    with col_card:
                        render_emotion_card(result["prediction"], result["confidence"], "文本模型 · 94.8%")
                    with col_bars:
                        st.markdown("#### 各情感概率")
                        render_prob_bars(result["all_probs"])
            else:
                st.warning("模型未加载，请先在本地运行训练脚本")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_img:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("### 人脸识别")
        uploaded_file = st.file_uploader(
            "上传包含人脸的图片",
            type=["jpg","jpeg","png","bmp"], label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and HAS_CV2:
        img_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        faces = detect_all_faces(img_bgr)

        col_orig, col_detected = st.columns([1, 1])
        with col_orig:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True)
        with col_detected:
            if faces:
                st.image(draw_face_boxes(img_bgr, faces), caption=f"检测到 {len(faces)} 张人脸", use_container_width=True)
            else:
                st.warning("未检测到人脸，将使用整张图片")

        if st.button("分析人脸", type="primary", key="btn_img"):
            if not faces:
                st.error("未检测到人脸")
            elif predictor:
                st.markdown("---")
                st.markdown(f"### 检测到 {len(faces)} 张人脸 — 图像模型")
                face_cols = st.columns(min(len(faces), 3))
                for i, (face_np, bbox) in enumerate(faces):
                    result = predictor.predict_image(face_np)
                    col = face_cols[i % 3]
                    with col:
                        color = EMOTION_COLORS[result["prediction"]]
                        cn = EMOTION_CN.get(result["prediction"], "")
                        emoji = EMOTION_EMOJI.get(result["prediction"], "")
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="font-size:1.8rem;">{emoji}</div>
                            <div style="font-weight:600;color:{color};">{cn}</div>
                            <div style="font-size:0.8rem;color:#8e8c95;">{result['confidence']:.1%}</div>
                        </div>""", unsafe_allow_html=True)

# ==================== Tab 2: 摄像头 ====================
with tab_cam:
    col_text_cam, col_cam = st.columns([1, 1])
    with col_text_cam:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("### 文本识别")
        text_input_cam = st.text_area(
            "输入描述情绪的英文文本...",
            value="I am feeling great today, everything is wonderful!",
            height=140, label_visibility="collapsed", key="text_cam",
        )
        if st.button("分析文本", type="primary", key="btn_text_cam"):
            if predictor:
                result = predictor.predict_text(text_input_cam.strip())
                if "error" not in result:
                    col_card, col_bars = st.columns([1, 2])
                    with col_card:
                        render_emotion_card(result["prediction"], result["confidence"], "文本模型 · 94.8%")
                    with col_bars:
                        st.markdown("#### 各情感概率")
                        render_prob_bars(result["all_probs"])
            else:
                st.warning("模型未加载")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cam:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("### 人脸识别")
        # Dynamic key to allow re-taking photos
        cam_key = f"webcam_{st.session_state.cam_counter}"
        camera_img = st.camera_input("拍照", key=cam_key, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    if camera_img and HAS_CV2:
        img_bytes = camera_img.getvalue()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        faces = detect_all_faces(img_bgr)

        col_orig, col_detected = st.columns([1, 1])
        with col_orig:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="拍摄画面", use_container_width=True)
        with col_detected:
            if faces:
                st.image(draw_face_boxes(img_bgr, faces), caption=f"检测到 {len(faces)} 张人脸", use_container_width=True)
            else:
                st.warning("未检测到人脸")

        if st.button("分析人脸", type="primary", key="btn_cam"):
            if not faces:
                st.error("未检测到人脸，请面向镜头")
            elif predictor:
                st.markdown("---")
                st.markdown(f"### 检测到 {len(faces)} 张人脸 — 图像模型")
                face_cols = st.columns(min(len(faces), 3))
                for i, (face_np, bbox) in enumerate(faces):
                    result = predictor.predict_image(face_np)
                    col = face_cols[i % 3]
                    with col:
                        color = EMOTION_COLORS[result["prediction"]]
                        cn = EMOTION_CN.get(result["prediction"], "")
                        emoji = EMOTION_EMOJI.get(result["prediction"], "")
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="font-size:1.8rem;">{emoji}</div>
                            <div style="font-weight:600;color:{color};">{cn}</div>
                            <div style="font-size:0.8rem;color:#8e8c95;">{result['confidence']:.1%}</div>
                        </div>""", unsafe_allow_html=True)
                # Reset camera for next photo
                st.session_state.cam_counter += 1
                st.rerun()

st.markdown('<div class="footer-spacer"></div>', unsafe_allow_html=True)
