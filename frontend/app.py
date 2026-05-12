"""
app.py — Streamlit frontend for Multimodal Emotion Recognition System.
"""

import base64
import io
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:5000"

EMOTION_COLORS = {
    "anger":    "#ff4b4b",
    "disgust":  "#9b59b6",
    "fear":     "#e67e22",
    "joy":      "#00cc88",
    "neutral":  "#7f8c8d",
    "sadness":  "#3498db",
    "surprise": "#f1c40f",
}

st.set_page_config(page_title="Emotion Recognition", layout="wide", page_icon="🎭")

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    section[data-testid="stSidebar"] { background-color: #161a25; }
    h1, h2, h3 { color: #e0e0e0; }
    div[data-testid="metric-container"] {
        background: #1e2130;
        border-radius: 10px;
        padding: 12px 20px;
        border-left: 4px solid #4f8bf9;
    }
    .emotion-badge {
        font-size: 1.8rem;
        font-weight: 800;
        padding: 8px 20px;
        border-radius: 12px;
        display: inline-block;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


def score_bars(scores: dict):
    for emo, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        c = EMOTION_COLORS.get(emo, "#ffffff")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0'>"
            f"<span style='width:80px;color:{c};font-weight:600'>{emo}</span>"
            f"<div style='flex:1;background:#1e2130;border-radius:6px;height:14px'>"
            f"<div style='width:{score*100:.1f}%;background:{c};height:14px;border-radius:6px'></div></div>"
            f"<span style='width:50px;text-align:right;color:#ccc'>{score:.2%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def emotion_badge(emotion: str, emoji: str):
    color = EMOTION_COLORS.get(emotion, "#ffffff")
    st.markdown(
        f"<div class='emotion-badge' style='background:{color}22;color:{color};border:2px solid {color}'>"
        f"{emoji} {emotion.upper()}</div>",
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎭 Emotion Detector")
    st.divider()
    if st.button("🩺 Check API Health"):
        try:
            r = requests.get(f"{API_URL}/", timeout=5)
            st.success("✅ API online") if r.status_code == 200 else st.error(f"❌ Status {r.status_code}")
        except Exception:
            st.error("❌ Cannot reach API")
    st.divider()
    st.caption("**Models:**\n- Text: DistilRoBERTa\n- Facial: DeepFace (FER+)\n- Audio: wav2vec2 (RAVDESS)\n- Fusion: Weighted late fusion")

st.title("🎭 Multimodal Emotion Recognition")
st.markdown("Detect emotions from **text**, **facial expression**, **speech**, or **all combined**.")
st.divider()

tab_text, tab_face, tab_audio, tab_fused = st.tabs(
    ["📝 Text", "📷 Facial", "🎙️ Audio", "🔀 Fused"]
)

# ── Tab 1: Text ───────────────────────────────────────────────────────────────
with tab_text:
    st.subheader("Text Emotion Analysis")
    text_input = st.text_area(
        "Enter text to analyse",
        placeholder="e.g. I can't believe how amazing today has been!",
        height=120,
    )
    if st.button("Analyse Text 🔍", key="btn_text"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analysing …"):
                try:
                    r = requests.post(f"{API_URL}/predict/text", json={"text": text_input}, timeout=30)
                    if r.status_code == 200:
                        res = r.json()
                        emotion_badge(res["dominant_emotion"], res["emoji"])
                        st.markdown("**Confidence scores:**")
                        score_bars(res["scores"])
                    else:
                        st.error(f"API error {r.status_code}: {r.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect. Start the backend first.")

# ── Tab 2: Facial ─────────────────────────────────────────────────────────────
with tab_face:
    st.subheader("Facial Emotion Analysis")
    upload_col, cam_col = st.columns(2)

    with upload_col:
        uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    with cam_col:
        camera_img = st.camera_input("Or take a photo")

    img_source = camera_img if camera_img else uploaded

    if img_source and st.button("Analyse Face 🔍", key="btn_face"):
        with st.spinner("Analysing face …"):
            try:
                img_bytes = img_source.getvalue()
                b64_uri   = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()

                r = requests.post(f"{API_URL}/predict/facial", json={"image": b64_uri}, timeout=120)
                if r.status_code == 200:
                    res = r.json()
                    if not res.get("face_detected", True):
                        st.warning("⚠️ No face detected — showing best-guess result.")
                    col_img, col_result = st.columns([1, 1])
                    with col_img:
                        st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
                    with col_result:
                        emotion_badge(res["dominant_emotion"], res["emoji"])
                        score_bars(res["scores"])
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect. Start the backend first.")

# ── Tab 3: Audio ──────────────────────────────────────────────────────────────
with tab_audio:
    st.subheader("Speech Emotion Analysis")
    st.caption("Upload a voice recording or record directly from your mic. Supports WAV, MP3, OGG.")

    audio_upload = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg", "flac"], key="audio_upload")
    mic_audio    = st.audio_input("Or record from mic", key="mic_audio")

    audio_source = mic_audio if mic_audio else audio_upload

    if audio_source:
        st.audio(audio_source)

    if audio_source and st.button("Analyse Audio 🔍", key="btn_audio"):
        with st.spinner("Analysing speech …"):
            try:
                audio_bytes = audio_source.getvalue()
                b64_uri     = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode()

                r = requests.post(f"{API_URL}/predict/audio", json={"audio": b64_uri}, timeout=120)
                if r.status_code == 200:
                    res = r.json()
                    if "error" in res:
                        st.warning(f"⚠️ {res['error']}")
                    emotion_badge(res["dominant_emotion"], res["emoji"])
                    st.markdown("**Confidence scores:**")
                    score_bars(res["scores"])
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect. Start the backend first.")

# ── Tab 4: Fused ──────────────────────────────────────────────────────────────
with tab_fused:
    st.subheader("Fused Emotion Analysis")
    st.caption("Provide any combination of text, image, and audio. Weights: Text 34% · Facial 33% · Audio 33%.")

    fused_text = st.text_area(
        "Text input (optional)",
        placeholder="e.g. Everything is going wrong today …",
        height=80,
        key="fused_text",
    )

    f_img_col, f_cam_col = st.columns(2)
    with f_img_col:
        fused_upload = st.file_uploader("Face photo (optional)", type=["jpg", "jpeg", "png"], key="fused_upload")
    with f_cam_col:
        fused_cam = st.camera_input("Or take a photo", key="fused_cam")
    fused_img_source = fused_cam if fused_cam else fused_upload

    fused_audio_upload = st.file_uploader("Audio (optional)", type=["wav", "mp3", "ogg", "flac"], key="fused_audio_upload")
    fused_mic          = st.audio_input("Or record from mic", key="fused_mic")
    fused_audio_source = fused_mic if fused_mic else fused_audio_upload
    if fused_audio_source:
        st.audio(fused_audio_source)

    if st.button("Fused Analysis 🔀", key="btn_fused"):
        if not fused_text.strip() and not fused_img_source and not fused_audio_source:
            st.warning("Provide at least one input (text, image, or audio).")
        else:
            with st.spinner("Running fused analysis …"):
                try:
                    payload: dict = {}
                    if fused_text.strip():
                        payload["text"] = fused_text.strip()
                    if fused_img_source:
                        payload["image"] = "data:image/jpeg;base64," + base64.b64encode(fused_img_source.getvalue()).decode()
                    if fused_audio_source:
                        payload["audio"] = "data:audio/wav;base64," + base64.b64encode(fused_audio_source.getvalue()).decode()

                    r = requests.post(f"{API_URL}/predict/fused", json=payload, timeout=120)
                    if r.status_code == 200:
                        res    = r.json()
                        fused  = res["fused_result"]
                        text_r = res.get("text_result")
                        face_r = res.get("facial_result")
                        aud_r  = res.get("audio_result")

                        st.markdown("### Fused Result")
                        emotion_badge(fused["dominant_emotion"], fused["emoji"])
                        st.caption(f"Modalities used: {', '.join(fused.get('modalities_used', []))}")
                        st.markdown("---")

                        cols = st.columns(3)
                        for col, result, label in zip(
                            cols,
                            [text_r, face_r, aud_r],
                            ["Text", "Facial", "Audio"],
                        ):
                            if result:
                                with col:
                                    st.markdown(f"**{label}**")
                                    st.markdown(f"{result['emoji']} **{result['dominant_emotion'].upper()}**")
                                    for emo, sc in sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)[:3]:
                                        st.caption(f"{emo}: {sc:.2%}")

                        st.markdown("**Fused confidence scores:**")
                        score_bars(fused["scores"])
                    else:
                        st.error(f"API error {r.status_code}: {r.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect. Start the backend first.")
