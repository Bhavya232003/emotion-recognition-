"""
facial_emotion.py — Facial emotion analysis using DeepFace.
Accepts base64-encoded image string or raw numpy array.
DeepFace labels: angry, disgust, fear, happy, sad, surprise, neutral
"""

import base64
import io
import numpy as np
from PIL import Image
import cv2
from deepface import DeepFace

DEEPFACE_TO_STANDARD = {
    "angry":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "joy",
    "sad":      "sadness",
    "surprise": "surprise",
    "neutral":  "neutral",
}

EMOTION_EMOJI = {
    "anger":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "joy":      "😄",
    "neutral":  "😐",
    "sadness":  "😢",
    "surprise": "😲",
}


def _decode_image(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


DETECTORS = ["opencv"]


def _analyze(img: np.ndarray, detector: str) -> dict:
    analysis = DeepFace.analyze(
        img_path=img,
        actions=["emotion"],
        detector_backend=detector,
        enforce_detection=False,
        silent=True,
    )
    return analysis[0] if isinstance(analysis, list) else analysis


def warmup():
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    for detector in DETECTORS:
        try:
            _analyze(dummy, detector)
            print(f"[DONE] DeepFace ready (detector: {detector}).")
            return
        except Exception:
            continue
    print("[DONE] DeepFace ready (fallback).")


def predict(b64_image: str) -> dict:
    img = _decode_image(b64_image)
    last_err = None
    for detector in DETECTORS:
        try:
            result = _analyze(img, detector)
            face_detected = bool(result.get("face_detected", True))
            raw = result["emotion"]
            scores = {
                DEEPFACE_TO_STANDARD.get(k, k): float(round(v / 100.0, 4))
                for k, v in raw.items()
            }
            top = max(scores, key=scores.get)
            return {
                "dominant_emotion": top,
                "emoji": EMOTION_EMOJI.get(top, ""),
                "scores": scores,
                "face_detected": face_detected,
                "detector_used": detector,
            }
        except Exception as e:
            last_err = e
            continue
    return {
        "dominant_emotion": "neutral",
        "emoji": "😐",
        "scores": {"neutral": 1.0},
        "face_detected": False,
        "error": str(last_err),
    }
