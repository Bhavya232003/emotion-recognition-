"""
text_emotion.py — Text emotion classification using pretrained DistilRoBERTa.
Model: j-hartmann/emotion-english-distilroberta-base
Labels: anger, disgust, fear, joy, neutral, sadness, surprise
"""

import torch
from transformers import pipeline

_pipe = None
_DEVICE = 0 if torch.cuda.is_available() else -1

EMOTION_EMOJI = {
    "anger":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "joy":      "😄",
    "neutral":  "😐",
    "sadness":  "😢",
    "surprise": "😲",
}


def get_pipeline():
    global _pipe
    if _pipe is None:
        device_name = torch.cuda.get_device_name(0) if _DEVICE == 0 else "CPU"
        print(f"[TextEmotion] Loading DistilRoBERTa on {device_name} …")
        _pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=_DEVICE,
        )
        print("[TextEmotion] Model loaded.")
    return _pipe


def predict(text: str) -> dict:
    pipe = get_pipeline()
    results = pipe(text[:512])[0]
    scores = {r["label"]: round(r["score"], 4) for r in results}
    top = max(scores, key=scores.get)
    return {
        "dominant_emotion": top,
        "emoji": EMOTION_EMOJI.get(top, ""),
        "scores": scores,
    }
