"""
audio_emotion.py — Speech emotion recognition using wav2vec2-xlsr.
Model : ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (8 emotions)
Fix   : checkpoint uses a 2-layer head (dense→tanh→output); the standard
        Wav2Vec2ForSequenceClassification expects a single linear layer, so the
        pipeline silently randomises the head.  We load the backbone through
        from_pretrained (backbone keys load fine), then manually wire in the
        correct head weights from the raw checkpoint.
"""

import base64
import io

import librosa
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_ID  = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
TARGET_SR = 16_000
_DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

LABEL_MAP = {
    "angry":     "anger",
    "calm":      "neutral",
    "disgust":   "disgust",
    "fearful":   "fear",
    "happy":     "joy",
    "neutral":   "neutral",
    "sad":       "sadness",
    "surprised": "surprise",
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


class _Head(nn.Module):
    """Two-layer classification head matching the ehcalabres checkpoint."""
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dense  = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.dense(x)))


class _Wav2Vec2Emotion(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.wav2vec2   = backbone
        self.classifier = _Head(hidden_size, len(LABELS))

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
        return self.classifier(hidden)


_feature_extractor = None
_model: _Wav2Vec2Emotion | None = None


def _load_model() -> None:
    global _feature_extractor, _model
    if _model is not None:
        return

    device_name = torch.cuda.get_device_name(0) if _DEVICE.type == "cuda" else "CPU"
    print(f"[AudioEmotion] Loading wav2vec2-xlsr on {device_name} …", flush=True)

    _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)

    # Backbone weights (wav2vec2.*) load correctly; only the classifier head mismatches.
    base = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, ignore_mismatched_sizes=True
    )
    print("[AudioEmotion] Backbone loaded, wiring head …", flush=True)

    model = _Wav2Vec2Emotion(base.wav2vec2, base.config.hidden_size)
    del base

    # Pull classifier head weights from the cached checkpoint (avoids re-download).
    from huggingface_hub import try_to_load_from_cache
    ckpt = try_to_load_from_cache(MODEL_ID, "pytorch_model.bin")
    if ckpt is None:
        ckpt = hf_hub_download(MODEL_ID, "pytorch_model.bin")
    print(f"[AudioEmotion] Loading checkpoint from cache …", flush=True)
    full_sd = torch.load(ckpt, map_location="cpu")

    head_sd = {
        k[len("classifier."):]: v
        for k, v in full_sd.items()
        if k.startswith("classifier.")
    }
    model.classifier.load_state_dict(head_sd, strict=True)
    del full_sd

    _model = model.to(_DEVICE).eval()
    print("[AudioEmotion] Model loaded.", flush=True)


def warmup() -> None:
    _load_model()
    dummy = np.zeros(TARGET_SR, dtype=np.float32)
    _infer(dummy)
    print("[DONE] AudioEmotion ready.", flush=True)


def _infer(audio: np.ndarray) -> list[dict]:
    inputs = _feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
        max_length=TARGET_SR * 10,
        truncation=True,
    )
    with torch.no_grad():
        logits = _model(inputs.input_values.to(_DEVICE))
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return [{"label": LABELS[i], "score": float(probs[i])} for i in range(len(LABELS))]


def _decode_audio(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    audio, _ = librosa.load(io.BytesIO(base64.b64decode(b64)), sr=TARGET_SR, mono=True)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95  # peak-normalise
    return audio.astype(np.float32)


def predict(b64_audio: str) -> dict:
    try:
        audio   = _decode_audio(b64_audio)
        results = _infer(audio)

        raw: dict[str, float] = {}
        for r in results:
            label = LABEL_MAP.get(r["label"], r["label"])
            raw[label] = raw.get(label, 0.0) + r["score"]

        total  = sum(raw.values()) or 1.0
        scores = {k: round(v / total, 4) for k, v in raw.items()}
        top    = max(scores, key=scores.get)
        return {
            "dominant_emotion": top,
            "emoji":  EMOTION_EMOJI.get(top, ""),
            "scores": scores,
        }
    except Exception as exc:
        return {
            "dominant_emotion": "neutral",
            "emoji":  "😐",
            "scores": {"neutral": 1.0},
            "error":  str(exc),
        }
