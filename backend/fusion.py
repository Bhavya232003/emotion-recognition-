"""
fusion.py — Weighted late fusion of text, facial, and audio emotion scores.
Dynamically renormalizes when a modality is missing.
"""

TEXT_WEIGHT   = 0.34
FACIAL_WEIGHT = 0.33
AUDIO_WEIGHT  = 0.33

ALL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

EMOTION_EMOJI = {
    "anger":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "joy":      "😄",
    "neutral":  "😐",
    "sadness":  "😢",
    "surprise": "😲",
}


def fuse(
    text_result:   dict | None,
    facial_result: dict | None,
    audio_result:  dict | None = None,
) -> dict:
    weights: dict[str, float] = {}
    score_maps: dict[str, dict] = {}

    if text_result:
        weights["text"]    = TEXT_WEIGHT
        score_maps["text"] = text_result.get("scores", {})
    if facial_result:
        weights["facial"]    = FACIAL_WEIGHT
        score_maps["facial"] = facial_result.get("scores", {})
    if audio_result:
        weights["audio"]    = AUDIO_WEIGHT
        score_maps["audio"] = audio_result.get("scores", {})

    if not weights:
        return {"dominant_emotion": "neutral", "emoji": "😐", "scores": {}, "modalities_used": []}

    total_w = sum(weights.values())
    fused = {
        emotion: round(
            sum((weights[m] / total_w) * score_maps[m].get(emotion, 0.0) for m in weights),
            4,
        )
        for emotion in ALL_EMOTIONS
    }

    top = max(fused, key=fused.get)
    return {
        "dominant_emotion": top,
        "emoji": EMOTION_EMOJI.get(top, ""),
        "scores": fused,
        "modalities_used": list(weights.keys()),
    }
