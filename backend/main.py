"""
main.py — FastAPI backend for Multimodal Emotion Recognition System.

Endpoints:
  GET  /                  health check
  POST /predict/text      text-only emotion
  POST /predict/facial    facial-only emotion (base64 image)
  POST /predict/audio     audio-only emotion  (base64 audio)
  POST /predict/fused     text + facial + audio fused result
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

import text_emotion
import facial_emotion
import audio_emotion
import fusion

app = FastAPI(title="Multimodal Emotion Recognition API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warmup():
    text_emotion.get_pipeline()
    facial_emotion.warmup()
    try:
        audio_emotion.warmup()
    except Exception as exc:
        print(f"[AudioEmotion] Warmup ERROR: {exc}")
        import traceback; traceback.print_exc()
    print("[DONE] API ready.")


@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "message": "Emotion Recognition API is running"}


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

    model_config = {
        "json_schema_extra": {"example": {"text": "I am so happy today!"}}
    }


class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image (JPEG/PNG). Data-URI prefix optional.")

    model_config = {
        "json_schema_extra": {"example": {"image": "data:image/jpeg;base64,/9j/4AAQ..."}}
    }


class AudioRequest(BaseModel):
    audio: str = Field(..., description="Base64-encoded audio (WAV/MP3/OGG). Data-URI prefix optional.")

    model_config = {
        "json_schema_extra": {"example": {"audio": "data:audio/wav;base64,UklGRi..."}}
    }


class FusedRequest(BaseModel):
    text:  Optional[str] = Field(None, description="Input text (optional).")
    image: Optional[str] = Field(None, description="Base64-encoded image (optional).")
    audio: Optional[str] = Field(None, description="Base64-encoded audio (optional).")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text":  "I feel terrible today.",
                "image": "data:image/jpeg;base64,/9j/4AAQ...",
                "audio": "data:audio/wav;base64,UklGRi...",
            }
        }
    }


@app.post("/predict/text", summary="Text emotion prediction")
def predict_text(req: TextRequest):
    try:
        return text_emotion.predict(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/facial", summary="Facial emotion prediction")
def predict_facial(req: ImageRequest):
    try:
        return facial_emotion.predict(req.image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/audio", summary="Audio speech emotion prediction")
def predict_audio(req: AudioRequest):
    try:
        return audio_emotion.predict(req.audio)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/fused", summary="Fused text + facial + audio emotion")
def predict_fused(req: FusedRequest):
    if not req.text and not req.image and not req.audio:
        raise HTTPException(status_code=422, detail="Provide at least one of: text, image, audio.")
    try:
        text_res   = text_emotion.predict(req.text)    if req.text  else None
        facial_res = facial_emotion.predict(req.image) if req.image else None
        audio_res  = audio_emotion.predict(req.audio)  if req.audio else None
        fused_res  = fusion.fuse(text_res, facial_res, audio_res)
        return {
            "text_result":   text_res,
            "facial_result": facial_res,
            "audio_result":  audio_res,
            "fused_result":  fused_res,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
