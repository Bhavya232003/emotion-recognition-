# Multimodal Emotion Recognition

A real-time emotion detection system that analyses emotions from **text**, **facial expressions**, and **speech audio** — individually or all at once using late fusion.

## Models

| Modality | Model |
|----------|-------|
| Text | DistilRoBERTa (Hugging Face) |
| Facial | DeepFace (FER+) |
| Audio | wav2vec2 (RAVDESS) |
| Fusion | Weighted late fusion — Text 34% · Facial 33% · Audio 33% |

**Detectable emotions:** anger, disgust, fear, joy, neutral, sadness, surprise

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **ML:** PyTorch, Transformers, DeepFace, OpenCV

## Project Structure

```
emotion-recognition/
├── backend/
│   ├── main.py              # FastAPI app & endpoints
│   ├── text_emotion.py      # Text emotion pipeline
│   ├── facial_emotion.py    # Facial emotion pipeline
│   ├── audio_emotion.py     # Audio emotion pipeline
│   └── fusion.py            # Late fusion logic
├── frontend/
│   └── app.py               # Streamlit UI
├── requirements.txt
└── test_face_model.py
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Bhavya232003/emotion-recognition-.git
cd emotion-recognition-
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the App

### Start the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### Start the frontend (new terminal)

```bash
cd frontend
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict/text` | Text-only emotion |
| POST | `/predict/facial` | Facial emotion (base64 image) |
| POST | `/predict/audio` | Speech emotion (base64 audio) |
| POST | `/predict/fused` | Fused result from all modalities |

Interactive API docs available at [http://localhost:5000/docs](http://localhost:5000/docs).
