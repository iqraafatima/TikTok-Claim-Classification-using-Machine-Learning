from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import ClaimPredictor

app = FastAPI(title="TikTok Claim Classifier API")
predictor = ClaimPredictor()

class VideoInput(BaseModel):
    video_duration_sec: int
    video_view_count: float
    video_like_count: float
    video_share_count: float
    video_download_count: float
    video_comment_count: float
    verified_status_verified: bool
    author_ban_status_banned: bool
    author_ban_status_under_review: bool
    video_transcription_text: str

@app.post("/predict")
def predict(input: VideoInput):
    features = input.dict()
    features['author_ban_status_under review'] = features.pop('author_ban_status_under_review')
    features['text_length'] = len(features['video_transcription_text'])
    return predictor.predict(features)

@app.get("/health")
def health():
    return {"status": "ok"}