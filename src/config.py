from pathlib import Path

RANDOM_STATE = 0
DATA_PATH = Path("data/tiktok_dataset.csv")
MODEL_PATH = Path("models/champion_model.pkl")
VECTORIZER_PATH = Path("models/vectorizer.pkl")

NUMERIC_FEATURES = [
    'video_duration_sec', 'video_view_count', 'video_like_count',
    'video_share_count', 'video_download_count', 'video_comment_count',
    'text_length'
]
CATEGORICAL_FEATURES = ['verified_status', 'author_ban_status']
TARGET = 'claim_status'