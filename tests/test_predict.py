import pytest
from src.predict import ClaimPredictor

@pytest.fixture(scope="module")
def predictor():
    return ClaimPredictor()

def test_predict_returns_valid_label(predictor):
    sample = {
        'video_duration_sec': 30, 'video_view_count': 500000,
        'video_like_count': 20000, 'video_share_count': 3000,
        'video_download_count': 500, 'video_comment_count': 200,
        'text_length': 100, 'verified_status_verified': False,
        'author_ban_status_banned': False, 'author_ban_status_under review': False,
        'video_transcription_text': 'someone shared with me that this happened'
    }
    result = predictor.predict(sample)
    assert result['prediction'] in ['claim', 'opinion']
    assert 0.0 <= result['confidence'] <= 1.0