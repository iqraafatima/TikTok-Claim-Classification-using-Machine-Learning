import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="TikTok Claim Classifier", page_icon="🎥")

MODEL_PATH = Path("models/champion_model.pkl")
VECTORIZER_PATH = Path("models/vectorizer.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("🎥 TikTok Claim vs. Opinion Classifier")
st.write(
    "Predicts whether a TikTok video makes a **claim** (factual assertion, "
    "possibly against terms of service) or states an **opinion**, based on "
    "engagement metrics and transcript text."
)

st.header("Video details")

col1, col2 = st.columns(2)
with col1:
    duration = st.number_input("Video duration (seconds)", min_value=1, max_value=600, value=30)
    views = st.number_input("View count", min_value=0, value=10000)
    likes = st.number_input("Like count", min_value=0, value=500)
    shares = st.number_input("Share count", min_value=0, value=50)
with col2:
    downloads = st.number_input("Download count", min_value=0, value=10)
    comments = st.number_input("Comment count", min_value=0, value=20)
    verified = st.checkbox("Author is verified")
    banned = st.selectbox("Author ban status", ["active", "under review", "banned"])

transcript = st.text_area(
    "Video transcript text",
    placeholder="e.g. someone shared with me that drone deliveries are..."
)

if st.button("Predict", type="primary"):
    text_len = len(transcript)
    base_features = {
        'video_duration_sec': duration,
        'video_view_count': float(views),
        'video_like_count': float(likes),
        'video_share_count': float(shares),
        'video_download_count': float(downloads),
        'video_comment_count': float(comments),
        'text_length': text_len,
        'verified_status_verified': verified,
        'author_ban_status_banned': banned == "banned",
        'author_ban_status_under review': banned == "under review",
    }

    text_feats = pd.DataFrame(
        vectorizer.transform([transcript]).toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    final_df = pd.DataFrame([base_features])
    final_df = pd.concat([final_df, text_feats], axis=1)
    final_df = final_df.reindex(columns=model.feature_names_in_, fill_value=0)

    pred = model.predict(final_df)[0]
    proba = model.predict_proba(final_df)[0][1]

    label = "🚩 Claim" if pred == 1 else "💬 Opinion"
    st.subheader(f"Prediction: {label}")
    st.progress(float(proba))
    st.caption(f"Confidence (claim probability): {proba:.1%}")

st.divider()
st.caption("Model: Random Forest classifier | Trained on TikTok content moderation dataset")
