import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="TikTok Claim Classifier", page_icon="🎥", layout="wide")

MODEL_PATH = Path("models/champion_model.pkl")
VECTORIZER_PATH = Path("models/vectorizer.pkl")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_artifacts()

# ==============================================================
# DESIGN SYSTEM — trust & safety / moderation-dashboard aesthetic
# ==============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
    --bg:#12151A;
    --surface:#1B1F26;
    --surface-alt:#232833;
    --ink:#E7EAEE;
    --ink-soft:#8B93A1;
    --claim:#E2574C;
    --opinion:#4C9FE2;
    --warn:#E2A33C;
    --line:#2A2F3A;
}
html, body, [class*="css"]{ font-family:'Inter', sans-serif; color:var(--ink); }
.stApp{ background:var(--bg); }
h1, h2, h3, .display{ font-family:'Space Grotesk', sans-serif; color:var(--ink); }

.eyebrow{
    font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.14em;
    text-transform:uppercase; color:var(--ink-soft); font-weight:600;
}
hr{ border:none; height:1px; background:var(--line); margin:26px 0; }

.hero{
    background:linear-gradient(135deg,#1B1F26 0%, #20262F 100%);
    border:1px solid var(--line);
    padding:32px 36px; border-radius:18px; margin-bottom:24px;
}
.hero h1{ font-size:1.9rem; margin:6px 0 8px 0; }
.hero p{ color:var(--ink-soft); max-width:640px; margin:0; }

.section-card{
    background:var(--surface); border:1px solid var(--line);
    padding:22px 26px; border-radius:16px; margin-bottom:20px;
}
.section-card h4{ margin-top:0; margin-bottom:4px; color:var(--ink); }
.section-sub{ color:var(--ink-soft); font-size:0.85rem; margin-bottom:16px; }

/* Streamlit inputs, dark-mode friendly */
div[data-testid="stNumberInput"] input, div[data-testid="stTextArea"] textarea{
    background:var(--surface-alt) !important; color:var(--ink) !important;
    border:1.5px solid var(--line) !important; border-radius:10px !important;
}
div[data-testid="stSelectbox"] > div > div{
    background:var(--surface-alt) !important; border:1.5px solid var(--line) !important;
    border-radius:10px !important;
}
div[data-testid="stCheckbox"] label p{ color:var(--ink) !important; }

div.stButton > button{
    background:var(--claim) !important; color:#FFFFFF !important; border:none !important;
    border-radius:12px !important; padding:0.7em 1.4em !important; font-weight:600 !important;
    font-family:'Inter', sans-serif !important; transition:.2s ease;
}
div.stButton > button:hover{ filter:brightness(1.1); transform:translateY(-2px); }
div.stButton > button:focus-visible{ outline:3px solid var(--warn) !important; outline-offset:2px; }

.badge{
    display:inline-flex; align-items:center; gap:8px;
    padding:10px 18px; border-radius:10px; font-weight:600; font-size:1.05rem;
    margin-bottom:16px;
}
.badge-claim{ background:rgba(226,87,76,0.14); color:var(--claim); border:1px solid rgba(226,87,76,0.35); }
.badge-opinion{ background:rgba(76,159,226,0.14); color:var(--opinion); border:1px solid rgba(76,159,226,0.35); }

.meter-track{
    background:var(--surface-alt); border-radius:10px; height:14px; overflow:hidden;
    border:1px solid var(--line); position:relative;
}
.meter-fill{ height:100%; border-radius:10px 0 0 10px; }
.meter-threshold{
    position:absolute; top:-4px; bottom:-4px; width:2px; background:var(--warn);
}

.stat-readout{ font-family:'IBM Plex Mono', monospace; font-weight:600; color:var(--ink); }
.meta-label{
    font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.08em;
    text-transform:uppercase; color:var(--ink-soft); font-weight:600; margin-bottom:2px;
}
.diag-ok{ color:#4FBF7A; }
.diag-warn{ color:var(--warn); }
.diag-bad{ color:var(--claim); }
</style>
""", unsafe_allow_html=True)

# ==============================================================
# Layout
# ==============================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">Content Moderation · Screening Tool</div>
    <h1>🎥 TikTok Claim vs. Opinion Classifier</h1>
    <p>Predicts whether a video makes a <b>claim</b> (a factual assertion,
    possibly against platform terms) or states an <b>opinion</b>, based on
    engagement metrics, author signals, and transcript text.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
<h4>📊 Engagement Metrics</h4>
<div class="section-sub">Raw counts pulled from the video's public stats.</div>
""", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    duration = st.number_input("Video duration (seconds)", min_value=1, max_value=600, value=30)
    views = st.number_input("View count", min_value=0, value=10000)
with c2:
    likes = st.number_input("Like count", min_value=0, value=500)
    shares = st.number_input("Share count", min_value=0, value=50)
with c3:
    downloads = st.number_input("Download count", min_value=0, value=10)
    comments = st.number_input("Comment count", min_value=0, value=20)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
<h4>👤 Author Signals</h4>
<div class="section-sub">Trust indicators about the account posting the video.</div>
""", unsafe_allow_html=True)
a1, a2 = st.columns(2)
with a1:
    verified = st.checkbox("Author is verified")
with a2:
    banned = st.selectbox("Author ban status", ["active", "under review", "banned"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
<h4>📝 Transcript</h4>
<div class="section-sub">What the video's narration or on-screen text actually says.</div>
""", unsafe_allow_html=True)
transcript = st.text_area(
    "Video transcript text",
    placeholder="e.g. someone shared with me that drone deliveries are...",
    label_visibility="collapsed",
    height=110,
)
st.markdown("</div>", unsafe_allow_html=True)

run = st.button("Run Prediction", type="primary", use_container_width=True)

if run:
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

    # ---- Diagnostics: how much of what we built actually reaches the model ----
    pre_reindex_cols = set(final_df.columns)
    model_cols = set(model.feature_names_in_)
    matched = pre_reindex_cols & model_cols
    missing_from_input = model_cols - pre_reindex_cols  # will be filled with 0
    nonzero_text_feats = int((text_feats.values != 0).sum())

    final_df = final_df.reindex(columns=model.feature_names_in_, fill_value=0)

    pred = model.predict(final_df)[0]
    proba = model.predict_proba(final_df)[0][1]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Result</div>', unsafe_allow_html=True)

    if pred == 1:
        st.markdown('<div class="badge badge-claim">🚩 Claim</div>', unsafe_allow_html=True)
        fill_color = "var(--claim)"
    else:
        st.markdown('<div class="badge badge-opinion">💬 Opinion</div>', unsafe_allow_html=True)
        fill_color = "var(--opinion)"

    pct = proba * 100
    st.markdown(f"""
    <div class="meter-track">
        <div class="meter-fill" style="width:{pct:.1f}%; background:{fill_color};"></div>
        <div class="meter-threshold" style="left:50%;"></div>
    </div>
    <p style="margin-top:8px;" class="stat-readout">Claim probability: {proba:.1%}</p>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Feature diagnostics (debug: is the transcript actually being used?)"):
        st.markdown(f"""
        - **Model expects** `{len(model_cols)}` total features
        - **Your input produced** `{len(pre_reindex_cols)}` features before reindexing
        - **Matched features** (these actually reach the model): `{len(matched)}`
        - **Missing / zero-filled** (model expects these but your input didn't have them): `{len(missing_from_input)}`
        - **Non-zero values inside the transcript's TF-IDF vector**: `{nonzero_text_feats}`
        """)
        if nonzero_text_feats == 0 and text_len > 0:
            st.markdown(
                '<span class="diag-bad">⚠ Your transcript produced an all-zero TF-IDF vector. '
                'None of its words are in the vectorizer\'s vocabulary — check that this is the '
                'SAME fitted vectorizer used at training time.</span>',
                unsafe_allow_html=True,
            )
        if len(missing_from_input) > 0:
            st.markdown(
                f'<span class="diag-warn">⚠ {len(missing_from_input)} feature(s) the model expects '
                'were not found in your input and got filled with 0. If most of these are text/word '
                'features, the model is likely ignoring the transcript entirely.</span>',
                unsafe_allow_html=True,
            )
            st.code(sorted(missing_from_input)[:30])

    st.caption("Model: Random Forest classifier · Trained on TikTok content moderation dataset")
