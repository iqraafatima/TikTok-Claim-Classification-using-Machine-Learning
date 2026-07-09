import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib

st.set_page_config(page_title="Case File · Claim Review", page_icon="🗂️", layout="wide")

MODEL_PATH = Path("models/champion_model.pkl")
VECTORIZER_PATH = Path("models/vectorizer.pkl")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_artifacts()

# ==============================================================
# DESIGN SYSTEM — "Case File" aesthetic
# A trust & safety review desk: manila folders, stamped verdicts,
# typewritten case metadata, redaction bars. Built for the subject
# (content moderation), not a generic dashboard skin.
# ==============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Zilla+Slab:wght@500;600;700&family=Special+Elite&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
    --paper:#EDE4CC;
    --paper-alt:#E4D8B8;
    --folder:#C9A667;
    --folder-dark:#B08F52;
    --ink:#24211A;
    --ink-soft:#6B6350;
    --claim:#9C3B2C;
    --opinion:#2E5578;
    --warn:#A87422;
    --redact:#1C1A16;
    --line:#B9AC85;
    --card-shadow: 4px 6px 0px rgba(28,26,22,0.08);
}

html, body, [class*="css"]{ font-family:'IBM Plex Sans', sans-serif; color:var(--ink); }
.stApp{
    background:
        repeating-linear-gradient(0deg, rgba(0,0,0,0.015) 0px, rgba(0,0,0,0.015) 1px, transparent 1px, transparent 3px),
        var(--paper);
}
h1, h2, h3{ font-family:'Zilla Slab', serif; color:var(--ink); font-weight:700; }

.typewriter{ font-family:'Special Elite', monospace; }
.mono{ font-family:'IBM Plex Mono', monospace; }

.eyebrow{
    font-family:'Special Elite', monospace; font-size:11px; letter-spacing:0.12em;
    text-transform:uppercase; color:var(--ink-soft);
}

/* ---- Folder tab masthead ---- */
.folder-wrap{ margin-bottom:0; position:relative; }
.folder-tab{
    display:inline-block; background:var(--folder); color:var(--redact);
    padding:8px 26px 10px 26px; border-radius:10px 10px 0 0;
    font-family:'Special Elite', monospace; font-size:12px; letter-spacing:0.08em;
    text-transform:uppercase; border:1px solid var(--folder-dark); border-bottom:none;
    box-shadow: 0 -1px 0 rgba(0,0,0,0.05) inset;
}
.hero{
    background: var(--paper-alt);
    border:1px solid var(--line); border-top:3px solid var(--folder-dark);
    padding:30px 36px 26px 36px; border-radius:0 12px 12px 12px;
    box-shadow: var(--card-shadow); margin-bottom:26px;
}
.hero h1{ font-size:2rem; margin:6px 0 10px 0; }
.hero p{ color:var(--ink-soft); max-width:680px; margin:0; font-size:0.98rem; line-height:1.5; }
.case-meta{
    display:flex; gap:28px; margin-top:18px; padding-top:14px;
    border-top:1px dashed var(--line); flex-wrap:wrap;
}
.case-meta div{ font-family:'IBM Plex Mono', monospace; font-size:12px; color:var(--ink-soft); }
.case-meta b{ color:var(--ink); display:block; font-size:13px; margin-bottom:2px; }

/* ---- Index-card sections ---- */
.section-card{
    background:#F5EFDC; border:1px solid var(--line);
    padding:22px 26px 24px 26px; border-radius:4px; margin-bottom:22px;
    box-shadow: var(--card-shadow);
    position:relative;
    clip-path: polygon(0 0, 100% 0, 100% 100%, 12px 100%, 0 calc(100% - 12px));
}
.section-card h4{
    margin-top:0; margin-bottom:2px; color:var(--ink); font-family:'Zilla Slab', serif;
    font-size:1.15rem; display:flex; align-items:center; gap:8px;
}
.section-sub{ color:var(--ink-soft); font-size:0.83rem; margin-bottom:16px; font-style:italic; }
.tag-no{
    font-family:'IBM Plex Mono', monospace; font-size:11px; color:var(--ink-soft);
    background:var(--paper); border:1px solid var(--line); padding:1px 7px; border-radius:3px;
}

hr.cut-line{
    border:none; border-top:1px dashed var(--line); margin:28px 0;
}

/* ---- Inputs, dark-on-paper tuned ---- */
div[data-testid="stNumberInput"] input, div[data-testid="stTextArea"] textarea{
    background:#FBF7EA !important; color:var(--ink) !important;
    border:1.5px solid var(--line) !important; border-radius:6px !important;
    font-family:'IBM Plex Mono', monospace !important;
}
div[data-testid="stTextArea"] textarea{ font-family:'IBM Plex Sans', sans-serif !important; }
div[data-testid="stSelectbox"] > div > div{
    background:#FBF7EA !important; border:1.5px solid var(--line) !important; border-radius:6px !important;
}
div[data-testid="stCheckbox"] label p{ color:var(--ink) !important; }
label p { color: var(--ink-soft) !important; font-size:0.82rem !important; text-transform:uppercase; letter-spacing:0.03em; }

div.stButton > button{
    background:var(--redact) !important; color:#F5EFDC !important; border:none !important;
    border-radius:4px !important; padding:0.75em 1.4em !important; font-weight:600 !important;
    font-family:'Special Elite', monospace !important; letter-spacing:0.08em !important;
    text-transform:uppercase; font-size:0.85rem !important;
    transition:.15s ease; border:1px solid var(--redact) !important;
}
div.stButton > button:hover{ background:#332E24 !important; transform:translateY(-1px); }
div.stButton > button:focus-visible{ outline:3px solid var(--warn) !important; outline-offset:2px; }

/* ---- Verdict stamp: the signature element ---- */
.stamp-zone{
    display:flex; justify-content:center; align-items:center;
    padding:34px 10px 30px 10px; position:relative;
}
.stamp{
    display:inline-block; padding:16px 34px; border:5px double;
    font-family:'Special Elite', monospace; font-size:2.1rem; letter-spacing:0.12em;
    text-transform:uppercase; transform:rotate(-5deg); border-radius:6px;
    opacity:0.92; text-shadow: 0 0 1px currentColor;
    background: repeating-radial-gradient(circle at 30% 30%, rgba(0,0,0,0.02), transparent 2px);
}
.stamp-claim{ color:var(--claim); border-color:var(--claim); }
.stamp-opinion{ color:var(--opinion); border-color:var(--opinion); }
.stamp-sub{
    text-align:center; font-family:'IBM Plex Mono', monospace; font-size:11.5px;
    color:var(--ink-soft); letter-spacing:0.06em; margin-top:-6px;
}

/* ---- Confidence gauge styled as a polygraph strip ---- */
.gauge-label{
    display:flex; justify-content:space-between; font-family:'IBM Plex Mono', monospace;
    font-size:11px; color:var(--ink-soft); margin-bottom:5px; text-transform:uppercase;
}
.meter-track{
    background:#E4D8B8; border-radius:3px; height:16px; overflow:hidden;
    border:1px solid var(--line); position:relative;
}
.meter-fill{ height:100%; }
.meter-threshold{
    position:absolute; top:-4px; bottom:-4px; width:2px; background:var(--redact);
}
.meter-threshold::after{
    content:'DECISION LINE'; position:absolute; top:-16px; left:6px;
    font-family:'IBM Plex Mono', monospace; font-size:9px; color:var(--ink-soft); white-space:nowrap;
}

.stat-readout{ font-family:'IBM Plex Mono', monospace; font-weight:600; color:var(--ink); }

/* ---- Evidence log / diagnostics ---- */
.diag-ok{ color:#3E7D4C; }
.diag-warn{ color:var(--warn); }
.diag-bad{ color:var(--claim); font-weight:600; }

.redact-line{
    display:inline-block; background:var(--redact); color:var(--redact);
    padding:0 8px; border-radius:2px; user-select:none;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================
# Layout
# ==============================================================
case_id = f"AMU-TS-{datetime.now().strftime('%Y%m%d')}-{abs(hash(datetime.now())) % 900 + 100}"

st.markdown('<div class="folder-wrap"><span class="folder-tab">Exhibit &nbsp;·&nbsp; Content Review Desk</span></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">Trust &amp; Safety — Screening Assist</div>
    <h1>🗂️ Claim vs. Opinion Review</h1>
    <p>Estimates whether a video makes a <b>verifiable claim</b> (a factual
    assertion, possibly against platform policy) or states a <b>personal
    opinion</b> — using engagement signals, author trust indicators, and the
    transcript itself.</p>
    <div class="case-meta">
        <div><b>Case No.</b>{case_id}</div>
        <div><b>Reviewer</b>Automated Model</div>
        <div><b>Opened</b>{datetime.now().strftime('%d %b %Y, %H:%M')}</div>
        <div><b>Status</b>Awaiting Input</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
<h4>📊 Engagement Log <span class="tag-no">FORM 01</span></h4>
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
<h4>👤 Author Trust Indicators <span class="tag-no">FORM 02</span></h4>
<div class="section-sub">Signals about the account posting the video.</div>
""", unsafe_allow_html=True)
a1, a2 = st.columns(2)
with a1:
    verified = st.checkbox("Author is verified")
with a2:
    banned = st.selectbox("Author ban status", ["active", "under review", "banned"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">
<h4>📝 Transcript <span class="tag-no">EXHIBIT A</span></h4>
<div class="section-sub">What the video's narration or on-screen text actually says.</div>
""", unsafe_allow_html=True)
transcript = st.text_area(
    "Video transcript text",
    placeholder="e.g. someone shared with me that drone deliveries are...",
    label_visibility="collapsed",
    height=110,
)
st.markdown("</div>", unsafe_allow_html=True)

run = st.button("File for Review", type="primary", use_container_width=True)

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

    st.markdown('<hr class="cut-line">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow" style="text-align:center;">Verdict</div>', unsafe_allow_html=True)

    if pred == 1:
        st.markdown("""
        <div class="stamp-zone">
            <div style="text-align:center;">
                <div class="stamp stamp-claim">🚩 Claim</div>
                <div class="stamp-sub">FLAGGED FOR REVIEW — CASE %s</div>
            </div>
        </div>
        """ % case_id, unsafe_allow_html=True)
        fill_color = "var(--claim)"
    else:
        st.markdown("""
        <div class="stamp-zone">
            <div style="text-align:center;">
                <div class="stamp stamp-opinion">💬 Opinion</div>
                <div class="stamp-sub">CLEARED — CASE %s</div>
            </div>
        </div>
        """ % case_id, unsafe_allow_html=True)
        fill_color = "var(--opinion)"

    pct = proba * 100
    st.markdown(f"""
    <div class="gauge-label"><span>Opinion</span><span>Claim probability</span></div>
    <div class="meter-track">
        <div class="meter-fill" style="width:{pct:.1f}%; background:{fill_color};"></div>
        <div class="meter-threshold" style="left:50%;"></div>
    </div>
    <p style="margin-top:14px; text-align:center;" class="stat-readout">{proba:.1%} claim probability</p>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Chain of Custody — Feature Audit Trail"):
        st.caption("Every prediction should be traceable back to the inputs that produced it. This log shows exactly what reached the model.")
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
        st.markdown(
            '<span class="diag-warn">⚠ If probability barely moves across very different transcripts, '
            'suspect class imbalance or leakage in training rather than this UI — check the training '
            'label distribution and confirm no leaked column encodes the label.</span>',
            unsafe_allow_html=True,
        )

    st.caption(f"Model: Random Forest classifier · Case {case_id} · Not a substitute for human moderation review")
