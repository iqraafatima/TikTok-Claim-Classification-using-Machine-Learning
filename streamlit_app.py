import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Case File · Claim Review", page_icon="🗂️", layout="wide")

MODEL_PATH = Path("models/champion_model.pkl")
VECTORIZER_PATH = Path("models/vectorizer.pkl")

STUDENT_NAME = "Iqra Fatima Umang"
STUDENT_DEGREE = "Master of Computer Science and Applications"
STUDENT_UNIVERSITY = "Aligarh Muslim University"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_artifacts()

# session-stable case numbering: fixes the old bug where the case number
# was regenerated from datetime.now() on every Streamlit rerun and could
# mismatch between the header and the result.
if "case_seed" not in st.session_state:
    st.session_state.case_seed = datetime.now().strftime("%Y%m%d")
if "filing_count" not in st.session_state:
    st.session_state.filing_count = 0

CASE_ID = f"AMU-TS-{st.session_state.case_seed}-{st.session_state.filing_count:04d}"

# ==============================================================
# DESIGN SYSTEM — "Case File" aesthetic
# ==============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Zilla+Slab:wght@500;600;700&family=Special+Elite&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
    --paper:#EDE4CC;
    --paper-alt:#E4D8B8;
    --paper-card:#F5EFDC;
    --folder:#C9A667;
    --folder-dark:#B08F52;
    --ink:#24211A;
    --ink-soft:#6B6350;
    --claim:#9C3B2C;
    --opinion:#2E5578;
    --warn:#A87422;
    --redact:#1C1A16;
    --line:#B9AC85;
    --tab-amber:#B8862E;
    --tab-blue:#2E5578;
    --tab-red:#9C3B2C;
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

/* ---- Sidebar: case officer badge + intake ---- */
section[data-testid="stSidebar"]{
    background: var(--paper-alt) !important; border-right:2px solid var(--folder-dark);
}
section[data-testid="stSidebar"] .block-container{ padding-top:1.6rem; }

.id-badge{
    background:#FBF7EA; border:2px solid var(--redact); border-radius:8px;
    padding:14px 16px; margin-bottom:22px; transform:rotate(-1.2deg);
    box-shadow: 3px 4px 0 rgba(28,26,22,0.15);
    display:flex; align-items:center; gap:12px;
}
.id-badge-photo{
    width:44px; height:44px; border-radius:6px; background:var(--folder);
    display:flex; align-items:center; justify-content:center; font-size:22px;
    border:1.5px solid var(--redact); flex-shrink:0;
}
.id-badge-name{ font-family:'Zilla Slab', serif; font-weight:700; font-size:0.98rem; color:var(--ink); line-height:1.15; }
.id-badge-role{ font-family:'IBM Plex Mono', monospace; font-size:10.5px; color:var(--ink-soft); margin-top:3px; }
.id-badge-org{ font-family:'IBM Plex Mono', monospace; font-size:10.5px; color:var(--tab-blue); font-weight:600; }

.sidebar-eyebrow{
    font-family:'Special Elite', monospace; font-size:11px; letter-spacing:0.1em;
    text-transform:uppercase; color:var(--ink-soft); margin:18px 0 6px 0;
    border-top:1px dashed var(--line); padding-top:12px;
}
.sidebar-eyebrow:first-of-type{ border-top:none; padding-top:0; margin-top:0; }

/* ---- Folder tab masthead ---- */
.folder-tab{
    display:inline-block; background:var(--folder); color:var(--redact);
    padding:8px 26px 10px 26px; border-radius:10px 10px 0 0;
    font-family:'Special Elite', monospace; font-size:12px; letter-spacing:0.08em;
    text-transform:uppercase; border:1px solid var(--folder-dark); border-bottom:none;
}
.hero{
    background: var(--paper-alt);
    border:1px solid var(--line); border-top:3px solid var(--folder-dark);
    padding:30px 36px 26px 36px; border-radius:0 12px 12px 12px;
    box-shadow: 6px 6px 0 var(--paper-card), 12px 12px 0 var(--line);
    margin-bottom:36px; position:relative;
}
.hero h1{ font-size:2rem; margin:6px 0 10px 0; }
.hero p{ color:var(--ink-soft); max-width:660px; margin:0; font-size:0.98rem; line-height:1.5; }
.confidential-stamp{
    position:absolute; top:22px; right:34px; border:2.5px solid var(--claim); color:var(--claim);
    font-family:'Special Elite', monospace; font-size:11px; letter-spacing:0.1em;
    padding:4px 10px; border-radius:4px; transform:rotate(6deg); opacity:0.75;
    text-transform:uppercase;
}
.case-meta{
    display:flex; gap:26px; margin-top:18px; padding-top:14px;
    border-top:1px dashed var(--line); flex-wrap:wrap;
}
.case-meta div{ font-family:'IBM Plex Mono', monospace; font-size:12px; color:var(--ink-soft); }
.case-meta b{ color:var(--ink); display:block; font-size:13px; margin-bottom:2px; font-family:'IBM Plex Sans',sans-serif; }

/* ---- Section cards with colored folder-tab ribbon ---- */
.section-card{
    background:var(--paper-card); border:1px solid var(--line);
    padding:20px 22px 22px 22px; border-radius:4px; margin-bottom:20px;
    box-shadow: var(--card-shadow); position:relative;
    clip-path: polygon(0 0, 100% 0, 100% 100%, 12px 100%, 0 calc(100% - 12px));
}
.ribbon{
    position:absolute; top:10px; right:-9px; color:#fff;
    font-family:'IBM Plex Mono', monospace; font-size:10px; letter-spacing:0.06em;
    padding:3px 10px 3px 8px; border-radius:3px 0 0 3px; text-transform:uppercase;
    box-shadow: -2px 2px 0 rgba(0,0,0,0.12);
}
.section-card h4{
    margin-top:0; margin-bottom:2px; color:var(--ink); font-family:'Zilla Slab', serif;
    font-size:1.08rem;
}
.section-sub{ color:var(--ink-soft); font-size:0.8rem; margin-bottom:14px; font-style:italic; }

/* ---- Ledger (live receipt of current inputs) ---- */
.ledger{
    background:#FBF7EA; border:1px solid var(--line); border-radius:4px;
    padding:18px 22px; font-family:'IBM Plex Mono', monospace; font-size:12.5px;
    box-shadow: var(--card-shadow);
}
.ledger-title{
    font-family:'Special Elite', monospace; font-size:12px; text-transform:uppercase;
    letter-spacing:0.08em; color:var(--ink-soft); margin-bottom:10px;
    border-bottom:1px dashed var(--line); padding-bottom:8px;
}
.ledger-row{ display:flex; justify-content:space-between; padding:3px 0; color:var(--ink); }
.ledger-row span:first-child{ color:var(--ink-soft); }
.ledger-row.total{ border-top:1px dashed var(--line); margin-top:6px; padding-top:8px; font-weight:600; }

hr.cut-line{ border:none; border-top:1px dashed var(--line); margin:30px 0; }

/* ---- Inputs ---- */
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
label p { color: var(--ink-soft) !important; font-size:0.78rem !important; text-transform:uppercase; letter-spacing:0.03em; }

div.stButton > button{
    background:var(--redact) !important; color:#F5EFDC !important; border:none !important;
    border-radius:4px !important; padding:0.8em 1.4em !important; font-weight:600 !important;
    font-family:'Special Elite', monospace !important; letter-spacing:0.08em !important;
    text-transform:uppercase; font-size:0.85rem !important; width:100%;
    transition:.15s ease; border:1px solid var(--redact) !important;
}
div.stButton > button:hover{ background:#332E24 !important; transform:translateY(-1px); }
div.stButton > button:focus-visible{ outline:3px solid var(--warn) !important; outline-offset:2px; }

/* ---- Verdict stamp ---- */
.stamp-zone{ display:flex; justify-content:center; align-items:center; padding:30px 10px 26px 10px; }
.stamp{
    display:inline-block; padding:16px 34px; border:5px double;
    font-family:'Special Elite', monospace; font-size:2.1rem; letter-spacing:0.12em;
    text-transform:uppercase; transform:rotate(-5deg); border-radius:6px;
    opacity:0.92; text-shadow: 0 0 1px currentColor;
}
.stamp-claim{ color:var(--claim); border-color:var(--claim); }
.stamp-opinion{ color:var(--opinion); border-color:var(--opinion); }
.stamp-sub{
    text-align:center; font-family:'IBM Plex Mono', monospace; font-size:11.5px;
    color:var(--ink-soft); letter-spacing:0.06em; margin-top:-6px;
}

.gauge-label{
    display:flex; justify-content:space-between; font-family:'IBM Plex Mono', monospace;
    font-size:11px; color:var(--ink-soft); margin-bottom:5px; text-transform:uppercase;
}
.meter-track{
    background:#E4D8B8; border-radius:3px; height:16px; overflow:hidden;
    border:1px solid var(--line); position:relative;
}
.meter-fill{ height:100%; }
.meter-threshold{ position:absolute; top:-4px; bottom:-4px; width:2px; background:var(--redact); }
.meter-threshold::after{
    content:'DECISION LINE'; position:absolute; top:-16px; left:6px;
    font-family:'IBM Plex Mono', monospace; font-size:9px; color:var(--ink-soft); white-space:nowrap;
}
.stat-readout{ font-family:'IBM Plex Mono', monospace; font-weight:600; color:var(--ink); }

.diag-ok{ color:#3E7D4C; }
.diag-warn{ color:var(--warn); }
.diag-bad{ color:var(--claim); font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ==============================================================
# Sidebar — case officer badge + intake form
# ==============================================================
with st.sidebar:
    st.markdown(f"""
    <div class="id-badge">
        <div class="id-badge-photo">🕵️</div>
        <div>
            <div class="id-badge-name">{STUDENT_NAME}</div>
            <div class="id-badge-role">{STUDENT_DEGREE}</div>
            <div class="id-badge-org">{STUDENT_UNIVERSITY}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-eyebrow">📊 Engagement Log</div>', unsafe_allow_html=True)
    duration = st.number_input("Video duration (seconds)", min_value=1, max_value=600, value=30)
    views = st.number_input("View count", min_value=0, value=10000)
    likes = st.number_input("Like count", min_value=0, value=500)
    shares = st.number_input("Share count", min_value=0, value=50)
    downloads = st.number_input("Download count", min_value=0, value=10)
    comments = st.number_input("Comment count", min_value=0, value=20)

    st.markdown('<div class="sidebar-eyebrow">👤 Author Trust Indicators</div>', unsafe_allow_html=True)
    verified = st.checkbox("Author is verified")
    banned = st.selectbox("Author ban status", ["active", "under review", "banned"])

    st.markdown('<div class="sidebar-eyebrow">📝 Transcript</div>', unsafe_allow_html=True)
    transcript = st.text_area(
        "Video transcript text",
        placeholder="e.g. someone shared with me that drone deliveries are...",
        label_visibility="collapsed",
        height=110,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("File for Review", type="primary")

# ==============================================================
# Main — masthead, ledger, verdict
# ==============================================================
st.markdown('<span class="folder-tab">Exhibit &nbsp;·&nbsp; Content Review Desk</span>', unsafe_allow_html=True)
st.markdown(f"""
<div class="hero">
    <div class="confidential-stamp">Confidential</div>
    <div class="eyebrow">Trust &amp; Safety — Screening Assist</div>
    <h1>🗂️ Claim vs. Opinion Review</h1>
    <p>Estimates whether a video makes a <b>verifiable claim</b> (a factual
    assertion, possibly against platform policy) or states a <b>personal
    opinion</b> — using engagement signals, author trust indicators, and the
    transcript itself.</p>
    <div class="case-meta">
        <div><b>Case No.</b>{CASE_ID}</div>
        <div><b>Prepared by</b>{STUDENT_NAME}</div>
        <div><b>Opened</b>{datetime.now().strftime('%d %b %Y, %H:%M')}</div>
        <div><b>Status</b>{'Filed' if run else 'Awaiting Input'}</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_ledger, col_result = st.columns([1, 1.4], gap="large")

with col_ledger:
    st.markdown(f"""
    <div class="ledger">
        <div class="ledger-title">Intake Ledger — Current Entry</div>
        <div class="ledger-row"><span>Duration</span><span>{duration} sec</span></div>
        <div class="ledger-row"><span>Views</span><span>{views:,}</span></div>
        <div class="ledger-row"><span>Likes</span><span>{likes:,}</span></div>
        <div class="ledger-row"><span>Shares</span><span>{shares:,}</span></div>
        <div class="ledger-row"><span>Downloads</span><span>{downloads:,}</span></div>
        <div class="ledger-row"><span>Comments</span><span>{comments:,}</span></div>
        <div class="ledger-row"><span>Verified author</span><span>{"Yes" if verified else "No"}</span></div>
        <div class="ledger-row"><span>Ban status</span><span>{banned}</span></div>
        <div class="ledger-row total"><span>Transcript length</span><span>{len(transcript)} chars</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_result:
    if not run:
        st.markdown("""
        <div class="section-card" style="text-align:center; padding:38px 22px;">
            <div class="ribbon" style="background:var(--tab-amber);">Pending</div>
            <h4>No verdict yet</h4>
            <div class="section-sub">Fill the intake form in the sidebar, then click "File for Review."</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.session_state.filing_count += 1
        CASE_ID = f"AMU-TS-{st.session_state.case_seed}-{st.session_state.filing_count:04d}"

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

        pre_reindex_cols = set(final_df.columns)
        model_cols = set(model.feature_names_in_)
        matched = pre_reindex_cols & model_cols
        missing_from_input = model_cols - pre_reindex_cols
        nonzero_text_feats = int((text_feats.values != 0).sum())

        final_df = final_df.reindex(columns=model.feature_names_in_, fill_value=0)

        pred = model.predict(final_df)[0]
        proba = model.predict_proba(final_df)[0][1]

        if pred == 1:
            st.markdown(f"""
            <div class="stamp-zone">
                <div style="text-align:center;">
                    <div class="stamp stamp-claim">🚩 Claim</div>
                    <div class="stamp-sub">FLAGGED FOR REVIEW — CASE {CASE_ID}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            fill_color = "var(--claim)"
        else:
            st.markdown(f"""
            <div class="stamp-zone">
                <div style="text-align:center;">
                    <div class="stamp stamp-opinion">💬 Opinion</div>
                    <div class="stamp-sub">CLEARED — CASE {CASE_ID}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
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
            st.caption("Every prediction should be traceable back to the inputs that produced it.")
            st.markdown(f"""
            - **Model expects** `{len(model_cols)}` total features
            - **Your input produced** `{len(pre_reindex_cols)}` features before reindexing
            - **Matched features**: `{len(matched)}`
            - **Missing / zero-filled**: `{len(missing_from_input)}`
            - **Non-zero TF-IDF values in transcript**: `{nonzero_text_feats}`
            """)
            if nonzero_text_feats == 0 and text_len > 0:
                st.markdown(
                    '<span class="diag-bad">⚠ Transcript produced an all-zero TF-IDF vector — '
                    'confirm this is the SAME fitted vectorizer used at training time.</span>',
                    unsafe_allow_html=True,
                )
            if len(missing_from_input) > 0:
                st.markdown(
                    f'<span class="diag-warn">⚠ {len(missing_from_input)} feature(s) the model expects '
                    'were zero-filled. If mostly text features, the model may be ignoring the transcript.</span>',
                    unsafe_allow_html=True,
                )
                st.code(sorted(missing_from_input)[:30])
            st.markdown(
                '<span class="diag-warn">⚠ If probability barely moves across very different transcripts, '
                'suspect class imbalance or leakage in training, not this UI.</span>',
                unsafe_allow_html=True,
            )

st.markdown('<hr class="cut-line">', unsafe_allow_html=True)
st.caption(f"Model: Random Forest classifier · Prepared by {STUDENT_NAME}, {STUDENT_DEGREE}, {STUDENT_UNIVERSITY} · Not a substitute for human moderation review")
