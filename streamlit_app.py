"""
TikTok Claim vs. Opinion Classifier — Streamlit App
----------------------------------------------------
Companion app for the eda_and_modeling.ipynb notebook. Recreates the
EDA + modeling pipeline interactively, and — importantly — surfaces a
serious data leakage issue found in the original notebook: the model
was trained on post-publication engagement metrics (views, likes,
shares, downloads, comments) which are a near-perfect proxy for the
label, not a genuine signal available at prediction time.

Run with:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# --------------------------------------------------------------------------
# Page config & light styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="TikTok Claim Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .leak-banner {
        background-color: #3a1f1f;
        border-left: 5px solid #e05c5c;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .safe-banner {
        background-color: #1f3a24;
        border-left: 5px solid #5ce07f;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #262c36;
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ENGAGEMENT_COLS = [
    "video_view_count",
    "video_like_count",
    "video_share_count",
    "video_download_count",
    "video_comment_count",
]
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tiktok_dataset.csv")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.dropna(axis=0)
    df = df.drop_duplicates()
    df["text_length"] = df["video_transcription_text"].str.len()
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame, use_engagement: bool):
    X = df.copy()
    drop_cols = [c for c in ["#", "video_id", "text_length"] if c in X.columns]
    X = X.drop(columns=drop_cols)
    X["claim_status"] = X["claim_status"].map({"opinion": 0, "claim": 1})
    X = pd.get_dummies(X, columns=["verified_status", "author_ban_status"], drop_first=True)
    if not use_engagement:
        X = X.drop(columns=[c for c in ENGAGEMENT_COLS if c in X.columns])
    y = X["claim_status"]
    X = X.drop(columns=["claim_status"])
    return X, y


# --------------------------------------------------------------------------
# Model training (cached per config)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame, use_engagement: bool, model_choice: str):
    X, y = build_features(df, use_engagement)

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=0, stratify=y_tr
    )

    count_vec = CountVectorizer(ngram_range=(2, 3), max_features=15, stop_words="english")
    count_train = count_vec.fit_transform(X_train["video_transcription_text"]).toarray()
    count_val = count_vec.transform(X_val["video_transcription_text"]).toarray()
    count_test = count_vec.transform(X_test["video_transcription_text"]).toarray()
    feat_names = count_vec.get_feature_names_out()

    def merge(base_df, count_arr):
        return pd.concat(
            [
                base_df.drop(columns=["video_transcription_text"]).reset_index(drop=True),
                pd.DataFrame(count_arr, columns=feat_names),
            ],
            axis=1,
        )

    train_df = merge(X_train, count_train)
    val_df = merge(X_val, count_val)
    test_df = merge(X_test, count_test)

    scaler = None
    if model_choice == "Logistic Regression":
        scaler = StandardScaler()
        train_fit = scaler.fit_transform(train_df)
        val_fit = scaler.transform(val_df)
        test_fit = scaler.transform(test_df)
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(train_fit, y_train)
    elif model_choice == "XGBoost" and XGB_AVAILABLE:
        train_fit, val_fit, test_fit = train_df, val_df, test_df
        model = XGBClassifier(
            objective="binary:logistic",
            random_state=0,
            max_depth=8,
            min_child_weight=3,
            learning_rate=0.1,
            n_estimators=300,
            eval_metric="logloss",
        )
        model.fit(train_fit, y_train)
    else:  # Random Forest (default / fallback)
        train_fit, val_fit, test_fit = train_df, val_df, test_df
        model = RandomForestClassifier(
            random_state=0,
            n_estimators=200,
            max_features=0.6,
            min_samples_leaf=1,
            min_samples_split=2,
        )
        model.fit(train_fit, y_train)

    return {
        "model": model,
        "scaler": scaler,
        "count_vec": count_vec,
        "X_val": val_fit,
        "y_val": y_val,
        "X_test": test_fit,
        "y_test": y_test,
        "feature_names": train_df.columns.tolist(),
        "template_row": X_train.drop(columns=["video_transcription_text"]).iloc[0:1],
    }


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
    }
    if y_proba is not None:
        metrics["ROC AUC"] = roc_auc_score(y, y_proba)
    return metrics, y_pred, y_proba


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🎬 TikTok Claim Classifier")
st.sidebar.caption("Claim vs. opinion detection — with a data-leakage check built in.")

uploaded = st.sidebar.file_uploader("Upload tiktok_dataset.csv (optional)", type="csv")
data_source = uploaded if uploaded is not None else DEFAULT_DATA_PATH

df = load_data(data_source)

st.sidebar.markdown("---")
st.sidebar.subheader("Model configuration")

model_options = ["Random Forest", "Logistic Regression"]
if XGB_AVAILABLE:
    model_options.append("XGBoost")
model_choice = st.sidebar.selectbox("Model", model_options, index=0)

use_engagement = st.sidebar.checkbox(
    "Include engagement metrics (views/likes/shares/comments/downloads)",
    value=False,
    help="These columns are only known AFTER a video is posted and moderated. "
    "Turning this on recreates the leakage bug from the original notebook.",
)

if use_engagement:
    st.sidebar.markdown(
        '<div class="leak-banner">⚠️ Leaky mode: using post-publication '
        "engagement counts as predictors. Scores will look almost perfect "
        "but the model isn't learning anything useful about the content.</div>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div class="safe-banner">✅ Leakage-safe mode: model only sees '
        "information available before/at upload time.</div>",
        unsafe_allow_html=True,
    )

with st.spinner("Training model..."):
    bundle = train_model(df, use_engagement, model_choice)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_eda, tab_leakage, tab_model, tab_predict = st.tabs(
    ["📋 Overview", "📊 EDA", "🚨 Data Leakage", "🤖 Model Performance", "🔮 Try a Prediction"]
)

# ---- Overview ----
with tab_overview:
    st.header("TikTok Claim vs. Opinion Classifier")
    st.write(
        "This app reproduces the modeling notebook's pipeline (text n-grams + "
        "metadata → Random Forest / Logistic Regression / XGBoost) and adds an "
        "interactive check for a **serious data leakage issue** found in the "
        "original analysis. Use the tabs above to explore."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (after cleaning)", f"{len(df):,}")
    c2.metric("Claim videos", f"{(df['claim_status']=='claim').sum():,}")
    c3.metric("Opinion videos", f"{(df['claim_status']=='opinion').sum():,}")
    c4.metric("Class balance", f"{df['claim_status'].value_counts(normalize=True).iloc[0]*100:.1f}% / "
              f"{df['claim_status'].value_counts(normalize=True).iloc[1]*100:.1f}%")
    st.subheader("Sample data")
    st.dataframe(df.head(10), use_container_width=True)

# ---- EDA ----
with tab_eda:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transcription text length by class")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            data=df, x="text_length", hue="claim_status", multiple="dodge",
            palette="pastel", ax=ax
        )
        ax.set_xlabel("Text length (characters)")
        st.pyplot(fig)
        st.caption(
            f"Claims average {df.loc[df.claim_status=='claim','text_length'].mean():.1f} "
            f"chars vs. {df.loc[df.claim_status=='opinion','text_length'].mean():.1f} "
            "for opinions — a modest, genuine text signal."
        )

    with col2:
        st.subheader("Class balance")
        fig, ax = plt.subplots(figsize=(6, 4))
        df["claim_status"].value_counts().plot.bar(ax=ax, color=["#5c9ee0", "#e0a85c"])
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.subheader("Correlation heatmap — numeric features")
    numeric_cols = ["video_duration_sec"] + ENGAGEMENT_COLS
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("Engagement metrics by claim status")
    metric_pick = st.selectbox("Metric", ENGAGEMENT_COLS, index=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="claim_status", y=metric_pick, ax=ax)
    ax.set_yscale("symlog")
    st.pyplot(fig)

# ---- Leakage ----
with tab_leakage:
    st.header("🚨 The serious issue in the original notebook")
    st.markdown(
        """
The notebook builds its feature set (`X`) directly from the raw dataframe, which
still includes `video_view_count`, `video_like_count`, `video_share_count`,
`video_download_count`, and `video_comment_count`. These are **engagement
metrics measured after a video is posted and after TikTok's moderation system
has already reacted to it** — they are downstream consequences of a video
being a "claim," not independent evidence a real-world classifier would have
*before* making a decision. That makes them a form of **target leakage**.
        """
    )

    trivial_thresh = st.slider(
        "Try a single-feature rule: predict 'claim' if video_view_count is above...",
        min_value=0, max_value=int(df["video_view_count"].max()), value=10000, step=500,
    )
    guess = np.where(df["video_view_count"] > trivial_thresh, "claim", "opinion")
    trivial_acc = (guess == df["claim_status"]).mean()
    st.metric("Accuracy of this ONE-line rule (no ML, no text features)", f"{trivial_acc*100:.2f}%")
    st.caption(
        "If a single threshold on one column nearly matches your trained model's "
        "accuracy, the model isn't really learning from the video content — it's "
        "rediscovering this shortcut."
    )

    st.subheader("Why it happens: the two classes barely overlap on view count")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.kdeplot(data=df, x="video_view_count", hue="claim_status", fill=True, common_norm=False, ax=ax)
    ax.set_xlabel("video_view_count")
    st.pyplot(fig)
    st.write(
        pd.concat(
            [
                df.groupby("claim_status")["video_view_count"].mean().rename("mean views"),
                df.groupby("claim_status")["video_view_count"].median().rename("median views"),
            ],
            axis=1,
        )
    )

    st.info(
        "**Fix applied in this app:** the sidebar toggle 'Include engagement "
        "metrics' is OFF by default. With it off, the model is trained only on "
        "information available at/near upload time (transcription text, "
        "duration, verified status, author ban status) — a fair test of whether "
        "the video's *content* predicts claim vs. opinion. Flip the toggle on "
        "to reproduce the original (leaky) result and compare."
    )

# ---- Model performance ----
with tab_model:
    st.header(f"Model performance — {model_choice} ({'leaky' if use_engagement else 'leakage-safe'} features)")

    val_metrics, val_pred, val_proba = evaluate(bundle["model"], bundle["X_val"], bundle["y_val"])
    cols = st.columns(len(val_metrics))
    for c, (name, val) in zip(cols, val_metrics.items()):
        c.metric(name, f"{val:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion matrix (validation set)")
        cm = confusion_matrix(bundle["y_val"], val_pred)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["opinion", "claim"], yticklabels=["opinion", "claim"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        if val_proba is not None:
            st.subheader("ROC curve (validation set)")
            fpr, tpr, _ = roc_curve(bundle["y_val"], val_proba)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ax.plot(fpr, tpr, label=f"AUC = {val_metrics['ROC AUC']:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)

    st.subheader("Classification report (validation set)")
    st.text(classification_report(bundle["y_val"], val_pred, target_names=["opinion", "claim"]))

    if hasattr(bundle["model"], "feature_importances_"):
        st.subheader("Feature importances")
        imp = pd.Series(bundle["model"].feature_importances_, index=bundle["feature_names"])
        imp = imp.sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        if use_engagement and imp.index[0] in ENGAGEMENT_COLS:
            st.warning(
                f"Top feature is **{imp.index[0]}**, an engagement metric — "
                "confirming the model is leaning on leaked information rather "
                "than the video's text/content."
            )

    st.caption(
        "Held-out **test set** accuracy: "
        f"{accuracy_score(bundle['y_test'], bundle['model'].predict(bundle['X_test']))*100:.2f}%"
    )

# ---- Prediction playground ----
with tab_predict:
    st.header("🔮 Try a live prediction")
    st.write("Fill in a hypothetical video below. Fields shown depend on the current feature mode.")

    with st.form("predict_form"):
        text_in = st.text_area("Video transcription text", "someone shared with me that")
        duration = st.slider("Video duration (sec)", 5, 60, 30)
        verified = st.selectbox("Author verified status", ["not verified", "verified"])
        ban_status = st.selectbox("Author ban status", ["active", "under review", "banned"])

        engagement_inputs = {}
        if use_engagement:
            st.markdown("**Engagement metrics** (only meaningful for a video that's already live):")
            engagement_inputs["video_view_count"] = st.number_input("View count", 0, 2_000_000, 5000)
            engagement_inputs["video_like_count"] = st.number_input("Like count", 0, 1_000_000, 800)
            engagement_inputs["video_share_count"] = st.number_input("Share count", 0, 500_000, 150)
            engagement_inputs["video_download_count"] = st.number_input("Download count", 0, 50_000, 15)
            engagement_inputs["video_comment_count"] = st.number_input("Comment count", 0, 20_000, 5)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = bundle["template_row"].copy()
        row["video_transcription_text"] = text_in
        row["video_duration_sec"] = duration
        row["verified_status"] = verified
        row["author_ban_status"] = ban_status
        for k, v in engagement_inputs.items():
            row[k] = v

        row_enc = pd.get_dummies(row, columns=["verified_status", "author_ban_status"], drop_first=True)
        for col in bundle["feature_names"]:
            if col not in row_enc.columns and col not in bundle["count_vec"].get_feature_names_out():
                row_enc[col] = 0

        count_arr = bundle["count_vec"].transform(row_enc["video_transcription_text"]).toarray()
        count_df = pd.DataFrame(count_arr, columns=bundle["count_vec"].get_feature_names_out())
        final_row = pd.concat(
            [row_enc.drop(columns=["video_transcription_text"]).reset_index(drop=True), count_df],
            axis=1,
        )
        final_row = final_row.reindex(columns=bundle["feature_names"], fill_value=0)

        if bundle["scaler"] is not None:
            final_row = bundle["scaler"].transform(final_row)

        pred = bundle["model"].predict(final_row)[0]
        proba = (
            bundle["model"].predict_proba(final_row)[0][1]
            if hasattr(bundle["model"], "predict_proba")
            else None
        )
        label = "Claim" if pred == 1 else "Opinion"
        st.success(f"Prediction: **{label}**" + (f"  (P(claim) = {proba:.2f})" if proba is not None else ""))
