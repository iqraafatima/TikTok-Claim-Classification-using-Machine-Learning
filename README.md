# TikTok Claim vs. Opinion Classifier

A machine learning project that classifies TikTok videos as **claims**
(assertions of fact) or **opinions**, based on video transcription text and
metadata. Includes an interactive Streamlit app and a walkthrough of a real
**data leakage bug** found and fixed during the analysis.

## The problem

TikTok wants to prioritize videos that make factual claims for review by its
content moderation team, since claims are more likely to violate policy than
opinions. Given ~19K labeled videos with transcription text and account/video
metadata, the goal is to build a classifier that flags claim videos.

## ⚠️ Key finding: the original model had a data leakage bug

The initial notebook trained its models on the full feature set, including
`video_view_count`, `video_like_count`, `video_share_count`,
`video_download_count`, and `video_comment_count`. These are **engagement
metrics captured after a video is posted and after TikTok's own moderation
systems have already acted on it** — they are a *consequence* of a video
being a claim, not information a real classifier would have in advance.

The evidence is stark:

| Approach | Accuracy |
|---|---|
| Single rule: `video_view_count > 10,000` → "claim" | **99.5%** |
| Full model (Random Forest) trained **with** engagement metrics | **99.8%** |
| Full model (Random Forest) trained **without** engagement metrics (text + metadata only) | **82.6%** |

Claim videos in this dataset average ~500K views; opinion videos average
~5K views — the two classes are almost perfectly separated by view count
alone, for reasons unrelated to the video's content (e.g., moderation
throttling reach). Training on these columns produces a model that looks
excellent on paper but has **not learned anything about what makes a video
a claim** — it's just rediscovering the leak. This is a classic case of
**target leakage**, and it would fail immediately in production, where a
newly uploaded video has no engagement history yet.

**Fix:** the app trains on a leakage-safe feature set by default —
transcription text (via n-gram counts), video duration, verified status, and
author ban status only — all information available at/near upload time. The
leaky configuration is kept as an opt-in toggle purely for comparison and
demonstration.

## What's in this repo

```
.
├── app.py                     # Streamlit app (EDA, leakage demo, model comparison, live prediction)
├── requirements.txt
├── data/
│   └── tiktok_dataset.csv     # dataset (~19K rows)
└── eda_and_modeling.ipynb     # original exploratory notebook
```

## App features

- **Overview** — dataset summary and sample rows
- **EDA** — text length distributions, class balance, correlation heatmap,
  engagement metrics by class
- **Data Leakage** — interactive slider to test the "one-line rule" against
  the real labels, KDE plot showing the near-total separation by view count,
  and an explanation of the fix
- **Model Performance** — swap between Logistic Regression, Random Forest,
  and XGBoost; toggle leaky vs. leakage-safe features; view accuracy,
  precision, recall, F1, ROC AUC, confusion matrix, ROC curve, and feature
  importances
- **Try a Prediction** — enter a hypothetical video's transcription text and
  metadata and get a live claim/opinion prediction

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

You can also upload a different CSV with the same schema from the sidebar.

## Dataset

Each row is one TikTok video with:
- `claim_status` — target label (`claim` / `opinion`)
- `video_transcription_text` — transcribed spoken text from the video
- `video_duration_sec`, `verified_status`, `author_ban_status`
- `video_view_count`, `video_like_count`, `video_share_count`,
  `video_download_count`, `video_comment_count` — engagement metrics
  (excluded from the default model; see above)

## Modeling approach

1. Clean data (drop nulls/duplicates)
2. Feature engineering: text length, 2–3 word n-gram counts (top 15,
   English stop words removed) via `CountVectorizer`, one-hot encoding of
   categorical metadata
3. Train/validation/test split (60/20/20)
4. Models compared: Logistic Regression (baseline), Random Forest and
   XGBoost (hyperparameter-tuned via `GridSearchCV`, optimized for recall
   since missing a real claim is costlier than a false positive)
5. Evaluation: accuracy, precision, recall, F1, ROC AUC, confusion matrix,
   feature importance

 ## bullet points
Built and deployed an interactive Streamlit ML app classifying TikTok videos as factual claims vs. opinions from transcription text and metadata, comparing Logistic Regression, Random Forest, and XGBoost (82.6% accuracy on leakage-safe features)
Identified and fixed a critical data leakage bug where post-publication engagement metrics (views, likes, shares) allowed a single-feature rule to achieve 99.5% accuracy, masking the model's true content-based predictive power
Engineered an NLP feature pipeline using n-gram count vectorization (CountVectorizer, 2–3 grams) combined with categorical and numerical metadata for binary text classification
Applied GridSearchCV hyperparameter tuning across 100+ parameter combinations, optimizing for recall to prioritize catching true claim videos for moderation review
Communicated model risk clearly through visual diagnostics (ROC/PR curves, confusion matrices, feature importance, SHAP) to distinguish genuine signal from spurious correlation


## Possible next steps

- Add cross-validated leakage checks as a standard pipeline step (e.g.,
  automatically flag any feature with near-perfect single-variable
  separation before modeling)
- Try transformer-based text embeddings in place of n-gram counts
- Calibrate the recall/precision trade-off against real moderation team
  capacity
