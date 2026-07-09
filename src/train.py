"""
Train script: run as `python -m src.train`
Trains RF + XGBoost with cross-validation, selects champion by recall,
saves the model and vectorizer to disk.
"""

import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, MODEL_PATH, VECTORIZER_PATH
from src.data_loader import load_data
from src.features import (add_text_length, encode_target, encode_categoricals,
                            fit_vectorizer, transform_text)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_dataset():
    df = load_data()
    df = add_text_length(df)
    df = encode_target(df)

    X = df.drop(columns=['#', 'video_id'], errors='ignore')
    X = encode_categoricals(X)
    y = X.pop('claim_status')
    return X, y


def main():
    logger.info("Loading and preparing data...")
    X, y = build_dataset()

    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=RANDOM_STATE)

    logger.info("Fitting text vectorizer on training data only...")
    vec = fit_vectorizer(X_train['video_transcription_text'])

    def finalize(X_split):
        text_feats = transform_text(vec, X_split['video_transcription_text'])
        return pd.concat(
            [X_split.drop(columns=['video_transcription_text']).reset_index(drop=True),
             text_feats.reset_index(drop=True)],
            axis=1
        )

    X_train_final = finalize(X_train)
    X_val_final = finalize(X_val)
    X_test_final = finalize(X_test)

    logger.info("Running GridSearchCV for Random Forest...")
    rf_cv = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid={
            'max_depth': [5, 7, None], 'max_features': [0.3, 0.6],
            'max_samples': [0.7], 'min_samples_leaf': [1, 2],
            'min_samples_split': [2, 3], 'n_estimators': [75, 100, 200],
        },
        scoring=['accuracy', 'precision', 'recall', 'f1'], cv=5, refit='recall'
    )
    rf_cv.fit(X_train_final, y_train)
    logger.info(f"RF best recall (CV): {rf_cv.best_score_:.4f}")

    logger.info("Running GridSearchCV for XGBoost...")
    xgb_cv = GridSearchCV(
        XGBClassifier(objective='binary:logistic', random_state=RANDOM_STATE),
        param_grid={
            'max_depth': [4, 8, 12], 'min_child_weight': [3, 5],
            'learning_rate': [0.01, 0.1], 'n_estimators': [300, 500],
        },
        scoring=['accuracy', 'precision', 'recall', 'f1'], cv=5, refit='recall'
    )
    xgb_cv.fit(X_train_final, y_train)
    logger.info(f"XGB best recall (CV): {xgb_cv.best_score_:.4f}")

    # Champion selection on validation set (not just CV score)
    from sklearn.metrics import recall_score
    rf_val_recall = recall_score(y_val, rf_cv.best_estimator_.predict(X_val_final))
    xgb_val_recall = recall_score(y_val, xgb_cv.best_estimator_.predict(X_val_final))
    logger.info(f"RF val recall: {rf_val_recall:.4f} | XGB val recall: {xgb_val_recall:.4f}")

    champion = rf_cv.best_estimator_ if rf_val_recall >= xgb_val_recall else xgb_cv.best_estimator_
    champion_name = "RandomForest" if rf_val_recall >= xgb_val_recall else "XGBoost"
    logger.info(f"Champion model: {champion_name}")

    test_recall = recall_score(y_test, champion.predict(X_test_final))
    logger.info(f"Champion recall on held-out test set: {test_recall:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, MODEL_PATH)
    joblib.dump(vec, VECTORIZER_PATH)
    logger.info(f"Saved model to {MODEL_PATH} and vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()