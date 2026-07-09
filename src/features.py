import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from src.config import CATEGORICAL_FEATURES

def add_text_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['text_length'] = df['video_transcription_text'].str.len()
    return df

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['claim_status'] = df['claim_status'].replace({'opinion': 0, 'claim': 1})
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

def fit_vectorizer(train_text: pd.Series, max_features=15) -> CountVectorizer:
    vec = CountVectorizer(ngram_range=(2, 3), max_features=max_features, stop_words='english')
    vec.fit(train_text)
    return vec

def transform_text(vec: CountVectorizer, text: pd.Series) -> pd.DataFrame:
    arr = vec.transform(text).toarray()
    return pd.DataFrame(arr, columns=vec.get_feature_names_out())