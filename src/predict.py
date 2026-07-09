import joblib
import pandas as pd
from src.config import MODEL_PATH, VECTORIZER_PATH

class ClaimPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

    def predict(self, features: dict) -> dict:
        text = features.pop('video_transcription_text', '')
        base_df = pd.DataFrame([features])
        text_feats = pd.DataFrame(
            self.vectorizer.transform([text]).toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )
        final_df = pd.concat([base_df.reset_index(drop=True), text_feats], axis=1)
        final_df = final_df.reindex(columns=self.model.feature_names_in_, fill_value=0)

        pred = self.model.predict(final_df)[0]
        proba = self.model.predict_proba(final_df)[0][1]
        return {"prediction": "claim" if pred == 1 else "opinion", "confidence": float(proba)}