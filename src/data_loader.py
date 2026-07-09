import pandas as pd
from src.config import DATA_PATH

def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(axis=0)
    df = df.drop_duplicates()
    return df