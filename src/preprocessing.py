"""Validate the dataset and split unique observations."""
import pandas as pd
from sklearn.model_selection import train_test_split

from app.schema import FEATURES, HeartFeatures
from config.variables import DATA_PATH, RANDOM_STATE, TEST_SIZE


def preprocess_data(data_path=DATA_PATH):
    raw = pd.read_csv(data_path)
    if set(raw.columns) != set(FEATURES + ["target"]):
        raise ValueError("Unexpected dataset columns")
    if raw.isna().any().any() or set(raw.target.unique()) != {0, 1}:
        raise ValueError("Dataset must have no missing values and binary targets")
    df = raw.drop_duplicates().reset_index(drop=True)
    if df.groupby(FEATURES).target.nunique().max() > 1:
        raise ValueError("Conflicting labels for identical feature vectors")
    for row in df[FEATURES].to_dict(orient="records"):
        HeartFeatures.model_validate(row)
    split = train_test_split(
        df[FEATURES], df.target, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=df.target,
    )
    metadata = {
        "source": "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/",
        "raw_rows": len(raw), "unique_rows": len(df),
        "duplicates_removed": len(raw) - len(df),
        "train_rows": len(split[0]), "test_rows": len(split[1]),
        "random_state": RANDOM_STATE, "features": FEATURES,
    }
    return (*split, metadata)
