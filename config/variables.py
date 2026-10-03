"""Paths and reproducible training settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/heart.csv"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models/model.joblib"))
RANDOM_STATE = 42
TEST_SIZE = 0.2
