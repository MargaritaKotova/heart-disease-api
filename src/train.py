"""Train and save the classifier."""
from pathlib import Path
import sys

# Direct file execution puts src/, rather than the project root, on sys.path.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.ensemble import RandomForestClassifier

from config.variables import RANDOM_STATE


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)
    return model.fit(X_train, y_train)


def save_model(model, model_path):
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    from src.pipeline import main

    main()
