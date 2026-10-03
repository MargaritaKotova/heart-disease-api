"""Load the model and preserve feature order during inference."""
import joblib

from app.schema import FEATURES


def load_model(model_path):
    model = joblib.load(model_path)
    if list(model.feature_names_in_) != FEATURES or list(model.classes_) != [0, 1]:
        raise RuntimeError("Model schema does not match the API")
    return model


def predict(model, X):
    return model.predict(X.loc[:, FEATURES])
