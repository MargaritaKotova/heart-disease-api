"""Run preprocessing, training, serialization and evaluation."""
import json
from pathlib import Path

from config.variables import DATA_PATH, MODEL_PATH
from src.evaluate import evaluate_model
from src.inference import load_model
from src.preprocessing import preprocess_data
from src.train import save_model, train_model


def main(data_path=DATA_PATH, output_dir=None):
    model_path = MODEL_PATH if output_dir is None else Path(output_dir) / "model.joblib"
    X_train, X_test, y_train, y_test, metadata = preprocess_data(data_path)
    model = train_model(X_train, y_train)
    save_model(model, model_path)
    metrics = metadata | evaluate_model(load_model(model_path), X_test, y_test)
    model_path.with_name("metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
