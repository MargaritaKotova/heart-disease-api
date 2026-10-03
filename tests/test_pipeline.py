import pandas as pd
import pytest

from app.schema import FEATURES
from src.inference import load_model, predict
from src.pipeline import main
from src.preprocessing import preprocess_data


def test_split_has_no_duplicate_leakage():
    X_train, X_test, y_train, y_test, metadata = preprocess_data()
    train_rows = set(map(tuple, X_train.to_numpy()))
    test_rows = set(map(tuple, X_test.to_numpy()))
    assert train_rows.isdisjoint(test_rows)
    assert set(y_train) == set(y_test) == {0, 1}
    assert metadata["duplicates_removed"] > 0


def test_pipeline_saves_loadable_model_and_metrics(tmp_path):
    metrics = main(output_dir=tmp_path)
    model = load_model(tmp_path / "model.joblib")
    X_train, *_ = preprocess_data()
    assert list(model.feature_names_in_) == FEATURES
    assert predict(model, X_train.iloc[:2]).shape == (2,)
    assert (tmp_path / "metrics.json").exists()
    assert 0 <= metrics["accuracy"] <= 1


def test_conflicting_labels_are_rejected(tmp_path):
    X_train, *_ = preprocess_data()
    row = X_train.iloc[[0]].copy()
    first = row.assign(target=0)
    second = row.assign(target=1)
    path = tmp_path / "conflicting.csv"
    pd.concat([first, second]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Conflicting labels"):
        preprocess_data(path)
