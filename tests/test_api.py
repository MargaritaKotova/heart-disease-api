import joblib
import pandas as pd
import pytest

from app.main import MODEL_PATH, create_app
from app.schema import FEATURES
from test_request import EXAMPLE


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_prediction_matches_model_and_ignores_parameter_order(client):
    model = joblib.load(MODEL_PATH)
    expected = int(model.predict(pd.DataFrame([EXAMPLE], columns=FEATURES))[0])
    for params in (EXAMPLE, dict(reversed(list(EXAMPLE.items())))):
        response = client.get("/predict", query_string=params)
        assert response.status_code == 200
        assert response.get_json() == {"target": expected}


@pytest.mark.parametrize("change", [
    {"age": -1}, {"sex": 2}, {"oldpeak": "nan"}, {"chol": "inf"},
    {"cp": 4}, {"thal": 4}, {"age": "invalid"}, {"unexpected": 1},
])
def test_invalid_features(client, change):
    assert client.get("/predict", query_string=EXAMPLE | change).status_code == 422


def test_missing_features(client):
    assert client.get("/predict", query_string={"age": 63}).status_code == 422


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok", "model_loaded": True}
    assert client.get("/").get_json() == {"status": "ok", "model_loaded": True}


def test_missing_model_prevents_startup(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.MODEL_PATH", tmp_path / "missing.joblib")
    with pytest.raises(FileNotFoundError):
        create_app()


def test_duplicate_parameters_are_rejected(client):
    params = list(EXAMPLE.items()) + [("age", 50)]
    response = client.get("/predict", query_string=params)
    assert response.status_code == 422
    assert response.get_json()["detail"][0]["type"] == "duplicate_parameter"


def test_validation_errors_are_json(client):
    response = client.get("/predict", query_string=EXAMPLE | {"oldpeak": "nan"})
    assert response.status_code == 422
    assert response.is_json
    assert response.get_json()["detail"][0]["loc"] == ["query", "oldpeak"]


def test_post_matches_get_and_model(client):
    model = joblib.load(MODEL_PATH)
    expected = int(model.predict(pd.DataFrame([EXAMPLE], columns=FEATURES))[0])
    response = client.post("/predict", json=dict(reversed(list(EXAMPLE.items()))))
    assert response.status_code == 200
    assert response.get_json() == {"target": expected}
    assert response.get_json() == client.get("/predict", query_string=EXAMPLE).get_json()


@pytest.mark.parametrize("payload", [
    {}, {"age": 63}, EXAMPLE | {"sex": 2}, EXAMPLE | {"extra": 1},
    EXAMPLE | {"oldpeak": "nan"}, [], None,
])
def test_post_invalid_payload(client, payload):
    import json

    response = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 422
    assert response.is_json
    assert response.get_json()["detail"][0]["loc"][0] == "body"


def test_post_requires_valid_json(client):
    assert client.post("/predict", data="not json").status_code == 415
    response = client.post("/predict", data="{", content_type="application/json")
    assert response.status_code == 400
    assert response.is_json
