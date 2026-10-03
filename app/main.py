"""Flask application factory and prediction routes."""
import pandas as pd
from flask import Flask, current_app, jsonify, request
from pydantic import ValidationError
from werkzeug.exceptions import BadRequest

from app.schema import FEATURES, HeartFeatures, Prediction
from config.variables import MODEL_PATH
from src.inference import load_model, predict as predict_model


def create_app(model_path=None):
    application = Flask(__name__)
    application.config["MODEL"] = load_model(MODEL_PATH if model_path is None else model_path)

    @application.get("/")
    @application.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=current_app.config["MODEL"] is not None)

    @application.route("/predict", methods=["GET", "POST"])
    def predict():
        location = "body" if request.method == "POST" else "query"
        if request.method == "POST":
            if not request.is_json:
                return jsonify(detail="Content-Type must be application/json"), 415
            try:
                data = request.get_json()
            except BadRequest:
                return jsonify(detail="Invalid JSON body"), 400
        else:
            data = request.args.to_dict()
        repeated = [key for key in request.args if len(request.args.getlist(key)) > 1] if request.method == "GET" else []
        if repeated:
            return jsonify(detail=[{
                "loc": ["query", key], "msg": "Parameter must occur once",
                "type": "duplicate_parameter",
            } for key in repeated]), 422
        try:
            features = HeartFeatures.model_validate(data)
        except ValidationError as error:
            return jsonify(detail=[{
                "loc": [location, *item["loc"]],
                "msg": item["msg"], "type": item["type"],
            } for item in error.errors(include_url=False)]), 422
        frame = pd.DataFrame([features.model_dump()], columns=FEATURES)
        target = int(predict_model(current_app.config["MODEL"], frame)[0])
        return jsonify(Prediction(target=target).model_dump())

    return application
