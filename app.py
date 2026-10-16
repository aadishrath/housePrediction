"""Flask API for status-conditioned house price estimates."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FEATURE_DIR = ROOT / "features"
FEATURES_PATH = FEATURE_DIR / "selected_features.json"
SCHEMA_PATH = FEATURE_DIR / "input_schema.json"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
METADATA_PATH = MODEL_DIR / "training_metadata.json"
GUIDE_PATH = FEATURE_DIR / "modeling_notes.json"
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": r"http://(localhost|127\.0\.0\.1)(:\d+)?"}})


def model_id(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", " ").split())


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


@lru_cache(maxsize=1)
def get_features() -> dict[str, list[str]]:
    return read_json(FEATURES_PATH, {})


@lru_cache(maxsize=1)
def get_schema() -> list[dict]:
    return read_json(SCHEMA_PATH, [])


@lru_cache(maxsize=1)
def get_metrics() -> dict:
    return read_json(METRICS_PATH, {})


@lru_cache(maxsize=1)
def get_metadata() -> dict:
    return read_json(METADATA_PATH, {})


@lru_cache(maxsize=1)
def get_feature_guide() -> dict:
    return read_json(GUIDE_PATH, {"included": [], "excluded": []})


@lru_cache(maxsize=None)
def load_model(key: str):
    path = MODEL_DIR / f"{key}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Model {path.name} is missing. Run python train_models.py first.")
    return joblib.load(path)


def available_models() -> list[dict]:
    models = []
    for label in get_features():
        key = model_id(label)
        if not (MODEL_DIR / f"{key}.pkl").is_file():
            continue
        try:
            load_model(key)
        except Exception:
            app.logger.exception("Skipping unreadable model artifact for %s", label)
            continue
        score = get_metrics().get(label, {})
        models.append({"id": key, "label": label, "metrics": {name: score.get(name) for name in (
            "mae", "median_ae", "mape", "rmse", "r2", "mean_error",
            "validation_band_radius_80", "validation_band_coverage", "training_device")}})
    models.sort(key=lambda model: model["metrics"]["mae"] if model["metrics"]["mae"] is not None else float("inf"))
    for i, model in enumerate(models):
        model["best"] = i == 0
    return models


def validate_features(values: dict, features: list[str]) -> tuple[dict | None, str | None]:
    schema = {field["id"]: field for field in get_schema()}
    status = str(values.get("status", "")).strip().lower()
    parsed = {}
    for feature in features:
        field = schema.get(feature)
        if field is None:
            return None, f"Input definition for {feature.replace('_', ' ')} is missing. Retrain the models."
        raw = values.get(feature)
        required = field.get("required", True)
        if field.get("required_unless") and status == field["required_unless"]:
            required = False
        if raw is None or isinstance(raw, bool) or str(raw).strip() == "":
            if required:
                return None, f"Enter a value for {field['label'].lower()}."
            parsed[feature] = np.nan if field["type"] != "text" else ""
            continue
        if field["type"] == "select":
            option = next((item for item in field.get("options", []) if str(item["value"]) == str(raw)), None)
            if option is None:
                return None, f"Choose a valid {field['label'].lower()}."
            parsed[feature] = str(option["value"])
        elif field["type"] == "text":
            value = str(raw).strip()
            if len(value) < field.get("min_length", 0) or len(value) > field.get("max_length", 999):
                return None, f"{field['label']} has an invalid length."
            if feature == "state":
                value = value.upper()
                if not value.isalpha():
                    return None, "State must be a two-letter abbreviation."
            if feature == "city":
                value = value.title()
            if feature == "zip_code" and (len(value) != 5 or not value.isdigit()):
                return None, "ZIP code must contain five digits."
            parsed[feature] = value
        else:
            try:
                number = float(raw)
            except (TypeError, ValueError):
                return None, f"{field['label']} must be a number."
            if not np.isfinite(number) or number < field["min"] or number > field["max"]:
                return None, f"{field['label']} must be between {field['min']:,} and {field['max']:,}."
            if field["type"] == "integer" and not number.is_integer():
                return None, f"{field['label']} must be a whole number."
            parsed[feature] = number
    return parsed, None


@app.get("/health")
def health():
    return jsonify({"status": "ok", "models_available": len(available_models())})


@app.get("/models")
def get_models():
    return jsonify(available_models())


@app.get("/features/<model>")
def get_model_features(model: str):
    key = model_id(model)
    label = next((name for name in get_features() if model_id(name) == key), None)
    fields = get_features().get(label) if label else None
    if not fields:
        return jsonify({"error": f"Model '{model}' is not available."}), 404
    definitions = {field["id"]: field for field in get_schema()}
    return jsonify({"fields": [definitions[name] for name in fields if name in definitions],
                    "metadata": get_metadata(), "raw_features": get_metadata().get("raw_columns", []),
                    "feature_guide": get_feature_guide()})


@app.get("/sample/<model>")
def get_sample_input(model: str):
    key = model_id(model)
    if key not in {model_id(label) for label in get_features()}:
        return jsonify({"error": f"Model '{model}' is not available."}), 404
    return jsonify({"status": "sold", "bed": 3, "bath": 2, "house_size": 1800,
                    "acre_lot": 0.15, "state": "CA", "city": "San Jose", "zip_code": "95112"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), dict):
        return jsonify({"error": "Send a JSON object with a model and property details."}), 400
    requested = payload.get("model")
    if not isinstance(requested, str):
        return jsonify({"error": "Choose a prediction model."}), 400
    key = model_id(requested)
    label = next((name for name in get_features() if model_id(name) == key), None)
    features = get_features().get(label) if label else None
    if not features or label not in {item["label"] for item in available_models()}:
        return jsonify({"error": f"Model '{requested}' is unavailable. Train the models and try again."}), 404
    values, error = validate_features(payload["features"], features)
    if error:
        return jsonify({"error": error}), 400
    try:
        model_input = pd.DataFrame([{feature: values[feature] for feature in features}], columns=features)
        estimate = max(0.0, float(load_model(key).predict(model_input)[0]))
        metrics = get_metrics().get(label, {})
        status = values["status"]
        status_metrics = metrics.get("by_status", {}).get(status, {})
        radius = float(status_metrics.get("validation_band_radius_80", metrics.get("validation_band_radius_80", 0)))
        return jsonify({"predicted_price": estimate, "estimated_low": max(0.0, estimate - radius),
                        "estimated_high": estimate + radius, "validation_band_radius": radius,
                        "validation_coverage": status_metrics.get("validation_band_coverage", metrics.get("validation_band_coverage", 0.8)),
                        "price_context": status, "model": label})
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "The estimate could not be calculated. Check the trained model files and try again."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
