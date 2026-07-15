"""
Rising Waters — Flask web application.

Serves the flood-risk dashboard and a JSON prediction API backed by the
best-performing model saved by train_model.py.
"""
import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "flood_model.pkl")
SCALER_PATH = os.path.join(APP_DIR, "model", "scaler.pkl")
METADATA_PATH = os.path.join(APP_DIR, "model", "metadata.json")

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
NEEDS_SCALING = metadata["needs_scaling"]


def build_feature_vector(payload):
    """Pull required features from the request payload in the right order."""
    values = []
    for feat in FEATURES:
        if feat not in payload:
            raise ValueError(f"Missing required field: {feat}")
        values.append(float(payload[feat]))
    return np.array(values).reshape(1, -1)


def risk_band(probability):
    if probability >= 0.75:
        return "severe", "Severe flood risk — issue evacuation advisory"
    if probability >= 0.5:
        return "high", "High flood risk — activate monitoring & response teams"
    if probability >= 0.25:
        return "moderate", "Moderate risk — continue close monitoring"
    return "low", "Low risk — routine monitoring sufficient"


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_name=metadata["best_model"],
        results=metadata["results"],
        xgboost_available=metadata["xgboost_available"],
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    try:
        X = build_feature_vector(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    X_input = scaler.transform(X) if NEEDS_SCALING else X
    probability = float(model.predict_proba(X_input)[0, 1])
    prediction = int(probability >= 0.5)
    band, advisory = risk_band(probability)

    return jsonify({
        "region": payload.get("region", "Unspecified region"),
        "flood_predicted": bool(prediction),
        "probability_percent": round(probability * 100, 2),
        "risk_band": band,
        "advisory": advisory,
        "model_used": metadata["best_model"],
    })


@app.route("/api/model-info")
def model_info():
    return jsonify(metadata)


if __name__ == "__main__":
    # For local dev. On IBM Cloud (Cloud Foundry / Code Engine) this is
    # typically served via gunicorn — see README for deployment notes.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
