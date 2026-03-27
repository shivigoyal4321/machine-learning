from typing import Any, Dict, List

import joblib
import pandas as pd
from flask import Flask, jsonify, request


MODEL_PATH = "breast_cancer_model.pkl"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_names: List[str] = artifact["feature_names"]
target_names: List[str] = artifact["target_names"]

app = Flask(__name__)


@app.get("/")
def root() -> Any:
    return jsonify(
        {
            "message": "Breast cancer prediction API is running.",
            "endpoints": ["/health", "/schema", "/predict"],
        }
    )


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/schema")
def schema() -> Any:
    return jsonify(
        {
            "required_features_count": len(feature_names),
            "required_features": feature_names,
            "target_names": target_names,
            "prediction_mapping": {"0": "malignant", "1": "benign"},
        }
    )


@app.post("/predict")
def predict() -> Any:
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    features = payload.get("features")

    if features is None:
        return (
            jsonify(
                {
                    "error": "Missing 'features' in JSON body.",
                    "expected": {
                        "features": {name: 0.0 for name in feature_names[:3]},
                        "note": "Include all required feature names with numeric values.",
                    },
                }
            ),
            400,
        )

    if not isinstance(features, dict):
        return jsonify({"error": "'features' must be an object/dictionary."}), 400

    missing = [f for f in feature_names if f not in features]
    if missing:
        return (
            jsonify(
                {
                    "error": "Missing required features.",
                    "missing_features": missing,
                    "required_features": feature_names,
                }
            ),
            400,
        )

    row = {name: float(features[name]) for name in feature_names}
    input_df = pd.DataFrame([row], columns=feature_names)

    pred_num = int(model.predict(input_df)[0])
    pred_label = target_names[pred_num]
    probabilities = model.predict_proba(input_df)[0]

    return jsonify(
        {
            "prediction": pred_label,
            "prediction_code": pred_num,
            "probabilities": {
                target_names[0]: float(probabilities[0]),
                target_names[1]: float(probabilities[1]),
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
