from typing import Any, Dict, List

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request


MODEL_PATH = "breast_cancer_model.pkl"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_names: List[str] = artifact["feature_names"]
target_names: List[str] = artifact["target_names"]

app = Flask(__name__)

UI_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Breast Cancer Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f7f9fc; color: #1f2937; }
    .wrap { max-width: 1100px; margin: 0 auto; }
    h1 { margin-bottom: 8px; }
    p { margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }
    .field { background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 10px; }
    label { font-size: 12px; display: block; margin-bottom: 6px; color: #374151; }
    input { width: 100%; padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; }
    .actions { margin-top: 16px; display: flex; gap: 10px; }
    button { padding: 10px 14px; border: 0; border-radius: 8px; cursor: pointer; }
    .primary { background: #2563eb; color: #fff; }
    .secondary { background: #e5e7eb; color: #111827; }
    .result, .error { margin-top: 16px; padding: 12px; border-radius: 8px; background: #fff; border: 1px solid #d1d5db; }
    .error { border-color: #ef4444; color: #b91c1c; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Breast Cancer Prediction</h1>
    <p>Fill all feature values and click <code>Predict</code>.</p>
    <form id="predict-form">
      <div class="grid" id="fields"></div>
      <div class="actions">
        <button class="primary" type="submit">Predict</button>
        <button class="secondary" type="button" id="fill-defaults">Fill Sample Values</button>
      </div>
    </form>
    <div id="result" class="result" style="display:none;"></div>
    <div id="error" class="error" style="display:none;"></div>
  </div>

  <script>
    const form = document.getElementById("predict-form");
    const fieldsContainer = document.getElementById("fields");
    const resultBox = document.getElementById("result");
    const errorBox = document.getElementById("error");
    const fillDefaultsBtn = document.getElementById("fill-defaults");
    let requiredFeatures = [];

    const sampleValues = {
      "mean radius": 14.6, "mean texture": 22.7, "mean perimeter": 96.4, "mean area": 657,
      "mean smoothness": 0.085, "mean compactness": 0.133, "mean concavity": 0.103, "mean concave points": 0.04,
      "mean symmetry": 0.1654, "mean fractal dimension": 0.05147, "radius error": 0.3354, "texture error": 1.108,
      "perimeter error": 2.244, "area error": 19.74, "smoothness error": 0.004342, "compactness error": 0.04649,
      "concavity error": 0.06578, "concave points error": 0.01506, "symmetry error": 0.01738, "fractal dimension error": 0.00454,
      "worst radius": 13.48, "worst texture": 37.27, "worst perimeter": 105.9, "worst area": 734.5,
      "worst smoothness": 0.1206, "worst compactness": 0.317, "worst concavity": 0.3682, "worst concave points": 0.1305,
      "worst symmetry": 0.2348, "worst fractal dimension": 0.08004
    };

    function showError(msg) {
      errorBox.textContent = msg;
      errorBox.style.display = "block";
      resultBox.style.display = "none";
    }

    function showResult(data) {
      resultBox.innerHTML = `
        <strong>Prediction:</strong> ${data.prediction}<br/>
        <strong>Prediction Code:</strong> ${data.prediction_code}<br/>
        <strong>Probabilities:</strong><br/>
        - malignant: ${Number(data.probabilities.malignant).toFixed(4)}<br/>
        - benign: ${Number(data.probabilities.benign).toFixed(4)}
      `;
      resultBox.style.display = "block";
      errorBox.style.display = "none";
    }

    async function loadSchema() {
      const res = await fetch("/schema");
      const schema = await res.json();
      requiredFeatures = schema.required_features;
      fieldsContainer.innerHTML = "";
      for (const name of requiredFeatures) {
        const div = document.createElement("div");
        div.className = "field";
        div.innerHTML = `
          <label for="${name}">${name}</label>
          <input type="number" id="${name}" name="${name}" step="any" required />
        `;
        fieldsContainer.appendChild(div);
      }
    }

    fillDefaultsBtn.addEventListener("click", () => {
      for (const name of requiredFeatures) {
        const input = document.getElementById(name);
        if (input && sampleValues[name] !== undefined) input.value = sampleValues[name];
      }
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const features = {};
      for (const name of requiredFeatures) {
        const value = document.getElementById(name)?.value;
        if (value === "" || value === undefined) {
          showError(`Please enter a value for: ${name}`);
          return;
        }
        features[name] = Number(value);
      }

      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features })
      });
      const data = await res.json();
      if (!res.ok) {
        showError(data.error || "Prediction failed.");
        return;
      }
      showResult(data);
    });

    loadSchema().catch(() => showError("Failed to load schema."));
  </script>
</body>
</html>
"""


@app.get("/")
def root() -> Any:
    return jsonify(
        {
            "message": "Breast cancer prediction API is running.",
            "endpoints": ["/health", "/schema", "/predict"],
        }
    )

@app.get("/ui")
def ui() -> Any:
    return render_template_string(UI_HTML)


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
