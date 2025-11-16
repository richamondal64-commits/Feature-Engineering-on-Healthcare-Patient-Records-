from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sys

app = Flask(__name__)
CORS(app)

# Load saved model and features
try:
    model = joblib.load("models/patient_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    print(f"✓ Model loaded. Features: {len(feature_names)}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "features_count": len(feature_names)})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204
    
    data = request.json or {}
    
    try:
        # Map input to expected feature names (fill missing with 0)
        row = {}
        for col in feature_names:
            # Default value 0 for missing encoded features
            row[col] = data.get(col, 0)
        
        df = pd.DataFrame([row])
        # Ensure all are numeric
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        
        # Predict
        prediction = model.predict(df)[0]
        
        return jsonify({
            "readmission_prediction": int(prediction),
            "message": "Readmission Risk" if prediction == 1 else "No Readmission Risk"
        })
    except Exception as e:
        return jsonify({"error": str(e), "message": "Prediction failed"}), 400

if __name__ == "__main__":
    print("Starting Flask app on http://0.0.0.0:3000")
    app.run(debug=False, host="0.0.0.0", port=3000, use_reloader=False)
