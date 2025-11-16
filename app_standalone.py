#!/usr/bin/env python3
"""
Simple Flask app that:
1. Accepts raw patient data from HTML form
2. Applies min/max scaling based on hardcoded ranges from training data
3. Encodes categorical variables as one-hot
4. Returns prediction
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load model and feature names
try:
    model = joblib.load("models/patient_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    print(f"✓ Loaded model with {len(feature_names)} features")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# Hardcoded scaling ranges from training data
AGE_MIN, AGE_MAX = 25.0, 78.0
STAY_MIN, STAY_MAX = 1.0, 10.0
COST_MIN, COST_MAX = 100.0, 25000.0

# Valid category values
GENDERS = ["Female", "Male"]
CONDITIONS = ["Allergic Reaction", "Appendicitis", "Cancer", "Childbirth", "Diabetes",
              "Fractured Arm", "Fractured Leg", "Heart Attack", "Heart Disease",
              "Hypertension", "Kidney Stones", "Osteoarthritis", "Prostate Cancer",
              "Respiratory Infection", "Stroke"]
TREATMENTS = ["Angioplasty", "Antibiotics and Rest", "Appendectomy", 
              "CT Scan and Medication", "Cardiac Catheterization", "Cast and Physical Therapy",
              "Delivery and Postnatal Care", "Epinephrine Injection", "Insulin Therapy",
              "Lithotripsy", "Medication and Counseling", "Physical Therapy and Pain Management",
              "Radiation Therapy", "Surgery and Chemotherapy", "X-Ray and Splint"]

def scale_value(value, min_val, max_val):
    """Min-max scaling: (x - min) / (max - min)"""
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204
    
    data = request.json or {}
    
    try:
        # Get input values
        age = float(data.get("Age", 50))
        stay_length = float(data.get("Stay_Length", 5))
        total_cost = float(data.get("Total_Cost", 10000))
        gender = str(data.get("Gender", "Male")).strip()
        condition = str(data.get("Condition", "Heart Disease")).strip()
        treatment = str(data.get("Treatment", "Angioplasty")).strip()
        
        # Scale numeric features
        age_scaled = scale_value(age, AGE_MIN, AGE_MAX)
        stay_scaled = scale_value(stay_length, STAY_MIN, STAY_MAX)
        cost_scaled = scale_value(total_cost, COST_MIN, COST_MAX)
        
        # Build feature dict with all one-hot encoded columns
        features = {}
        
        # Add numeric features (scaled)
        features["Age"] = age_scaled
        features["Stay_Length"] = stay_scaled
        features["Total_Cost"] = cost_scaled
        
        # One-hot encode gender
        for g in GENDERS:
            features[f"Gender_{g}"] = 1.0 if gender == g else 0.0
        
        # One-hot encode condition
        for c in CONDITIONS:
            features[f"Condition_{c}"] = 1.0 if condition == c else 0.0
        
        # One-hot encode treatment
        for t in TREATMENTS:
            features[f"Medication_{t}"] = 1.0 if treatment == t else 0.0
        
        # Also add Age_Group (placeholder)
        features["Age_Group"] = 0.0  # Not used in numeric features
        
        # Create dataframe with only the features used during training
        df = pd.DataFrame([features])
        
        # Ensure all training features are present (fill missing with 0)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        
        # Select only features used during training in correct order
        df = df[feature_names]
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return jsonify({
            "readmission_prediction": int(prediction),
            "message": "⚠️ Readmission Risk" if prediction == 1 else "✓ No Readmission Risk"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("Starting Flask app on http://0.0.0.0:8080")
    app.run(debug=False, host="0.0.0.0", port=8080, use_reloader=False)
