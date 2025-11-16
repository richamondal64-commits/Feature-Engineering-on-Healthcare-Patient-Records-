from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from cleaner import clean_data
from engineer_demographics import engineer_demographics
from encoder import encode_categoricals
from scaler import scale_numerics
from config import CATEGORICAL_COLS, NUMERIC_COLS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and feature names saved by `train_model`
MODEL_PATH = "models/patient_model.pkl"
FEATURES_PATH = "models/feature_names.pkl"
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise RuntimeError("Model or feature names not found. Run `train_model()` first to create them.")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept raw patient data, apply preprocessing, and return readmission prediction.
    Expected input JSON: {Age, Gender, Condition, Medication, ...}
    """
    data = request.json or {}
    
    # Build a minimal DataFrame row with raw patient data
    # Map form field names to CSV column names
    raw_df = pd.DataFrame([{
        "Patient_ID": 999,
        "Age": data.get("Age", 50),
        "Gender": data.get("Gender", "Male"),
        "Condition": data.get("Condition", "Heart Disease"),
        "Medication": data.get("Treatment", "Angioplasty"),  # form uses "Treatment"
        "Stay_Length": data.get("Stay_Length", 5),
        "Total_Cost": data.get("Total_Cost", 10000),
        "Admission_Date": "2025-01-01",  # placeholder
        "Discharge_Date": "2025-01-06",  # placeholder
        "Patient_State": "Delhi",
        "Year_of_Admission": 2025,
        "Length_of_Stay": data.get("Stay_Length", 5),
        "Readmission": "No",
        "Outcome": "Recovered",
        "Satisfaction": 4,
        "Insurance_Claimed": "Yes",
    }])
    
    try:
        # Apply the same preprocessing steps as during training
        df = clean_data(raw_df)
        df = engineer_demographics(df)
        
        # Encode categorical columns
        encoded_df = encode_categoricals(df, CATEGORICAL_COLS)
        
        # Scale numeric columns
        scaled_df = scale_numerics(df, NUMERIC_COLS)
        
        # Drop the original categoricals and numerics, concat encoded+scaled
        cols_to_drop = [c for c in (CATEGORICAL_COLS + NUMERIC_COLS) if c in df.columns]
        df = df.drop(columns=cols_to_drop + ["Patient_ID"])
        
        # Combine all features
        final_df = pd.concat([df.reset_index(drop=True), encoded_df, scaled_df], axis=1)
        
        # Select only the features used during training, filling missing cols with 0
        for col in feature_names:
            if col not in final_df.columns:
                final_df[col] = 0
        final_df = final_df[feature_names]
        
        # Predict
        prediction = model.predict(final_df)[0]
        
        return jsonify({
            "readmission_prediction": int(prediction),
            "message": "Readmission Risk" if prediction == 1 else "No Readmission Risk"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)
