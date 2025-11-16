import joblib

def save_model(model, path="models/patient_model.pkl"):
    joblib.dump(model, path)
    print(f"✅ Model saved at {path}")

def load_model(path="models/patient_model.pkl"):
    model = joblib.load(path)
    print("✅ Model loaded successfully")
    return model
