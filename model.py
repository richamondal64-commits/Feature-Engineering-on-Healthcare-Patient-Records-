import numpy as np
import pandas as pd
from config import TARGET_COL
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_model(data_path="data/engineered_features.csv"):
    """Train a Logistic Regression model on the engineered dataset.

    This function is defensive: it detects the target column name, converts
    common string labels (e.g. 'Yes'/'No') to numeric, selects numeric
    features only, and trains a logistic regression.
    """
    df = pd.read_csv(data_path)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Determine target column
    target_candidates = [TARGET_COL, "Readmission", "Readmission_Flag", "ReadmissionFlag", "Readmission_Flag "]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"No target column found. Checked: {target_candidates}")

    # Extract target and convert common string labels to numeric
    y = df[target_col]
    if y.dtype == object or pd.api.types.is_categorical_dtype(y):
        # map common yes/no labels
        mapping = {"Yes": 1, "No": 0, "Y": 1, "N": 0, "yes": 1, "no": 0}
        y_mapped = y.map(mapping)
        if y_mapped.isnull().all():
            # fallback to label encoding
            y_mapped, _ = pd.factorize(y)
        y = y_mapped

    # Drop identifier and target columns from features
    X = df.drop(columns=[c for c in ["Patient_ID", target_col] if c in df.columns])

    # Keep numeric features only (LogisticRegression requires numeric input)
    X_numeric = X.select_dtypes(include=[np.number]).copy()

    if X_numeric.shape[1] == 0:
        raise ValueError("No numeric features available for training. Check engineered features.")

    # Ensure no missing labels
    mask = ~pd.isna(y)
    X_numeric = X_numeric.loc[mask]
    y = y.loc[mask].astype(int)

    # Verify we have at least two classes
    if y.nunique() < 2:
        raise ValueError("Target has fewer than 2 classes — can't train classifier.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression (increase iterations to reduce convergence warnings)
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model and feature names for serving
    try:
        import joblib, os
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/patient_model.pkl")
        joblib.dump(list(X_numeric.columns), "models/feature_names.pkl")
        print("Saved model to models/patient_model.pkl and feature names to models/feature_names.pkl")
    except Exception:
        pass

    return model
