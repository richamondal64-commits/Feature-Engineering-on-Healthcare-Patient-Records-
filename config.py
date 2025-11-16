DATA_PATH = "Hospital Patient.csv"
DATE_COLS = ["Admission_Date", "Discharge_Date"]
# Match the actual CSV headers (whitespace is stripped during load)
CATEGORICAL_COLS = ["Gender", "Condition", "Medication"]
NUMERIC_COLS = ["Age", "Stay_Length", "Total_Cost"]
TARGET_COL = "Readmission"
