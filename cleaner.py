import pandas as pd


def clean_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)
    if "Satisfaction_Rating" in df.columns:
        df["Satisfaction_Rating"].fillna(df["Satisfaction_Rating"].median(), inplace=True)
    # compute stay length only if both date columns exist
    # ensure date columns are datetimes
    if "Admission_Date" in df.columns:
        df["Admission_Date"] = pd.to_datetime(df["Admission_Date"], dayfirst=True, errors="coerce")
    if "Discharge_Date" in df.columns:
        df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"], dayfirst=True, errors="coerce")

    # If dataset already has a length column, use it; otherwise compute from dates
    if "Length_of_Stay" in df.columns:
        df["Stay_Length"] = pd.to_numeric(df["Length_of_Stay"], errors="coerce")
    elif "Discharge_Date" in df.columns and "Admission_Date" in df.columns:
        df["Stay_Length"] = (df["Discharge_Date"] - df["Admission_Date"]).dt.days
    else:
        df["Stay_Length"] = pd.NA
    return df
