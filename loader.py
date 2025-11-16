import pandas as pd
from config import DATA_PATH, DATE_COLS

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=DATE_COLS)
    # strip whitespace in column names (some CSV headers have trailing spaces)
    df.columns = df.columns.str.strip()
    return df
