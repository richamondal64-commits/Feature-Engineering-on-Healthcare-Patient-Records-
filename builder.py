import pandas as pd
from config import CATEGORICAL_COLS, NUMERIC_COLS


def build_feature_table(df, encoded_df, scaled_df):
    # drop only columns that exist in the dataframe (avoid KeyError)
    cols_to_drop = [c for c in (CATEGORICAL_COLS + NUMERIC_COLS) if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    final_df = pd.concat([df.reset_index(drop=True), encoded_df, scaled_df], axis=1)
    return final_df
