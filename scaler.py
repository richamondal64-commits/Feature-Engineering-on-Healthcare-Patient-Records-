import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_numerics(df, cols):
    # only scale numeric columns that exist
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols])
    scaled_df = pd.DataFrame(scaled, columns=cols)
    return scaled_df
