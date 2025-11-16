import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_categoricals(df, cols):
    # only use columns that actually exist
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    # scikit-learn changed `sparse` -> `sparse_output` in newer versions
    try:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cols))
    return encoded_df
