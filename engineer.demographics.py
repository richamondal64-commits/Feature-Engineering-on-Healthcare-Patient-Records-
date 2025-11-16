def engineer_demographics(df):
    df["Age_Group"] = pd.cut(df["Age"], bins=[0,18,40,60,100], labels=["Child","Adult","Middle-aged","Senior"])
    return df
