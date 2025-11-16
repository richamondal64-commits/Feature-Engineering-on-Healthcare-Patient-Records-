from loader import load_data
from cleaner import clean_data
from engineer_demographics import engineer_demographics
from encoder import encode_categoricals
from scaler import scale_numerics
from builder import build_feature_table
from config import CATEGORICAL_COLS, NUMERIC_COLS

def main():
	df = load_data()
	df = clean_data(df)
	df = engineer_demographics(df)

	encoded_df = encode_categoricals(df, CATEGORICAL_COLS)
	scaled_df = scale_numerics(df, NUMERIC_COLS)

	final_df = build_feature_table(df, encoded_df, scaled_df)
	final_df.to_csv("data/engineered_features.csv", index=False)


if __name__ == "__main__":
	main()
