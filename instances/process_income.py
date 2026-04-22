import pandas as pd

# --- Load ---
df = pd.read_csv("acs_income_2018_1-Year_person_CA_features.csv")

# --- Encode SEX column: 1.0 -> 0, 2.0 -> 1 ---
df["SEX"] = df["SEX"].map({1.0: 0, 2.0: 1}).astype(int)

# --- Write to CSV ---
df.to_csv("acs_income_2018_1-Year_person_CA_features_processed.csv", index=False)
print("Written to acs_income_2018_1-Year_person_CA_features_processed.csv")
