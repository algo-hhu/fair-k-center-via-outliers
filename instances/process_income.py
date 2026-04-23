import argparse
import pandas as pd

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Process the ACS income dataset.")
parser.add_argument("--normalize", action="store_true", help="Normalize numerical columns to [-1, 1]")
args = parser.parse_args()

# --- Load ---
df = pd.read_csv("acs_income_2018_1-Year_person_CA_features.csv")

# --- Encode SEX column: 1.0 -> 0, 2.0 -> 1 ---
df["SEX"] = df["SEX"].map({1.0: 0, 2.0: 1}).astype(int)

critical_cols = ["SEX"]

# --- Normalize all numerical columns to [-1, 1] ---
num_cols = [c for c in df.columns if c not in critical_cols]

if args.normalize:
    for col in num_cols:
        col_centered = df[col] - df[col].mean()
        col_min = col_centered.min()
        col_max = col_centered.max()
        df[col] = 2 * (col_centered - col_min) / (col_max - col_min) - 1

# --- Write to CSV ---
suffix = "-normalized" if args.normalize else ""
outfile = f"acs_income_2018_1-Year_person_CA_features_processed{suffix}.csv"
df.to_csv(outfile, index=False)
print(f"Written to {outfile}")
