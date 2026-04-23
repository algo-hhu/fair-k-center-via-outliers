import argparse
import pandas as pd

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Process the census/adult dataset.")
parser.add_argument("--normalize", action="store_true", help="Normalize numerical columns to [-1, 1]")
args = parser.parse_args()

# --- Load ---
df = pd.read_csv(
    "ba-krollmann/census-normalized.csv",
    skipinitialspace=True,  # strips leading spaces after commas
)
df["row_id"] = df.index

# --- Keep all numerical columns + sex ---
cols = ["row_id", "sex", "race", "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
df = df[cols]

critical_cols = ["row_id", "sex", "race"]

# --- Encode sex column: Male -> 0, Female -> 1 ---
df["sex"] = df["sex"].str.strip().str.lower().map({"male": 0, "female": 1}).astype(int)

# --- Encode race column ---
df["race"] = df["race"].str.strip().str.lower().map({
    "other": 0,
    "amer-indian-eskimo": 1,
    "asian-pac-islander": 2,
    "black": 3,
    "white": 4
}).astype(int)


# --- Discard rows with any NaN values ---
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
print(f"Discarded {rows_before - rows_after} rows with non-numerical values ({rows_after} remaining)")

# --- Normalize all numerical columns to [-1, 1] ---
num_cols = [c for c in df.columns if c not in critical_cols]

if args.normalize:
    for col in num_cols:
        col_centered = df[col] - df[col].mean()
        col_min = col_centered.min()
        col_max = col_centered.max()
        df[col] = 2 * (col_centered - col_min) / (col_max - col_min) - 1

print(df.head())
print("\nData types:")
print(df.dtypes)
if args.normalize:
    print("\nValue range check:")
    print(df[num_cols].agg(["min", "max"]))

# --- Write to CSV ---
suffix = "-normalized" if args.normalize else ""
outfile = f"census_processed{suffix}.csv"
df.to_csv(outfile, index=False)
print(f"\nWritten to {outfile}")
