import argparse
import pandas as pd

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Process the bank marketing dataset.")
parser.add_argument("--normalize", action="store_true", help="Normalize numerical columns to [-1, 1]")
args = parser.parse_args()

# --- Load ---
df = pd.read_csv(
    "ba-krollmann/bank-normalized.csv",
    skipinitialspace=True,
)

# --- Keep all numerical columns + marital ---
cols = ["marital", "age", "balance", "day", "duration", "campaign", "pdays", "previous"]
df = df[cols]

# --- Add married column based on marital status ---
df["married"] = df["marital"].str.strip().str.lower().map(
    {"married": 1, "single": 0, "divorced": 0}
).astype(int)

# --- Drop the original marital column ---
df = df.drop(columns=["marital"])

critical_cols = ["married"]

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
outfile = f"bank_processed{suffix}.csv"
df.to_csv(outfile, index=False)
print(f"\nWritten to {outfile}")
