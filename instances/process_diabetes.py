import argparse
import pandas as pd

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Process the diabetes dataset.")
parser.add_argument("--normalize", action="store_true", help="Normalize numerical columns to [-1, 1]")
args = parser.parse_args()

# --- Load ---
df = pd.read_csv(
    "ba-krollmann/diabetes-normalized.csv",
    skipinitialspace=True,
)

# --- Keep relevant columns ---
# Note: max_glu_serum and A1Cresult are categorical (e.g. '>300', 'Norm') and are excluded
cols = [
    "gender", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3",
    "number_diagnoses",
]
df = df[cols]

# --- Map age intervals to numerical categories ---
age_map = {
    "[0-10)":   0,
    "[10-20)":  1,
    "[20-30)":  2,
    "[30-40)":  3,
    "[40-50)":  4,
    "[50-60)":  5,
    "[60-70)":  6,
    "[70-80)":  7,
    "[80-90)":  8,
    "[90-100)": 9,
}
df["age"] = df["age"].map(age_map)

# --- Encode gender column: Male -> 0, Female -> 1 (unknown values become NaN) ---
df["gender"] = df["gender"].str.strip().str.lower().map({"male": 0, "female": 1})

# --- Convert diag_1/2/3 to numeric, coercing non-numerical values to NaN ---
df["diag_1"] = pd.to_numeric(df["diag_1"], errors="coerce")
df["diag_2"] = pd.to_numeric(df["diag_2"], errors="coerce")
df["diag_3"] = pd.to_numeric(df["diag_3"], errors="coerce")

# --- Discard rows with any NaN values (catches unknown gender, bad diag codes, etc.) ---
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
print(f"Discarded {rows_before - rows_after} rows with non-numerical values ({rows_after} remaining)")

# --- Cast gender to int after NaNs are removed ---
df["gender"] = df["gender"].astype(int)

# --- Normalize all numerical columns to [-1, 1] ---
num_cols = [c for c in df.columns if c != "gender"]

if args.normalize:
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        df[col] = 2 * (df[col] - col_min) / (col_max - col_min) - 1

print(df.head())
print("\nData types:")
print(df.dtypes)
if args.normalize:
    print("\nValue range check:")
    print(df[num_cols].agg(["min", "max"]))

# --- Write to CSV ---
suffix = "-normalized" if args.normalize else ""
outfile = f"diabetes_processed{suffix}.csv"
df.to_csv(outfile, index=False)
print(f"\nWritten to {outfile}")
