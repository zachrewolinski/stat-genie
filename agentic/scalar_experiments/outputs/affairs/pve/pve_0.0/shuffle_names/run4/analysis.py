import pandas as pd

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

print("columns:", df.columns.tolist())
print("shape:", df.shape)
print("dtypes:\n", df.dtypes)

# summary stats
print("\nunique counts:")
print(df.nunique())

print("\nhead:")
print(df.head())

# check unique values for low-cardinality columns
for col in df.columns:
    nun = df[col].nunique(dropna=False)
    if nun <= 10:
        print(f"\n{col} unique values:", sorted(df[col].dropna().unique()))

# quick correlation with potential affairs column: maybe one with many zeros? Not sure.

