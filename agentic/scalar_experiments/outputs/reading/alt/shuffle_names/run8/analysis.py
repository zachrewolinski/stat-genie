import pandas as pd
import numpy as np

path = 'reading.csv'

df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:', df.dtypes)

# summary of unique counts and sample values per column
summary = []
for col in df.columns:
    s = df[col]
    uniq = s.nunique(dropna=True)
    sample_vals = s.dropna().unique()[:5]
    summary.append((col, s.dtype, uniq, sample_vals))

print('\nSummary:')
for col, dtype, uniq, sample_vals in summary:
    print(f"{col}: dtype={dtype}, unique={uniq}, samples={sample_vals}")

