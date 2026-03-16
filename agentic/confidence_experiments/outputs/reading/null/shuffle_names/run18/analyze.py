import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())

# summary of each column: dtype, unique count, sample values
summary = []
for col in df.columns:
    series = df[col]
    n_unique = series.nunique(dropna=True)
    sample = series.dropna().unique()[:5]
    summary.append((col, series.dtype, n_unique, sample))

for col, dtype, n_unique, sample in summary:
    print(f"\n{col}\n dtype={dtype} n_unique={n_unique} sample={sample}")

# Identify likely binary columns
print('\nBinary-like columns:')
for col in df.columns:
    vals = sorted(df[col].dropna().unique())
    if len(vals) <= 5:
        print(col, vals)

# Check numeric columns stats
print('\nNumeric stats:')
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        print(col, df[col].describe())
