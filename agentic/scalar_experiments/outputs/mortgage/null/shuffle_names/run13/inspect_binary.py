import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')

binary_cols = []
for col in df.columns:
    vals = df[col].dropna().unique()
    if set(vals).issubset({0, 1}):
        binary_cols.append(col)

means = {col: df[col].mean() for col in binary_cols}

print('Binary columns and means:')
for col, mean in sorted(means.items(), key=lambda x: x[1]):
    print(f"{col}: {mean:.4f}")
