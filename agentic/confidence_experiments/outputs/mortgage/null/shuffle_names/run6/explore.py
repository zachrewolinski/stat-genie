import pandas as pd
import numpy as np

_df = pd.read_csv('mortgage.csv')

# identify binary columns
binary_cols = []
for c in _df.columns:
    vals = _df[c].dropna().unique()
    if set(vals).issubset({0,1}):
        binary_cols.append(c)

print('Binary columns and proportion of 1s:')
for c in binary_cols:
    prop = _df[c].mean()
    print(f"{c}: mean={prop:.3f} count1={_df[c].sum()} n={_df[c].notna().sum()}")

# check correlations between binary columns to see complement pairs
print('\nCorrelation matrix (binary columns):')
print(_df[binary_cols].corr())

# check unique counts for non-binary columns to match plausible gender? not needed
