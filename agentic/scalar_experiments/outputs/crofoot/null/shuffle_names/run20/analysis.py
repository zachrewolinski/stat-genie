import pandas as pd
import numpy as np

# Load data
_df = pd.read_csv('crofoot.csv')

print(_df.head())
print(_df.dtypes)

# Summary of min/max/unique counts
summary = []
for col in _df.columns:
    series = _df[col]
    summary.append({
        'col': col,
        'min': series.min(),
        'max': series.max(),
        'unique': series.nunique(),
        'mean': series.mean(),
        'std': series.std(),
    })

print(pd.DataFrame(summary).sort_values('unique'))

# Correlations with binary columns
binary_cols = [c for c in _df.columns if set(_df[c].unique()).issubset({0,1})]
print('binary cols', binary_cols)
if binary_cols:
    for b in binary_cols:
        print('corrs for', b)
        print(_df.corr(numeric_only=True)[b].sort_values())

