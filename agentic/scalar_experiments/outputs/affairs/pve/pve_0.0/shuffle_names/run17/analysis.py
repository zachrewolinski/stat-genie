import pandas as pd
import json
import numpy as np

df = pd.read_csv('affairs.csv')

print(df.head())
print(df.dtypes)

summary = {}
for col in df.columns:
    series = df[col]
    # convert min/max to python scalar
    min_val = series.min()
    max_val = series.max()
    if isinstance(min_val, (np.generic,)):
        min_val = min_val.item()
    if isinstance(max_val, (np.generic,)):
        max_val = max_val.item()
    sample_vals = series.dropna().unique()[:10]
    sample_vals = [v.item() if isinstance(v, np.generic) else v for v in sample_vals]
    summary[col] = {
        'dtype': str(series.dtype),
        'nunique': int(series.nunique(dropna=True)),
        'min': min_val,
        'max': max_val,
        'sample_values': sample_vals,
    }

print(json.dumps(summary, indent=2))

for col in df.columns:
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) <= 6:
        sorted_vals = sorted([v.item() if isinstance(v, np.generic) else v for v in unique_vals])
        print(col, sorted_vals)
