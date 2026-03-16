import pandas as pd
import numpy as np

path = 'reading.csv'

df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print('\nhead')
print(df.head())

# basic summary
summary = []
for col in df.columns:
    s = df[col]
    dtype = s.dtype
    nunique = s.nunique(dropna=True)
    sample = s.dropna().unique()[:5]
    if pd.api.types.is_numeric_dtype(s):
        summary.append((col, str(dtype), nunique, float(s.min()), float(s.max()), float(s.mean()), float(s.std())))
    else:
        summary.append((col, str(dtype), nunique, sample.tolist()))

print('\nsummary')
for item in summary:
    print(item)

# identify binary columns
print('\nBinary-ish columns (unique values <=3):')
for col in df.columns:
    nunique = df[col].nunique(dropna=True)
    if nunique <= 3:
        print(col, df[col].value_counts(dropna=False).head(10).to_dict())

