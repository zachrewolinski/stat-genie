import pandas as pd
import numpy as np

# Load data

df = pd.read_csv('soccer.csv')
print('shape', df.shape)
print('columns', df.columns.tolist())

# basic summary
summary = []
for col in df.columns:
    s = df[col]
    summary.append({
        'col': col,
        'dtype': s.dtype,
        'nunique': s.nunique(dropna=True),
        'min': s.min() if pd.api.types.is_numeric_dtype(s) else None,
        'max': s.max() if pd.api.types.is_numeric_dtype(s) else None,
        'mean': s.mean() if pd.api.types.is_numeric_dtype(s) else None,
        'std': s.std() if pd.api.types.is_numeric_dtype(s) else None,
    })
summary_df = pd.DataFrame(summary)
print(summary_df)

# Identify likely skin tone columns: numeric with small set of values between 0 and 1.
skin_candidates = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        uniq = sorted(s.dropna().unique())
        if len(uniq) <= 6 and len(uniq) > 1 and min(uniq) >= 0 and max(uniq) <= 1:
            skin_candidates.append((col, uniq))
print('skin_candidates', skin_candidates)

# Identify red card column: numeric count with many zeros, maybe small integer; also maybe called redCards.
# Print distributions for numeric integer-like columns.
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        if np.all(np.isclose(s.dropna() % 1, 0)):
            counts = s.value_counts().sort_index()
            print('\ninteger-like col', col)
            print(counts.head(10))
            print('max', s.max())
