import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())

summary = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        summary.append({
            'col': col,
            'dtype': 'num',
            'min': float(s.min()),
            'max': float(s.max()),
            'mean': float(s.mean()),
            'std': float(s.std()),
            'n_unique': int(s.nunique()),
            'pct_zero': float((s == 0).mean()),
        })
    else:
        summary.append({
            'col': col,
            'dtype': str(s.dtype),
            'n_unique': int(s.nunique()),
            'sample_vals': list(s.dropna().unique()[:3])
        })

summary_df = pd.DataFrame(summary)
print(summary_df)

skin_candidates = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        if s.min() >= 0 and s.max() <= 1 and s.nunique() <= 10:
            skin_candidates.append(col)
print('skin_candidates', skin_candidates)

red_candidates = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        if s.min() >= 0 and s.max() <= 10 and s.nunique() <= 11:
            red_candidates.append(col)
print('red_candidates', red_candidates)

# candidate games columns: integer counts with max between 10 and 80
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        if s.min() >= 0 and s.max() <= 100 and s.nunique() > 10:
            if np.all(np.mod(s, 1) == 0):
                if s.max() >= 10:
                    print('candidate_games', col, 'min', s.min(), 'max', s.max(), 'unique', s.nunique())
