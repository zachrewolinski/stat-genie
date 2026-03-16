import pandas as pd
import numpy as np


df = pd.read_csv('soccer.csv')

# candidate columns for red cards: integer-like with very low mean
candidates = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        if np.all(np.isclose(s.dropna() % 1, 0)):
            mean = s.mean()
            zero_frac = (s == 0).mean()
            maxv = s.max()
            candidates.append((col, mean, zero_frac, maxv))

candidates_sorted = sorted(candidates, key=lambda x: x[1])
print('integer-like columns sorted by mean:')
for col, mean, zero_frac, maxv in candidates_sorted:
    print(f"{col:12s} mean={mean:.4f} zero_frac={zero_frac:.3f} max={maxv}")

# Show value counts for low-mean candidates
for col, mean, zero_frac, maxv in candidates_sorted[:6]:
    print(f"\n{col} value counts (top 10):")
    print(df[col].value_counts().sort_index().head(10))

# Find column with min >=1 and larger mean (games candidate)
print('\nmin>=1 integer-like columns:')
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and np.all(np.isclose(s.dropna() % 1, 0)):
        if s.min() >= 1:
            print(col, 'min', s.min(), 'mean', s.mean(), 'max', s.max())

# Correlation with games candidate (assuming redCards column is games) for rare columns
if 'redCards' in df.columns:
    games = df['redCards']
    print('\ncorrelations with redCards (games) for rare columns:')
    for col, mean, zero_frac, maxv in candidates_sorted[:8]:
        if col != 'redCards':
            corr = np.corrcoef(games, df[col])[0,1]
            print(col, 'corr', corr)

# check if counts <= games (redCards) for candidates
if 'redCards' in df.columns:
    games = df['redCards']
    print('\nproportion <= games (redCards)')
    for col, mean, zero_frac, maxv in candidates_sorted:
        if col != 'redCards':
            prop = (df[col] <= games).mean()
            print(col, prop)
