import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())

summary = []
for col in df.columns:
    s = df[col]
    # try numeric
    if pd.api.types.is_numeric_dtype(s):
        summary.append({
            'col': col,
            'dtype': 'num',
            'unique': s.nunique(dropna=True),
            'min': s.min(),
            'max': s.max(),
            'mean': s.mean(),
            'std': s.std(),
            'zeros': int((s==0).sum())
        })
    else:
        summary.append({
            'col': col,
            'dtype': 'cat',
            'unique': s.nunique(dropna=True),
            'sample': s.dropna().astype(str).head(3).tolist()
        })

summary_df = pd.DataFrame(summary)
print('\nSummary (first 30 rows):')
print(summary_df)

# identify potential skin tone columns: numeric with 5 unique values between 0 and 1
candidates = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        uniq = sorted(s.dropna().unique())
        if len(uniq) <= 6 and min(uniq) >= 0 and max(uniq) <= 1:
            candidates.append((col, len(uniq), uniq[:10]))
print('\nSkin tone candidates (0-1 small unique):')
for c in candidates:
    print(c)

# identify potential card count columns: numeric with many zeros and small max maybe <=5
card_candidates = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        if s.max() <= 10 and s.min() >= 0:
            zeros = (s==0).sum()
            card_candidates.append((col, s.max(), s.mean(), zeros))

print('\nCard-ish candidates (max<=10):')
for c in sorted(card_candidates, key=lambda x: x[2]):
    print(c)

