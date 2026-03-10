import pandas as pd
import numpy as np

_df = pd.read_csv('soccer.csv')

# compute basic stats for numeric columns
num_cols = _df.select_dtypes(include=[np.number]).columns
stats = []
for c in num_cols:
    s = _df[c]
    stats.append((c, s.min(), s.max(), s.mean(), s.std()))

print('numeric columns:', num_cols.tolist())
print('stats:')
for c, mn, mx, mean, std in stats:
    print(f"{c}: min={mn}, max={mx}, mean={mean}, std={std}")

print('\nvalue counts for rater1, rater2, nExp')
for c in ['rater1','rater2','nExp']:
    if c in _df.columns:
        print(c, _df[c].value_counts().head(10))

print('\nredCards summary')
if 'redCards' in _df.columns:
    print(_df['redCards'].describe())

print('\nyellowCards summary')
if 'yellowCards' in _df.columns:
    print(_df['yellowCards'].describe())
