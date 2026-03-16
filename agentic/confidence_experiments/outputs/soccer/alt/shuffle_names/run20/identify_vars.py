import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
num_cols = df.select_dtypes(include=[np.number]).columns

stats = []
for c in num_cols:
    s = df[c]
    zero_prop = (s == 0).mean() if pd.api.types.is_numeric_dtype(s) else np.nan
    stats.append((c, s.min(), s.max(), s.mean(), zero_prop, s.nunique()))

stats_sorted = sorted(stats, key=lambda x: (-x[4], x[2]))
print('Top zero proportion columns:')
for c, mn, mx, mean, zp, uniq in stats_sorted[:10]:
    print(c, 'min', mn, 'max', mx, 'mean', mean, 'zero_prop', zp, 'uniq', uniq)

# Show value counts for candidate small count columns
candidates = [c for c in num_cols if df[c].max() <= 20 and df[c].min() >= 0]
print('\nCandidates (max<=20):', candidates)
for c in candidates:
    counts = df[c].value_counts().sort_index().head(10)
    print('\n', c, 'value_counts head')
    print(counts)
